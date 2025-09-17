# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #
#        Comparing 2 Nano-Versions          #
#           Author: Gourab Saha             #
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

import os
import sys
import yaml
import dask
import time
import uproot
import awkward as ak
import numpy as np
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import mplhep as hep
hep.style.use("CMS")

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from dask.distributed import Client, LocalCluster
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed

import warnings
warnings.filterwarnings("ignore", message="Numba extension module 'awkward.numba'")
warnings.filterwarnings("ignore", module="coffea.*")


NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.version = "15"

def load_config(cfgfile):
    # load config file
    config = None
    with open(cfgfile, 'r') as f:
        config = yaml.safe_load(f)
    return config

def access_files_from_dataset(dataset):
    # get files from dataset
    files = []
    cmd = [
        "dasgoclient",
        f"-query=file dataset={dataset}",
        "system=rucio"
    ]
    cmdresult = subprocess.run(cmd, capture_output=True, text=True)
    if cmdresult.returncode == 0:
        files = cmdresult.stdout.strip().splitlines()
        #print("Found files :)")
    else:
        print("Error running dasgoclient :(", cmdresult.stderr)
        
    if len(files) == 0:
        raise RuntimeError("No files found ==> Check dataset name or XrootD operations :(")

    # adding xrdcp redirector for convenince
    files = [f"root://cmsxrootd.fnal.gov/{file}" for file in files]
        
    return files

def check_cols(events, cols):
    fields = np.array(events.fields)
    cols_retun = np.intersect1d(fields, np.array(cols)).to_list()

    if len(cols_retun) != len(cols):
        col_ext = [col for col in cols if col not in cols_retun]
        print(f"Columns {col_ext} not found")

    return cols_retun

def apply_selection(events, obj, field, cut):
    if field not in events[obj].fields:
        raise RuntimeError(f"{field} not found in {events[obj]} fields")
    
    mask = eval(
        cut,
        {"__builtins__": {"abs": abs}},
        {field : events[obj][field]})
    return mask

    
def get_selection_mask(events, selections):
    mask_return = None
    for i,(field,cut) in enumerate(selections.items()):
        fields = field.split('-')
        mask = apply_selection(events, fields[0], fields[1], cut)
        mask_return = mask if i == 0 else mask & mask_return

    return mask_return


def event_selection(events, selections):
    trigger_mask = get_selection_mask(events, selections['trigger'])

    tau_mask = get_selection_mask(events, selections['tau'])
    taus = events.Tau[tau_mask]
    events["Tau"] = taus

    nTau_mask = ak.sum(tau_mask, axis=1) >= 1
    event_mask = trigger_mask & nTau_mask

    return events[event_mask]


def dict_to_plot(vars):
    return {var: ak.Array([]) for var in vars}


def __basic_settings():
    return {"size": (8.9, 6.5),
            "heplogo": "Internal",
            "logoloc": 0,
            "histtype": "step",
            "linewidth": 2.0,
            "marker": "o",
            "markersize": 1.2,
            "capsize": 0.2,
            "ylim": None,
            "xlim": None,
            "markeredgewidth": 1.5,
            "markerstyles": ['o', 's', 'D', '*', '^', 'v', 'P', 'X', '<', '>'],
            "colors": ["#165a86","#cc660b","#217a21","#a31e1f","#6e4e92","#6b443e","#b85fa0","#666666","#96971b","#1294a6","#8c1c62","#144d3a"],
            "linestyles": ["-","--","-.",":","(0, (3, 1))", "(0, (3, 1, 1, 1))","(0, (5, 5))","(0, (1, 1))","(0, (6, 2))","(0, (4, 2, 1, 2))"]}


def doplot():
    pass

def dohist(data1, data2,
           leg1, leg2,
           bins,
           xlabel,
           title,
           outdir,
           **kwargs):

    logy = kwargs.get('logy', False)
    
    basics = __basic_settings()
    fig, (ax, rax) = plt.subplots(
        2, 1, 
        figsize=basics["size"], 
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}, 
        sharex=True
    )

    # --- Top panel
    hep.cms.text(basics["heplogo"], loc=basics["logoloc"], ax=ax)  # CMS label

    binning = np.linspace(bins[1], bins[2], bins[0])

    hist_vals = []
    hist_errs = []
    bin_edges = None

    counts, edges = np.histogram(data1, bins=binning, density=False)
    hist_vals.append(counts)
    hist_errs.append(np.sqrt(counts))  # Poisson errors

    counts, edges = np.histogram(data2, bins=binning, density=False)
    hist_vals.append(counts)
    hist_errs.append(np.sqrt(counts))  # Poisson errors

    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    bin_edges = edges
    
    ax.hist(data1,
            bins=binning,
            histtype='step',
            density=True,
            linewidth=basics['linewidth'],
            color=basics['colors'][0],
            linestyle=basics['linestyles'][0],
            label=leg1,
            log=logy)

    ax.hist(data2,
            bins=binning,
            histtype='step',
            density=True,
            linewidth=basics['linewidth'],
            color=basics['colors'][1],
            linestyle=basics['linestyles'][0],
            label=leg2,
            log=logy)
    
    ax.grid(True, color='gray', linestyle='--', linewidth=0.3, zorder=0)
    ax.legend(fontsize=14, framealpha=1, facecolor='white')
    ax.set_title(f"{title}", fontsize=13, loc='right')

    ax.tick_params(axis="y", labelsize=12)
    #ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    ax.set_ylabel(f"nEvents/{round((bins[2]-bins[1])/bins[0], 2)}", fontsize=14)
    
    h1, h2 = hist_vals
    e1, e2 = hist_errs
    
    ratio = np.divide(h1/np.sum(h1), h2/np.sum(h2),
                      out=np.zeros_like(h1, dtype=float),
                      where=h2 != 0)

    #e1 = e1/np.sum(h1)
    #e2 = e2/np.sum(h2)
    
    ratio_err = np.zeros_like(ratio, dtype=float)
    mask = (h1 > 0) & (h2 > 0)
    ratio_err[mask] = ratio[mask] * np.sqrt(
        (e1[mask]/h1[mask])**2 + (e2[mask]/h2[mask])**2
    )

    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    rax.errorbar(
        bin_centers, ratio, yerr=ratio_err,
        fmt='o', color='black', markersize=4, capsize=2
    )

    rax.axhline(1.0, color='red', linestyle='--')
    rax.set_ylabel("Ratio", fontsize=14)
    rax.set_ylim(0.5, 1.5)  # adjust depending on your data

    rax.set_xlabel(xlabel, fontsize=14)
    rax.tick_params(axis="both", labelsize=12)
    
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    hname = f"{outdir}/{title}_log.pdf" if logy == True else f"{outdir}/{title}.pdf"
    fig.savefig(hname, dpi=300)
    plt.close()
    

def __process_file(file, cfg, nano_key):
    columns_to_plot = cfg.get('columns_to_plot')

    events = NanoEventsFactory.from_root(
        file,
        schemaclass=NanoAODSchema,
    ).events()
    events = events[cfg.get('columns_to_keep')]

    if cfg.get('selectEvents'):
        events = event_selection(events, cfg.get('selections'))

    local_plot_dict = {nano_key: {}}
    for plot_var in columns_to_plot.keys():
        obj_fld = plot_var.split('-')
        array = events[obj_fld[0]][obj_fld[1]]
        local_plot_dict[nano_key][plot_var] = array # awkard array can’t be pickled
    
    return local_plot_dict

            
    

def main():
    real_start = time.perf_counter()
    cpu_start  = time.process_time()
    
    args = sys.argv
    if len(args) < 2:
        raise RuntimeError("config missing")

    cfgfile = args[-1]
    cfg = load_config(cfgfile)
    
    plot_dict = {}
    columns_to_plot = cfg.get('columns_to_plot')

    outdir = cfg.get('output')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # Loop over dataset keys
    dataset_dict = cfg.get('datasets')
    for nano_key in dataset_dict.keys():

        plot_dict[nano_key] = {} #dict_to_plot(list(cfg.get('columns_to_plot').keys()))
        
        print(f"Nano AOD format: {nano_key}")
        datasets = dataset_dict[nano_key]
        files = []
        for dataset in datasets:
            print(f"Dataset : {dataset}")
            files = files + access_files_from_dataset(dataset)

        #print(files)
        print(f"nROOTFiles: {len(files)}")
        print(f"Analysing first {cfg.get('limit_nfiles')} files")
        files = np.array(files)[:cfg.get('limit_nfiles')].tolist()

        """
        # loop over files
        for file in tqdm(files):
            events = NanoEventsFactory.from_root(
                file,
                schemaclass=NanoAODSchema,
            ).events()
            events = events[cfg.get('columns_to_keep')]

            if cfg.get('selectEvents'):
                events = event_selection(events, cfg.get('selections'))

            for plot_var in columns_to_plot.keys():
                obj_fld = plot_var.split('-')
                array = events[obj_fld[0]][obj_fld[1]]
                plot_dict[nano_key][plot_var] = array if plot_var not in plot_dict[nano_key] else ak.concatenate([array, plot_dict[nano_key][plot_var]], axis=0)
        """
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(__process_file, file, cfg, nano_key): file for file in files}

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                # Merge into global plot_dict
                for plot_var, array in result[nano_key].items():
                    if plot_var not in plot_dict[nano_key]:
                        plot_dict[nano_key][plot_var] = array
                    else:
                        plot_dict[nano_key][plot_var] = ak.concatenate(
                            [plot_dict[nano_key][plot_var], array], axis=0
                        )

            
                
    nano_keys = list(dataset_dict.keys())
    for plot_var,bins in columns_to_plot.items():
        #from IPython import embed; embed(); exit()
        print(f"plotting : {plot_var}")
        data1 = plot_dict[nano_keys[0]][plot_var]
        data2 = plot_dict[nano_keys[1]][plot_var]

        #from IPython import embed; embed()
        for j in range(1):
            data1_ = ak.to_numpy(ak.fill_none(ak.firsts(data1[:, j:j+1], axis=1), -99.9))
            data2_ = ak.to_numpy(ak.fill_none(ak.firsts(data2[:, j:j+1], axis=1), -99.9))

            dohist(data1_, data2_,
                   nano_keys[0], nano_keys[1],
                   bins,
                   f"{plot_var.split('-')[1]}",
                   f"{plot_var}-{j+1}",
                   outdir)
            # logy
            dohist(data1_, data2_,
                   nano_keys[0], nano_keys[1],
                   bins,
                   f"{plot_var.split('-')[1]}",
                   f"{plot_var}-{j+1}",
                   outdir,
                   logy=True)
            
            
    print("Done")
    real_stop = time.perf_counter()
    cpu_stop  = time.process_time()
    print(f"Real time : {real_stop - real_start:.3f} seconds")
    print(f"CPU time  : {cpu_stop - cpu_start:.3f} seconds")


if __name__ == "__main__":
    main()
