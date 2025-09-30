# NANOv15LightValidation

```bash
# voms proxy required
# activate the LCG105 venv
source setup.sh
# main function with a config file
python main.py <yaml-config>
```

Two `yaml` configs are there, one for `v9-v15L` comparison and the other one is for `v15-v15L`.

Some basic options one can tune to apply different selections or to produce different plots

```bash
#selections
<Add or remove selections>

#columns_to_plot
<Add or remove plots>
```