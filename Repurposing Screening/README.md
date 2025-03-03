## Repurposing Screening Dataset
The datasets belong to the repository of another paper [1](https://doi.org/10.7554/eLife.92491.4.sa0), and are available on Zenodo [2](https://doi.org/10.5281/zenodo.13909390). \
Once downloaded, the required files can be found in the folder DataSets\DrugRepurposing

**References**

[1] [https://doi.org/10.7554/eLife.92491.4.sa0/](https://doi.org/10.7554/eLife.92491.4.sa0/) \
[2] [https://doi.org/10.5281/zenodo.13909390/](https://doi.org/10.5281/zenodo.13909390/) 


## Installation
To reproduce the results you must install the dependencies.
You can do this in a conda environment:
```bash
conda create -n myenv python=3.8
source activate myenv
pip install -r requirements.txt
```
Install tierpsytools
```bash
git clone https://github.com/Tierpsy/tierpsy-tools-python.git
pip install imgstore
cd tierpsy-tools-python
pip install -e .
```

