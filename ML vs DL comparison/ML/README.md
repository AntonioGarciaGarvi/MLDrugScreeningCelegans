This is the code used to perform the comparison of ML algorithms (logistic regreession, random forest and XGBoost) to classify wells of the N2 and unc-80 strains.
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
## Usage
Run MLClassification.py to train the ML models and get the results by passing as argument the algorithm to use (RF, LR or XGB).
For example for random forest the command would be
```bash
python MLClassification.py --ML_algorithm RF
```
