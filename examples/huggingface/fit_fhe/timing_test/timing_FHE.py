#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.append(os.path.abspath("../../../../molvault"))
import numpy as np
import random
import warnings

# Local utility functions and regression models
from chemdata import *
from regress_utils import *
import json
import matplotlib
import pandas as pd
from itertools import product
matplotlib.use("Agg")  # Set the backend to 'Agg'
import pdb
import matplotlib.pyplot as plt

# Set a fixed seed for reproducibility
random.seed(42)

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


SMILES_train, SMILES_test, X_train, X_test, y_train, y_test = load_ADME_data("LOG HLM_CLint (mL/min/kg)", bits=1024, radius=2)


param_grid_xgboost = {
    "n_estimators": [10],
    "max_depth": [3, 5, 20],
    "learning_rate":[0.138949549],
    "reg_alpha": [0.625], 
    "reg_lambda": [0.54545454544], 
}

parameter_combinations_simplified = list(product(
    param_grid_xgboost['n_estimators'],
    param_grid_xgboost['max_depth'],
    param_grid_xgboost['learning_rate'],
    param_grid_xgboost['reg_alpha'],
    param_grid_xgboost['reg_lambda']
))

parameter_combinations_dicts = [
    {'n_estimators': combo[0], 'max_depth': combo[1], 'learning_rate': combo[2], 'reg_alpha': combo[3], 'reg_lambda': combo[4]}
    for combo in parameter_combinations_simplified
]



RESULTS = []

for task_ind, params in enumerate(parameter_combinations_dicts):

    model_dev = train_zama(X_train, y_train, params, regressor="XGB")
    print(params)
    run_time, error, pearsons_r = compare_timings(model_dev, X_test[:10])
    print(run_time, error, pearsons_r)

    #RESULTS.append([FHE_timings, FHE_errors])

#with open(f"models/FHE_timings.json", "w") as fp:
    #json.dump(RESULTS, fp, default=convert_numpy)