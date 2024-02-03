#!/usr/bin/env python
# coding: utf-8

import os
import sys

sys.path.append(os.path.abspath("../../molvault"))
import numpy as np
import random
import warnings

# Local utility functions and regression models
from chemdata import *
from regress_utils import *
import json
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # Set the backend to 'Agg'

import matplotlib.pyplot as plt

# Set a fixed seed for reproducibility
random.seed(42)

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


param_grid_linear = {
    "epsilon": [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5],
    "C": [1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0],
    "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
    "tol": [1e-3, 1e-4, 1e-5, 1e-6],
    "max_iter": [int(4e4)],
    "dual": [True],
}

param_grid_xgboost = {
    "n_estimators": np.arange(5, 120, 20),
    "max_depth": [3, 5, 10, 20],
    "learning_rate": np.logspace(-3, 0, 4),
    "reg_alpha": np.linspace(0, 0.5, 6), 
    "reg_lambda": np.linspace(0, 0.5, 6), 
}

all_tasks = [
    "LOG HLM_CLint (mL/min/kg)",
    "LOG MDR1-MDCK ER (B-A/A-B)",
    "LOG SOLUBILITY PH 6.8 (ug/mL)",
    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)",
    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)",
    "LOG RLM_CLint (mL/min/kg)",
]

RESULTS = {}

for reg_type in ["XGB", "SVR"]:
    if reg_type == "SVR":
        param_grid = param_grid_linear
    elif reg_type == "XGB":
        param_grid = param_grid_xgboost

    FHE_timings, FHE_errors = [], []
    for task_ind, task in enumerate(all_tasks):
        work_dir = f"models/{reg_type}/deployment_{task_ind}"
        # create subfolder for each task

        if not os.path.exists(f"{work_dir}"):
            os.makedirs(work_dir, exist_ok=True)

        SMILES_train, SMILES_test, X_train, X_test, y_train, y_test = load_ADME_data(task, bits=1024, radius=2)

        train_df, test_df = pd.DataFrame({"SMILES": SMILES_train.tolist(), "y": y_train}), pd.DataFrame(
            {"SMILES": SMILES_test.tolist(), "y": y_test}
        )
        #save the train and test data to the deployment folder
        train_df.to_csv(f"{work_dir}/train.csv", index=False)
        test_df.to_csv(f"{work_dir}/test.csv", index=False)

        best_params, best_score, best_model = hyper_opt(
            X_train, y_train, param_grid=param_grid, regressor=reg_type, verbose=10
        )

        with open(f"{work_dir}/best_params_demo_{task_ind}.json", "w") as fp:
            json.dump(best_params, fp, default=convert_numpy)
            print(best_params)

        pearsonr_score = evaluate_model(best_model, X_test, y_test)
        print(pearsonr_score)
        y_pred = best_model.predict(X_test)
        x = np.linspace(min(y_test), max(y_test), 100)
        plt.close()
        plt.plot(x, x, color="black", ls="--")
        plt.ylabel("Predicted")
        plt.xlabel("Experimental")
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.savefig(f"{work_dir}/regression_{task_ind}.png")
        plt.close()

        # write a file to deployment folder with name of task
        with open(f"{work_dir}/task_name.txt", "w") as fp:
            fp.write(task)

        try:
            with open(f"{work_dir}/best_params_demo_{task_ind}.json", "r") as fp:
                best_params = json.load(fp)
                print(best_params)
        except:
            print("No hyperparameter file found. Please run the cell above first.")

        model_dev = train_zama(X_train, y_train, best_params, regressor=reg_type)
        prediction_time = time_prediction(model_dev, X_test[0])

        print(f"Time to predict one sample: {prediction_time:.2f} seconds")

        network, fhemodel_dev = setup_network(model_dev)
        copied, error_message = copy_directory(
            network.dev_dir.name, destination=f"{work_dir}"
        )
        if not copied:
            print(f"Error copying directory: {error_message}")

        network.dev_send_model_to_server()
        network.dev_send_clientspecs_and_modelspecs_to_client()

        fhemodel_client, serialized_evaluation_keys = setup_client(
            network, network.client_dir.name
        )
        print(f"Evaluation keys size: {len(serialized_evaluation_keys) } B")

        network.client_send_evaluation_key_to_server(serialized_evaluation_keys)

        mae_fhe, pearsonr_score_fhe = compare_predictions(
            network, fhemodel_client, best_model, X_test[-8:]
        )

        VITAMIN = "C=C1CCC(O)CC1=CC=C1CCCC2(C)C1CCC2C(C)C=CC(C)C(C)C"
        ETHANOL = "CCO"
        IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"

        pred = predict_ADME(
            network,
            fhemodel_client,
            IBUPROFEN,
            bits=1024,
            radius=2,
        )
        print("Prediction: {:.1f}".format(pred[0][0][0]))

        if reg_type == "XGB":
            FHE_timings.append(
                [best_params["n_estimators"], best_params["max_depth"], prediction_time]
            )
        elif reg_type == "SVR":
            FHE_timings.append(
                [best_params["C"], best_params["epsilon"], prediction_time]
            )
        else:
            raise ValueError("Unknown regressor type")

        FHE_errors.append([mae_fhe, pearsonr_score_fhe])

    RESULTS[reg_type] = [FHE_timings, FHE_errors]

    with open(f"models/{reg_type}/FHE_timings.json", "w") as fp:
        json.dump(RESULTS, fp, default=convert_numpy)
