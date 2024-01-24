import sys
import os

import numpy as np
import random
import json
import shutil
import time
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR as LinearSVR
from sklearn.model_selection import KFold
from chemdata import (
    convert_numpy,
    LinearSVRZAMA,
    XGBRegressorZAMA,
    OnDiskNetwork,
    FHEModelDev,
    FHEModelClient,
    get_ECFP_AND_FEATURES,
)
import matplotlib.pyplot as plt
import xgboost as xgb

random.seed(42)


def hyper_opt(X_train, y_train, param_grid, regressor, verbose=10):
    if regressor == "SVR":
        if param_grid is None:
            param_grid = {
                "epsilon": [1e-2, 1e-1, 0.5],
                "C": [1e-4,1e-3, 1e-2, 1e-1],
                "loss": ["squared_epsilon_insensitive"],
                "tol": [0.0001],
                "max_iter": [50000],
                "dual": [True],
            }
        regressor_fct = LinearSVR()
    elif regressor == "XGB":
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "n_estimators": [10, 20, 50, 100],
                "colsample_bytree": [0.3, 0.7],
            }
        regressor_fct = xgb.XGBRegressor(objective="reg:squarederror")
    else:
        raise ValueError("Unknown regressor type")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=regressor_fct,
        param_grid=param_grid,
        cv=kfold,
        verbose=verbose,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return (
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.best_estimator_,
    )


def train_xgb_regressor(X_train, y_train, param_grid=None, verbose=10):
    if param_grid is None:
        param_grid = {
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [20],
            "colsample_bytree": [0.3, 0.7],
        }

    xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_regressor,
        param_grid=param_grid,
        cv=kfold,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    return (
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.best_estimator_,
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    pearsonr_score = pearsonr(y_test, y_pred).statistic
    return pearsonr_score


def performance_bits():
    """
    Test the model performance for different number of bits = feature vector length
    """
    bits = np.array([2**i for i in range(4, 12)])
    plt.close("all")
    fig, ax = plt.subplots()

    for r in [2, 3, 4]:
        performance = []
        for bit in bits:
            X_train, X_test, y_train, y_test = load_data(
                "LOG HLM_CLint (mL/min/kg)", bits=bit, radius=r
            )
            param_grid = {
                "epsilon": [0.0, 0.1, 0.2, 0.5, 1.0],
                "C": [0.1, 1, 10, 100],
                "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
                "tol": [1e-4, 1e-3, 1e-2],
                "max_iter": [1000, 5000, 10000],
            }
            best_params, best_score, best_model = hyper_opt(
                X_train, y_train, param_grid, regressor="SVR", verbose=10
            )
            if not os.path.exists("data"):
                os.makedirs("data")

            with open("data/best_params_{}.json".format(bit), "w") as fp:
                json.dump(best_params, fp, default=convert_numpy)

            pearsonr_score = evaluate_model(best_model, X_test, y_test)
            performance.append(pearsonr_score)

        performance = np.array(performance)
        ax.plot(bits, performance, marker="o", label=f"radius={r}")

    ax.set_xlabel("Number of Bits")
    ax.set_ylabel("Pearson's r Correlation Coefficient")
    ax.legend()
    plt.grid(True)
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/performance_bits.png")

    return bits, performance


def predict_fhe(model, X_test):
    y_pred_fhe = model.predict(X_test, fhe="execute")
    return y_pred_fhe


def setup_network(model_dev):
    network = OnDiskNetwork()
    fhemodel_dev = FHEModelDev(network.dev_dir.name, model_dev)
    fhemodel_dev.save()
    return network, fhemodel_dev


def copy_directory(source, destination="deployment"):
    try:
        # Check if the source directory exists
        if not os.path.exists(source):
            return False, "Source directory does not exist."

        # Check if the destination directory exists
        if not os.path.exists(destination):
            os.makedirs(destination)

        # Copy each item in the source directory
        for item in os.listdir(source):
            s = os.path.join(source, item)
            d = os.path.join(destination, item)
            if os.path.isdir(s):
                shutil.copytree(
                    s, d, dirs_exist_ok=True
                )  # dirs_exist_ok is available from Python 3.8
            else:
                shutil.copy2(s, d)

        return True, None

    except Exception as e:
        return False, str(e)


def client_server_interaction(network, fhemodel_client, X_client):
    decrypted_predictions = []
    execution_time = []
    for i in range(X_client.shape[0]):
        clear_input = X_client[[i], :]
        encrypted_input = fhemodel_client.quantize_encrypt_serialize(clear_input)
        execution_time.append(
            network.client_send_input_to_server_for_prediction(encrypted_input)
        )
        encrypted_prediction = network.server_send_encrypted_prediction_to_client()
        decrypted_prediction = fhemodel_client.deserialize_decrypt_dequantize(
            encrypted_prediction
        )[0]
        decrypted_predictions.append(decrypted_prediction)
    return decrypted_predictions, execution_time


def train_zama(X_train, y_train, best_params, regressor="SVR"):
    if regressor == "SVR":
        best_params["n_bits"] = 12
        model_dev = LinearSVRZAMA(**best_params)
    elif regressor == "XGB":
        best_params["n_bits"] = 6
        model_dev = XGBRegressorZAMA(**best_params)

    print("Training Zama model...")
    model_dev.fit(X_train, y_train)
    print("compiling model...")
    model_dev.compile(X_train)
    print("done")

    return model_dev


def time_prediction(model, X_sample):
    time_begin = time.time()
    y_pred_fhe = model.predict(X_sample, fhe="execute")
    time_end = time.time()
    return time_end - time_begin


def setup_client(network, key_dir):
    fhemodel_client = FHEModelClient(network.client_dir.name, key_dir=key_dir)
    fhemodel_client.generate_private_and_evaluation_keys()
    serialized_evaluation_keys = fhemodel_client.get_serialized_evaluation_keys()
    return fhemodel_client, serialized_evaluation_keys


def compare_predictions(network, fhemodel_client, sklearn_model, X_client):
    fhe_predictions_decrypted, _ = client_server_interaction(
        network, fhemodel_client, X_client
    )
    fhe_predictions_decrypted = [
        item for sublist in fhe_predictions_decrypted for item in sublist
    ]
    fhe_predictions_decrypted = np.array(fhe_predictions_decrypted)

    sklearn_predictions = sklearn_model.predict(X_client)

    # try:
    mae = np.mean(
        np.abs(sklearn_predictions.flatten() - fhe_predictions_decrypted.flatten())
    )
    # and pearson correlation
    pearsonr_score = pearsonr(
        sklearn_predictions.flatten(), fhe_predictions_decrypted.flatten()
    ).statistic
    # pearsons r
    print("sklearn_predictions")
    print(sklearn_predictions)
    print("fhe_predictions_decrypted:")
    print(fhe_predictions_decrypted)

    print("Pearson's r between sklearn and fhe predictions: " f"{pearsonr_score:.2f}")

    return mae, pearsonr_score


def predict_ADME(network, fhemodel_client, molecule, bits=256, radius=2):
    encodings = get_ECFP_AND_FEATURES(molecule, bits=bits, radius=radius).reshape(1, -1)
    # generate_fingerprint(molecule, radius=radius, bits=bits).reshape(1, -1)
    enc_inp = fhemodel_client.quantize_encrypt_serialize(encodings)
    network.client_send_input_to_server_for_prediction(enc_inp)
    encrypted_prediction = network.server_send_encrypted_prediction_to_client()
    decrypted_prediction = fhemodel_client.deserialize_decrypt_dequantize(
        encrypted_prediction
    )
    return np.array([decrypted_prediction])


def fit_final_model(HYPER=True):
    task = "LOG HLM_CLint (mL/min/kg)"
    bits, radius = 1024, 2
    X_train, X_test, y_train, y_test = load_data(task, bits=bits, radius=radius)

    if HYPER:
        param_grid = {
            "epsilon": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
            "C": [0.1, 0.5, 1, 5, 10, 50, 100],
            "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
            "tol": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
            "max_iter": [5000, 1e4, 2e4],
        }
        best_params, best_score, best_model = hyper_opt(
            X_train, y_train, param_grid, regressor="SVR", verbose=10
        )
        with open("best_params.json", "w") as fp:
            json.dump(best_params, fp, default=convert_numpy)
            print(best_params)
        pearsonr_score = evaluate_model(best_model, X_test, y_test)
        print(pearsonr_score)

    try:
        with open("best_params.json", "r") as fp:
            best_params = json.load(fp)
            print(best_params)
    except:
        print(
            "No hyperparameter file found. Please run function with HYPER=True first."
        )
        exit()

    model_dev = train_zama(X_train, y_train, best_params)

    prediction_time = time_prediction(model_dev, X_test[0])
    print(f"Time to predict one sample: {prediction_time:.2f} seconds")

    network, fhemodel_dev = setup_network(model_dev)
    copied, error_message = copy_directory(network.dev_dir.name)
    if not copied:
        print(f"Error copying directory: {error_message}")

    network.dev_send_model_to_server()
    network.dev_send_clientspecs_and_modelspecs_to_client()

    fhemodel_client, serialized_evaluation_keys = setup_client(
        network, network.client_dir.name
    )
    print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")

    network.client_send_evaluation_key_to_server(serialized_evaluation_keys)

    mae_fhe, pearsonr_score_fhe = compare_predictions(
        network, fhemodel_client, best_model, X_test[-10:]
    )

    pred = predict_with_fingerprint(
        network, fhemodel_client, "CC(=O)OC1=CC=CC=C1C(=O)O", bits=1024, radius=2
    )
    print(f"Prediction: {pred}")


if __name__ == "__main__":
    fit_final_model(HYPER=True)
    bits, performance = performance_bits()
