import sys
import os
import random
import json
import argparse
import sklearn.metrics
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Simplify path setup by combining steps
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "molvault")))

from chemdata import ProcessGenericChemData, convert_numpy
from regress_utils import (
    hyper_opt,
    train_zama,
    copy_directory,
    setup_network,
    setup_client,
    compare_predictions,
)

# Set a fixed seed for reproducibility
random.seed(42)

# Setup argument parser
parser = argparse.ArgumentParser(description="Process CSV data for regression analysis.")
parser.add_argument("--data", type=str, required=True, help="Path to the CSV file")
parser.add_argument("--target", type=str, required=True, help="Target column name")
parser.add_argument("--regtype", type=str, default="SVR", help="Regressor type (default: SVR)")
parser.add_argument("--folder", type=str, default="deployment", help="Folder to save the model (default: deployment)")

args = parser.parse_args()

# Process data and split
data = ProcessGenericChemData(source_file=args.data, target=args.target)
X_train, X_test, y_train, y_test = data.get_split()

# Hyperparameter optimization
best_params, _, best_model = hyper_opt(X_train, y_train,param_grid=None, regressor=args.regtype)
print("Hyperparameter tuning results:", best_params)

# Save best parameters
with open("best_params_demo.json", "w") as fp:
    json.dump(best_params, fp, default=convert_numpy)

# Train model
model_dev_fhe = train_zama(X_train, y_train, best_params, regressor=args.regtype)

# Predict based on regressor type
if args.regtype in ["SVR", "XGB"]:
    mode = "execute" if args.regtype == "SVR" else "simulate"
    y_pred_FHE = model_dev_fhe.predict(X_test, fhe=mode).flatten()
else:
    raise ValueError("Unknown regressor type")

# Deploy model
network, _ = setup_network(model_dev_fhe)
copied, error_message = copy_directory(network.dev_dir.name, destination=args.folder)
if not copied:
    print(f"Error copying directory: {error_message}")

network.dev_send_model_to_server()
network.dev_send_clientspecs_and_modelspecs_to_client()

fhemodel_client, serialized_evaluation_keys = setup_client(network, network.client_dir.name)
print(f"Evaluation keys size: {len(serialized_evaluation_keys)} B")

network.client_send_evaluation_key_to_server(serialized_evaluation_keys)

# Compare predictions
_, pearsonr_score_fhe = compare_predictions(network, fhemodel_client, best_model, X_test[-10:])
pearsonr_score = pearsonr(y_test, y_pred_FHE).statistic
print(f"Pearson R score: {pearsonr_score:.3f}")
print(f"MAE score: {sklearn.metrics.mean_absolute_error(y_test, y_pred_FHE):.3f}")

# Plot results
plt.scatter(y_test, y_pred_FHE)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="black", ls="--")
plt.xlabel("Experimental")
plt.ylabel("FHE Predicted")
plt.show()