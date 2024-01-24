import numpy as np
import json
import matplotlib.pyplot as plt
import pdb

# load the results from the json file

with open("./models/XGB/FHE_timings.json") as json_file:
    xgb_data = json.load(json_file)["XGB"]
    xgb_timings, xgb_errors = np.array(xgb_data[0]), np.array(xgb_data[1])
    unique_keys, indices = np.unique(xgb_timings[:, :2], axis=0, return_inverse=True)
    averaged_timings_xgb = np.empty((unique_keys.shape[0], xgb_timings.shape[1]))
    for i, key in enumerate(unique_keys):
        averaged_timings_xgb[i, :2] = key
        averaged_timings_xgb[i, 2] = np.mean(xgb_timings[indices == i, 2])


depths, time_xgb = averaged_timings_xgb[:, 1], averaged_timings_xgb[:, 2] / 8


fig, ax = plt.subplots(figsize=(8, 6))
ax.tick_params(axis="both", which="major", labelsize=16)
ax.plot(depths, time_xgb, "o-", label="XGBoost")
ax.set_xlabel("Depth", fontsize=16)
ax.set_ylabel("Time (s)", fontsize=16)

text_str = "n_bits = 6\nn_estimators = 100\nlen(X) = 1024"
ax.text(
    0.05, 0.95, text_str, transform=ax.transAxes, fontsize=16, verticalalignment="top"
)

plt.savefig("./models/XGB/FHE_timings.png")


with open("./models/SVR/FHE_timings.json") as json_file:
    svr_data = json.load(json_file)["SVR"]
    svr_timings, svr_errors = np.array(svr_data[0]), np.array(svr_data[1])

    unique_keys, indices = np.unique(xgb_timings[:, :2], axis=0, return_inverse=True)
    averaged_timings_svr = np.empty((unique_keys.shape[0], xgb_timings.shape[1]))
    for i, key in enumerate(unique_keys):
        averaged_timings_svr[i, :2] = key
        averaged_timings_svr[i, 2] = np.mean(svr_timings[indices == i, 2])


pdb.set_trace()
print("Done")
