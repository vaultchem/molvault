# MolVault

<p align="center">
  <img src="molvault.png" width="40%" height="40%" />
</p>


**Predict properties of molecules using machine learning?**

 Yes, please!

**Do not want to share your molecules with the world?**

 We neither! 


*We at VaultChem created MolVault. An application that allows you to predict properties of your molecules without sharing them publicly.*


www.vaultchem.com

**Background**

MolVault is built on Zamas `concrete-ml` library for fully homomorpic encryption (FHE). Consider pharma company A to be interested in properties of drug candidate molecules.

For instance understanding pharmacokinetic properties is crucial for determining a drug candidate's concentration profile at its action site, significantly impacting the drug's effectiveness[1].

A does not have sufficient data available for reliable ML predictions. Instead, A will securely obtain predictions on molecular data from an untrusted party B that owns a secret database and an ML model with sufficient training data. This is only possible using FHE to guarantee party A will not reveal the secret query to party B.

Here this scenario is simulated with open-source chemistry datasets and tools based on cheminformatics `rdkit` and FHE `concrete-ml`. We give an end-to-end solution to the problem of privacy-preserving prediction for molecules. 

We have also published an app on hugging face that allows you to make predictions on your molecules without sharing them with the world.

(link ..)

## Installation

Navigate to the `molvault` folder and run `pip install -e .` to install the package.

## Usage:
Here we quickly explain how to use MolVault to fit a fully homomorphic encrypted (FHE) model to your data and make predictions on new molecules. Your CSV file `data.csv` with the chemistry data should be of the following format:
```
SMILES, target_name
CC, 2.312
...
```
Note that the name of the column containing the SMILES strings must be "SMILES" and cannot be changed. The name of the target column can be changed by using the `--target` option.


To perform regression, including hyperparameter optimization, and deployment of the FHE model into a subfolder called "deploy", run the following command:

`python regress.py --data data.csv --target "target_name" --regtype "SVR" --folder "deploy"`

The default regression model is support vector regression `"SVR"`, passed in the `--regtype` option. To change to XGBoost, use `"XGB"`.

As an output you will first get the hyperparameters of the best model. If needed you can change the hyperparameter grid in the `regress_utils.py` file.

Next, predictions on the same points using the sklearn model and its FHE counterpart will be printed as well as their Pearson correlation coefficient. 
Also the error and Pearson correlation with the true values will be printed for the FHE model.
Finally the model is saved in the `deploy` folder.


## Tutorial

The `tutorial.ipynb` notebook for MolVault provides a guide on using the application to predict properties of molecules using machine learning. The notebook includes the following sections:

1) Data Preparation: This part demonstrates how to prepare the dataset for machine learning. It includes loading the data, preprocessing it, and splitting it into training and testing sets.

2) Model Training and Evaluation: In this section, the notebook outlines the process of training a machine learning model. It includes selecting a model (either Support Vector Regression or XGBoost), performing hyperparameter optimization, and evaluating the model's performance.

3) Fully Homomorphic Encryption (FHE) Model Deployment: How to deploy the trained model using Fully Homomorphic Encryption, ensuring the privacy of the data. It shows how to encrypt the model, evaluate it using encrypted data, and compare the predictions of the encrypted model with those of the plaintext model.

4) Prediction with New Molecules: The notebook concludes by demonstrating how to use the deployed model to make predictions on new molecules, including examples with molecules like Vitamin D, Ethanol, and Ibuprofen. This section highlights the application's capability to make predictions without revealing the molecular structures to the server.

Overall, the notebook serves as a practical guide for users to utilize MolVault for secure and private molecular property prediction, showcasing both machine learning and encryption techniques

## Huggingface space

Here we show all the scripts that are needed to run the MolVault app on huggingface and fit to obtain the set of FHE models used in the app.

To fit SVR and XGB to all properties contained in the ADME dataset [1], run:

`python fit_all.py`

This will result in a subfolder `models` containing the fitted models that can readily be used for the app and a file `FHE_timings.json` containing the timings for the FHE models.

#### References

[1] Fang, C., Wang, Y., Grater, R., Kapadnis, S., Black, C., Trapa, P., & Sciabola, S. (2023). Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective. _Journal of Chemical Information and Modeling, 63_(11), 3263-3274. [https://doi.org/10.1021/acs.jcim.3c00160](https://doi.org/10.1021/acs.jcim.3c00160)

The dataset can be found at:
`https://github.com/molecularinformatics/Computational-ADME`