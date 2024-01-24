# MolVault

<p align="center">
  <img src="logo_app.png" width="30%" height="40%" />
</p>


## Predict properties of molecules using machine learning? 
**Yes, please!** 

### Concerned about sharing your molecules with the world?
**We too!** 

#### Introducing MolVault from <a href='https://vaultchem.com/'>VaultChem</a></h6>
*MolVault* is an application that enables you to predict properties of your molecules **without sharing them publicly**.
You can send an encrypted version of your molecule to our server, which will perform the prediction and send back the encrypted result.

 🪄 **The magic?** 🪄

The server will never see the molecule in clear text, and you will be the only one able to decrypt the result.

#### How?
Find out below!


<p align="center">
  <img src="./examples/huggingface/app/ablauf.png" width="120%" height="40%" />
</p>


Here we present the first public demonstration of encrypted prediction of pharmacokinetic properties. As shown above, the steps are as follows:

0) Select a pharmacokinetic property of interest
1) Define the molecule for the prediction
2) Generate a key pair
3) Encrypt molecule, send it to server
4) 🪄**Magic happens here**🪄 The prediction is performed while the data is still encrypted
5) Encrypted prediction is sent back and user can decrypt

**Background**

MolVault is built on Zamas `concrete-ml` library for fully homomorpic encryption (FHE). Consider pharma company A to be interested in properties of drug candidate molecules: for instance understanding pharmacokinetic properties is crucial for determining a drug candidate's concentration profile at its action site, significantly impacting the drug's effectiveness [1].

Party A may not have sufficient data available for reliable ML predictions. Instead, A will securely obtain predictions on molecular data from an untrusted party B that owns a secret database and an ML model with sufficient training data. This is only possible using FHE to guarantee party A will not reveal the secret query to party B. Here this scenario is simulated with open-source chemistry datasets and tools based on cheminformatics `rdkit` and FHE `concrete-ml`. We give an end-to-end solution to the problem of privacy-preserving prediction for molecules. 

We have also published an app on hugging face that allows you to make predictions on your molecules without sharing them with the world.

(link ..)

## Installation
First clone the repository:

`git clone https://github.com/vaultchem/molvault.git`

Navigate to the `molvault` folder. 
Create a new python `3.10` environment, for instance using miniconda:

`conda create -n molvault python=3.10`

Now activate the environment:
`conda activate molvault`

and install the requirements using

`pip install -r requirements.txt` 

and then run 

`pip install -e .` 

to install the package.

## Usage:
How to use MolVault to fit a fully homomorphic encrypted (FHE) model to your data and make predictions on new molecules: Your CSV file `data.csv` with the chemistry data should be of the following format:
```
SMILES, target_name
CC, 2.312
...
```
Note that the name of the column containing the SMILES strings must be "SMILES" and cannot be changed. The name of the target column can be changed by using the `--target` option. To perform regression, including hyperparameter optimization, and deployment of the FHE model into a subfolder called "deploy", run the following command:

`python regress.py --data example_data.csv --target "example_target"`

The default regression model is support vector regression `"SVR"`, passed in the `--regtype` option. To change to XGBoost, use `"XGB"`.
As an output you will first get the hyperparameters of the best model. If needed you can change the hyperparameter grid in the `regress_utils.py` file.

Next, predictions on the same points using the sklearn model and its FHE counterpart will be printed as well as their Pearson correlation coefficient. Also the error and Pearson correlation with the true values will be printed for the FHE model.
Finally the model is saved in the `deploy` folder. You may change the folder name using the `--deploy` option.


## Tutorial

The `tutorial.ipynb` notebook for MolVault provides a guide on using the application to predict properties of molecules using machine learning. The notebook includes the following sections:

1) How to prepare the chemistry dataset for machine learning. Load the data, preprocess it, and split into training and testing sets.

2) Outlines process of model training. It includes selecting a model (either Support Vector Regression or XGBoost), performing hyperparameter optimization, and evaluating the model's performance.

3) Deploy the model with Fully Homomorphic Encryption. How to encrypt the model, evaluate it using encrypted data, and compare the predictions of the encrypted model with those of the plaintext model.

4) Use the model to make predictions on new molecules, including examples with molecules like Vitamin D, Ethanol, and Ibuprofen. This section highlights the application's capability to make predictions without revealing the molecular structures to the server.

## Huggingface space

Here we show all the scripts that are needed to run the MolVault app on huggingface and fit to obtain the set of FHE models used in the app.

To fit SVR and XGB to all properties contained in the ADME dataset [1], run:

`python fit_all.py`

This will result in a subfolder `models` containing the fitted models that can readily be used for the app and a file `FHE_timings.json` containing the timings for the FHE models.

Note: You can download the pretrained models from the huggingface space.

#### References

[1] Fang, C., Wang, Y., Grater, R., Kapadnis, S., Black, C., Trapa, P., & Sciabola, S. (2023). Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective. _Journal of Chemical Information and Modeling, 63_(11), 3263-3274. [https://doi.org/10.1021/acs.jcim.3c00160](https://doi.org/10.1021/acs.jcim.3c00160)

The dataset used to traing the ML models for the app can be found at:
`https://github.com/molecularinformatics/Computational-ADME`


<p align="center">
  <img src="./examples/huggingface/app/VaultChem.png" width="30%" height="40%" />
  <h6 style='text-align: center; color: grey;'>Visit our website : <a href='https://vaultchem.com/'>VaultChem</a></h6>
</p>
