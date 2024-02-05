<p align="center">
  <img src="logo_app.png" width="30%" height="40%" />
</p>


## Predict properties of 🤫CONFIDENTIAL🤫 molecules🧬 using machine learning🤖? 
## Concerned about leaking your data to a competitor🕵️‍♂️?


#### Introducing MolVault from <a href='https://vaultchem.com/'>VaultChem</a></h6>
*MolVault* is an application that enables predicting properties of confidential molecules using an untrusted server **without sharing them publicly**.


Machine learning (ML) has become a cornerstone of modern drug discovery. However, the data used to evaluate the ML models is often confidential. This is especially true for the pharmaceutical industry  where new drug candidates 💊 are considered as the most valuable asset. Therefore chemical companies are reluctant to share their data with third parties, for instance to use ML services provided by other companies. 

Using *MolVault* only an encrypted version of the molecular data is sent to as server. **The data will remain encrypted EVEN when the prediction is computed.** Only the party that holds a private key can decrypt the result.


 🪄 **The magic?** 🪄

The server on which the prediction is computed will never see the molecule in clear text, but will still compute an encrypted prediction. Why is this magic? Because this is equivalent to computing the prediction on the molecule in clear text, but without sharing the molecule with the server. Even if organization "B" - or in fact any other party - would try to steal the data, they would only see the encrypted molecular data. Only the party that has the private key (organization "A") can decrypt the prediction. This is possible using a method called "Fully homomorphic encryption" (FHE). This special encryption scheme allows to perform computations on encrypted data.


**What are the steps involved?**

Find out below! 👇
Or try it out yourself on huggingface:

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/VaultChem/molvault)

<p align="center">
  <img src="./examples/huggingface/app/scheme2.png" width="120%" height="40%" />
</p>

## Background

MolVault is built on [Zama](https://www.zama.ai/) library [Concrete-ml](https://github.com/zama-ai/concrete-ml) for fully homomorphic encryption (FHE), to learn more about FHE, click [here](https://fhe.org/resources/).
 
Consider pharma company "A" to be interested in the properties of drug candidate molecules: for instance, understanding pharmacokinetic properties is crucial for determining a drug candidate's effectiveness [1].

The organization "A" may not have sufficient data available for reliable ML predictions. Instead, "A" can securely obtain predictions on molecular data from an untrusted party "B" which owns a secret database and an ML model with sufficient training data. This is only possible using FHE to guarantee "A" will not reveal the secret query to party "B". Here this scenario is simulated with open-source chemistry datasets and tools based on cheminformatics `rdkit` and FHE `Concrete-ml`. We give an end-to-end solution to the problem of privacy-preserving prediction for molecules. 

In this [**tutorial**](https://github.com/vaultchem/molvault/blob/main/examples/tutorial/tutorial.ipynb) we will walk you through the details of training a machine learning model on a chemistry dataset, deploying the model with fully homomorphic encryption, and using the model to make predictions on new molecules.

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

## Docker Installation

Apart from Conda you can use MolVault through a Docker environment. See commands below to install.

```bash
docker build -t molvault .
docker run -it -v /media/vaultchem/molvault:/app -p 8501:8501 molvault
export PYTHONPATH=$PYTHONPATH:/app/examples/huggingface/app/
```

## Usage

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

Predictions on the same points using the sklearn model and its FHE counterpart will be printed as well as their Pearson correlation coefficient. Also the error and Pearson correlation  compared to the true values will be printed for the FHE model. Finally, the model is saved in the `deploy` folder. You may change the folder name using the `--deploy` option.


## Tutorial

The `tutorial.ipynb` notebook for MolVault provides a guide on using the application to predict properties of molecules using machine learning. The notebook includes the following sections:

1) How to prepare the chemistry dataset for machine learning. Load the data, preprocess it, and split into training and testing sets.

2) Outlines process of model training. It includes selecting a model (either Support Vector Regression or XGBoost), performing hyperparameter optimization, and evaluating the model's performance.

3) Deploy the model with Fully Homomorphic Encryption. How to encrypt the model, evaluate it using encrypted data, and compare the predictions of the encrypted model with those of the plaintext model.

4) Use the model to make predictions on new molecules, including examples with molecules like Vitamin D, Ethanol, and Ibuprofen. This section highlights the application's capability to make predictions without revealing the molecular structures to the server.

## User Interface

The easiest way to run the app is via the huggingface space (see above).
Alternatively, after following the installation instruction you can launch the app with:

1. Navigate to the `examples/huggingface/app` folder
2. If you want to use the SVR go to `SVR` folder, if you want to use the XGB go to `XGB` folder
3. Move the `deployment.zip` file to the `app` folder
4. Unzip the `deployment.zip` file using `unzip deployment.zip`
5. Run `bash run_app.sh` to start the app
6. Open you browser and go to `http://localhost:8501/`

Note that the huggingface space is already set up to use the SVR model.

## Model fitting

To fit FHE models with SVR and XGB to all properties contained in the ADME dataset [1], run:

`python fit_all.py`

This will result in a subfolder `models` containing the fitted models that can readily be used for the app and a file `FHE_timings.json` containing the timings for the FHE models.

Note: By default, this involves extensive hyperparameter search. In particular, computing the XGB models can be computationally demanding. You can also download the pretrained models from the huggingface space.

#### References

[1] Fang, C., Wang, Y., Grater, R., Kapadnis, S., Black, C., Trapa, P., & Sciabola, S. (2023). Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective. _Journal of Chemical Information and Modeling, 63_(11), 3263-3274. [https://doi.org/10.1021/acs.jcim.3c00160](https://doi.org/10.1021/acs.jcim.3c00160)

The dataset used to traing the ML models for the app can be found at:

https://github.com/molecularinformatics/Computational-ADME




<p align="center">
  <img src="./examples/huggingface/app/VaultChem.png" width="30%" height="40%" />
  <h6 style='text-align: center; color: grey;'>Visit our website : <a href='https://vaultchem.com/'>VaultChem</a></h6>
</p>
