import gzip
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn.svm import LinearSVR as LinearSVRZAMA
from concrete.ml.sklearn import XGBClassifier as XGBClassifierZAMA
from concrete.ml.sklearn import XGBRegressor as XGBRegressorZAMA
from concrete.ml.sklearn import LogisticRegression as LogisticRegressionZAMA

from sklearn.svm import LinearSVR as LinearSVR
import time
from shutil import copyfile
from tempfile import TemporaryDirectory
import pickle
import os
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
import numpy as np
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class OnDiskNetwork:
    """Simulate a network on disk."""

    def __init__(self):
        # Create 3 temporary folder for server, client and dev with tempfile
        self.server_dir = TemporaryDirectory()
        self.client_dir = TemporaryDirectory()
        self.dev_dir = TemporaryDirectory()

    def client_send_evaluation_key_to_server(self, serialized_evaluation_keys):
        """Send the public key to the server."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "wb") as f:
            f.write(serialized_evaluation_keys)

    def client_send_input_to_server_for_prediction(self, encrypted_input):
        """Send the input to the server and execute on the server in FHE."""
        with open(self.server_dir.name + "/serialized_evaluation_keys.ekl", "rb") as f:
            serialized_evaluation_keys = f.read()
        time_begin = time.time()
        encrypted_prediction = FHEModelServer(self.server_dir.name).run(
            encrypted_input, serialized_evaluation_keys
        )
        time_end = time.time()
        with open(self.server_dir.name + "/encrypted_prediction.enc", "wb") as f:
            f.write(encrypted_prediction)
        return time_end - time_begin

    def dev_send_model_to_server(self):
        """Send the model to the server."""
        copyfile(
            self.dev_dir.name + "/server.zip", self.server_dir.name + "/server.zip"
        )

    def server_send_encrypted_prediction_to_client(self):
        """Send the encrypted prediction to the client."""
        with open(self.server_dir.name + "/encrypted_prediction.enc", "rb") as f:
            encrypted_prediction = f.read()
        return encrypted_prediction

    def dev_send_clientspecs_and_modelspecs_to_client(self):
        """Send the clientspecs and evaluation key to the client."""
        copyfile(
            self.dev_dir.name + "/client.zip", self.client_dir.name + "/client.zip"
        )

    def cleanup(self):
        """Clean up the temporary folders."""
        self.server_dir.cleanup()
        self.client_dir.cleanup()
        self.dev_dir.cleanup()


def generate_fingerprint(smiles, radius=2, bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits)

    return np.array(fp)


def compute_descriptors_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    MDlist = []

    MDlist.append(rdMolDescriptors.CalcTPSA(mol))
    MDlist.append(rdMolDescriptors.CalcFractionCSP3(mol))
    MDlist.append(rdMolDescriptors.CalcNumAliphaticCarbocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumAliphaticHeterocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumAliphaticRings(mol))
    MDlist.append(rdMolDescriptors.CalcNumAmideBonds(mol))
    MDlist.append(rdMolDescriptors.CalcNumAromaticCarbocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumAromaticHeterocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    MDlist.append(rdMolDescriptors.CalcNumLipinskiHBA(mol))
    MDlist.append(rdMolDescriptors.CalcNumLipinskiHBD(mol))
    MDlist.append(rdMolDescriptors.CalcNumHeteroatoms(mol))
    MDlist.append(rdMolDescriptors.CalcNumHeterocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumRings(mol))
    MDlist.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    MDlist.append(rdMolDescriptors.CalcNumSaturatedCarbocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumSaturatedHeterocycles(mol))
    MDlist.append(rdMolDescriptors.CalcNumSaturatedRings(mol))
    MDlist.append(rdMolDescriptors.CalcHallKierAlpha(mol))
    MDlist.append(rdMolDescriptors.CalcKappa1(mol))
    MDlist.append(rdMolDescriptors.CalcKappa2(mol))
    MDlist.append(rdMolDescriptors.CalcKappa3(mol))
    MDlist.append(rdMolDescriptors.CalcChi0n(mol))
    MDlist.append(rdMolDescriptors.CalcChi0v(mol))
    MDlist.append(rdMolDescriptors.CalcChi1n(mol))
    MDlist.append(rdMolDescriptors.CalcChi1v(mol))
    MDlist.append(rdMolDescriptors.CalcChi2n(mol))
    MDlist.append(rdMolDescriptors.CalcChi2v(mol))
    MDlist.append(rdMolDescriptors.CalcChi3n(mol))
    MDlist.append(rdMolDescriptors.CalcChi3v(mol))
    MDlist.append(rdMolDescriptors.CalcChi4n(mol))
    MDlist.append(rdMolDescriptors.CalcChi4v(mol))
    MDlist.append(rdMolDescriptors.CalcExactMolWt(mol) / 100)
    MDlist.append(Lipinski.HeavyAtomCount(mol))
    MDlist.append(Lipinski.NumHAcceptors(mol))
    MDlist.append(Lipinski.NumHDonors(mol))
    MDlist.append(Lipinski.NOCount(mol))

    return MDlist


def get_ECFP_AND_FEATURES(smiles, radius=2, bits=512):
    fp = generate_fingerprint(smiles, radius=radius, bits=bits)
    MDlist = np.array(compute_descriptors_from_smiles(smiles))
    return np.hstack([MDlist, fp])


def compute_descriptors_from_smiles_list(SMILES):
    X = [compute_descriptors_from_smiles(smi) for smi in SMILES]
    return np.array(X)


class ProcessADMEChemData:
    def __init__(self, bits=512, radius=2):
        self.bits = int(bits)
        self.radius = int(radius)
        if not os.path.exists("data"):
            os.makedirs("data")
        self.save_file = "data/" + "save_file_ADME_{}_{}.pkl".format(
            self.bits, self.radius
        )

        if os.path.exists(self.save_file):
            with open(self.save_file, "rb") as file:
                self.adjusted_valid_entries_per_task = pickle.load(file)
        else:
            url = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"
            self.df = pd.read_csv(url)
            self.all_tasks = self.df.columns[4:].values
            self.process()
            self.save_adjusted_data()

    def process(self):
        SMILES = self.df["SMILES"].values
        MOLS = [Chem.MolFromSmiles(smi) for smi in SMILES]
        self.df["MOL"] = MOLS
        self.df["smiles"] = [Chem.MolToSmiles(mol) for mol in MOLS]
        self.adjusted_valid_entries_per_task = {}

        # Iterating through each task column and extracting valid entries
        for task in self.all_tasks:  # Excluding mol_id and smiles from the iteration
            valid_entries = self.df.dropna(subset=[task])[["Vendor ID", "smiles", task]]

            valid_entries["fps"] = valid_entries["smiles"].apply(
                lambda x: generate_fingerprint(x, radius=self.radius, bits=self.bits)
            )
            valid_entries = valid_entries.dropna(subset=["fps"])

            valid_entries["descriptors"] = valid_entries["smiles"].apply(
                lambda x: compute_descriptors_from_smiles_list([x])[0]
            )

            valid_entries = valid_entries.dropna(subset=["descriptors"])

            ## now stack the fps and descriptors
            valid_entries["combined"] = valid_entries.apply(
                lambda row: np.hstack([row["descriptors"], row["fps"]]), axis=1
            )
            valid_entries = valid_entries.sample(frac=1, random_state=42).reset_index(
                drop=True
            )
            self.adjusted_valid_entries_per_task[task] = valid_entries
            self.adjusted_valid_entries_per_task[
                task
            ] = self.adjusted_valid_entries_per_task[task].rename(columns={task: "y"})

    def save_adjusted_data(self):
        with open(self.save_file, "wb") as file:
            pickle.dump(self.adjusted_valid_entries_per_task, file)

    def get_X_y(self, task):
        X = np.float_(
            np.stack(self.adjusted_valid_entries_per_task[task].combined.values)
        )
        y = self.adjusted_valid_entries_per_task[task].y.values.astype(float)
        return X, y


def load_ADME_data(task, bits=256, radius=2):
    """
    Load and split data for a specified task in cheminformatics.

    This function processes chemical data for a given task using specified parameters for bits and radius.
    It then splits the data into training and test sets.

    Parameters:
    task (str): The specific ADME task for which data needs to be processed.
    bits (int, optional): The number of bits to be used in the fingerprint representation. Default is 256.
    radius (int, optional): The radius parameter for the fingerprint calculation. Default is 2.

    Returns:
    tuple: A tuple containing the split data in the form (X_train, X_test, y_train, y_test),
           where X_train and X_test are the features and y_train and y_test are the labels.
    """
    data = ProcessADMEChemData(bits=bits, radius=radius)
    X, y = data.get_X_y(task)
    SMILES = data.adjusted_valid_entries_per_task[task]["smiles"].values
    return train_test_split(SMILES,X, y, test_size=0.2, random_state=42)


class ProcessGenericChemData:
    def __init__(self, source_file, target="y", bits=512, radius=2):
        self.source_file = source_file
        self.target = target

        self.bits = int(bits)
        self.radius = int(radius)

        self.df = pd.read_csv(self.source_file)
        # check if a column called y exists in the csv file
        if self.target not in self.df.columns:
            raise ValueError(
                "The target column {} does not exist in the source file {}".format(
                    self.target, self.source_file
                )
            )
        
        self.process()

    def process(self):
        self.df = self.df.dropna()
        SMILES = self.df["SMILES"].values
        MOLS = [Chem.MolFromSmiles(smi) for smi in SMILES]
        self.df["MOL"] = MOLS
        self.df["smiles"] = [Chem.MolToSmiles(mol) for mol in MOLS]
        self.df["fps"] = self.df["SMILES"].apply(
            lambda x: generate_fingerprint(x, radius=self.radius, bits=self.bits)
        )

    def get_X_y(self):
        X = np.float_(np.stack(self.df.fps.values))
        y = self.df[self.target].values.astype(float)

        return X, y

    def get_split(self):
        X, y = self.get_X_y()
        return train_test_split(X, y, test_size=0.2, random_state=42)


# main function
if __name__ == "__main__":
    data = ProcessGenericChemData(source_file="output.csv")
    X_train, X_test, y_train, y_test = data.get_split()
