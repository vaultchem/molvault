"""Server that will listen for GET requests from the client."""
from fastapi import FastAPI
from joblib import load
from concrete.ml.deployment import FHEModelServer
from pydantic import BaseModel
import base64
from pathlib import Path

current_dir = Path(__file__).parent

# Load the model
fhe_model_HLM = FHEModelServer(
    Path.joinpath(current_dir, "deployment/deployment_0")
)
fhe_model_MDR1MDCK = FHEModelServer(
    Path.joinpath(current_dir, "deployment/deployment_1")
)
fhe_model_SOLUBILITY = FHEModelServer(
    Path.joinpath(current_dir, "deployment/deployment_2")
)
fhe_model_PROTEIN_BINDING_HUMAN = FHEModelServer(
    Path.joinpath(current_dir, "deployment/deployment_3")
)
fhe_model_PROTEIN_BINDING_RAT = FHEModelServer(
    Path.joinpath(current_dir, "deployment/deployment_4")
)
fhe_model_RLM_CLint = FHEModelServer(
    Path.joinpath(current_dir, "deployment/deployment_5")
)


class PredictRequest(BaseModel):
    evaluation_key: str
    encrypted_encoding: str


# Initialize an instance of FastAPI
app = FastAPI()


# Define the default route
@app.get("/")
def root():
    return {"message": "Welcome to Your Molecular Property prediction FHE Server!"}


@app.post("/predict_HLM")
def predict_HLM(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_HLM.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}

@app.post("/predict_MDR1MDCK")
def predict_MDR1MDCK(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_MDR1MDCK.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}

@app.post("/predict_SOLUBILITY")
def predict_SOLUBILITY(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_SOLUBILITY.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}

@app.post("/predict_PROTEIN_BINDING_HUMAN")
def predict_PROTEIN_BINDING_HUMAN(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_PROTEIN_BINDING_HUMAN.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}


@app.post("/predict_PROTEIN_BINDING_RAT")
def predict_PROTEIN_BINDING_RAT(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_PROTEIN_BINDING_RAT.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}

def predict_RLM_CLint(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_RLM_CLint.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}

@app.post("/predict_RLM_CLint")
def predict_RLM_CLint(query: PredictRequest):
    encrypted_encoding = base64.b64decode(query.encrypted_encoding)
    evaluation_key = base64.b64decode(query.evaluation_key)
    prediction = fhe_model_RLM_CLint.run(encrypted_encoding, evaluation_key)

    # Encode base64 the prediction
    encoded_prediction = base64.b64encode(prediction).decode()
    return {"encrypted_prediction": encoded_prediction}