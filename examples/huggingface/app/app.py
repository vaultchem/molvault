# uncollemt if run locally
import sys, os

sys.path.append(os.path.abspath("../../../molvault"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from requests import head
from concrete.ml.deployment import FHEModelClient
import numpy
import os
from pathlib import Path
import requests
import json
import base64
import subprocess
import shutil
import time
from chemdata import get_ECFP_AND_FEATURES
import streamlit as st
import subprocess
import cairosvg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd

st.set_page_config(layout="wide")


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, width=None):
    img_bytes = img_to_bytes(img_path)
    if width:
        img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='width:{};'>".format(
            img_bytes, width
        )
    else:
        img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
            img_bytes
        )
    return img_html


# start timing
formatted_text = (
    "<h1 style='text-align: center; color: blue;'>VaultChem</h1>"
    "<h1 style='text-align: center;'>"
    "<span style='color: red;'>Pharmacokinetics</span>"
    "<span style='color: black;'> of </span>"
    "<span style='color: blue;'>confidential</span>"
    "<span style='color: black;'> molecules</span>"
    "</h1>"
)


st.markdown(formatted_text, unsafe_allow_html=True)

interesting_text = """
## Predict properties of molecules using machine learning? 
**Yes, please!** 

### Concerned about sharing your molecules with the world?
**We too!** 

#### Introducing MolVault from VaultChem
*MolVault* is an application that enables you to predict properties of your molecules **without sharing them publicly**.

You can send an encrypted version of your molecule to our server, which will perform the prediction and send back the encrypted result.
\n 
🪄 **The magic?** 🪄
\n
The server will never see the molecule in clear text, and you will be the only one able to decrypt the result.

##### How?
Find out below!
"""
st.markdown(interesting_text)
st.divider()
# st.image("ablauf.png", width=1200)
st.markdown(
    "<p style='text-align: center; color: grey;'>"
    + img_to_html("ablauf.png", width="80%")
    + "</p>",
    unsafe_allow_html=True,
)

st.divider()
# read text from file

with open("description.txt", "r") as f:
    readme_text = f.read()

st.markdown(
    f"<div style='max-height: 600px; overflow-y: auto;'>{readme_text}</div>",
    unsafe_allow_html=True,
)
# Define your data

st.divider()


time_begin = time.time()


# This repository's directory
REPO_DIR = Path(__file__).parent
subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)

# if not exists, create a directory for the FHE keys called .fhe_keys
if not os.path.exists(".fhe_keys"):
    os.mkdir(".fhe_keys")
# if not exists, create a directory for the tmp files called tmp
if not os.path.exists("tmp"):
    os.mkdir("tmp")


# Wait 5 sec for the server to start
time.sleep(4)

# Encrypted data limit for the browser to display
# (encrypted data is too large to display in the browser)
ENCRYPTED_DATA_BROWSER_LIMIT = 500
N_USER_KEY_STORED = 20


def clean_tmp_directory():
    # Allow 20 user keys to be stored.
    # Once that limitation is reached, deleted the oldest.
    path_sub_directories = sorted(
        [f for f in Path(".fhe_keys/").iterdir() if f.is_dir()], key=os.path.getmtime
    )

    user_ids = []
    if len(path_sub_directories) > N_USER_KEY_STORED:
        n_files_to_delete = len(path_sub_directories) - N_USER_KEY_STORED
        for p in path_sub_directories[:n_files_to_delete]:
            user_ids.append(p.name)
            shutil.rmtree(p)

    list_files_tmp = Path("tmp/").iterdir()
    # Delete all files related to user_id
    for file in list_files_tmp:
        for user_id in user_ids:
            if file.name.endswith(f"{user_id}.npy"):
                file.unlink()


def keygen():
    # Clean tmp directory if needed
    clean_tmp_directory()

    print("Initializing FHEModelClient...")
    task = st.session_state["task"]
    # Let's create a user_id
    user_id = numpy.random.randint(0, 2**32)
    fhe_api = FHEModelClient(f"deployment/deployment_{task}", f".fhe_keys/{user_id}")
    fhe_api.load()

    # Generate a fresh key
    fhe_api.generate_private_and_evaluation_keys(force=True)
    evaluation_key = fhe_api.get_serialized_evaluation_keys()

    numpy.save(f"tmp/tmp_evaluation_key_{user_id}.npy", evaluation_key)

    return [list(evaluation_key)[:ENCRYPTED_DATA_BROWSER_LIMIT], user_id]


def encode_quantize_encrypt(text, user_id):
    task = st.session_state["task"]
    fhe_api = FHEModelClient(f"deployment/deployment_{task}", f".fhe_keys/{user_id}")
    fhe_api.load()

    encodings = get_ECFP_AND_FEATURES(text, radius=2, bits=1024).reshape(1, -1)

    quantized_encodings = fhe_api.model.quantize_input(encodings).astype(numpy.uint8)
    encrypted_quantized_encoding = fhe_api.quantize_encrypt_serialize(encodings)

    # Save encrypted_quantized_encoding in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    numpy.save(
        f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy",
        encrypted_quantized_encoding,
    )

    # Compute size
    encrypted_quantized_encoding_shorten = list(encrypted_quantized_encoding)[
        :ENCRYPTED_DATA_BROWSER_LIMIT
    ]
    encrypted_quantized_encoding_shorten_hex = "".join(
        f"{i:02x}" for i in encrypted_quantized_encoding_shorten
    )
    return (
        encodings[0],
        quantized_encodings[0],
        encrypted_quantized_encoding_shorten_hex,
    )


def run_fhe(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_quantized_encoding_{user_id}.npy")
    # if not user_id:
    #     print("You need to generate FHE keys first.")
    # if not encoded_data_path.is_file():
    #     print("No encrypted data was found. Encrypt the data before trying to predict.")

    # Read encrypted_quantized_encoding from the file

    task = st.session_state["task"]
    if st.session_state["fhe_prediction"] == "":
        encrypted_quantized_encoding = numpy.load(encoded_data_path)

        # Read evaluation_key from the file
        evaluation_key = numpy.load(f"tmp/tmp_evaluation_key_{user_id}.npy")

        # Use base64 to encode the encodings and evaluation key
        encrypted_quantized_encoding = base64.b64encode(
            encrypted_quantized_encoding
        ).decode()
        encoded_evaluation_key = base64.b64encode(evaluation_key).decode()

        query = {}
        query["evaluation_key"] = encoded_evaluation_key
        query["encrypted_encoding"] = encrypted_quantized_encoding
        headers = {"Content-type": "application/json"}
        # pdb.set_trace()
        if task == "0":
            response = requests.post(
                "http://localhost:8000/predict_HLM",
                data=json.dumps(query),
                headers=headers,
            )
        elif task == "1":
            response = requests.post(
                "http://localhost:8000/predict_MDR1MDCK",
                data=json.dumps(query),
                headers=headers,
            )
        elif task == "2":
            response = requests.post(
                "http://localhost:8000/predict_SOLUBILITY",
                data=json.dumps(query),
                headers=headers,
            )
        elif task == "3":
            response = requests.post(
                "http://localhost:8000/predict_PROTEIN_BINDING_HUMAN",
                data=json.dumps(query),
                headers=headers,
            )
        elif task == "4":
            response = requests.post(
                "http://localhost:8000/predict_PROTEIN_BINDING_RAT",
                data=json.dumps(query),
                headers=headers,
            )
        elif task == "5":
            response = requests.post(
                "http://localhost:8000/predict_RLM_CLint",
                data=json.dumps(query),
                headers=headers,
            )
        else:
            print("Invalid task number")
        # pdb.set_trace()
        encrypted_prediction = base64.b64decode(response.json()["encrypted_prediction"])

        # Save encrypted_prediction in a file, since too large to pass through regular Gradio
        # buttons, https://github.com/gradio-app/gradio/issues/1877
        numpy.save(f"tmp/tmp_encrypted_prediction_{user_id}.npy", encrypted_prediction)
        encrypted_prediction_shorten = list(encrypted_prediction)[
            :ENCRYPTED_DATA_BROWSER_LIMIT
        ]
        encrypted_prediction_shorten_hex = "".join(
            f"{i:02x}" for i in encrypted_prediction_shorten
        )
        st.session_state["fhe_prediction"] = encrypted_prediction_shorten_hex

        st.session_state["fhe_done"] = True


def decrypt_prediction(user_id):
    encoded_data_path = Path(f"tmp/tmp_encrypted_prediction_{user_id}.npy")

    # Read encrypted_prediction from the file
    task = st.session_state["task"]
    if st.session_state["decryption_done"] == False:
        encrypted_prediction = numpy.load(encoded_data_path).tobytes()

        fhe_api = FHEModelClient(
            f"deployment/deployment_{task}", f".fhe_keys/{user_id}"
        )
        fhe_api.load()

        # We need to retrieve the private key that matches the client specs (see issue #18)
        fhe_api.generate_private_and_evaluation_keys(force=False)

        predictions = fhe_api.deserialize_decrypt_dequantize(encrypted_prediction)
        st.session_state["decryption_done"] = True
        st.session_state["decrypted_prediction"] = predictions


def process_text(text):
    # Replace the following line with your actual processing code
    processed_text = f"Processed: {text}"
    return processed_text


def init_session_state():
    if "molecule_submitted" not in st.session_state:
        st.session_state["molecule_submitted"] = False

    if "input_molecule" not in st.session_state:
        st.session_state["input_molecule"] = ""

    if "key_generated" not in st.session_state:
        st.session_state["key_generated"] = False

    if "evaluation_key" not in st.session_state:
        st.session_state["evaluation_key"] = []

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = -100

    if "encrypt" not in st.session_state:
        st.session_state["encrypt"] = False

    if "molecule_info_list" not in st.session_state:
        st.session_state["molecule_info_list"] = []

    if "encrypt_tuple" not in st.session_state:
        st.session_state["encrypt_tuple"] = ()

    if "fhe_prediction" not in st.session_state:
        st.session_state["fhe_prediction"] = ""

    if "fhe_done" not in st.session_state:
        st.session_state["fhe_done"] = False

    if "decryption_done" not in st.session_state:
        st.session_state["decryption_done"] = False
    if "decrypted_prediction" not in st.session_state:
        st.session_state[
            "decrypted_prediction"
        ] = ""  # actually a list of list. But python takes care as it is dynamically typed.


def molecule_submitted(text: str):
    msg_to_user = ""
    if len(text) == 0:
        msg_to_user = "Enter a non-empty molecule formula."
        molecule_present = False

    elif Chem.MolFromSmiles(text) == None:
        msg_to_user = "Invalid Molecule. Please enter a valid molecule. How about trying Aspirin or Ibuprofen?"
        molecule_present = False

    else:
        st.session_state["molecule_submitted"] = True
        st.session_state["input_molecule"] = text
        molecule_present = True
        msg_to_user = "Molecule Submitted for Prediction"

    st.session_state["molecule_info_list"].clear()
    st.session_state["molecule_info_list"].append(molecule_present)
    st.session_state["molecule_info_list"].append(msg_to_user)


def keygen_util():
    if st.session_state["molecule_submitted"] == False:
        pass
    else:
        if st.session_state["user_id"] == -100:
            (st.session_state["evaluation_key"], st.session_state["user_id"]) = keygen()
        st.session_state["key_generated"] = True


def encrpyt_data_util():
    if st.session_state["key_generated"] == False:
        pass
    else:
        if len(st.session_state["encrypt_tuple"]) == 0:
            st.session_state["encrypt_tuple"] = encode_quantize_encrypt(
                st.session_state["input_molecule"], st.session_state["user_id"]
            )
        st.session_state["encrypt"] = True


def mol_to_img(mol):
    mol = Chem.MolFromSmiles(mol)
    mol = AllChem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return cairosvg.svg2png(bytestring=svg.encode("utf-8"))


def FHE_util():
    run_fhe(st.session_state["user_id"])


def decrypt_util():
    decrypt_prediction(st.session_state["user_id"])


def clear_session_state():
    st.session_state.clear()


if __name__ == "__main__":
    user_id = 0
    # Set up the Streamlit interface
    # Button
    init_session_state()

    with st.container():
        st.header(":green[Start]")
        st.text(
            "Run all the steps in order to predict the molecule's property. Why not all steps at once? Because we want to show you the steps involved in the process."
        )
        st.subheader("Step 0: Which property do you want to predict?")
        st.text(
            "This app can predict the following properties of confidential molecules:"
        )
        task_options = ["0", "1", "2", "3", "4", "5"]

        task_mapping = {
            "0": "HLM",
            "1": "MDR-1-MDCK-ER",
            "2": "Solubility",
            "3": "Protein bind. human",
            "4": "Protein bind. rat",
            "5": "RLM",
        }

        unit_mapping = {
            "0": "(mL/min/kg)",
            "1": " ",
            "2": "(ug/mL)",
            "3": " (%)",
            "4": " (%)",
            "5": "(mL/min/kg)",
        }
        task_options = list(task_mapping.values())
        # Check if 'task' is not already in session_state
        if "task" not in st.session_state:
            # Initialize it with the first value of your options
            st.session_state["task"] = "0"
            # task_options[0]

        # Create the dropdown menu
        data_dict = {
            "HLM": "Human Liver Microsomes: drug is metabolized by the liver",
            "MDR-1-MDCK-ER": "MDR-1-MDCK-ER: drug is transported by the P-glycoprotein",
            "Solubility": "How soluble a drug is in water",
            "Protein bind. human": "Drug binding to human plasma proteins",
            "Protein bind. rat": "Drug binding to rat plasma proteins",
            "RLM": "Rat Liver Microsomes: Drug metabolism by a rat liver",
        }

        # Convert the dictionary to a DataFrame
        data = pd.DataFrame(
            list(data_dict.items()), columns=["Property", "Explanation"]
        )

        # Custom HTML and CSS styling
        # Custom HTML and CSS styling
        html = data.to_html(index=False, classes="table table-striped table-hover")

        # Custom styling
        st.markdown(
            """
            <style>
                .table {
                    width: 100%;
                    margin: 10px 0 20px 0;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: rgba(0,0,0,.05);
                }
                .table-hover tbody tr:hover {
                    color: #563d7c;
                    background-color: rgba(0,0,0,.075);
                }
                .table thead th, .table tbody td {
                    text-align: center;
                    max-width: 150px;  # Adjust this value as needed
                    word-wrap: break-word;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display the HTML table
        st.write(html, unsafe_allow_html=True)
        st.text("Which to predict?")
        selected_label = st.selectbox(
            "Choose a property",
            task_options,
            index=task_options.index(task_mapping[st.session_state["task"]]),
        )
        st.session_state["task"] = list(task_mapping.keys())[
            task_options.index(selected_label)
        ]

        st.subheader("Step 1: Submit a molecule")

        x, y, z = st.columns(3)

        with x:
            st.text("")

        with y:
            submit_molecule = st.button(
                "Try Aspirin",
                on_click=molecule_submitted,
                args=("CC(=O)OC1=CC=CC=C1C(=O)O",),
            )

        with z:
            submit_molecule = st.button(
                "Try Ibuprofen",
                on_click=molecule_submitted,
                args=("CC(Cc1ccc(cc1)C(C(=O)O)C)C",),
            )

        molecule_to_test = st.text_input(
            "Press Try Aspirin or Ibuprofen - or enter below a molecular SMILES string for predicting its properties. Then press ENTER then CLICK submit "
        )

        submit_molecule = st.button(
            "Submit",
            on_click=molecule_submitted,
            args=(molecule_to_test,),
        )

        if len(st.session_state["molecule_info_list"]) != 0:
            if st.session_state["molecule_info_list"][0] == True:
                st.success(st.session_state["molecule_info_list"][1])
                mol_image = mol_to_img(st.session_state["input_molecule"])
                # center the image
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(mol_image)
                    st.caption(f"Input molecule {st.session_state['input_molecule']}")

            else:
                st.warning(st.session_state["molecule_info_list"][1], icon="⚠️")

    with st.container():
        st.subheader(
            f"Step 2 : Generate encryption key (private to you) and an evaluation key (public)."
        )
        bullet_points = """
        - Evaluation key is public and accessible by server.
        - Private Keys are solely accessible by client for encrypting the information 
        before sending to the server. The same key is used for decryption after FHE inference.
        """
        st.markdown(bullet_points, unsafe_allow_html=True)
        button_gen_key = st.button(
            "Click Here to generate Keys for this session", on_click=keygen_util
        )
        if st.session_state["key_generated"] == True:
            st.success("Keys generated successfully", icon="🙌")
            st.code(f'The user id for this session is {st.session_state["user_id"]} ')
        else:
            task = st.session_state["task"]
            task_label = task_mapping[task]
            st.warning(
                f"Please submit the molecule first to test its {task_label} value",
                icon="⚠️",
            )

    with st.container():
        st.subheader(
            f"Step 3 : Encrypt molecule using private key and send it to server."
        )
        encrypt_button = st.button("Click to Encrypt", on_click=encrpyt_data_util)
        if st.session_state["encrypt"] == True:
            st.success("Successfully Encrypted Data", icon="🙌")
            st.text("The server can only see the encrypted data:")
            st.code(
                f"The encrypted quantized encoding is \n {st.session_state['encrypt_tuple'][2]}..."
            )
        else:
            st.warning(
                "Keys Not Yet Generated. Encryption can be done only after you generate keys."
            )

    with st.container():
        st.subheader(f"Step 4 : Run encrypted prediction on server side.")
        fhe_button = st.button("Click to Predict in FHE domain", on_click=FHE_util)
        if st.session_state["fhe_done"]:
            st.success("Prediction Done Successfuly in FHE domain", icon="🙌")
            st.code(
                f"The encrypted prediction is {st.session_state['fhe_prediction']}..."
            )
        else:
            st.warning("Check if you have generated keys correctly.")

    with st.container():
        st.subheader(f"Step 5 : Decrypt the predictions with your private key.")
        decrypt_button = st.button(
            "Perform Decryption on FHE inferred prediction", on_click=decrypt_util
        )
        if st.session_state["decryption_done"]:
            st.success("Decryption Done successfully!", icon="🙌")
            value = st.session_state["decrypted_prediction"][0][0]
            # 2 digit precision
            value = round(value, 2)
            unit = unit_mapping[st.session_state["task"]]
            task_label = task_mapping[st.session_state["task"]]
            st.code(
                f"The Molecule {st.session_state['input_molecule']} has a {task_label} value of {value} {unit}"
            )
            st.toast("Session successfully completed!!!")
        else:
            st.warning("Check if FHE computation has been done.")

    with st.container():
        st.subheader(f"Step 6 : Reset to predict a new molecule")
        reset_button = st.button("Click Here to Reset", on_click=clear_session_state)
        x, y, z = st.columns(3)
        with x:
            st.write("")
        with y:
            st.markdown(
                "<p style='text-align: center; color: grey;'>"
                + img_to_html("VaultChem.png", width="50%")
                + "</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h6 style='text-align: center; color: grey;'>Visit our website : <a href='https://vaultchem.com/'>VaultChem</a></h6>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h6 style='text-align: center; color: grey;'>Visit our Github Repo : <a href='https://github.com/vaultchem'>Github</a></h6>",
                unsafe_allow_html=True,
            )

            # new_link_url = "https://github.com/vaultchem"
            # new_alias_text = "Github"
            # st.markdown(f"Visit our Github Repo : [{new_alias_text}]({new_link_url})")
            # link_url = "https://streamlit.io/"
            # alias_text = "Streamlit"
            # st.markdown(f"Built with [{alias_text}]({link_url})🎈")
            st.markdown(
                "<h6 style='text-align: center; color: grey;'>Built with <a href='https://streamlit.io/'>Streamlit</a>🎈</h6>",
                unsafe_allow_html=True,
            )
        with z:
            st.write("")

st.markdown(
    """
    <div style="width: 100%; text-align: center; padding: 10px;">
        The app was built with <a href="https://docs.zama.ai/concrete-ml" target="_blank">Concrete ML</a>,
        a Privacy-Preserving Machine Learning (PPML) open-source set of tools by Zama.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    ":red[Please Note]: The content of your app is purely for educational and illustrative purposes and is not intended for the management of sensitive information. We disclaim any liability for potential financial or other damages. This platform is not a substitute for professional health advice, diagnosis, or treatment. Health-related inquiries should be directed to qualified medical professionals. Use of this app implies acknowledgment of these terms and understanding of its intended educational use."
)