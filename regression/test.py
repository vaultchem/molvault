import pandas as pd
import requests
from io import StringIO

def download_and_process_csv(url, output_file):
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # This will raise an error for a failed request

    # Read the content of the file into a pandas DataFrame
    data = pd.read_csv(StringIO(response.text))

    # Select only the columns you need
    selected_columns = data[['SMILES', 'LOG MDR1-MDCK ER (B-A/A-B)']]

    # Write these columns to a new CSV file
    selected_columns.to_csv(output_file, index=False)

# Example usage
url = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"
#"YOUR_CSV_FILE_URL_HERE"
output_file = "output.csv"
download_and_process_csv(url, output_file)

