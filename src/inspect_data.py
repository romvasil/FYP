import os
import zipfile
import glob
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'DataSetsRaw')
EXTRACT_PATH = os.path.join(RAW_PATH, 'AGAIN_Extracted')

def inspect_again():
    print("--- INSPECTING AGAIN DATASET ---")
    
    zip_files = glob.glob(os.path.join(RAW_PATH, "*.zip"))
    if not zip_files:
        print(f"ERROR: No .zip file found in {RAW_PATH}")
        return
    
    zip_file_path = zip_files[0]
    print(f"Found zip file: {os.path.basename(zip_file_path)}")
    
    if not os.path.exists(EXTRACT_PATH):
        print(f"Unzipping to {EXTRACT_PATH}...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
    else:
        print("Folder already unzipped. Skipping extraction.")
        
    csv_files = glob.glob(os.path.join(EXTRACT_PATH, "**", "*.csv"), recursive=True)
    if csv_files:
        first_csv = csv_files[0]
        print(f"Reading file: {os.path.basename(first_csv)}")
        df = pd.read_csv(first_csv, nrows=5)
        print("COLUMNS FOUND:")
        print(df.columns.tolist())
        
        if 'game' in df.columns:
            print(f"Game Column found: {df['game'].unique()}")
    else:
        print("No CSV files found inside the zip!")

def inspect_recola():
    print("\n--- INSPECTING RECOLA DATASET ---")
    csv_path = os.path.join(RAW_PATH, "all_participants_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=5)
        print(f"Confirmed RECOLA file found. Features: {len(df.columns)}")
    else:
        print("RECOLA csv not found.")

if __name__ == "__main__":
    inspect_recola()
    inspect_again()