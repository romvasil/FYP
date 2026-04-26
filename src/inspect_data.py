import os
import zipfile
import glob
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'DataSetsRaw')
EXTRACT_PATH = os.path.join(RAW_PATH, 'AGAIN_Extracted')

def inspect_again():
    zip_files = glob.glob(os.path.join(RAW_PATH, "*.zip"))
    if not zip_files:
        return
    
    if not os.path.exists(EXTRACT_PATH):
        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
            
    csv_files = glob.glob(os.path.join(EXTRACT_PATH, "**", "*clean*.csv"), recursive=True)
    if not csv_files:
        csv_files = glob.glob(os.path.join(EXTRACT_PATH, "**", "*.csv"), recursive=True)
        
    if csv_files:
        df = pd.read_csv(csv_files[0], low_memory=False)
        
        print("AGAIN DATASET STATISTICS")
        print("=" * 50)
        print(f"Total Rows: {len(df)}")
        
        if '[control]genre' in df.columns and '[control]player_id' in df.columns:
            print("\nGenre Distribution:")
            genre_counts = df['[control]genre'].value_counts()
            for genre, count in genre_counts.items():
                players = df[df['[control]genre'] == genre]['[control]player_id'].nunique()
                print(f"{genre:<15} | Data Points: {count:<8} | Unique Players: {players}")
                
        if '[output]arousal' in df.columns:
            print("\nArousal Distribution:")
            print(df['[output]arousal'].describe().to_string())
        print("=" * 50)

def inspect_recola():
    recola_path = os.path.join(RAW_PATH, 'RECOLA')
    feature_files = glob.glob(os.path.join(recola_path, "**", "*.csv"), recursive=True)
    
    if feature_files:
        df = pd.read_csv(feature_files[0])
        
        print("\nRECOLA DATASET STATISTICS")
        print("=" * 50)
        print(f"Total Feature Files: {len(feature_files)}")
        print(f"Sample File Shape: {df.shape}")
        
        if 'frameTime' in df.columns:
            framerate = df['frameTime'].iloc[1] - df['frameTime'].iloc[0]
            print(f"Time Step: {framerate:.3f} seconds")
        print("=" * 50)

if __name__ == "__main__":
    inspect_again()
    inspect_recola()