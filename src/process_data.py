import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

RAW_PATH = "DataSetsRaw"
PROC_PATH = "DataSetsProcessed"
GENRE = "Shooter"
np.random.seed(42)

def get_bins(a):
    # Add tiny jitter to avoid unique bin edge errors
    a_jit = a + np.random.uniform(-1e-6, 1e-6, size=len(a))
    return [-np.inf] + sorted(np.percentile(a_jit, [20, 40, 60, 80])) + [np.inf]

def split_and_save(name, X, y_dict, groups):
    """
    y_dict: Dictionary containing {'arousal_reg': ..., 'arousal_class': ..., 'valence_reg': ...}
    """
    # 1. Split indices based on Groups (Subjects)
    s1 = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_i, temp_i = next(s1.split(X, y_dict['arousal_reg'], groups))
    
    s2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_sub, test_sub = next(s2.split(X[temp_i], y_dict['arousal_reg'][temp_i], groups[temp_i]))
    val_i, test_i = temp_i[val_sub], temp_i[test_sub]
    
    # 2. Save Features (X)
    for n, i in [('train', train_i), ('val', val_i), ('test', test_i)]:
        np.save(f"{PROC_PATH}/{name}_X_{n}.npy", X[i])
        
        # 3. Save ALL Targets (Arousal & Valence)
        for target_name, target_data in y_dict.items():
            # target_name e.g., 'arousal_reg', 'valence_class'
            # Save as: RECOLA_y_arousal_reg_train.npy
            np.save(f"{PROC_PATH}/{name}_y_{target_name}_{n}.npy", target_data[i])

def process_recola():
    path = os.path.join(RAW_PATH, "all_participants_data.csv")
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    
    # Features
    X = StandardScaler().fit_transform(df[[c for c in df.columns if "ComParE13" in c]].fillna(0))
    groups = df['Participant'].values
    
    # Targets
    y_ar_reg = df['median_arousal'].values
    y_val_reg = df['median_valence'].values
    
    y_dict = {
        'arousal_reg': y_ar_reg,
        'arousal_class': pd.cut(y_ar_reg, bins=get_bins(y_ar_reg), labels=[0,1,2,3,4], include_lowest=True).astype(int),
        'valence_reg': y_val_reg,
        'valence_class': pd.cut(y_val_reg, bins=get_bins(y_val_reg), labels=[0,1,2,3,4], include_lowest=True).astype(int)
    }
    
    split_and_save("RECOLA", X, y_dict, groups)

def process_again():
    files = glob.glob(os.path.join(RAW_PATH, "AGAIN_Extracted", "**", "*clean*.csv"), recursive=True)
    if not files: return
    
    df = pd.read_csv(files[0], low_memory=False)
    
    df = df[df['[control]genre'] == GENRE].copy().dropna(subset=['[output]arousal'])
    X = StandardScaler().fit_transform(df.drop(columns=[c for c in df.columns if '[' in c]).fillna(0))
    groups = df['[control]player_id'].values
    
    y_ar_reg = df['[output]arousal'].values
    
    y_dict = {
        'arousal_reg': y_ar_reg,
        # Use .values to ensure it's a numpy array, avoiding Index Error
        'arousal_class': df.groupby('[control]player_id')['[output]arousal'].transform(lambda x: pd.cut(x, bins=get_bins(x.values), labels=[0,1,2,3,4], include_lowest=True)).astype(int).values
    }
    
    split_and_save("AGAIN", X, y_dict, groups)

if __name__ == "__main__":
    os.makedirs(PROC_PATH, exist_ok=True)
    process_recola()
    process_again()
    print("Done. Processed Arousal (AGAIN/RECOLA) and Valence (RECOLA only).")