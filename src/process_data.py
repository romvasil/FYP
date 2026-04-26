import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

RAW_PATH = "DataSetsRaw"
PROC_PATH = "DataSetsProcessed"
GENRE = "Shooter"
WINDOW_SIZE = 20
np.random.seed(42)


def get_bins(a):
    a_jit = a + np.random.uniform(-1e-6, 1e-6, size=len(a))
    return [-np.inf] + sorted(np.percentile(a_jit, [20, 40, 60, 80])) + [np.inf]


def make_windows(X, labels_dict, groups):
    X_out, groups_out = [], []
    y_out = {k: [] for k in labels_dict}

    for pid in np.unique(groups):
        mask = groups == pid
        Xp = X[mask]
        if len(Xp) < WINDOW_SIZE:
            continue
        for start in range(len(Xp) - WINDOW_SIZE + 1):
            X_out.append(Xp[start:start + WINDOW_SIZE])
            groups_out.append(pid)
            for k in labels_dict:
                y_out[k].append(labels_dict[k][mask][start + WINDOW_SIZE - 1])

    return (
        np.array(X_out),
        {k: np.array(v) for k, v in y_out.items()},
        np.array(groups_out)
    )


def split_and_save(name, X_raw, y_reg_dict, groups):
    ref_y = next(iter(y_reg_dict.values()))

    s1 = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_i, temp_i = next(s1.split(X_raw, ref_y, groups))

    s2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_sub, test_sub = next(s2.split(X_raw[temp_i], ref_y[temp_i], groups[temp_i]))
    val_i, test_i = temp_i[val_sub], temp_i[test_sub]

    scaler = StandardScaler()
    X_scaled = {
        'train': scaler.fit_transform(X_raw[train_i]),
        'val':   scaler.transform(X_raw[val_i]),
        'test':  scaler.transform(X_raw[test_i]),
    }
    idx_splits = {'train': train_i, 'val': val_i, 'test': test_i}

    bins = {target: get_bins(y_reg_dict[target][train_i]) for target in y_reg_dict}

    for split_name, idx in idx_splits.items():
        split_groups = groups[idx]
        labels = {}
        for target, y_arr in y_reg_dict.items():
            labels[f'{target}_reg'] = y_arr[idx]
            labels[f'{target}_class'] = pd.cut(
                y_arr[idx], bins=bins[target],
                labels=[0, 1, 2, 3, 4], include_lowest=True
            ).astype(int)

        X_win, y_win, _ = make_windows(X_scaled[split_name], labels, split_groups)

        np.save(f"{PROC_PATH}/{name}_X_{split_name}.npy", X_win)
        for target_name, target_data in y_win.items():
            np.save(f"{PROC_PATH}/{name}_y_{target_name}_{split_name}.npy", target_data)


def process_recola():
    path = os.path.join(RAW_PATH, "all_participants_data.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if "ComParE13" in c]
    X_raw = df[feature_cols].fillna(0).values
    groups = df['Participant'].values

    y_reg_dict = {
        'arousal': df['median_arousal'].values,
        'valence': df['median_valence'].values,
    }

    split_and_save("RECOLA", X_raw, y_reg_dict, groups)


def process_again():
    files = glob.glob(os.path.join(RAW_PATH, "AGAIN_Extracted", "**", "*clean*.csv"), recursive=True)
    if not files:
        return
    df = pd.read_csv(files[0], low_memory=False)
    df = df[df['[control]genre'] == GENRE].copy().dropna(subset=['[output]arousal'])

    feature_cols = [c for c in df.columns if '[' not in c]
    X_raw = df[feature_cols].fillna(0).values
    groups = df['[control]player_id'].values

    y_reg_dict = {
        'arousal': df['[output]arousal'].values,
    }

    split_and_save("AGAIN", X_raw, y_reg_dict, groups)


if __name__ == "__main__":
    os.makedirs(PROC_PATH, exist_ok=True)
    process_recola()
    process_again()