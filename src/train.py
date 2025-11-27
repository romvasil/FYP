import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import LinearBaseline, MLPBaseline, LSTMBaseline

DATA_PATH = "DataSetsProcessed"
MODELS_PATH = "models"
RESULTS_FILE = "baseline_results.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

def load_data(dataset, target, task):
    X_train = np.load(f"{DATA_PATH}/{dataset}_X_train.npy")
    X_val = np.load(f"{DATA_PATH}/{dataset}_X_val.npy")
    y_train = np.load(f"{DATA_PATH}/{dataset}_y_{target}_{task}_train.npy")
    y_val = np.load(f"{DATA_PATH}/{dataset}_y_{target}_{task}_val.npy")
    
    X_train = torch.FloatTensor(X_train).to(DEVICE)
    X_val = torch.FloatTensor(X_val).to(DEVICE)
    
    if task == 'reg':
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)
    else:
        y_train = torch.LongTensor(y_train).to(DEVICE)
        y_val = torch.LongTensor(y_val).to(DEVICE)
        
    return X_train, y_train, X_val, y_val

def calculate_ccc(y_true, y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true, unbiased=False)
    var_pred = torch.var(y_pred, unbiased=False)
    covar = torch.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2)

def train_model(run_name, dataset, target, task, ModelClass):
    X_train, y_train, X_val, y_val = load_data(dataset, target, task)
    
    input_dim = X_train.shape[1]
    output_dim = 1 if task == 'reg' else 5
    
    model = ModelClass(input_dim, output_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() if task == 'reg' else nn.CrossEntropyLoss()
    
    best_score = -999.0
    
    print(f"Training {run_name}...")
    
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        
        for i in range(0, X_train.size(0), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            if task == 'reg':
                score = calculate_ccc(y_val, val_out).item()
            else:
                predictions = torch.argmax(val_out, dim=1)
                score = (predictions == y_val).float().mean().item()
        
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"{MODELS_PATH}/{run_name}.pth")

    result_string = f"{run_name}: {'CCC' if task == 'reg' else 'Acc'} = {best_score:.4f}"
    print(result_string)
    with open(RESULTS_FILE, "a") as f:
        f.write(result_string + "\n")

if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    experiments = []
    
    models = [
        (LinearBaseline, "Linear"),
        (MLPBaseline, "MLP"),
        (LSTMBaseline, "LSTM")
    ]
    
    for target in ['arousal', 'valence']:
        for task in ['reg', 'class']:
            for ModelClass, model_name in models:
                experiments.append(('RECOLA', target, task, ModelClass, model_name))
                
    for task in ['reg', 'class']:
        for ModelClass, model_name in models:
            experiments.append(('AGAIN', 'arousal', task, ModelClass, model_name))

    for dataset, target, task, ModelClass, model_name in experiments:
        run_name = f"{dataset}_{target}_{task}_{model_name}"
        train_model(run_name, dataset, target, task, ModelClass)