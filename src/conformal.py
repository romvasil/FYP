import os
import torch
import numpy as np
from crepes import ConformalRegressor, ConformalClassifier
from crepes.extras import hinge
from models import LinearBaseline, MLPBaseline, LSTMBaseline

DATA_PATH = "DataSetsProcessed"
MODELS_PATH = "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.40]

def load_data_for_cp(dataset, target, task):
    X_cal = np.load(f"{DATA_PATH}/{dataset}_X_val.npy")
    y_cal = np.load(f"{DATA_PATH}/{dataset}_y_{target}_{task}_val.npy")
    
    X_test = np.load(f"{DATA_PATH}/{dataset}_X_test.npy")
    y_test = np.load(f"{DATA_PATH}/{dataset}_y_{target}_{task}_test.npy")

    X_cal_tensor = torch.FloatTensor(X_cal).to(DEVICE)
    X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
    
    return X_cal, y_cal, X_test, y_test, X_cal_tensor, X_test_tensor

def get_model_predictions(model, X_tensor, task):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        
        if task == 'reg':
            return outputs.cpu().numpy().flatten()
        else:
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

def evaluate_conformal():
    models_dict = {
        "Linear": LinearBaseline,
        "MLP": MLPBaseline,
        "LSTM": LSTMBaseline
    }
    
    experiments = [
        ('RECOLA', 'arousal', 'reg'), ('RECOLA', 'valence', 'reg'),
        ('RECOLA', 'arousal', 'class'), ('RECOLA', 'valence', 'class'),
        ('AGAIN', 'arousal', 'reg'), ('AGAIN', 'arousal', 'class')
    ]

    for dataset, target, task in experiments:
        print(f"\n{'='*50}\nEvaluating {dataset} - {target} - {task.upper()}\n{'='*50}")
        
        X_cal, y_cal, X_test, y_test, X_cal_tensor, X_test_tensor = load_data_for_cp(dataset, target, task)
        input_dim = X_cal.shape[-1]
        output_dim = 1 if task == 'reg' else 5 
        
        for model_name, ModelClass in models_dict.items():
            run_name = f"{dataset}_{target}_{task}_{model_name}"
            model_path = f"{MODELS_PATH}/{run_name}.pth"
            
            if not os.path.exists(model_path):
                continue
                
            print(f"\n--- Model: {model_name} ---")
            
            model = ModelClass(input_dim, output_dim).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            
            cal_preds = get_model_predictions(model, X_cal_tensor, task)
            test_preds = get_model_predictions(model, X_test_tensor, task)
            
            if task == 'reg':
                residuals = y_cal - cal_preds
                cr = ConformalRegressor()
                cr.fit(residuals=residuals)
                
                for alpha in ALPHAS:
                    conf_level = (1 - alpha) * 100
                    intervals = cr.predict_int(test_preds, confidence=1 - alpha)
                    
                    avg_width = np.mean(intervals[:, 1] - intervals[:, 0])
                    print(f"Confidence {conf_level:.0f}% (alpha={alpha}): Average Interval Width = {avg_width:.4f}")
                    
            else:
                classes = np.array([0, 1, 2, 3, 4])
                alphas_cal = hinge(cal_preds, classes, y_cal)
                
                cc = ConformalClassifier()
                cc.fit(alphas_cal) 
                
                alphas_test = hinge(test_preds)
                
                for alpha in ALPHAS:
                    conf_level = (1 - alpha) * 100
                    prediction_sets = cc.predict_set(alphas_test, confidence=1 - alpha)
                    
                    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))
                    print(f"Confidence {conf_level:.0f}% (alpha={alpha}): Average Set Size = {avg_set_size:.2f} classes")

if __name__ == "__main__":
    evaluate_conformal()