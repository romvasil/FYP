# Quantifying Uncertainty in Affect Modelling
## Roman Vasilets

This repository contains the source code for a Final Year Project (FYP) investigating the application of **Conformal Prediction** to quantify uncertainty in affective computing and emotion recognition models. 

The project evaluates deep learning baselines (Linear, MLP, LSTM) on continuous arousal and valence prediction tasks (both regression and classification) using multimodal and gameplay datasets (RECOLA and AGAIN). It utilizes Normalized Conformal Prediction to generate prediction intervals and sets with mathematically guaranteed coverage, shifting the focus from point predictions to reliable uncertainty estimation.

## 📂 Repository Structure

Based on the required data pipeline and model architecture, the repository is structured as follows:

```text
├── DataSetsRaw/            # (Not tracked in Git) Raw dataset files (AGAIN .zips, RECOLA features)
├── DataSetsProcessed/      # (Not tracked in Git) Preprocessed, scaled, and split .npy files
├── models/                 # Saved PyTorch model weights (.pth files)
├── src/                    # Main source code directory
│   ├── inspect_data.py     # Exploratory data analysis and dataset statistics extraction
│   ├── process_data.py     # Data cleaning, windowing, and GroupShuffleSplit preprocessing
│   ├── models.py           # PyTorch definitions for Linear, MLP, and LSTM architectures
│   ├── train.py            # Model training loop, validation, and baseline metric logging
│   └── conformal.py        # Conformal prediction evaluation (using the `crepes` library)
├── .gitignore              # Ignores raw data, virtual environments, and caches
├── baseline_results.txt    # Output logs for baseline model performance (CCC, Accuracy, MSE)
└── requirements.txt        # Python dependencies
