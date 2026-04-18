import matplotlib.pyplot as plt
import numpy as np

# --- 1. The Data ---
# Confidence levels from lowest to highest for a standard x-axis
conf_levels = [60, 80, 90, 95, 99]

# Data extracted from your conformal prediction run
# Reversed from your terminal output so it aligns with the 60 -> 99 x-axis
results = {
    "RECOLA_Arousal": {
        "Regression": {
            "Linear": [0.3337, 0.4862, 0.5931, 0.7058, 0.8831],
            "MLP":    [0.3217, 0.4665, 0.5851, 0.6998, 0.8997],
            "LSTM":   [0.3178, 0.4764, 0.6061, 0.7133, 0.9182]
        },
        "Classification": {
            "Linear": [2.22, 3.22, 3.87, 4.23, 4.70],
            "MLP":    [2.02, 3.03, 3.61, 4.09, 4.85],
            "LSTM":   [2.02, 3.01, 3.59, 4.09, 4.80]
        }
    },
    "RECOLA_Valence": {
        "Regression": {
            "Linear": [0.2222, 0.3454, 0.4619, 0.5818, 0.7139],
            "MLP":    [0.2174, 0.3280, 0.4462, 0.5708, 0.8119],
            "LSTM":   [0.2474, 0.3911, 0.5206, 0.6374, 0.8367]
        },
        "Classification": {
            "Linear": [2.68, 3.73, 4.27, 4.62, 4.87],
            "MLP":    [2.62, 3.67, 4.24, 4.61, 4.91],
            "LSTM":   [2.71, 3.85, 4.42, 4.70, 4.91]
        }
    },
    "AGAIN_Arousal": {
        "Regression": {
            "Linear": [0.4506, 0.6285, 0.7913, 0.9285, 1.2286],
            "MLP":    [0.4405, 0.6270, 0.7953, 0.9477, 1.3087],
            "LSTM":   [0.4392, 0.6479, 0.8221, 0.9788, 1.3516]
        },
        "Classification": {
            "Linear": [1.89, 2.87, 3.63, 4.13, 4.79],
            "MLP":    [1.75, 2.80, 3.57, 4.14, 4.76],
            "LSTM":   [1.78, 2.85, 3.59, 4.18, 4.78]
        }
    }
}

# Styling variables
colors = {"Linear": "#1f77b4", "MLP": "#ff7f0e", "LSTM": "#2ca02c"}
markers = {"Linear": "o", "MLP": "s", "LSTM": "^"}

# --- 2. Generate the Plots ---
for dataset_name, tasks in results.items():
    # Create a figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Conformal Prediction Uncertainty: {dataset_name.replace('_', ' ')}", fontsize=16, y=1.05)

    # Plot Regression (Left Panel)
    ax_reg = axes[0]
    for model_name, widths in tasks["Regression"].items():
        ax_reg.plot(conf_levels, widths, label=model_name, color=colors[model_name], 
                    marker=markers[model_name], linewidth=2, markersize=8)
    
    ax_reg.set_title("Regression: Average Interval Width", fontsize=14)
    ax_reg.set_xlabel("Confidence Level (%)", fontsize=12)
    ax_reg.set_ylabel("Interval Width (CCC scale)", fontsize=12)
    ax_reg.set_xticks(conf_levels)
    ax_reg.grid(True, linestyle='--', alpha=0.7)
    ax_reg.legend(fontsize=11)

    # Plot Classification (Right Panel)
    ax_class = axes[1]
    for model_name, sizes in tasks["Classification"].items():
        ax_class.plot(conf_levels, sizes, label=model_name, color=colors[model_name], 
                      marker=markers[model_name], linewidth=2, markersize=8)
    
    ax_class.set_title("Classification: Average Set Size", fontsize=14)
    ax_class.set_xlabel("Confidence Level (%)", fontsize=12)
    ax_class.set_ylabel("Number of Classes Included", fontsize=12)
    ax_class.set_xticks(conf_levels)
    ax_class.set_yticks([1, 2, 3, 4, 5]) # Since there are 5 classes
    ax_class.grid(True, linestyle='--', alpha=0.7)
    ax_class.legend(fontsize=11)

    # Adjust layout and save as a high-resolution PNG
    plt.tight_layout()
    file_name = f"CP_Results_{dataset_name}.png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print(f"Successfully saved {file_name}")
    plt.close()