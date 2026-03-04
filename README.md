# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Python and scikit-learn. The model is trained on an imbalanced dataset and uses under-sampling plus logistic regression to classify transactions as legitimate or fraudulent.

## Overview

The notebook loads the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), explores the data, handles class imbalance via under-sampling, and trains a logistic regression classifier to predict fraud.

## Dataset

- **Source**: Credit card transactions (anonymized; features V1–V28 are PCA components).
- **Size**: 284,807 transactions.
- **Columns**: `Time`, `V1`–`V28` (PCA features), `Amount`, `Class`.
- **Target**: `Class` — `0` = legitimate, `1` = fraudulent.
- **Imbalance**: 284,315 legitimate vs 492 fraudulent (~0.17% fraud).

## Project Structure

```
fraud-detection/
├── credit_card_fraud_detection.ipynb   # Main analysis and model notebook
├── creditcard.csv                      # Dataset (place in project root)
├── README.md
└── .gitignore
```

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn

Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

Or with a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install numpy pandas scikit-learn
```

## Getting Started

1. **Get the data**  
   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root (same folder as the notebook).

2. **Run the notebook**  
   Open `credit_card_fraud_detection.ipynb` in Jupyter Notebook or JupyterLab and run all cells.

   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

## Methodology

1. **Exploration** — Load data, check shape, missing values, and class distribution.
2. **Under-sampling** — Balance classes by sampling 492 legitimate transactions to match the 492 fraud cases (total 984 samples).
3. **Train/test split** — 80% train / 20% test with stratification on `Class`.
4. **Model** — Logistic regression (no scaling in the current notebook; consider `StandardScaler` for better convergence).
5. **Evaluation** — Accuracy on training and test sets.

## Results

Reported in the notebook:

- **Training accuracy**: ~93.8%
- **Test accuracy**: ~94.4%

For production use, consider reporting precision, recall, F1, and ROC-AUC, especially for the positive (fraud) class, since false negatives are costly in fraud detection.

## License

Dataset usage is subject to the Kaggle dataset license. Code in this repository is provided as-is for learning and experimentation.
