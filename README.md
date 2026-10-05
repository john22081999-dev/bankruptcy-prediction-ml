# Bankruptcy Prediction — Classification & Model Comparison

A machine-learning project that predicts company bankruptcy from financial
indicators, comparing eight classifiers and studying the effect of class
balancing on performance.

## Problem

Given a set of yearly company financial indicators, classify each company as
**healthy** or **bankrupt**. The dataset is imbalanced (far more healthy than
bankrupt companies), so the project also measures how balancing the training
data changes each model's behaviour.

## Approach

- **Exploratory analysis:** distribution of companies per year and summary
  statistics of the financial indicators (see `figures/`).
- **Cross-validation:** `StratifiedKFold` so each fold keeps the class ratio.
- **Class balancing:** every model is trained and evaluated twice — on the
  original (unbalanced) data and on a balanced version — to compare results.
- **Models compared (8):** Linear Discriminant Analysis, Logistic Regression,
  Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors,
  Gaussian Naive Bayes, and Support Vector Classifier.
- **Metrics:** accuracy, precision, recall, F1 and ROC-AUC, plus a confusion
  matrix per fold (train and test), for both the balanced and unbalanced runs.

## What's in this repo

- `assignment1_classification.py` — main pipeline (training, evaluation, plots)
- `build_excel_report.py` — builds an Excel summary of the results
- `balancedDataOutcomes.csv` / `.xlsx` — the metrics output across models/folds
- `figures/` — EDA plots and all confusion matrices (balanced & unbalanced)

## Dataset

The input dataset was provided as part of a university course and is **not
redistributed here**. To run the pipeline yourself, place the Excel file at:

```
data/Dataset2Use_Assignment1.xlsx
```

(The `data/` folder is git-ignored.) The code expects a tabular file of company
financial indicators with a binary bankruptcy label.

## Tech stack

- Python 3
- scikit-learn, pandas, numpy
- matplotlib (plots), openpyxl (Excel I/O)

## How to run

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Put the dataset at `data/Dataset2Use_Assignment1.xlsx`.
3. Run the pipeline:
   ```
   python assignment1_classification.py
   ```

## Notes

Started as a university machine-learning assignment; cleaned up and documented
as a standalone portfolio project.
