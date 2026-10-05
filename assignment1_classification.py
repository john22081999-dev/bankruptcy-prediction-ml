"""
Bankruptcy Prediction - Classification & Model Comparison

Compares eight classifiers for predicting company bankruptcy using
stratified k-fold cross-validation, and evaluates performance on both
balanced and unbalanced versions of the training data.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

DATA_PATH = "data/Dataset2Use_Assignment1.xlsx"

OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CM_DIR = os.path.join(FIG_DIR, "confusion_matrices")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

RANDOM_STATE = 42         
N_FOLDS = 4                 
TARGET_RATIO = 3            
RESULTS_CSV = os.path.join(OUTPUT_DIR, "balancedDataOutcomes.csv")

np.random.seed(RANDOM_STATE)

# Positive class = "bankrupt" (status == 2) ->  1
# Negative class = "healthy"  (status == 1) ->  0
POSITIVE_LABEL = "bankrupt"
NEGATIVE_LABEL = "healthy"


def status_to_binary(series):
    return (series == 2).astype(int)


print("=" * 90)
print("STEP 1: Reading data from Excel file:", DATA_PATH)
print("=" * 90)

df = pd.read_excel(DATA_PATH)


PERFORMANCE_COLS = df.columns[0:8].tolist()
ACTIVITY_COLS = df.columns[8:11].tolist()
STATUS_COL = df.columns[11]
YEAR_COL = df.columns[12]
FEATURE_COLS = PERFORMANCE_COLS + ACTIVITY_COLS  

df["target"] = status_to_binary(df[STATUS_COL])

print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Performance indicators (A-H): {PERFORMANCE_COLS}")
print(f"Activity indicators   (I-K): {ACTIVITY_COLS}")
print(f"Status column: '{STATUS_COL}'  |  Year column: '{YEAR_COL}'")
print(f"Healthy companies : {(df['target'] == 0).sum()}")
print(f"Bankrupt companies: {(df['target'] == 1).sum()}")


print("\n" + "=" * 90)
print("STEP 2: Generating Figure 1 (healthy / bankrupt companies per year)")
print("=" * 90)

year_status = (
    df.groupby([YEAR_COL, "target"]).size().unstack(fill_value=0).rename(
        columns={0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}
    )
)
print(year_status)

fig, ax = plt.subplots(figsize=(8, 5))
years = year_status.index.astype(str)
width = 0.35
x = np.arange(len(years))
ax.bar(x - width / 2, year_status[NEGATIVE_LABEL], width, label="Healthy", color="#2a9d8f")
ax.bar(x + width / 2, year_status[POSITIVE_LABEL], width, label="Bankrupt", color="#e76f51")
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_xlabel("Year")
ax.set_ylabel("Number of companies")
ax.set_title("Figure 1: Healthy vs Bankrupt companies per year")
ax.legend()
for i, v in enumerate(year_status[NEGATIVE_LABEL]):
    ax.text(i - width / 2, v + 20, str(v), ha="center", fontsize=8)
for i, v in enumerate(year_status[POSITIVE_LABEL]):
    ax.text(i + width / 2, v + 20, str(v), ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure1_companies_per_year.png"), dpi=150)
plt.close()
print(f"Saved: {FIG_DIR}/figure1_companies_per_year.png")


print("\n" + "=" * 90)
print("STEP 3: Generating Figure 2 (min/max/avg per indicator, healthy vs bankrupt)")
print("=" * 90)

healthy_df = df[df["target"] == 0][PERFORMANCE_COLS]
bankrupt_df = df[df["target"] == 1][PERFORMANCE_COLS]

healthy_stats = healthy_df.agg(["min", "max", "mean"]).T
bankrupt_stats = bankrupt_df.agg(["min", "max", "mean"]).T
print("\nHealthy companies - indicator statistics:")
print(healthy_stats.round(2))
print("\nBankrupt companies - indicator statistics:")
print(bankrupt_stats.round(2))

short_labels = [f"Ind.{i+1}" for i in range(len(PERFORMANCE_COLS))]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
for ax, stats, title, color_set in zip(
    axes,
    [healthy_stats, bankrupt_stats],
    ["Healthy companies", "Bankrupt companies"],
    [("#264653", "#2a9d8f", "#8ab17d"), ("#9d0208", "#e76f51", "#f4a261")],
):
    xpos = np.arange(len(short_labels))
    w = 0.27
    ax.bar(xpos - w, stats["min"], width=w, label="Min", color=color_set[0])
    ax.bar(xpos, stats["mean"], width=w, label="Average", color=color_set[1])
    ax.bar(xpos + w, stats["max"], width=w, label="Max", color=color_set[2])
    ax.set_xticks(xpos)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_yscale("symlog")  
    ax.legend()

fig.suptitle("Figure 2: Min / Max / Average value per performance indicator")
legend_txt = "  |  ".join(f"{s}={c}" for s, c in zip(short_labels, PERFORMANCE_COLS))
fig.text(0.5, -0.05, legend_txt, ha="center", fontsize=7, wrap=True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure2_indicator_stats.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_DIR}/figure2_indicator_stats.png")


print("\n" + "=" * 90)
print("STEP 4: Checking for missing values (NaNs)")
print("=" * 90)

missing_counts = df[FEATURE_COLS + [STATUS_COL, YEAR_COL]].isna().sum()
total_missing = missing_counts.sum()
if total_missing > 0:
    print("WARNING: missing values detected in the dataset!")
    print(missing_counts[missing_counts > 0])
else:
    print("No missing values (NaNs) were found in the dataset. Data is complete.")


print("\n" + "=" * 90)
print("STEP 5: Min-Max normalization of the performance indicators to [0, 1]")
print("=" * 90)


df_norm = df.copy()
minmax_params = {}
for col in PERFORMANCE_COLS:
    col_min, col_max = df[col].min(), df[col].max()
    minmax_params[col] = (col_min, col_max)
    if col_max > col_min:
        df_norm[col] = (df[col] - col_min) / (col_max - col_min)
    else:
        df_norm[col] = 0.0 

print("Min-max normalization parameters (per indicator):")
for col, (mn, mx) in minmax_params.items():
    print(f"  {col:45s} min={mn:12.4f}  max={mx:12.4f}")

print("\nPost-normalization sanity check (should all be within [0, 1]):")
print(df_norm[PERFORMANCE_COLS].describe().loc[["min", "max"]].round(3))

X_all = df_norm[FEATURE_COLS].reset_index(drop=True)
y_all = df_norm["target"].reset_index(drop=True)


def build_models():
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred, y_score):
    """Return accuracy, precision, recall, f1, auc and the confusion matrix
    (tn, fp, fn, tp) for the positive class = bankrupt = 1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return acc, prec, rec, f1, auc, tn, fp, fn, tp


def plot_confusion_grid(cm_dict, title, filepath):
    """cm_dict: {model_name: (tn, fp, fn, tp)} -> combined grid figure with
    one confusion-matrix heatmap per classifier."""
    n = len(cm_dict)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, (tn, fp, fn, tp)) in zip(axes, cm_dict.items()):
        mat = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(mat, cmap="Blues")
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, int(val), ha="center", va="center",
                     color="white" if val > mat.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Healthy", "Bankrupt"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Healthy", "Bankrupt"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close()



print("\n" + "=" * 90)
print(f"STEP 6: Stratified {N_FOLDS}-Fold split, class balancing and model training")
print("=" * 90)


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results_rows = []
fold_num = 0

for train_idx, test_idx in skf.split(X_all, y_all):
    fold_num += 1
    print("\n" + "-" * 90)
    print(f"FOLD {fold_num}/{N_FOLDS}")
    print("-" * 90)

    X_train_full, y_train_full = X_all.iloc[train_idx].reset_index(drop=True), y_all.iloc[train_idx].reset_index(drop=True)
    X_test, y_test = X_all.iloc[test_idx].reset_index(drop=True), y_all.iloc[test_idx].reset_index(drop=True)

    # train/test class distribution
    n_train_bankrupt = int((y_train_full == 1).sum())
    n_train_healthy = int((y_train_full == 0).sum())
    n_test_bankrupt = int((y_test == 1).sum())
    n_test_healthy = int((y_test == 0).sum())
    print(f"[Unbalanced] Train set -> healthy: {n_train_healthy:5d} | bankrupt: {n_train_bankrupt:5d}")
    print(f"             Test  set -> healthy: {n_test_healthy:5d} | bankrupt: {n_test_bankrupt:5d}")

 
    ratio = n_train_healthy / max(n_train_bankrupt, 1)
    rng = np.random.RandomState(RANDOM_STATE + fold_num)
    if ratio > TARGET_RATIO:
        n_healthy_keep = TARGET_RATIO * n_train_bankrupt
        healthy_idx = np.where(y_train_full.values == 0)[0]
        bankrupt_idx = np.where(y_train_full.values == 1)[0]
        healthy_keep_idx = rng.choice(healthy_idx, size=n_healthy_keep, replace=False)
        balanced_idx = np.concatenate([healthy_keep_idx, bankrupt_idx])
        rng.shuffle(balanced_idx)
        X_train_bal = X_train_full.iloc[balanced_idx].reset_index(drop=True)
        y_train_bal = y_train_full.iloc[balanced_idx].reset_index(drop=True)
    else:
        X_train_bal = X_train_full.copy()
        y_train_bal = y_train_full.copy()

    n_train_bal_healthy = int((y_train_bal == 0).sum())
    n_train_bal_bankrupt = int((y_train_bal == 1).sum())
    print(f"[Balanced 3:1] Train set -> healthy: {n_train_bal_healthy:5d} | bankrupt: {n_train_bal_bankrupt:5d}")
    print(f"               Test  set -> healthy: {n_test_healthy:5d} | bankrupt: {n_test_bankrupt:5d}  (unchanged)")
    print("               (Note: healthy companies removed from the training set are "
          "DISCARDED, i.e. NOT moved into the test set, as required.)")

    train_scenarios = {
        "unbalanced": (X_train_full, y_train_full),
        "balanced": (X_train_bal, y_train_bal),
    }

    # train & evaluate all models, for both scenarios 
    for scenario_name, (X_tr, y_tr) in train_scenarios.items():
        models = build_models()
        cm_train_grid, cm_test_grid = {}, {}

        print(f"\n  >> Training scenario: '{scenario_name}' "
              f"(train samples={len(X_tr)}, bankrupt={int((y_tr==1).sum())})")

        for model_name, model in models.items():
            model.fit(X_tr, y_tr)

            #  predictions & scores on TRAIN 
            y_pred_tr = model.predict(X_tr)
            y_score_tr = model.predict_proba(X_tr)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_tr)
            acc_tr, prec_tr, rec_tr, f1_tr, auc_tr, tn_tr, fp_tr, fn_tr, tp_tr = evaluate(y_tr, y_pred_tr, y_score_tr)

            #  predictions & scores on TEST 
            y_pred_te = model.predict(X_test)
            y_score_te = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
            acc_te, prec_te, rec_te, f1_te, auc_te, tn_te, fp_te, fn_te, tp_te = evaluate(y_test, y_pred_te, y_score_te)

            cm_train_grid[model_name] = (tn_tr, fp_tr, fn_tr, tp_tr)
            cm_test_grid[model_name] = (tn_te, fp_te, fn_te, tp_te)

            print(f"     {model_name:18s} | TRAIN  Acc={acc_tr:.2f} Prec={prec_tr:.2f} Rec={rec_tr:.2f} "
                  f"F1={f1_tr:.2f} AUC={auc_tr:.2f}")
            print(f"     {'':18s} | TEST   Acc={acc_te:.2f} Prec={prec_te:.2f} Rec={rec_te:.2f} "
                  f"F1={f1_te:.2f} AUC={auc_te:.2f}")

            # log rows into the results DataFrame  
            results_rows.append({
                "Fold": fold_num,
                "Classifier": model_name,
                "Set": "train",
                "Balance": scenario_name,
                "N_train_samples": len(X_tr),
                "N_train_bankrupt": int((y_tr == 1).sum()),
                "TP": tp_tr, "TN": tn_tr, "FP": fp_tr, "FN": fn_tr,
                "Accuracy": round(acc_tr, 2), "Precision": round(prec_tr, 2),
                "Recall": round(rec_tr, 2), "F1": round(f1_tr, 2),
                "ROC_AUC": round(auc_tr, 2),
            })
            results_rows.append({
                "Fold": fold_num,
                "Classifier": model_name,
                "Set": "test",
                "Balance": scenario_name,
                "N_train_samples": len(X_tr),
                "N_train_bankrupt": int((y_tr == 1).sum()),
                "TP": tp_te, "TN": tn_te, "FP": fp_te, "FN": fn_te,
                "Accuracy": round(acc_te, 2), "Precision": round(prec_te, 2),
                "Recall": round(rec_te, 2), "F1": round(f1_te, 2),
                "ROC_AUC": round(auc_te, 2),
            })

        plot_confusion_grid(
            cm_train_grid,
            f"Fold {fold_num} | {scenario_name} training set | Confusion matrices (TRAIN)",
            os.path.join(CM_DIR, f"fold{fold_num}_{scenario_name}_train_confusion.png"),
        )
        plot_confusion_grid(
            cm_test_grid,
            f"Fold {fold_num} | {scenario_name} training set | Confusion matrices (TEST)",
            os.path.join(CM_DIR, f"fold{fold_num}_{scenario_name}_test_confusion.png"),
        )

print("\nAll folds processed. Confusion-matrix grid figures saved under:", CM_DIR)


results_df = pd.DataFrame(results_rows)
results_df.to_csv(RESULTS_CSV, index=False)
print("\n" + "=" * 90)
print(f"STEP 7: Results saved to {RESULTS_CSV}  ({len(results_df)} rows)")
print("=" * 90)
print(results_df.head(10))
