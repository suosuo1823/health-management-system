# -*- coding: utf-8 -*-
"""
  Feature Selection Comparison: Plan A (Strict 9-Feature) vs Plan B (Full 15-Feature)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, auc as sk_auc
)
from sklearn.feature_selection import mutual_info_classif, RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "obesity_level.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

print("=" * 70)
print("  Feature Selection Comparison Experiment")
print("  Plan A (Strict 9-Feature) vs Plan B (Full 15-Feature)")
print("=" * 70)

# ========== Step 1: Data Preprocessing ==========
print("\n[Step 1] Data Loading & Preprocessing")

df_raw = pd.read_csv(DATA_PATH, index_col="id")
dup_count = df_raw.duplicated().sum()
if dup_count > 0:
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)

df = df_raw.copy()
df["0be1dad"] = df["0be1dad"].str.strip()
df["0be1dad"] = df["0be1dad"].replace({
    "0rmal_Weight": "Normal_Weight",
    "Ormal_Weight": "Normal_Weight"
})

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
caec_map = {"no": 0, "0": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc_map = {"no": 0, "0": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
df["CAEC"] = df["CAEC"].map(caec_map).fillna(df["CAEC"].mode()[0])
df["CALC"] = df["CALC"].map(calc_map).fillna(df["CALC"].mode()[0])

if "MTRANS" in df.columns:
    df = df.drop(columns=["MTRANS"])

le = LabelEncoder()
y = le.fit_transform(df["0be1dad"])
class_names = le.classes_

feature_cols = [c for c in df.columns if c != "0be1dad"]
X = df[feature_cols].copy()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

continuous_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

scaler_full = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_cols] = scaler_full.fit_transform(X_train[continuous_cols])
X_test_scaled[continuous_cols] = scaler_full.transform(X_test[continuous_cols])

feature_names = list(X_train_scaled.columns)
print("  Total features: %d" % len(feature_names))
print("  Train: %d | Test: %d" % (len(y_train), len(y_test)))

# ========== Step 2: Feature Ranking ==========
print("\n[Step 2] Feature Ranking Analysis")

# RF importance
rf_fs = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1)
rf_fs.fit(X_train_scaled, y_train)
rf_imp = rf_fs.feature_importances_

# XGBoost importance
xgb_fs = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           use_label_encoder=False, eval_metric="mlogloss",
                           random_state=RANDOM_STATE, n_jobs=-1)
xgb_fs.fit(X_train_scaled, y_train)
xgb_imp = xgb_fs.feature_importances_

# Mutual Information
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=RANDOM_STATE)

# Comprehensive ranking
rf_r = {f: i+1 for i, f in enumerate(sorted(feature_names, key=lambda f: -rf_imp[feature_names.index(f)]))}
xr = {f: i+1 for i, f in enumerate(sorted(feature_names, key=lambda f: -xgb_imp[feature_names.index(f)]))}
mr = {f: i+1 for i, f in enumerate(sorted(feature_names, key=lambda f: -mi_scores[feature_names.index(f)]))}

rank_data = []
for f in feature_names:
    idx = feature_names.index(f)
    rank_data.append({
        "Feature": f,
        "Rank_Mean": (rf_r[f] + xr[f] + mr[f]) / 3.0,
        "RF_Rank": rf_r[f],
        "XGB_Rank": xr[f],
        "MI_Rank": mr[f],
        "RF_Imp": rf_imp[idx],
        "XGB_Imp": xgb_imp[idx],
        "MI_Score": mi_scores[idx],
    })
rank_df = pd.DataFrame(rank_data).sort_values("Rank_Mean").reset_index(drop=True)

print("\n  Comprehensive Ranking:")
for _, row in rank_df.iterrows():
    print("    #%-2d %-30s  Rank=%.1f  RF=%.4f  XGB=%.4f  MI=%.4f" %
          (_+1, row["Feature"], row["Rank_Mean"], row["RF_Imp"], row["XGB_Imp"], row["MI_Score"]))

# RFE
rfe_est = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rfe = RFE(estimator=rfe_est, n_features_to_select=max(5, len(feature_names)//2), step=1)
rfe.fit(X_train_scaled, y_train)
rfe_selected = rank_df[rfe.support_]["Feature"].tolist()

N_PLAN_A = 9
plan_a_features = rank_df.head(N_PLAN_A)["Feature"].tolist()
plan_b_features = list(feature_names)

print("\n" + "=" * 70)
print("  PLAN A: Strict Top-%d features" % N_PLAN_A)
for i, f in enumerate(plan_a_features):
    print("    #%d: %s" % (i+1, f))

excluded = [f for f in plan_b_features if f not in plan_a_features]
print("  Excluded (%d): %s" % (len(excluded), excluded))

print("\n  PLAN B: Full %d features" % len(plan_b_features))

# ========== Step 3: Build Datasets ==========
print("\n[Step 3] Building Training Sets")
X_tr_A = X_train_scaled[plan_a_features].copy()
X_te_A = X_test_scaled[plan_a_features].copy()
X_tr_B = X_train_scaled.copy()
X_te_B = X_test_scaled.copy()
print("  Plan A shape: %s" % str(X_tr_A.shape))
print("  Plan B shape: %s" % str(X_tr_B.shape))

# ========== Step 4: Train & Tune Both Plans ==========
print("\n[Step 4] Hyperparameter Tuning & Model Training")

param_dist_xgb = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.1, 1, 5, 10]
}

results_all = {}

for pname, Xtr, Xte in [("PlanA_9feat", X_tr_A, X_te_A), ("PlanB_15feat", X_tr_B, X_te_B)]:
    print("\n  Tuning %s (%d features)..." % (pname, Xtr.shape[1]))

    base_m = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                random_state=RANDOM_STATE, n_jobs=-1)
    searcher = RandomizedSearchCV(base_m, param_dist_xgb, n_iter=80, cv=5,
                                  scoring="accuracy", random_state=RANDOM_STATE,
                                  n_jobs=-1, verbose=0, refit=True)
    searcher.fit(Xtr, y_train)
    best_m = searcher.best_estimator_

    y_pred = best_m.predict(Xte)
    y_proba = best_m.predict_proba(Xte)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_val = np.nan

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_m, Xtr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    results_all[pname] = {
        "model": best_m, "best_params": searcher.best_params_,
        "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
        "acc": float(acc), "f1_weighted": float(f1_w), "f1_macro": float(f1_m),
        "precision": float(prec), "recall": float(rec), "auc": float(auc_val) if not np.isnan(auc_val) else None,
        "y_pred": y_pred, "y_proba": y_proba,
        "n_features": int(Xtr.shape[1]), "features_used": list(Xtr.columns),
    }

    print("    Best params: %s" % str(searcher.best_params_))
    print("    CV Acc: %.4f +/- %.4f" % (cv_scores.mean(), cv_scores.std()))
    print("    Test Acc: %.4f | F1(w): %.4f | AUC: %.4f" % (acc, f1_w, auc_val))


# ========== Step 5: Comparison Results ==========
ra = results_all["PlanA_9feat"]
rb = results_all["PlanB_15feat"]

print("\n" + "=" * 70)
print("  COMPARISON RESULTS")
print("=" * 70)

metrics_info = [
    ("cv_mean", "CV Accuracy"),
    ("acc", "Test Accuracy"),
    ("f1_weighted", "F1(Weighted)"),
    ("f1_macro", "F1(Macro)"),
    ("precision", "Precision(W)"),
    ("recall", "Recall(W)"),
    ("auc", "AUC(OvR)"),
]

header_fmt = "%-25s %18s %18s %12s %8s"
row_fmt   = "%-25s %18s %18s %+12s %8s"

print("\n" + header_fmt % ("Metric", "PlanA (9 feat)", "PlanB (15 feat)", "Diff", "Winner"))
print("-" * 85)

winner_counts = {"A": 0, "B": 0, "Tie": 0}

for mkey, mname in metrics_info:
    va = ra[mkey]
    vb = rb[mkey]
    diff = vb - va
    winner = "B" if diff > 0.001 else ("A" if diff < -0.001 else "Tie")
    if winner == "A":
        winner_counts["A"] += 1
    elif winner == "B":
        winner_counts["B"] += 1
    else:
        winner_counts["Tie"] += 1

    sa = "%.4f" % va if va is not None else "N/A"
    sb = "%.4f" % vb if vb is not None else "N/A"
    sd = "%+.4f" % diff if (va is not None and vb is not None) else "N/A"
    print(row_fmt % (mname, sa, sb, sd, winner))

print("\n  Winner Summary: A=%d / B=%d / Tie=%d" % (winner_counts["A"], winner_counts["B"], winner_counts["Tie"]))

final_winner = "B" if rb["acc"] >= ra["acc"] else "A"
print("\n  >>> FINAL DECISION: PLAN %s <<<" % final_winner)
print("  Plan A accuracy: %.4f (%.2f%%)" % (ra["acc"], ra["acc"]*100))
print("  Plan B accuracy: %.4f (%.2f%%)" % (rb["acc"], rb["acc"]*100))

# ========== Step 6: Generate Charts ==========
print("\n[Step 6] Generating Comparison Charts")

# Chart 1: Overall Performance Comparison
fig, ax = plt.subplots(figsize=(14, 7))
plot_metrics = ["cv_mean", "acc", "f1_weighted", "auc"]
plot_labels = ["CV Accuracy", "Test Accuracy", "F1(Weighted)", "AUC(OvR)"]
x_pos = np.arange(len(plot_metrics))
width = 0.35

vals_a = [ra[m]*100 if ra[m] is not None else 0 for m in plot_metrics]
vals_b = [rb[m]*100 if rb[m] is not None else 0 for m in plot_metrics]

la = "Plan A (9 Features, Acc={:.2f}%)".format(ra["acc"]*100)
lb = "Plan B (15 Features, Acc={:.2f}%)".format(rb["acc"]*100)

ax.bar(x_pos - width/2, vals_a, width, label=la, color='#3498db', edgecolor='white', linewidth=1.5)
ax.bar(x_pos + width/2, vals_b, width, label=lb, color='#e74c3c', edgecolor='white', linewidth=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_labels, fontsize=11)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Feature Selection Comparison:\nStrict 9 Features vs Full 15 Features', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_ylim(min(min(vals_a), min(vals_b)) - 2, 102)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fs_comparison_overall_performance.png"), dpi=150)
plt.close()
print("  -> fs_comparison_overall_performance.png")


# Chart 2: Feature Ranking + Cutoff
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
ax1 = axes[0]
colors_rank = ['#e74c3c' if f in plan_a_features else '#95a5a6' for f in rank_df['Feature']]
bars = ax1.barh(rank_df['Feature'][::-1], rank_df['Rank_Mean'][::-1], color=colors_rank[::-1], edgecolor='white')
ax1.axvline(x=N_PLAN_A + 0.5, color='#e74c3c', linestyle='--', linewidth=2, label='Top-%d cutoff' % N_PLAN_A)
ax1.set_xlabel('Average Rank (Lower = More Important)', fontsize=11)
title_str = 'Comprehensive Feature Ranking\n(Red = Selected by Plan A, Top-%d)' % N_PLAN_A
ax1.set_title(title_str, fontsize=13, fontweight='bold')

for bar, val in zip(bars, rank_df['Rank_Mean'][::-1]):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, "%.1f" % val, va='center', fontsize=8)
ax1.legend(loc='lower right')

ax2 = axes[1]
rank_matrix = rank_df[['RF_Rank', 'XGB_Rank', 'MI_Rank']].values.T
im = ax2.imshow(rank_matrix, cmap='YlOrRd_r', aspect='auto')
ax2.set_xticks(range(len(rank_df)))
ax2.set_xticklabels(rank_df['Feature'], rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(3))
ax2.set_yticklabels(['Random Forest', 'XGBoost', 'Mutual Info'])
ax2.set_title('Individual Method Rankings\n(Darker = Higher Priority)', fontsize=13, fontweight='bold')

for i in range(3):
    for j in range(len(rank_df)):
        c = 'white' if rank_matrix[i,j] > 10 else 'black'
        ax2.text(j, i, "%d" % int(rank_matrix[i,j]), ha='center', va='center', fontsize=7, color=c)

plt.colorbar(im, ax=ax2, label='Rank Position')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fs_comparison_feature_ranking.png"), dpi=150)
plt.close()
print("  -> fs_comparison_feature_ranking.png")


# Chart 3: Confusion Matrix Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(confusion_matrix(y_test, ra["y_pred"]), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=axes[0])
axes[0].set_title('Plan A (9 Features)\nAccuracy=%.4f' % ra["acc"], fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Label'); axes[0].set_ylabel('True Label')
axes[0].tick_params(axis='x', rotation=30)

sns.heatmap(confusion_matrix(y_test, rb["y_pred"]), annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', ax=axes[1])
axes[1].set_title('Plan B (15 Features)\nAccuracy=%.4f' % rb["acc"], fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Label'); axes[1].set_ylabel('True Label')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fs_comparison_confusion_matrix.png"), dpi=150)
plt.close()
print("  -> fs_comparison_confusion_matrix.png")


# Chart 4: ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
n_classes = len(class_names)
y_bin = label_binarize(y_test, classes=np.arange(n_classes))

for pkey, rdict, clr, lstyle, lname in [
    ("PlanA_9feat", ra, '#3498db', '-', 'Plan A (9 Features)'),
    ("PlanB_15feat", rb, '#e74c3c', '--', 'Plan B (15 Features)'),
]:
    fpr, tpr, _ = roc_curve(y_bin.ravel(), rdict["y_proba"].ravel())
    roc_a = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, color=clr, linewidth=2.5, linestyle=lstyle,
            label='%s (Micro-AUC=%.4f)' % (lname, roc_a))

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax.set_title('ROC Curve Comparison\n(Plan A vs Plan B)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fs_comparison_roc_curves.png"), dpi=150)
plt.close()
print("  -> fs_comparison_roc_curves.png")


# Chart 5: Per-Class Metrics
fig, ax = plt.subplots(figsize=(16, 7))
report_a = classification_report(y_test, ra["y_pred"], target_names=class_names, output_dict=True, zero_division=0)
report_b = classification_report(y_test, rb["y_pred"], target_names=class_names, output_dict=True, zero_division=0)
classes_for_plot = [c for c in class_names if c in report_a]
x = np.arange(len(classes_for_plot))
w = 0.22

for mi, mname in enumerate(["precision", "recall", "f1-score"]):
    va = [report_a[c][mname]*100 for c in classes_for_plot]
    vb = [report_b[c][mname]*100 for c in classes_for_plot]
    off = (mi - 1) * w
    ax.bar(x + off, va, w*0.85, color=sns.color_palette("Blues_d", 3)[mi], edgecolor='white', label='A_%s' % mname)
    ax.bar(x + off + w*0.85, vb, w*0.85, color=sns.color_palette("Reds_d", 3)[mi], edgecolor='white', alpha=0.75, label='B_%s' % mname)

short_names = [c.replace('_', '\n')[:15] for c in classes_for_plot]
ax.set_xticks(x + w); ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Per-Class Performance Comparison\n(Solid=PlanA Semi-trans=PlanB)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 105); ax.legend(fontsize=8, ncol=3, loc='lower right'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fs_comparison_per_class_metrics.png"), dpi=150)
plt.close()
print("  -> fs_comparison_per_class_metrics.png")


# Chart 6: Complexity vs Accuracy (Key Thesis Figure)
fig, ax = plt.subplots(figsize=(10, 7))
nf_list = [ra["n_features"], rb["n_features"]]
acc_list = [ra["acc"]*100, rb["acc"]*100]
auc_list = [ra["auc"]*100 if ra["auc"] else 0, rb["auc"]*100 if rb["auc"] else 0]

ax.scatter(nf_list, acc_list, s=[400, 400], c=['#3498db', '#e74c3c'], edgecolors='black', linewidths=2, zorder=5)
ax.plot(nf_list, acc_list, 'o-', color='gray', linewidth=2, markersize=0, zorder=3)

for i, (nf, av, auv, nm) in enumerate(zip(nf_list, acc_list, auc_list, ['Plan A\n(Top-9)', 'Plan B\n(Full-15)'])):
    ann_text = '%s\nAcc=%.2f%%\nAUC=%.2f%%' % (nm, av, auv)
    ax.annotate(ann_text, xy=(nf, av), xytext=(nf+0.8, av+0.8),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

ax.set_xlabel('Number of Features', fontsize=12)
ax.set_ylabel('Test Set Accuracy (%)', fontsize=12)
ax.set_title('Model Complexity vs Accuracy Trade-off\n(Feature Selection Impact)', fontsize=13, fontweight='bold')
ax.set_xlim(5, 19); ax.set_ylim(min(acc_list)-2, max(acc_list)+3)
ax.set_xticks([6, 9, 12, 15, 18]); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fs_comparison_complexity_vs_accuracy.png"), dpi=150)
plt.close()
print("  -> fs_comparison_complexity_vs_accuracy.png")


# ========== Step 7: Save Models & Results ==========
print("\n[Step 7] Saving Models & Results")

joblib.dump(ra["model"], os.path.join(OUTPUT_DIR, "plan_a_model_9feat.joblib"))
joblib.dump(rb["model"], os.path.join(OUTPUT_DIR, "plan_b_model_15feat.joblib"))

plan_a_cont = [c for c in plan_a_features if c in continuous_cols]
scaler_a_data = {
    "mean": scaler_full.mean_[[continuous_cols.index(c) for c in plan_a_cont]],
    "scale": scaler_full.scale_[[continuous_cols.index(c) for c in plan_a_cont]],
    "continuous_features": plan_a_cont,
}
joblib.dump(scaler_a_data, os.path.join(OUTPUT_DIR, "plan_a_scaler_info.joblib"))

comparison_summary = {
    "experiment_description": "Feature selection strict comparison: Top-9 selected features vs full 15 features",
    "experiment_date": "2026-04-14",
    "plan_a": {
        "description": "Strict top-9 feature selection (comprehensive ranking from RF+XGBoost+MI)",
        "n_features": int(ra["n_features"]),
        "selected_features": plan_a_features,
        "excluded_features": [f for f in plan_b_features if f not in plan_a_features],
        "best_hyperparameters": {str(k): v for k, v in ra["best_params"].items()},
        "cv_accuracy_mean": ra["cv_mean"],
        "test_accuracy": ra["acc"],
        "f1_weighted": ra["f1_weighted"],
        "auc_ovr_macro": ra["auc"],
        "model_file": "plan_a_model_9feat.joblib",
    },
    "plan_b": {
        "description": "All 15 original features (current production approach)",
        "n_features": int(rb["n_features"]),
        "selected_features": plan_b_features,
        "best_hyperparameters": {str(k): v for k, v in rb["best_params"].items()},
        "cv_accuracy_mean": rb["cv_mean"],
        "test_accuracy": rb["acc"],
        "f1_weighted": rb["f1_weighted"],
        "auc_ovr_macro": rb["auc"],
        "model_file": "plan_b_model_15feat.joblib",
    },
    "comparison": {
        "accuracy_diff_pct": round((rb["acc"] - ra["acc"]) * 100, 4),
        "recommended_plan": final_winner,
        "recommendation_reason": (
            "Plan B wins: higher accuracy (%.4f > %.4f)" % (rb["acc"], ra["acc"])
            if final_winner == "B" else
            "Plan A wins: comparable accuracy but simpler model (%.4f)" % ra["acc"]
        ),
    },
    "full_ranking_table": rank_df.to_dict(orient="records"),
}
with open(os.path.join(OUTPUT_DIR, "feature_selection_comparison.json"), "w", encoding="utf-8") as f:
    json.dump(comparison_summary, f, ensure_ascii=False, indent=2, default=str)

print("  Saved:")
print("    -> plan_a_model_9feat.joblib")
print("    -> plan_b_model_15feat.joblib")
print("    -> plan_a_scaler_info.joblib")
print("    -> feature_selection_comparison.json")

# Classification reports
print("\n%s" % "=" * 70)
print("  Classification Report - Plan A (%d features)" % ra["n_features"])
print("%s" % "=" * 70)
print(classification_report(y_test, ra["y_pred"], target_names=class_names, digits=4))

print("\n%s" % "=" * 70)
print("  Classification Report - Plan B (%d features)" % rb["n_features"])
print("%s" % "=" * 70)
print(classification_report(y_test, rb["y_pred"], target_names=class_names, digits=4))


# ========== Final Summary Table ==========
print("""
%s
              FEATURE SELECTION COMPARISON RESULT
%s
  +----------------------------+----------+----------+---------+
  | Metric                     | Plan A   | Plan B   | Winner  |
  |                            | (9 feat) | (15 feat)|         |
  +----------------------------+----------+----------+---------+
  | CV Accuracy (mean)         | %.4f   | %.4f   | %s       |
  | Test Set Accuracy          | %.4f   | %.4f   | %s       |
  | F1 Score (weighted)        | %.4f   | %.4f   | %s       |
  | AUC (OvR macro)            | %.4f   | %.4f   | %s       |
  | Precision (weighted)       | %.4f   | %.4f   | %s       |
  | Recall (weighted)          | %.4f   | %.4f   | %s       |
  +----------------------------+----------+----------+---------+
  | Number of Features         | %8d | %8d |         |
  +----------------------------+----------+----------+---------+

  RECOMMENDED: PLAN %s
""" % (
    "=", "=",
    ra["cv_mean"], rb["cv_mean"], "B" if rb["cv_mean"]>ra["cv_mean"] else "A ",
    ra["acc"],     rb["acc"],     final_winner,
    ra["f1_weighted"], rb["f1_weighted"], "B" if rb["f1_weighted"]>ra["f1_weighted"] else "A ",
    ra["auc"],     rb["auc"],     "B" if (ra["auc"] and rb["auc"] and rb["auc"]>ra["auc"]) else "A ",
    ra["precision"], rb["precision"], "B" if rb["precision"]>ra["precision"] else "A ",
    ra["recall"],    rb["recall"],    "B" if rb["recall"]>ra["recall"] else "A ",
    ra["n_features"], rb["n_features"],
    final_winner,
))

print("Experiment Complete!")
