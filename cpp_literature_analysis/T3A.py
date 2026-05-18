import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# =========================
# 0. Basic settings
# =========================

INPUT_FILE = "data_backfilled.xlsx"
OUTPUT_DIR = "figures"
RESULT_XLSX = "analysis_results.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure style
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# =========================
# 1. Load data
# =========================

df = pd.read_excel(INPUT_FILE)

# =========================
# 2. Define outcome columns
#    Empty cells are treated as 0
# =========================

OUTCOME_VIVO = "in_vivo_flag"
OUTCOME_VITRO = "in_vitro_functional_effect"
SEQ_COL = "peptide_backbone_clean"

for col in [OUTCOME_VIVO, OUTCOME_VITRO]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df[col] = (df[col] > 0).astype(int)

# =========================
# 3. Clean peptide sequence
# =========================

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWY")
AROMATIC = set("FWY")
ACIDIC = set("DE")
BASIC = set("KRH")

def clean_sequence(seq):
    """
    Keep only canonical amino acids.
    Return uppercase cleaned sequence.
    """
    if pd.isna(seq):
        return ""
    seq = str(seq).upper().strip()
    return "".join([aa for aa in seq if aa in AA_SET])

df["sequence_clean"] = df[SEQ_COL].apply(clean_sequence)

# =========================
# 4. Extract sequence features
# =========================

def seq_features(seq):
    """
    Extract simple sequence-derived features from the peptide backbone.
    """
    L = len(seq)
    if L == 0:
        return {
            "seq_length": 0,
            "net_charge_proxy": 0,
            "charge_density": 0,
            "arg_fraction": 0,
            "lys_fraction": 0,
            "his_fraction": 0,
            "acidic_fraction": 0,
            "hydrophobic_fraction": 0,
            "aromatic_fraction": 0,
            "proline_fraction": 0,
            "glycine_fraction": 0,
            "basic_fraction": 0,
        }

    count = {aa: seq.count(aa) for aa in AA_SET}

    net_charge_proxy = (
        count.get("K", 0)
        + count.get("R", 0)
        + count.get("H", 0)
        - count.get("D", 0)
        - count.get("E", 0)
    )

    return {
        "seq_length": L,
        "net_charge_proxy": net_charge_proxy,
        "charge_density": net_charge_proxy / L,
        "arg_fraction": count.get("R", 0) / L,
        "lys_fraction": count.get("K", 0) / L,
        "his_fraction": count.get("H", 0) / L,
        "acidic_fraction": (count.get("D", 0) + count.get("E", 0)) / L,
        "hydrophobic_fraction": sum(count.get(x, 0) for x in HYDROPHOBIC) / L,
        "aromatic_fraction": sum(count.get(x, 0) for x in AROMATIC) / L,
        "proline_fraction": count.get("P", 0) / L,
        "glycine_fraction": count.get("G", 0) / L,
        "basic_fraction": sum(count.get(x, 0) for x in BASIC) / L,
    }

seq_feat_df = df["sequence_clean"].apply(seq_features).apply(pd.Series)

# =========================
# 5. Extract modification features
# =========================

mod_cols = [c for c in df.columns if c.startswith("mod_")]

for c in mod_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df[c] = (df[c] > 0).astype(int)

mod_feat_df = df[mod_cols].copy()

if len(mod_cols) > 0:
    mod_feat_df["modification_count"] = mod_feat_df.sum(axis=1)
else:
    mod_feat_df["modification_count"] = 0

# =========================
# 6. Build final feature matrix
# =========================

X = pd.concat([seq_feat_df, mod_feat_df], axis=1)
y_vivo = df[OUTCOME_VIVO]
y_vitro = df[OUTCOME_VITRO]

print("Feature matrix shape:", X.shape)
print("Number of in vivo positives:", y_vivo.sum())
print("Number of in vitro positives:", y_vitro.sum())
print("Feature columns:")
print(list(X.columns))

# =========================
# 7. Train two separate logistic regression models
# =========================

def fit_logistic_model(X, y, label_name):
    """
    Fit a logistic regression model and print metrics.
    """
    if y.nunique() < 2:
        print(f"\n===== {label_name} =====")
        print("Skipped: outcome has only one class.")
        coef_df = pd.DataFrame({
            "feature": X.columns,
            "coefficient": np.nan,
            "abs_coefficient": np.nan
        })
        return None, coef_df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n===== {label_name} =====")
    try:
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    coef = model.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef)
    }).sort_values("abs_coefficient", ascending=False)

    print("\nTop features by absolute coefficient:")
    print(coef_df.head(15))

    return model, coef_df

model_vitro, coef_vitro = fit_logistic_model(X, y_vitro, "In vitro model")
model_vivo, coef_vivo = fit_logistic_model(X, y_vivo, "In vivo model")

# =========================
# 8. Compare feature effects between in vitro and in vivo
# =========================

coef_compare = coef_vitro[["feature", "coefficient"]].rename(
    columns={"coefficient": "coef_in_vitro"}
).merge(
    coef_vivo[["feature", "coefficient"]].rename(
        columns={"coefficient": "coef_in_vivo"}
    ),
    on="feature",
    how="outer"
)

coef_compare["direction_same"] = (
    np.sign(coef_compare["coef_in_vitro"].fillna(0)) ==
    np.sign(coef_compare["coef_in_vivo"].fillna(0))
)
coef_compare["coef_diff"] = coef_compare["coef_in_vitro"] - coef_compare["coef_in_vivo"]
coef_compare = coef_compare.sort_values("coef_diff", key=np.abs, ascending=False)

print("\n===== Feature effect comparison =====")
print(coef_compare.head(20))

# =========================
# 9. Focus on key subgroup
# =========================

df_analysis = df.copy()
df_analysis["in_vitro"] = y_vitro
df_analysis["in_vivo"] = y_vivo
df_analysis["vitro_success_vivo_fail"] = (
    (df_analysis["in_vitro"] == 1) & (df_analysis["in_vivo"] == 0)
).astype(int)

summary_df = pd.concat(
    [df_analysis[["in_vitro", "in_vivo", "vitro_success_vivo_fail"]], X],
    axis=1
)

group_summary = summary_df.groupby("vitro_success_vivo_fail").mean(numeric_only=True).T

if group_summary.shape[1] == 2:
    group_summary.columns = ["Others", "InVitroSuccess_But_InVivoFail"]
elif group_summary.shape[1] == 1:
    if 0 in summary_df["vitro_success_vivo_fail"].unique():
        group_summary.columns = ["Others"]
    else:
        group_summary.columns = ["InVitroSuccess_But_InVivoFail"]

# =========================
# 10. Figure 1: In vitro -> In vivo funnel
# =========================

total_n = len(df)
vitro_n = int(y_vitro.sum())
vivo_n = int(y_vivo.sum())
vitro_success_vivo_fail_n = int(((y_vitro == 1) & (y_vivo == 0)).sum())

fig1 = plt.figure(figsize=(7, 5))
stages = ["Total peptides", "In vitro positive", "In vivo positive"]
counts = [total_n, vitro_n, vivo_n]

bars = plt.bar(stages, counts)
plt.ylabel("Number of peptides")
plt.title("Drop-off from in vitro functionality to in vivo success")

offset = max(counts) * 0.02 if max(counts) > 0 else 1
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        f"{count}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, "figure_1_funnel.png"), bbox_inches="tight")
plt.show()
plt.close(fig1)

# ========================================================
# PCA visualization is handled separately in T3B.py.
# 12. Figure 2: Coefficient comparison
# ========================================================

fig2 = plt.figure(figsize=(16, 14)) 

plt.axhline(0, linestyle="--", linewidth=1, color='gray', alpha=0.3)
plt.axvline(0, linestyle="--", linewidth=1, color='gray', alpha=0.3)

plot_coef_compare = coef_compare.dropna(subset=["coef_in_vitro", "coef_in_vivo"]).copy()
plot_coef_compare["sig_score"] = np.abs(plot_coef_compare["coef_in_vitro"]) + np.abs(plot_coef_compare["coef_in_vivo"])

# Label the 30 features with the largest combined coefficient magnitude.
top_n_to_label = 30 
labeled_df = plot_coef_compare.sort_values("sig_score", ascending=False).head(top_n_to_label).copy()

def get_color(name):
    return "#1f77b4" if name.startswith("mod_") else "#ff7f0e"

# Background points
plt.scatter(plot_coef_compare["coef_in_vitro"], plot_coef_compare["coef_in_vivo"],
            c=plot_coef_compare["feature"].apply(get_color), alpha=0.15, s=40)

# Labeled points
plt.scatter(labeled_df["coef_in_vitro"], labeled_df["coef_in_vivo"],
            c=labeled_df["feature"].apply(get_color), alpha=0.9, s=120, edgecolors='white', zorder=5)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='#1f77b4', label='Modification Features (Blue)'),
    Patch(facecolor='#ff7f0e', label='Sequence Features (Orange)'),
    Patch(facecolor='none', label='-'*20)
]

# Sort by x-axis position to make label placement easier to scan.
labeled_df = labeled_df.sort_values("coef_in_vitro")

for i, (idx, row) in enumerate(labeled_df.iterrows()):
    point_num = i + 1
    
    x_val = row["coef_in_vitro"]
    y_val = row["coef_in_vivo"]
    
    # Alternate vertical offsets to reduce label overlap.
    offset_y = 0.02 if point_num % 2 == 0 else -0.02
    
    plt.text(
        x_val, 
        y_val + offset_y, 
        str(point_num),
        fontsize=12,
        fontweight='bold',
        ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
        zorder=6
    )
    
    legend_elements.append(Patch(facecolor='none', label=f"{point_num}: {row['feature']}"))

plt.xlabel("Coefficient in In Vitro Model", fontsize=14, fontweight='bold')
plt.ylabel("Coefficient in In Vivo Model", fontsize=14, fontweight='bold')
plt.title("Feature Effects Comparison", fontsize=16, fontweight='bold', pad=25)

plt.legend(handles=legend_elements, 
           loc='upper left', 
           bbox_to_anchor=(1.02, 1), 
           fontsize=11,
           frameon=True, 
           ncol=1,
           title="Feature Index Map", 
           title_fontsize=13,
           labelspacing=1.2)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "figure_2_final_clear.png"), bbox_inches="tight", dpi=300)
plt.show()
plt.close(fig2)

# =========================
# 13. Figure 3 (Original 4): Modification enrichment
# =========================

key_group_mask = (y_vitro == 1) & (y_vivo == 0)
other_group_mask = ~key_group_mask
mod_feature_cols = [c for c in X.columns if c.startswith("mod_")]

top_mods = pd.DataFrame()

if len(mod_feature_cols) > 0:
    mod_freq_df = pd.DataFrame({
        "feature": mod_feature_cols,
        "Others": X.loc[other_group_mask, mod_feature_cols].mean().values,
        "InVitroSuccess_But_InVivoFail": X.loc[key_group_mask, mod_feature_cols].mean().values
    })
    mod_freq_df["difference"] = mod_freq_df["InVitroSuccess_But_InVivoFail"] - mod_freq_df["Others"]
    mod_freq_df = mod_freq_df.sort_values("difference", ascending=False)
    top_mods = mod_freq_df.head(12).copy()

    x = np.arange(len(top_mods))
    width = 0.4
    fig3 = plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, top_mods["Others"], width=width, label="Others")
    plt.bar(x + width / 2, top_mods["InVitroSuccess_But_InVivoFail"], width=width, label="IVS-IVF Subgroup")

    plt.xticks(x, top_mods["feature"], rotation=45, ha="right")
    plt.ylabel("Mean frequency")
    plt.legend()
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "figure_3_mod_enrichment.png"), bbox_inches="tight")
    plt.show()
    plt.close(fig3)

# =========================
# 14. Figure 4-5 (Original 5-6): Top coefficient bar plots
# =========================

def plot_top_coefficients(coef_df, title, save_path, top_n=12):
    if coef_df["coefficient"].isna().all(): return
    plot_df = coef_df.head(top_n).copy().sort_values("coefficient")
    fig = plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["coefficient"])
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

plot_top_coefficients(coef_vitro, "In vitro Top features", os.path.join(OUTPUT_DIR, "figure_4_vitro_coef.png"))
plot_top_coefficients(coef_vivo, "In vivo Top features", os.path.join(OUTPUT_DIR, "figure_5_vivo_coef.png"))

# =========================
# 15. Save results
# =========================

with pd.ExcelWriter(RESULT_XLSX, engine="openpyxl") as writer:
    seq_feat_df.to_excel(writer, sheet_name="sequence_features", index=False)
    mod_feat_df.to_excel(writer, sheet_name="modification_features", index=False)
    X.to_excel(writer, sheet_name="final_feature_matrix", index=False)
    coef_vitro.to_excel(writer, sheet_name="coef_in_vitro", index=False)
    coef_vivo.to_excel(writer, sheet_name="coef_in_vivo", index=False)
    coef_compare.to_excel(writer, sheet_name="coef_comparison", index=False)
    group_summary.to_excel(writer, sheet_name="group_summary")
    if not top_mods.empty:
        top_mods.to_excel(writer, sheet_name="top_mod_enrichment", index=False)
    df_analysis.to_excel(writer, sheet_name="analysis_with_labels", index=False)

print("\nDone. All results saved.")
