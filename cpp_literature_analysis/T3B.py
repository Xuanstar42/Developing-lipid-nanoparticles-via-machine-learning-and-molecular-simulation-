import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import umap


# =========================
# 0. Basic settings
# =========================

INPUT_FILE = "data_backfilled.xlsx"
OUTPUT_DIR = "figures_pca_umap"

OUTCOME_VIVO = "in_vivo_flag"
OUTCOME_VITRO = "in_vitro_functional_effect"
SEQ_COL = "peptide_backbone_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# =========================
# 1. Amino acid settings
# =========================

AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AVILMFWY")
AROMATIC = set("FWY")
BASIC = set("KRH")


# =========================
# 2. Helper functions
# =========================

def clean_sequence(seq):
    """
    Keep only canonical amino acids.
    Return uppercase cleaned sequence.
    """
    if pd.isna(seq):
        return ""
    seq = str(seq).upper().strip()
    return "".join([aa for aa in seq if aa in AA_SET])


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


def build_feature_matrix(df):
    """
    Build final feature matrix from sequence-derived features + modification flags.
    """
    df = df.copy()

    for col in [OUTCOME_VIVO, OUTCOME_VITRO]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = (df[col] > 0).astype(int)

    df["sequence_clean"] = df[SEQ_COL].apply(clean_sequence)

    seq_feat_df = df["sequence_clean"].apply(seq_features).apply(pd.Series)

    mod_cols = [c for c in df.columns if c.startswith("mod_")]

    for c in mod_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[c] = (df[c] > 0).astype(int)

    if len(mod_cols) > 0:
        mod_feat_df = df[mod_cols].copy()
        mod_feat_df["modification_count"] = mod_feat_df.sum(axis=1)
    else:
        mod_feat_df = pd.DataFrame(index=df.index)
        mod_feat_df["modification_count"] = 0

    X = pd.concat([seq_feat_df, mod_feat_df], axis=1)
    y_vitro = df[OUTCOME_VITRO]
    y_vivo = df[OUTCOME_VIVO]

    return df, X, y_vitro, y_vivo


def preprocess_features(X):
    """
    Impute missing values and standardize features.
    """
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled


def scatter_by_label(ax, X_emb, y, title, xlabel, ylabel):
    """
    Scatter plot colored by binary label.
    """
    for label, name in [(0, "Negative"), (1, "Positive")]:
        mask = (y == label)
        ax.scatter(
            X_emb[mask, 0],
            X_emb[mask, 1],
            alpha=0.7,
            label=name
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


# =========================
# 3. Main
# =========================

def main():
    print("Loading data...")
    df = pd.read_excel(INPUT_FILE)

    print("Building feature matrix...")
    df, X, y_vitro, y_vivo = build_feature_matrix(df)

    print("Feature matrix shape:", X.shape)
    print("Number of in vitro positives:", int(y_vitro.sum()))
    print("Number of in vivo positives:", int(y_vivo.sum()))
    print("Feature columns:")
    print(list(X.columns))

    print("\nPreprocessing features...")
    X_scaled = preprocess_features(X)

    # =========================
    # 4. PCA
    # =========================
    print("\nRunning PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    print("PCA explained variance ratio:")
    print(f"PC1: {explained_var[0]:.4f}")
    print(f"PC2: {explained_var[1]:.4f}")
    print(f"PC1 + PC2: {explained_var.sum():.4f}")

    # =========================
    # 5. UMAP
    # =========================
    print("\nRunning UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    X_umap = umap_model.fit_transform(X_scaled)

    # =========================
    # 6. Save coordinate tables
    # =========================
    print("\nSaving coordinate tables...")

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    umap_df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])

    result_df = df.copy()
    result_df["in_vitro"] = y_vitro.values
    result_df["in_vivo"] = y_vivo.values

    output_xlsx = os.path.join(OUTPUT_DIR, "pca_umap_results.xlsx")

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        X.to_excel(writer, sheet_name="feature_matrix", index=False)
        result_df.to_excel(writer, sheet_name="data_with_labels", index=False)
        pca_df.to_excel(writer, sheet_name="pca_coordinates", index=False)
        umap_df.to_excel(writer, sheet_name="umap_coordinates", index=False)

    # =========================
    # 7. Plot PCA only
    # =========================
    print("\nSaving PCA figure...")
    fig_pca, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter_by_label(
        axes[0],
        X_pca,
        y_vitro,
        "PCA space colored by in vitro outcome",
        f"PC1 ({explained_var[0] * 100:.1f}% variance)",
        f"PC2 ({explained_var[1] * 100:.1f}% variance)"
    )

    scatter_by_label(
        axes[1],
        X_pca,
        y_vivo,
        "PCA space colored by in vivo outcome",
        f"PC1 ({explained_var[0] * 100:.1f}% variance)",
        f"PC2 ({explained_var[1] * 100:.1f}% variance)"
    )

    plt.tight_layout()
    fig_pca.savefig(os.path.join(OUTPUT_DIR, "pca_space.png"), bbox_inches="tight")
    plt.show()
    plt.close(fig_pca)

    # =========================
    # 8. Plot UMAP only
    # =========================
    print("\nSaving UMAP figure...")
    fig_umap, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter_by_label(
        axes[0],
        X_umap,
        y_vitro,
        "UMAP space colored by in vitro outcome",
        "UMAP1",
        "UMAP2"
    )

    scatter_by_label(
        axes[1],
        X_umap,
        y_vivo,
        "UMAP space colored by in vivo outcome",
        "UMAP1",
        "UMAP2"
    )

    plt.tight_layout()
    fig_umap.savefig(os.path.join(OUTPUT_DIR, "umap_space.png"), bbox_inches="tight")
    plt.show()
    plt.close(fig_umap)

    # =========================
    # 9. Plot PCA vs UMAP comparison
    # =========================
    print("\nSaving PCA vs UMAP comparison figure...")
    fig_compare, axes = plt.subplots(2, 2, figsize=(10, 10))

    scatter_by_label(
        axes[0, 0],
        X_pca,
        y_vitro,
        "PCA - in vitro",
        f"PC1 ({explained_var[0] * 100:.1f}% variance)",
        f"PC2 ({explained_var[1] * 100:.1f}% variance)"
    )

    scatter_by_label(
        axes[0, 1],
        X_pca,
        y_vivo,
        "PCA - in vivo",
        f"PC1 ({explained_var[0] * 100:.1f}% variance)",
        f"PC2 ({explained_var[1] * 100:.1f}% variance)"
    )

    scatter_by_label(
        axes[1, 0],
        X_umap,
        y_vitro,
        "UMAP - in vitro",
        "UMAP1",
        "UMAP2"
    )

    scatter_by_label(
        axes[1, 1],
        X_umap,
        y_vivo,
        "UMAP - in vivo",
        "UMAP1",
        "UMAP2"
    )

    plt.tight_layout()
    fig_compare.savefig(
        os.path.join(OUTPUT_DIR, "pca_vs_umap_comparison.png"),
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig_compare)

    print("\nDone.")
    print("All figures saved to:", OUTPUT_DIR)
    print("All tables saved to:", output_xlsx)


if __name__ == "__main__":
    main()