import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import pairwise2
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


# =========================
# 1. basic settings
# =========================
INPUT_FILE = "data.xlsx"
SHEET_NAME = "Sheet1"

PAPER_COL = "paper_id"
BACKBONE_COL = "peptide_backbone_clean"
PEPTIDE_NAME_COL = "peptide_name"

MAX_LEN = 60
TOP_N_TREE = 20
TOP_N_BAR = 20
FIG_DPI = 300

TREE_FIG_W = 13
TREE_BASE_H = 7
TREE_PER_ITEM_H = 0.42
TREE_LEAF_FONT = 9

BAR_FIG_W = 10
BAR_BASE_H = 6
BAR_PER_ITEM_H = 0.32


# =========================
# 2. load data
# =========================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

df = df[[PAPER_COL, BACKBONE_COL, PEPTIDE_NAME_COL]].copy()
df = df.dropna(subset=[PAPER_COL, BACKBONE_COL]).copy()

df[PAPER_COL] = df[PAPER_COL].astype(str).str.strip()
df[BACKBONE_COL] = df[BACKBONE_COL].astype(str)
df[PEPTIDE_NAME_COL] = df[PEPTIDE_NAME_COL].fillna("").astype(str).str.strip()


# =========================
# 3. clean backbone
# =========================
def clean_backbone(seq: str) -> str:
    seq = seq.strip()
    seq = re.sub(r"\s+", "", seq)
    seq = seq.upper()
    return seq

df["backbone"] = df[BACKBONE_COL].apply(clean_backbone)
df["length"] = df["backbone"].str.len()

valid_pattern = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

df = df[df["length"] <= MAX_LEN].copy()
df = df[df["backbone"].apply(lambda x: bool(valid_pattern.fullmatch(x)))].copy()

print(f"Rows after cleaning: {len(df)}")
print(f"Unique cleaned backbones: {df['backbone'].nunique()}")


# =========================
# 4. paper-level deduplication
# =========================
df_unique = df[[PAPER_COL, "backbone"]].drop_duplicates().copy()

paper_count = (
    df_unique.groupby("backbone")[PAPER_COL]
    .nunique()
    .sort_values(ascending=False)
)

paper_count_df = paper_count.reset_index()
paper_count_df.columns = ["backbone", "paper_count"]


# =========================
# 5. one representative peptide name per backbone
# =========================
name_summary = (
    df.loc[df[PEPTIDE_NAME_COL].str.strip() != "", ["backbone", PEPTIDE_NAME_COL]]
    .drop_duplicates()
    .groupby("backbone")[PEPTIDE_NAME_COL]
    .first()
    .reset_index()
    .rename(columns={PEPTIDE_NAME_COL: "peptide_name"})
)

summary_df = paper_count_df.merge(name_summary, on="backbone", how="left")
summary_df["peptide_name"] = summary_df["peptide_name"].fillna("")

print("\nTop 20 backbones by unique paper count:")
print(summary_df.head(20)[["backbone", "paper_count", "peptide_name"]])


# =========================
# 6. pairwise sequence identity
# =========================
def seq_identity(seq1: str, seq2: str) -> float:
    if seq1 == seq2:
        return 1.0
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    matches = aln.score
    denom = max(len(seq1), len(seq2))
    return matches / denom


# =========================
# 7. full distance matrix for all unique backbones
# =========================
all_sequences = summary_df["backbone"].tolist()
n_all = len(all_sequences)

dist_mat_all = np.zeros((n_all, n_all), dtype=float)

for i in range(n_all):
    if i % 50 == 0:
        print(f"Computing full distance matrix: {i}/{n_all}")
    for j in range(i + 1, n_all):
        identity = seq_identity(all_sequences[i], all_sequences[j])
        distance = 1.0 - identity
        dist_mat_all[i, j] = distance
        dist_mat_all[j, i] = distance

dist_df_all = pd.DataFrame(dist_mat_all, index=all_sequences, columns=all_sequences)


# =========================
# 8. dendrogram only for top 20
# =========================
tree_df = summary_df.head(TOP_N_TREE).copy()
tree_sequences = tree_df["backbone"].tolist()
n_tree = len(tree_sequences)

dist_mat_tree = np.zeros((n_tree, n_tree), dtype=float)

for i in range(n_tree):
    for j in range(i + 1, n_tree):
        identity = seq_identity(tree_sequences[i], tree_sequences[j])
        distance = 1.0 - identity
        dist_mat_tree[i, j] = distance
        dist_mat_tree[j, i] = distance

condensed_dist_tree = squareform(dist_mat_tree)
Z_tree = linkage(condensed_dist_tree, method="average")

labels_tree = []
for _, row in tree_df.iterrows():
    rep_name = row["peptide_name"] if row["peptide_name"] else "NA"
    label = f"{row['backbone']} | n={row['paper_count']} | {rep_name}"
    labels_tree.append(label)

fig_h = max(TREE_BASE_H, n_tree * TREE_PER_ITEM_H)
plt.figure(figsize=(TREE_FIG_W, fig_h))
ax = plt.gca()

# Use colored branches for the top-20 dendrogram.
max_d = np.max(Z_tree[:, 2])
color_thr = 0.7 * max_d

dendrogram(
    Z_tree,
    labels=labels_tree,
    orientation="right",
    leaf_font_size=TREE_LEAF_FONT,
    color_threshold=color_thr,
    above_threshold_color="gray",
    ax=ax
)

ax.set_title(
    f"CPP backbone dendrogram (top {TOP_N_TREE} by paper count)",
    fontsize=14,
    pad=12
)
ax.set_xlabel("Sequence identity distance (1 - identity)", fontsize=11)
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=TREE_LEAF_FONT)

plt.tight_layout()
plt.savefig("cpp_backbone_dendrogram_top20_colored.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()


# =========================
# 9. top 20 usage barplot
# =========================
bar_df = summary_df.head(TOP_N_BAR).copy()
bar_df = bar_df.sort_values("paper_count", ascending=True)

fig_h = max(BAR_BASE_H, TOP_N_BAR * BAR_PER_ITEM_H)
plt.figure(figsize=(BAR_FIG_W, fig_h))
ax = plt.gca()

bars = ax.barh(bar_df["backbone"], bar_df["paper_count"])

ax.set_title(f"Top {TOP_N_BAR} most-used CPP backbones", fontsize=14, pad=12)
ax.set_xlabel("Number of unique papers", fontsize=11)
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=9)

for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width)}",
        va="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("cpp_backbone_top20_usage_coloredstyle.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()


# =========================
# 10. save outputs
# =========================
top20_detail_df = summary_df.head(TOP_N_TREE).copy()

with pd.ExcelWriter("cpp_backbone_analysis_outputs.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="cleaned_rows", index=False)
    df_unique.to_excel(writer, sheet_name="paper_backbone_unique", index=False)
    summary_df.to_excel(writer, sheet_name="backbone_summary", index=False)
    top20_detail_df.to_excel(writer, sheet_name="top20_backbones", index=False)
    dist_df_all.to_excel(writer, sheet_name="pairwise_distance_matrix_all")

print("\nDone.")
print("Generated files:")
print("- cpp_backbone_dendrogram_top20_colored.png")
print("- cpp_backbone_top20_usage_coloredstyle.png")
print("- cpp_backbone_analysis_outputs.xlsx")
