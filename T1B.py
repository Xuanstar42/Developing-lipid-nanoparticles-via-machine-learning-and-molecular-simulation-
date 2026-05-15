# -----------------------------
# Q2 Step 1: Build modification signature and count
# -----------------------------

import pandas as pd

df_mod = pd.read_csv("data2.csv")

# Identify modification columns
mod_columns = [c for c in df_mod.columns if c.startswith("mod_")]

# Separate columns
mod_specific = [c for c in mod_columns if c not in ["mod_none", "mod_other"]]

# Count number of specific modifications
df_mod["n_modifications"] = df_mod[mod_specific].sum(axis=1)

# Build modification signature
def build_signature(row):

    # If "other modification" is present
    if row["mod_other"] == 1:
        return "other"

    active_mods = [
        col.replace("mod_", "")
        for col in mod_specific
        if row[col] == 1
    ]

    if len(active_mods) > 0:
        active_mods = sorted(active_mods)
        return "+".join(active_mods)

    # If explicitly marked as no modification
    if row["mod_none"] == 1:
        return "none"

    # fallback safety
    return "none"


df_mod["modification_signature"] = df_mod.apply(
    build_signature,
    axis=1
)

print("\nExample rows:")
print(
    df_mod[
        [
            "peptide_backbone_clean",
            "peptide_name",
            "modification_signature",
            "n_modifications"
        ]
    ].head()
)

# Save dataset
df_mod.to_csv("data2_with_mod_signature.csv", index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Q2 Step 2: Paper-level modification statistics
# -----------------------------
# This step summarizes:
# 1. modification strategy usage (combination-level)
# 2. single modification type usage
# 3. modification complexity
# All counts are based on unique paper usage, not row usage.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
df_mod = pd.read_csv("data2_with_mod_signature.csv")

# Identify modification columns
mod_columns = [c for c in df_mod.columns if c.startswith("mod_")]
mod_specific = [c for c in mod_columns if c not in ["mod_none", "mod_other"]]

# -----------------------------
# Build backbone name map
# -----------------------------
# Assign the most frequent peptide_name to each backbone for readability.

def most_frequent_name(series):
    series = series.dropna()
    if len(series) == 0:
        return pd.NA
    return series.value_counts().index[0]

backbone_name_map = (
    df_mod.groupby("peptide_backbone_clean")["peptide_name"]
    .apply(most_frequent_name)
    .reset_index(name="representative_peptide_name")
)

# Helper function for plot labels
def make_backbone_label(row):
    backbone = row["peptide_backbone_clean"]
    name = row["representative_peptide_name"]
    if pd.isna(name):
        return str(backbone)
    return f"{backbone}\n{name}"

# -----------------------------
# STEP 3
# Backbone + modification strategy reuse
# -----------------------------
# Exclude "other" because it is not interpretable as a specific strategy.

df_strategy = df_mod[df_mod["modification_signature"] != "other"].copy()

backbone_mod_strategy = (
    df_strategy.groupby(
        ["peptide_backbone_clean", "modification_signature"]
    )["paper_id"]
    .nunique()
    .reset_index(name="paper_count")
    .sort_values("paper_count", ascending=False)
    .merge(backbone_name_map, on="peptide_backbone_clean", how="left")
)

backbone_mod_strategy.to_csv(
    "backbone_modification_strategy_paper_count.csv",
    index=False
)

print("\nTop backbone-modification strategies:")
print(
    backbone_mod_strategy[
        [
            "peptide_backbone_clean",
            "representative_peptide_name",
            "modification_signature",
            "paper_count"
        ]
    ].head(20).to_string(index=False)
)

# -----------------------------
# Plot 1
# Top 15 backbone-modification strategies
# -----------------------------
top15_strategy = backbone_mod_strategy.head(15).copy()

top15_strategy["plot_label"] = top15_strategy.apply(
    lambda row: (
        f"{row['peptide_backbone_clean']}\n"
        f"{row['representative_peptide_name']}\n"
        f"{row['modification_signature']}"
        if pd.notna(row["representative_peptide_name"])
        else f"{row['peptide_backbone_clean']}\n{row['modification_signature']}"
    ),
    axis=1
)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top15_strategy)))

plt.figure(figsize=(14, 7))

bars = plt.bar(
    top15_strategy["plot_label"],
    top15_strategy["paper_count"],
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

plt.ylabel("Paper count", fontsize=12)
plt.xlabel("Backbone / Representative peptide name / Modification strategy", fontsize=12)
plt.title("Top backbone-modification strategies", fontsize=14)

plt.xticks(rotation=45, ha="right")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.15,
        f"{int(h)}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "top_backbone_modification_strategies.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# STEP 4A
# Number of modification TYPES used per backbone
# -----------------------------
# Exclude "other" from type-level statistics.

records = []

for backbone, group in df_mod.groupby("peptide_backbone_clean"):
    active_mods = set()

    for mod in mod_specific:
        papers = group.loc[group[mod] == 1, "paper_id"].unique()
        if len(papers) > 0:
            active_mods.add(mod.replace("mod_", ""))

    records.append({
        "peptide_backbone_clean": backbone,
        "n_modification_types": len(active_mods)
    })

backbone_mod_types = (
    pd.DataFrame(records)
    .merge(backbone_name_map, on="peptide_backbone_clean", how="left")
    .sort_values("n_modification_types", ascending=False)
)

backbone_mod_types.to_csv(
    "backbone_modification_type_count.csv",
    index=False
)

print("\nBackbones with most modification types:")
print(
    backbone_mod_types[
        [
            "peptide_backbone_clean",
            "representative_peptide_name",
            "n_modification_types"
        ]
    ].head(20).to_string(index=False)
)

# -----------------------------
# Plot 2
# Top 20 backbones by modification types
# -----------------------------
top20_types = backbone_mod_types.head(20).copy()
top20_types["plot_label"] = top20_types.apply(make_backbone_label, axis=1)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top20_types)))

plt.figure(figsize=(14, 7))

bars = plt.bar(
    top20_types["plot_label"],
    top20_types["n_modification_types"],
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

plt.ylabel("Number of modification types", fontsize=12)
plt.xlabel("Backbone / Representative peptide name", fontsize=12)
plt.title("Modification diversity per peptide backbone", fontsize=14)

plt.xticks(rotation=45, ha="right")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.08,
        f"{int(h)}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "backbone_modification_type_diversity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# STEP 4B
# Number of modification STRATEGIES used per backbone
# -----------------------------
# Exclude "other" from strategy-level statistics.

backbone_mod_strategy_count = (
    df_strategy.groupby("peptide_backbone_clean")["modification_signature"]
    .nunique()
    .reset_index(name="n_modification_strategies")
    .merge(backbone_name_map, on="peptide_backbone_clean", how="left")
    .sort_values("n_modification_strategies", ascending=False)
)

backbone_mod_strategy_count.to_csv(
    "backbone_modification_strategy_count.csv",
    index=False
)

print("\nBackbones with most modification strategies:")
print(
    backbone_mod_strategy_count[
        [
            "peptide_backbone_clean",
            "representative_peptide_name",
            "n_modification_strategies"
        ]
    ].head(20).to_string(index=False)
)

# -----------------------------
# Plot 3
# Top 20 backbones by modification strategies
# -----------------------------
top20_strategy = backbone_mod_strategy_count.head(20).copy()
top20_strategy["plot_label"] = top20_strategy.apply(make_backbone_label, axis=1)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top20_strategy)))

plt.figure(figsize=(14, 7))

bars = plt.bar(
    top20_strategy["plot_label"],
    top20_strategy["n_modification_strategies"],
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

plt.ylabel("Number of modification strategies", fontsize=12)
plt.xlabel("Backbone / Representative peptide name", fontsize=12)
plt.title("Modification strategies per peptide backbone", fontsize=14)

plt.xticks(rotation=45, ha="right")

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.08,
        f"{int(h)}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "backbone_modification_strategy_diversity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Print: 'X peptides are used in Y% of studies or systems'
# -----------------------------
# This summary is based on the final backbone summary table (summary_df).
# "studies" = paper-level usage
# "systems" = row-level usage
# -----------------------------
# Build backbone summary (paper-level and row-level)
# -----------------------------

row_counts = (
    df_mod.groupby("peptide_backbone_clean")
    .size()
    .reset_index(name="row_count")
)

paper_counts = (
    df_mod.groupby("peptide_backbone_clean")["paper_id"]
    .nunique()
    .reset_index(name="paper_count")
)

summary_df = (
    paper_counts
    .merge(row_counts, on="peptide_backbone_clean", how="outer")
)

summary_df = summary_df.sort_values(
    "paper_count",
    ascending=False
).reset_index(drop=True)

# Calculate fractions
total_rows = summary_df["row_count"].sum()
total_papers = summary_df["paper_count"].sum()

summary_df["row_fraction"] = summary_df["row_count"] / total_rows * 100
summary_df["paper_fraction"] = summary_df["paper_count"] / total_papers * 100
for x in [5, 10, 20]:
    if len(summary_df) >= x:
        topx = summary_df.head(x)

        study_pct = topx["paper_fraction"].sum()
        system_pct = topx["row_fraction"].sum()

        print(
            f"Top {x} peptides are used in {study_pct:.2f}% of studies "
            f"(paper-level backbone occurrences)."
        )
        print(
            f"Top {x} peptides are used in {system_pct:.2f}% of systems "
            f"(row-level entries)."
        )
        print()

# Optional: print the exact peptide names for top X
x = 10
topx_names = summary_df.head(x)["peptide_backbone_clean"].tolist()
print(f"Top {x} peptide backbones:")
print(", ".join(topx_names))