import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 1. Load data
# -----------------------------
input_file = Path("data.xlsx")
output_file = Path("data_1.xlsx")

df = pd.read_excel(input_file)

# -----------------------------
# 2. Basic cleaning
# -----------------------------
# Standardize text columns
text_cols = ["peptide_name", "peptide_backbone_clean", "peptide_modifications"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype("string").str.strip()

# Remove internal spaces from backbone sequences
df["peptide_backbone_clean"] = (
    df["peptide_backbone_clean"]
    .str.replace(r"\s+", "", regex=True)
)

# Standardize missing values
for col in ["peptide_backbone_clean", "peptide_modifications", "peptide_name"]:
    df[col] = df[col].replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "NA": pd.NA
    })

# -----------------------------
# 3. Define chemical variant
# -----------------------------
# Template is directly defined by peptide_backbone_clean.
# Only chemical_variant is added as a new column.
df["chemical_variant"] = (
    df["peptide_backbone_clean"].fillna("missing_backbone")
    + " || " +
    df["peptide_modifications"].fillna("unmodified_or_not_reported")
)

# -----------------------------
# 4. Save the updated dataset
# -----------------------------
df.to_excel(output_file, index=False)
print(f"Saved updated Excel file: {output_file}")

# -----------------------------
# 5. Keep valid rows for backbone-based analysis
# -----------------------------
df_valid = df.dropna(subset=["peptide_backbone_clean"]).copy()

# -----------------------------
# 6. Step 1 and Step 2: basic diversity statistics
# -----------------------------
n_unique_backbones = df_valid["peptide_backbone_clean"].nunique()
n_unique_variants = df_valid["chemical_variant"].nunique()

print(f"\nNumber of unique peptide backbones (templates): {n_unique_backbones}")
print(f"Number of unique chemical variants: {n_unique_variants}")

# -----------------------------
# 7. Build backbone name map
# -----------------------------
# For each backbone, assign the most frequent peptide_name for readability.
def most_frequent_name(series):
    series = series.dropna()
    if len(series) == 0:
        return pd.NA
    return series.value_counts().index[0]

backbone_name_map = (
    df_valid.groupby("peptide_backbone_clean")["peptide_name"]
    .apply(most_frequent_name)
    .reset_index(name="representative_peptide_name")
)

# -----------------------------
# 8. Row-level backbone count
# -----------------------------
# This counts how many rows each backbone appears in.
row_level_counts = (
    df_valid.groupby("peptide_backbone_clean")
    .size()
    .reset_index(name="row_count")
    .sort_values("row_count", ascending=False)
)

row_level_counts = row_level_counts.merge(
    backbone_name_map,
    on="peptide_backbone_clean",
    how="left"
)

print("\nTop 10 backbones by row-level count:")
print(row_level_counts.head(10).to_string(index=False))

# -----------------------------
# 9. Step 3: Count backbone reuse across independent papers
# -----------------------------
# Each backbone is counted once per paper, regardless of how many
# modifications or repeated rows it has within the same paper.
df_paper_backbone = df.dropna(subset=["paper_id", "peptide_backbone_clean"]).copy()

paper_backbone_unique = df_paper_backbone.drop_duplicates(
    subset=["paper_id", "peptide_backbone_clean"]
).copy()

backbone_paper_counts = (
    paper_backbone_unique.groupby("peptide_backbone_clean")
    .agg(unique_paper_count=("paper_id", "nunique"))
    .reset_index()
    .sort_values("unique_paper_count", ascending=False)
)

backbone_paper_counts = backbone_paper_counts.merge(
    backbone_name_map,
    on="peptide_backbone_clean",
    how="left"
)

print("\nTop 10 backbones by unique paper count:")
print(backbone_paper_counts.head(10).to_string(index=False))

# -----------------------------
# 10. Step 4: Compare row-level and paper-level usage
# -----------------------------
backbone_usage_comparison = (
    row_level_counts[["peptide_backbone_clean", "representative_peptide_name", "row_count"]]
    .merge(
        backbone_paper_counts[["peptide_backbone_clean", "unique_paper_count"]],
        on="peptide_backbone_clean",
        how="outer"
    )
)

backbone_usage_comparison["row_count"] = (
    backbone_usage_comparison["row_count"].fillna(0).astype(int)
)
backbone_usage_comparison["unique_paper_count"] = (
    backbone_usage_comparison["unique_paper_count"].fillna(0).astype(int)
)

# Sort primarily by paper-level usage, then by row-level usage
backbone_usage_comparison = backbone_usage_comparison.sort_values(
    ["unique_paper_count", "row_count"],
    ascending=False
).reset_index(drop=True)

print("\nTop 20 backbones: row-level vs paper-level comparison")
print(backbone_usage_comparison.head(20).to_string(index=False))

# -----------------------------
# 11. Add ranking columns
# -----------------------------
row_rank_df = row_level_counts[["peptide_backbone_clean", "row_count"]].copy()
row_rank_df = row_rank_df.sort_values("row_count", ascending=False).reset_index(drop=True)
row_rank_df["row_rank"] = np.arange(1, len(row_rank_df) + 1)

paper_rank_df = backbone_paper_counts[["peptide_backbone_clean", "unique_paper_count"]].copy()
paper_rank_df = paper_rank_df.sort_values("unique_paper_count", ascending=False).reset_index(drop=True)
paper_rank_df["paper_rank"] = np.arange(1, len(paper_rank_df) + 1)

backbone_usage_comparison = (
    backbone_usage_comparison
    .merge(row_rank_df[["peptide_backbone_clean", "row_rank"]], on="peptide_backbone_clean", how="left")
    .merge(paper_rank_df[["peptide_backbone_clean", "paper_rank"]], on="peptide_backbone_clean", how="left")
)

print("\nTop 20 backbones with ranking comparison:")
print(
    backbone_usage_comparison[
        [
            "peptide_backbone_clean",
            "representative_peptide_name",
            "row_count",
            "unique_paper_count",
            "row_rank",
            "paper_rank",
        ]
    ].head(20).to_string(index=False)
)

# -----------------------------
# 12. Plot 1: Top 10 backbones by row-level count
# -----------------------------
top10_row = row_level_counts.head(10).copy()

def make_label(row):
    backbone = row["peptide_backbone_clean"]
    name = row["representative_peptide_name"]
    if pd.isna(name):
        return f"{backbone}\n(name unavailable)"
    return f"{backbone}\n{name}"

top10_row["plot_label"] = top10_row.apply(make_label, axis=1)

plt.figure(figsize=(12, 6))
plt.bar(top10_row["plot_label"], top10_row["row_count"])
plt.ylabel("Row count")
plt.xlabel("Backbone / Representative peptide name")
plt.title("Top 10 peptide backbones by row-level count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top10_backbone_row_count.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 13. Plot 2: Top 10 backbones by unique paper count
# -----------------------------
top10_paper = backbone_paper_counts.head(10).copy()
top10_paper["plot_label"] = top10_paper.apply(make_label, axis=1)

plt.figure(figsize=(12, 6))
plt.bar(top10_paper["plot_label"], top10_paper["unique_paper_count"])
plt.ylabel("Unique paper count")
plt.xlabel("Backbone / Representative peptide name")
plt.title("Top 10 peptide backbones by unique paper count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top10_backbone_unique_paper_count.png", dpi=300, bbox_inches="tight")
plt.show()
# -----------------------------
# 14. Step 5: Concentration analysis based on unique paper count
# -----------------------------
# This step quantifies how much of the literature is covered by the most
# frequently reused peptide backbones.

total_unique_paper_occurrences = backbone_paper_counts["unique_paper_count"].sum()

top5_count = backbone_paper_counts.head(5)["unique_paper_count"].sum()
top10_count = backbone_paper_counts.head(10)["unique_paper_count"].sum()
top20_count = backbone_paper_counts.head(20)["unique_paper_count"].sum()

top5_fraction = top5_count / total_unique_paper_occurrences if total_unique_paper_occurrences > 0 else np.nan
top10_fraction = top10_count / total_unique_paper_occurrences if total_unique_paper_occurrences > 0 else np.nan
top20_fraction = top20_count / total_unique_paper_occurrences if total_unique_paper_occurrences > 0 else np.nan

print("\nStep 5: Concentration analysis based on unique paper count")
print(f"Total backbone-paper occurrences: {total_unique_paper_occurrences}")
print(f"Top 5 backbones account for {top5_fraction:.2%} of all backbone-paper occurrences.")
print(f"Top 10 backbones account for {top10_fraction:.2%} of all backbone-paper occurrences.")
print(f"Top 20 backbones account for {top20_fraction:.2%} of all backbone-paper occurrences.")

# -----------------------------
# 15. Print ready-to-use interpretation sentences
# -----------------------------
n_total_backbones = backbone_paper_counts["peptide_backbone_clean"].nunique()

print("\nSuggested result statements:")
print(
    f"Across the dataset, {n_total_backbones} unique peptide backbones were identified."
)
print(
    f"The top 5 most frequently reused backbones accounted for {top5_fraction:.1%} "
    f"of all backbone-paper occurrences."
)
print(
    f"The top 10 most frequently reused backbones accounted for {top10_fraction:.1%} "
    f"of all backbone-paper occurrences."
)
print(
    f"This suggests that the CPP literature is concentrated around a limited number "
    f"of peptide backbones rather than being broadly distributed across many distinct templates."
)
# -----------------------------
# -----------------------------
# 16. Pie chart: Top 10 individual backbones vs Others
# -----------------------------
# The pie itself does not display labels or percentages.
# All names, counts, and percentages are shown in the legend.

top10 = backbone_paper_counts.head(10).copy()

top10_counts = top10["unique_paper_count"].tolist()
others_count = total_unique_paper_occurrences - sum(top10_counts)

pie_sizes = top10_counts + [others_count]

legend_labels = []

for _, row in top10.iterrows():
    backbone = row["peptide_backbone_clean"]
    name = row["representative_peptide_name"]
    count = row["unique_paper_count"]
    pct = count / total_unique_paper_occurrences * 100

    if pd.isna(name):
        label = f"{backbone}: {count} ({pct:.1f}%)"
    else:
        label = f"{backbone} - {name}: {count} ({pct:.1f}%)"

    legend_labels.append(label)

others_pct = others_count / total_unique_paper_occurrences * 100
legend_labels.append(f"Others: {others_count} ({others_pct:.1f}%)")

plt.figure(figsize=(9, 8))

wedges, _ = plt.pie(
    pie_sizes,
    startangle=90
)

plt.title("Contribution of top 10 peptide backbones\n(based on unique paper count)")

plt.legend(
    wedges,
    legend_labels,
    title="Backbones",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    frameon=False
)

plt.tight_layout()
plt.savefig(
    "top10_backbone_pie_individual_unique_paper_count.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# -----------------------------
# 17. Step 7: Temporal trends using globally dominant backbones
# -----------------------------
# This step evaluates how peptide backbone usage changes over time.
# The analysis is based on unique year-paper-backbone combinations.
# The current incomplete year (2026) is excluded.

# Keep valid year, paper_id, and backbone entries
df_year_paper_backbone = df.dropna(
    subset=["year", "paper_id", "peptide_backbone_clean"]
).copy()

# Convert year to numeric
df_year_paper_backbone["year"] = pd.to_numeric(
    df_year_paper_backbone["year"],
    errors="coerce"
)

df_year_paper_backbone = df_year_paper_backbone.dropna(subset=["year"]).copy()
df_year_paper_backbone["year"] = df_year_paper_backbone["year"].astype(int)

# Remove duplicated year-paper-backbone combinations
year_paper_backbone_unique = df_year_paper_backbone.drop_duplicates(
    subset=["year", "paper_id", "peptide_backbone_clean"]
).copy()

# Exclude the incomplete current year
year_paper_backbone_unique = year_paper_backbone_unique[
    year_paper_backbone_unique["year"] < 2026
].copy()

# -----------------------------
# 18. Define global top 5 / top 10 / top 20 backbones
# -----------------------------
# These sets are determined from the full paper-level backbone usage table.

global_top5_backbones = set(
    backbone_paper_counts.head(5)["peptide_backbone_clean"]
)

global_top10_backbones = set(
    backbone_paper_counts.head(10)["peptide_backbone_clean"]
)

global_top20_backbones = set(
    backbone_paper_counts.head(20)["peptide_backbone_clean"]
)

print("\nGlobal top 5 backbones:")
print(sorted(global_top5_backbones))

print("\nGlobal top 10 backbones:")
print(sorted(global_top10_backbones))

print("\nGlobal top 20 backbones:")
print(sorted(global_top20_backbones))

# -----------------------------
# 19. Build yearly summary
# -----------------------------
# For each year, calculate:
# - total backbone-paper occurrences
# - number of unique backbones
# - occurrences contributed by the global top 5 / top 10 / top 20
# - Shannon diversity index

yearly_summary_records = []

for year, group in year_paper_backbone_unique.groupby("year"):
    yearly_counts = (
        group.groupby("peptide_backbone_clean")["paper_id"]
        .nunique()
        .sort_index()
    )

    total_occurrences = yearly_counts.sum()
    n_unique_backbones = yearly_counts.shape[0]

    global_top5_count = yearly_counts[yearly_counts.index.isin(global_top5_backbones)].sum()
    global_top10_count = yearly_counts[yearly_counts.index.isin(global_top10_backbones)].sum()
    global_top20_count = yearly_counts[yearly_counts.index.isin(global_top20_backbones)].sum()

    p = yearly_counts / total_occurrences
    shannon_index = -np.sum(p * np.log(p))

    yearly_summary_records.append({
        "year": year,
        "total_occurrences": total_occurrences,
        "n_unique_backbones": n_unique_backbones,
        "global_top5_count": global_top5_count,
        "global_top10_count": global_top10_count,
        "global_top20_count": global_top20_count,
        "global_top5_fraction": global_top5_count / total_occurrences if total_occurrences > 0 else np.nan,
        "global_top10_fraction": global_top10_count / total_occurrences if total_occurrences > 0 else np.nan,
        "global_top20_fraction": global_top20_count / total_occurrences if total_occurrences > 0 else np.nan,
        "shannon_index": shannon_index
    })

yearly_summary_df = pd.DataFrame(yearly_summary_records).sort_values("year")

print("\nYearly temporal summary:")
print(yearly_summary_df.to_string(index=False))

# -----------------------------
# 20. Plot 1: Area + line chart
# -----------------------------
# Area = total backbone-paper occurrences per year
# Lines = unique backbones, global top 20 count, global top 10 count, global top 5 count

plt.figure(figsize=(10, 6))

plt.fill_between(
    yearly_summary_df["year"],
    yearly_summary_df["total_occurrences"],
    alpha=0.2,
    label="Total backbone-paper occurrences"
)

plt.plot(
    yearly_summary_df["year"],
    yearly_summary_df["n_unique_backbones"],
    marker="o",
    linewidth=2,
    label="Number of unique backbones"
)

plt.plot(
    yearly_summary_df["year"],
    yearly_summary_df["global_top20_count"],
    marker="o",
    linewidth=2,
    label="Occurrences from global top 20 backbones"
)

plt.plot(
    yearly_summary_df["year"],
    yearly_summary_df["global_top10_count"],
    marker="o",
    linewidth=2,
    label="Occurrences from global top 10 backbones"
)

plt.plot(
    yearly_summary_df["year"],
    yearly_summary_df["global_top5_count"],
    marker="o",
    linewidth=2,
    label="Occurrences from global top 5 backbones"
)

plt.xlabel("Year")
plt.ylabel("Count")
plt.title("Temporal trend in peptide backbone usage")
plt.legend()
plt.tight_layout()
plt.savefig("yearly_backbone_usage_global_top_area_and_lines.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 21. Plot 2: Yearly Shannon diversity index
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    yearly_summary_df["year"],
    yearly_summary_df["shannon_index"],
    marker="o",
    linewidth=2
)

plt.xlabel("Year")
plt.ylabel("Shannon diversity index")
plt.title("Temporal trend in backbone diversity")
plt.tight_layout()
plt.savefig("yearly_shannon_diversity_index.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# 22. Print interpretation-ready summary
# -----------------------------
latest_year = yearly_summary_df["year"].max()
latest_row = yearly_summary_df[yearly_summary_df["year"] == latest_year].iloc[0]

print("\nSuggested interpretation:")
print(
    "The temporal analysis was anchored to the globally most reused peptide backbones "
    "and excluded the incomplete year 2026."
)
print(
    f"In {latest_year}, the global top 5 backbones accounted for "
    f"{latest_row['global_top5_fraction']:.1%} of all backbone-paper occurrences."
)
print(
    f"In {latest_year}, the global top 10 backbones accounted for "
    f"{latest_row['global_top10_fraction']:.1%} of all backbone-paper occurrences."
)
print(
    f"In {latest_year}, the global top 20 backbones accounted for "
    f"{latest_row['global_top20_fraction']:.1%} of all backbone-paper occurrences."
)
print(
    "This framework shows whether a limited set of historically dominant CPP backbones "
    "continued to shape the field over time."
)
# -----------------------------
# 25. Build final backbone summary table
# -----------------------------
# This table summarizes backbone usage statistics across the dataset.
# Each row represents one unique peptide backbone.

# Row-level counts
row_counts = (
    df_valid.groupby("peptide_backbone_clean")
    .size()
    .reset_index(name="row_count")
)

# Paper-level counts
paper_counts = (
    paper_backbone_unique.groupby("peptide_backbone_clean")["paper_id"]
    .nunique()
    .reset_index(name="paper_count")
)

# -----------------------------
# 26. Calculate first year, last year, and usage duration
# -----------------------------
# These statistics are based on unique paper-backbone occurrences.

paper_backbone_year = df.dropna(
    subset=["paper_id", "peptide_backbone_clean", "year"]
).copy()

paper_backbone_year["year"] = pd.to_numeric(
    paper_backbone_year["year"],
    errors="coerce"
)

paper_backbone_year = paper_backbone_year.dropna(subset=["year"]).copy()
paper_backbone_year["year"] = paper_backbone_year["year"].astype(int)

# Remove duplicated paper-backbone pairs before calculating year statistics
paper_backbone_year_unique = paper_backbone_year.drop_duplicates(
    subset=["paper_id", "peptide_backbone_clean"]
).copy()

backbone_year_stats = (
    paper_backbone_year_unique.groupby("peptide_backbone_clean")["year"]
    .agg(
        first_year_used="min",
        last_year_used="max"
    )
    .reset_index()
)

# Inclusive duration across years
backbone_year_stats["usage_duration_years"] = (
    backbone_year_stats["last_year_used"] - backbone_year_stats["first_year_used"] + 1
)

# -----------------------------
# 27. Merge all summary information
# -----------------------------
summary_df = (
    paper_counts
    .merge(row_counts, on="peptide_backbone_clean", how="outer")
    .merge(backbone_name_map, on="peptide_backbone_clean", how="left")
    .merge(backbone_year_stats, on="peptide_backbone_clean", how="left")
)

summary_df["row_count"] = summary_df["row_count"].fillna(0).astype(int)
summary_df["paper_count"] = summary_df["paper_count"].fillna(0).astype(int)

# -----------------------------
# 28. Sort by paper_count
# -----------------------------
# Paper-level occurrence is the main ranking metric.
summary_df = summary_df.sort_values(
    ["paper_count", "row_count"],
    ascending=False
).reset_index(drop=True)

# -----------------------------
# 29. Add ranking columns
# -----------------------------
summary_df["paper_rank"] = np.arange(1, len(summary_df) + 1)

summary_df["row_rank"] = (
    summary_df["row_count"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# -----------------------------
# 30. Calculate fractions
# -----------------------------
total_rows = summary_df["row_count"].sum()
total_papers = summary_df["paper_count"].sum()

summary_df["row_fraction"] = summary_df["row_count"] / total_rows
summary_df["paper_fraction"] = summary_df["paper_count"] / total_papers

# -----------------------------
# 31. Calculate cumulative fractions
# -----------------------------
summary_df["cumulative_paper_fraction"] = summary_df["paper_fraction"].cumsum()
summary_df["cumulative_row_fraction"] = summary_df["row_fraction"].cumsum()

# -----------------------------
# 32. Convert fractions to percentages
# -----------------------------
percentage_cols = [
    "row_fraction",
    "paper_fraction",
    "cumulative_paper_fraction",
    "cumulative_row_fraction"
]

for col in percentage_cols:
    summary_df[col] = summary_df[col] * 100

# -----------------------------
# 33. Round percentage columns for readability
# -----------------------------
summary_df[percentage_cols] = summary_df[percentage_cols].round(2)

# -----------------------------
# 34. Reorder columns
# -----------------------------
summary_df = summary_df[
    [
        "paper_rank",
        "row_rank",
        "peptide_backbone_clean",
        "representative_peptide_name",
        "paper_count",
        "row_count",
        "paper_fraction",
        "row_fraction",
        "cumulative_paper_fraction",
        "cumulative_row_fraction",
        "first_year_used",
        "last_year_used",
        "usage_duration_years"
    ]
]

# -----------------------------
# 35. Print top rows for inspection
# -----------------------------
print("\nFinal backbone summary (top 20):")
print(summary_df.head(20).to_string(index=False))

# -----------------------------
# 36. Print a few dataset-level statistics
# -----------------------------
n_unique_backbones_summary = summary_df["peptide_backbone_clean"].nunique()
median_paper_count = summary_df["paper_count"].median()
median_row_count = summary_df["row_count"].median()
median_duration = summary_df["usage_duration_years"].median()

top5_paper_pct = summary_df.head(5)["paper_fraction"].sum()
top10_paper_pct = summary_df.head(10)["paper_fraction"].sum()
top20_paper_pct = summary_df.head(20)["paper_fraction"].sum()

print("\nDataset-level statistics:")
print(f"Number of unique backbones: {n_unique_backbones_summary}")
print(f"Median paper count per backbone: {median_paper_count:.1f}")
print(f"Median row count per backbone: {median_row_count:.1f}")
print(f"Median usage duration (years): {median_duration:.1f}")
print(f"Top 5 backbones account for {top5_paper_pct:.2f}% of all backbone-paper occurrences.")
print(f"Top 10 backbones account for {top10_paper_pct:.2f}% of all backbone-paper occurrences.")
print(f"Top 20 backbones account for {top20_paper_pct:.2f}% of all backbone-paper occurrences.")
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# -----------------------------
# 36A. Plot usage duration for the 20 longest-lasting peptide backbones
# -----------------------------
# Identify the 20 backbones with the longest usage duration.
# If multiple backbones have the same duration, rank them by paper_count.

top20_longest_duration = (
    summary_df
    .sort_values(
        ["usage_duration_years", "paper_count"],
        ascending=False
    )
    .head(20)
    .copy()
)

print("\nTop 20 longest-lasting peptide backbones:")
print(
    top20_longest_duration[
        [
            "peptide_backbone_clean",
            "representative_peptide_name",
            "usage_duration_years",
            "first_year_used",
            "last_year_used",
            "paper_count"
        ]
    ].to_string(index=False)
)

# Build x-axis labels with both sequence and peptide name
def make_duration_label(row):
    backbone = row["peptide_backbone_clean"]
    name = row["representative_peptide_name"]
    if pd.isna(name):
        return backbone
    return f"{backbone}\n{name}"

top20_longest_duration["plot_label"] = top20_longest_duration.apply(
    make_duration_label,
    axis=1
)

# Science-style colormap
colors = plt.cm.viridis(
    np.linspace(0.2, 0.9, len(top20_longest_duration))
)

plt.figure(figsize=(16, 7))

bars = plt.bar(
    top20_longest_duration["plot_label"],
    top20_longest_duration["usage_duration_years"],
    color=colors,
    edgecolor="black",
    linewidth=0.8
)

plt.xlabel("Peptide backbone / representative peptide name", fontsize=12)
plt.ylabel("Usage duration (years)", fontsize=12)
plt.title("Usage duration of the 20 longest-lasting peptide backbones", fontsize=14)

plt.xticks(rotation=45, ha="right")

# Add duration labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.2,
        f"{int(height)} yr",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "top20_longest_backbone_duration_barplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# -----------------------------
# 37A. Bubble plot: backbone longevity vs popularity
# -----------------------------
# x-axis: usage duration
# y-axis: paper count
# bubble size: row count
# color: usage duration
# Only the most important points are labeled by index to avoid overlap.


plt.figure(figsize=(20, 12)) 



bubble_sizes = summary_df["row_count"] * 18

scatter = plt.scatter(
    summary_df["usage_duration_years"],
    summary_df["paper_count"],
    s=bubble_sizes,
    c=summary_df["usage_duration_years"],
    cmap="viridis",
    alpha=0.75,
    edgecolors="black",
    linewidths=0.5
)

plt.xlabel("Usage duration (years)", fontsize=12)
plt.ylabel("Paper count", fontsize=12)
plt.title("Backbone longevity vs popularity", fontsize=14)

cbar = plt.colorbar(scatter)
cbar.set_label("Usage duration (years)", fontsize=11)

# Select only the top 15 backbones for annotation
top_for_annotation = (
    summary_df
    .sort_values(["paper_count", "row_count"], ascending=False)
    .head(15)
    .copy()
    .reset_index(drop=True)
)

# Add numeric labels on the plot
for i, (_, row) in enumerate(top_for_annotation.iterrows(), start=1):
    plt.annotate(
        str(i),
        (row["usage_duration_years"], row["paper_count"]),
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center"
    )

# Build legend text shown on the right
legend_lines = []
for i, (_, row) in enumerate(top_for_annotation.iterrows(), start=1):
    backbone = row["peptide_backbone_clean"]
    name = row["representative_peptide_name"]

    if pd.isna(name):
        label = f"{i}. {backbone}"
    else:
        label = f"{i}. {backbone} | {name}"

    legend_lines.append(label)

legend_text = "\n".join(legend_lines)

# Put the text block outside the axes
plt.gcf().text(
    0.78, 0.5,
    legend_text,
    fontsize=9,
    va="center",
    ha="left"
)

plt.tight_layout(rect=[0, 0, 0.75, 1])

plt.savefig(
    "backbone_longevity_vs_popularity_bubbleplot_labeled.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# -----------------------------
# 37. Export CSV
# -----------------------------
summary_output = "backbone_summary_final.csv"
summary_df.to_csv(summary_output, index=False)

print(f"\nSaved summary CSV: {summary_output}")