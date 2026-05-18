from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0. Basic settings
# =========================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data_backfilled.xlsx"
OUTPUT_DIR = BASE_DIR / "figures_T4"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "T4_top10_peptide_stage_success_summary.csv"

STAGES = [
    ("uptake_confirmed", "Uptake confirmed"),
    ("in_vitro_functional_effect", "In vitro functional effect"),
    ("endosomal_escape_evidence", "Endosomal escape evidence"),
    ("in_vivo_flag", "In vivo evidence"),
    ("delivery_success_class", "Delivery success class"),
]

# Ordered and labeled for the supplementary figure.
PEPTIDES = [
    {
        "label": "3K",
        "backbone": "KFFKFFKFFK",
        "name": "(KFF)3K",
    },
    {
        "label": "PF14",
        "backbone": "AGYLLGKLLOOLAAAALOOLL",
        "name": "PF14",
    },
    {
        "label": "R8",
        "backbone": "RRRRRRRR",
        "name": "Octaarginine (R8)",
    },
    {
        "label": "R9",
        "backbone": "RRRRRRRRR",
        "name": "R9",
    },
    {
        "label": "penetratin",
        "backbone": "RQIKIWFQNRRMKWKK",
        "name": "Penetratin",
    },
    {
        "label": "pepfect",
        "backbone": "AGYLLGKINLKALAALAKKIL",
        "name": "PepFect6 (PF6)",
    },
    {
        "label": "tat1",
        "backbone": "YGRKKRRQRRR",
        "name": "TAT",
    },
    {
        "label": "tat2",
        "backbone": "RKKRRQRRR",
        "name": "TAT",
    },
    {
        "label": "tat3",
        "backbone": "GRKKRRQRRR",
        "name": "Tat",
    },
    {
        "label": "transpotin",
        "backbone": "GWTLNSAGYLLGKINLKALAALAKKIL",
        "name": "Transportan",
    },
]


# =========================
# 1. Helper functions
# =========================

def normalize_binary_series(series: pd.Series) -> pd.Series:
    """Convert common binary labels to 1/0/NaN."""
    series = series.copy()
    series = series.apply(lambda x: x.strip() if isinstance(x, str) else x)

    series = series.replace(
        {
            "": np.nan,
            " ": np.nan,
            "NA": np.nan,
            "na": np.nan,
            "NaN": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "none": np.nan,
        }
    )

    series = series.replace(
        {
            "yes": 1,
            "Yes": 1,
            "YES": 1,
            "y": 1,
            "Y": 1,
            "true": 1,
            "True": 1,
            True: 1,
            "positive": 1,
            "Positive": 1,
            "success": 1,
            "Success": 1,
            "successful": 1,
            "Successful": 1,
            "high": 1,
            "medium": 1,
            "low": 1,
            "no": 0,
            "No": 0,
            "NO": 0,
            "n": 0,
            "N": 0,
            "false": 0,
            "False": 0,
            False: 0,
            "negative": 0,
            "Negative": 0,
            "failure": 0,
            "Failure": 0,
            "failed": 0,
            "Failed": 0,
        }
    )

    return pd.to_numeric(series, errors="coerce")


def clean_backbone(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", "", regex=True)


def compute_stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for peptide in PEPTIDES:
        backbone = peptide["backbone"]
        mask = df["peptide_backbone_clean_normalized"] == backbone
        sub = df.loc[mask].copy()

        for stage_col, stage_label in STAGES:
            values = sub[stage_col]
            tested_mask = values.notna()
            total_n = int(len(values))
            tested_n = int(tested_mask.sum())
            success_n = int((values == 1).sum())
            fail_n = int((values == 0).sum())
            missing_n = int(values.isna().sum())

            success_rate_tested = (
                success_n / tested_n if tested_n > 0 else np.nan
            )
            success_rate_all_rows = (
                success_n / total_n if total_n > 0 else np.nan
            )

            records.append(
                {
                    "stage": stage_col,
                    "stage_label": stage_label,
                    "peptide_label": peptide["label"],
                    "backbone": backbone,
                    "representative_name": peptide["name"],
                    "total_rows": total_n,
                    "tested_n": tested_n,
                    "success_n": success_n,
                    "fail_n": fail_n,
                    "missing_n": missing_n,
                    "success_rate_tested": success_rate_tested,
                    "success_rate_all_rows": success_rate_all_rows,
                }
            )

    return pd.DataFrame(records)


def plot_stage(stage_df: pd.DataFrame, stage_col: str, stage_label: str) -> None:
    plot_df = stage_df[stage_df["stage"] == stage_col].copy()
    plot_df["success_percent"] = plot_df["success_rate_tested"] * 100

    x = np.arange(len(plot_df))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(plot_df)))

    fig, ax = plt.subplots(figsize=(12, 5.8))
    bars = ax.bar(
        x,
        plot_df["success_percent"].fillna(0),
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Success probability among assessed rows (%)", fontsize=11)
    ax.set_xlabel("Peptide backbone", fontsize=11)
    ax.set_title(f"{stage_label}: success probability for top 10 backbones", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["peptide_label"], rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        if pd.isna(row["success_rate_tested"]):
            label = "NA\n0/0"
            y = 2
        else:
            label = (
                f"{row['success_rate_tested'] * 100:.0f}%\n"
                f"{int(row['success_n'])}/{int(row['tested_n'])}"
            )
            y = bar.get_height() + 2

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    note = (
        "Denominator excludes missing / not assessed entries. "
        "Input values are from data_backfilled.xlsx."
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9, color="#555555")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path = OUTPUT_DIR / f"T4_{stage_col}_success_probability.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =========================
# 2. Main analysis
# =========================

def main() -> None:
    df = pd.read_excel(INPUT_FILE)

    required_cols = ["peptide_backbone_clean"] + [stage for stage, _ in STAGES]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    df["peptide_backbone_clean_normalized"] = clean_backbone(
        df["peptide_backbone_clean"]
    )

    for stage_col, _ in STAGES:
        df[stage_col] = normalize_binary_series(df[stage_col])

    summary_df = compute_stage_summary(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    for stage_col, stage_label in STAGES:
        plot_stage(summary_df, stage_col, stage_label)

    print("Saved summary table:", SUMMARY_CSV)
    print("Saved figures to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
