import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# =========================
# Load data
# =========================
df = pd.read_csv("data2_with_mod_signature.csv")

# =========================
# Columns in pipeline order
# =========================
cols = [
    "uptake_confirmed",
    "in_vitro_functional_effect",
    "endosomal_escape_evidence",
    "in_vivo_flag",
    "delivery_success_class",
]

# Output file names
backfilled_xlsx_path = "data_backfilled.xlsx"
csv_path = "branched_funnel_stage_summary.csv"
out_path = "branched_funnel_delivery_pipeline_colored.png"

# =========================
# Normalize values to 1 / 0 / NaN
# =========================
def norm_binary_series(s):
    s = s.copy()
    s = s.apply(lambda x: x.strip() if isinstance(x, str) else x)

    s = s.replace({
        "": np.nan,
        " ": np.nan,
        "NA": np.nan,
        "na": np.nan,
        "NaN": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "none": np.nan,
    })

    s = s.replace({
        "yes": 1, "Yes": 1, "YES": 1,
        "y": 1, "Y": 1,
        "true": 1, "True": 1, True: 1,
        "positive": 1, "Positive": 1,
        "success": 1, "Success": 1,
        "successful": 1, "Successful": 1,
        "high": 1, "medium": 1, "low": 1,

        "no": 0, "No": 0, "NO": 0,
        "n": 0, "N": 0,
        "false": 0, "False": 0, False: 0,
        "negative": 0, "Negative": 0,
        "failure": 0, "Failure": 0,
        "failed": 0, "Failed": 0,
    })

    return pd.to_numeric(s, errors="coerce")

# Make sure all required columns exist
missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns in CSV: {missing_cols}")

for c in cols:
    df[c] = norm_binary_series(df[c])

# =========================
# Backfill rule:
# If a later stage is 1, all previous stages are set to 1
# =========================
df_before_backfill = df.copy()

for i in range(len(cols) - 1, -1, -1):
    later = cols[i]
    mask = df[later] == 1
    for j in range(i):
        earlier = cols[j]
        df.loc[mask, earlier] = 1

changed_cells = (df_before_backfill[cols] != df[cols]).sum().sum()

# =========================
# Save backfilled dataframe
# =========================
df.to_excel(backfilled_xlsx_path, index=False)

# =========================
# Sequential branched-funnel counts
# =========================
records = []
current_mask = pd.Series(True, index=df.index)
current_n = int(current_mask.sum())

for col in cols:
    s = df.loc[current_mask, col]

    pos = int((s == 1).sum())
    neg = int((s == 0).sum())
    na = int(s.isna().sum())

    records.append(
        {
            "stage": col,
            "entered_n": current_n,
            "continue_n": pos,
            "failed_n": neg,
            "not_tested_n": na,
            "continue_p": pos / current_n if current_n else np.nan,
            "failed_p": neg / current_n if current_n else np.nan,
            "not_tested_p": na / current_n if current_n else np.nan,
        }
    )

    current_mask = current_mask & (df[col] == 1)
    current_n = int(current_mask.sum())

table_df = pd.DataFrame(records)

# =========================
# Additional conditional probabilities
# =========================
def safe_mean(mask_base, col_target):
    sub = df.loc[mask_base, col_target]
    if len(sub) == 0:
        return np.nan
    return float((sub == 1).mean())

summary_stats = {
    "P(in_vivo_flag = 1 | in_vitro_functional_effect = 1)": safe_mean(
        df["in_vitro_functional_effect"] == 1, "in_vivo_flag"
    ),
    "P(delivery_success_class = 1 | endosomal_escape_evidence = 1)": safe_mean(
        df["endosomal_escape_evidence"] == 1, "delivery_success_class"
    ),
    "P(delivery_success_class = 1 | in_vivo_flag = 1)": safe_mean(
        df["in_vivo_flag"] == 1, "delivery_success_class"
    ),
    "P(endosomal_escape_evidence = 1 | in_vitro_functional_effect = 1)": safe_mean(
        df["in_vitro_functional_effect"] == 1, "endosomal_escape_evidence"
    ),
}

# =========================
# Save summary table
# =========================
table_df.to_csv(csv_path, index=False)

# =========================
# Colors
# =========================
main_fill = "#DCEEFF"
main_edge = "#2F6DB3"
main_arrow = "#6FA8DC"

not_test_fill = "#FFF1CC"
not_test_edge = "#D6A400"

fail_fill = "#FDE0DD"
fail_edge = "#C65A5A"

stage_fill = "#EAF4FF"
stage_edge = "#245A9C"

text_dark = "#1F1F1F"

# =========================
# Draw figure
# =========================
fig = plt.figure(figsize=(23, 11), facecolor="white")
ax = fig.add_axes([0.03, 0.08, 0.74, 0.84])
ax2 = fig.add_axes([0.80, 0.10, 0.18, 0.80])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax2.axis("off")

x_positions = [0.06, 0.21, 0.36, 0.51, 0.66, 0.81]
box_w = 0.11
box_h = 0.12
y_main = 0.50
y_top = 0.83
y_bottom = 0.17

# Start box
start_box = FancyBboxPatch(
    (x_positions[0] - box_w / 2, y_main - box_h / 2),
    box_w,
    box_h,
    boxstyle="round,pad=0.02,rounding_size=0.02",
    facecolor=stage_fill,
    edgecolor=stage_edge,
    linewidth=2,
)
ax.add_patch(start_box)
ax.text(
    x_positions[0],
    y_main + 0.020,
    "All rows",
    ha="center",
    va="center",
    fontsize=13,
    color=text_dark,
)
ax.text(
    x_positions[0],
    y_main - 0.022,
    f"n = {len(df)}",
    ha="center",
    va="center",
    fontsize=13,
    color=text_dark,
)

prev_x = x_positions[0]

for i, rec in enumerate(records, start=1):
    x = x_positions[i]

    # stage box
    stage_box = FancyBboxPatch(
        (x - box_w / 2, y_main - box_h / 2),
        box_w,
        box_h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=stage_fill,
        edgecolor=stage_edge,
        linewidth=2,
    )
    ax.add_patch(stage_box)

    ax.text(
        x,
        y_main + 0.018,
        rec["stage"],
        ha="center",
        va="center",
        fontsize=10,
        color=text_dark,
        wrap=True,
    )
    ax.text(
        x,
        y_main - 0.024,
        f"n = {rec['continue_n']}",
        ha="center",
        va="center",
        fontsize=12,
        color=text_dark,
    )

    # main colored connector block between boxes
    left = prev_x + box_w / 2 + 0.01
    right = x - box_w / 2 - 0.01
    width = right - left

    connector = FancyBboxPatch(
        (left, y_main - 0.022),
        width,
        0.044,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=main_fill,
        edgecolor=main_arrow,
        linewidth=1.6,
        alpha=1.0,
    )
    ax.add_patch(connector)

    # little arrow head
    ax.annotate(
        "",
        xy=(right, y_main),
        xytext=(left + width * 0.80, y_main),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=main_edge),
    )

    midx = (prev_x + x) / 2

    # continue label in independent box above connector
    label_y = 0.63 if i % 2 == 1 else 0.69
    ax.text(
        midx,
        label_y,
        f"continue: {rec['continue_n']} / {rec['entered_n']} = {rec['continue_p']:.1%}",
        ha="center",
        va="center",
        fontsize=9.8,
        color=text_dark,
        bbox=dict(
            boxstyle="round,pad=0.28,rounding_size=0.12",
            facecolor=main_fill,
            edgecolor=main_edge,
            linewidth=1.2,
        ),
    )

    # subtle guide line from label box to connector
    ax.plot(
        [midx, midx],
        [label_y - 0.028, y_main + 0.030],
        color=main_arrow,
        linewidth=1.0,
        alpha=0.8,
    )

    # failed branch
    if rec["failed_n"] > 0:
        ax.annotate(
            "",
            xy=(midx, y_bottom + 0.055),
            xytext=(midx, y_main - 0.03),
            arrowprops=dict(arrowstyle="->", lw=1.8, color=fail_edge),
        )
        ax.text(
            midx,
            y_bottom,
            f"failed\n{rec['failed_n']} / {rec['entered_n']} = {rec['failed_p']:.1%}",
            ha="center",
            va="center",
            fontsize=9.6,
            color=text_dark,
            bbox=dict(
                boxstyle="round,pad=0.30,rounding_size=0.12",
                facecolor=fail_fill,
                edgecolor=fail_edge,
                linewidth=1.2,
            ),
        )

    # not tested branch
    if rec["not_tested_n"] > 0:
        ax.annotate(
            "",
            xy=(midx, y_top - 0.055),
            xytext=(midx, y_main + 0.03),
            arrowprops=dict(arrowstyle="->", lw=1.8, color=not_test_edge),
        )
        ax.text(
            midx,
            y_top,
            f"not tested / not assessed\n{rec['not_tested_n']} / {rec['entered_n']} = {rec['not_tested_p']:.1%}",
            ha="center",
            va="center",
            fontsize=9.2,
            color=text_dark,
            bbox=dict(
                boxstyle="round,pad=0.30,rounding_size=0.12",
                facecolor=not_test_fill,
                edgecolor=not_test_edge,
                linewidth=1.2,
            ),
        )

    prev_x = x

# Title
fig.suptitle(
    "Branched funnel of delivery pipeline",
    fontsize=20,
    y=0.975,
    color=text_dark,
)
ax.text(
    0.5,
    0.94,
    "Rule: if a later stage = 1, all earlier stages are backfilled to 1.\n"
    "Main path uses strict sequential filtering: only rows with all previous stages = 1 enter the next stage.",
    ha="center",
    va="center",
    fontsize=11,
    color=text_dark,
)

# Summary panel
ax2.text(0.0, 0.97, "Backfill summary", fontsize=15, va="top", color=text_dark)
ax2.text(0.0, 0.91, f"Changed cells: {int(changed_cells)}", fontsize=12, va="top", color=text_dark)
ax2.text(0.0, 0.85, f"Backfilled file:\n{backfilled_xlsx_path}", fontsize=11, va="top", color=text_dark)

ax2.text(0.0, 0.77, "Key conditional probabilities", fontsize=15, va="top", color=text_dark)
y = 0.71
for k, v in summary_stats.items():
    ax2.text(0.0, y, k, fontsize=10.2, va="top", color=text_dark)
    if np.isnan(v):
        ax2.text(0.0, y - 0.05, "NA", fontsize=16, va="top", color=text_dark)
    else:
        ax2.text(0.0, y - 0.05, f"{v:.1%}", fontsize=16, va="top", color=main_edge)
    y -= 0.16

ax2.text(0.0, 0.08, "Sequential stage counts", fontsize=14, va="top", color=text_dark)
y = 0.02
for rec in records:
    ax2.text(
        0.0,
        y,
        f"{rec['stage']}: entered {rec['entered_n']}, continue {rec['continue_n']}",
        fontsize=9.0,
        va="top",
        color=text_dark,
    )
    y -= 0.055

# Save figure
plt.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved colored figure to {out_path}")
print(f"Saved summary table to {csv_path}")
print(f"Saved backfilled xlsx to {backfilled_xlsx_path}")
print(f"Backfilled cells: {int(changed_cells)}")