"""
GIBD analysis workflow — Figure 3 ablation-performance plot

This script generates the final composite Figure 3 from previously generated
ablation result tables. Panel A summarizes TCGA out-of-fold and post-lock CGGA
external AUC values. Panel B summarizes post-lock CGGA sensitivity, specificity,
and balanced accuracy at each model-specific TCGA-derived operating threshold.

Analysis guardrails:
- This is a plotting-only script.
- It does not train models, select features, select thresholds, or modify model
  outputs.
- The plotted values are read from the final ablation result CSV files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

OUT_DIR = Path("Data") / "Revision_Ablation"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    OUT_DIR / "revision_ablation_master_v3_median_os_k100_g035_final_no_k80_summary.csv",
    OUT_DIR / "revision_ablation_master_v3_median_os_k100_g035_final_no_k80_external_locked_eval.csv",
    OUT_DIR / "revision_ablation_master_v3_median_os_k100_g035_final_summary.csv",
    OUT_DIR / "revision_ablation_master_v3_median_os_k100_g035_final_external_locked_eval.csv",
    OUT_DIR / "revision_ablation_master_v3_summary.csv",
    OUT_DIR / "revision_ablation_master_v3_external_locked_eval.csv",
]

input_path = None
for p in INPUT_CANDIDATES:
    if p.exists():
        input_path = p
        break

if input_path is None:
    raise FileNotFoundError(
        "Could not find the final ablation summary/external evaluation CSV.\n"
        "Expected one of:\n"
        + "\n".join(str(p) for p in INPUT_CANDIDATES)
    )

print(f"Reading ablation results from:\n{input_path}")
df = pd.read_csv(input_path)


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def normalize_colname(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
        .replace(".", "")
    )


def find_col(df_in, candidates, required=True):
    normalized = {normalize_colname(c): c for c in df_in.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            "Could not find required column. Tried: "
            + ", ".join(candidates)
            + "\nAvailable columns:\n"
            + "\n".join(df_in.columns.astype(str))
        )

    return None


def get_numeric_col(df_in, candidates, required=True):
    col = find_col(df_in, candidates, required=required)
    if col is None:
        return None
    return pd.to_numeric(df_in[col], errors="coerce")


# --------------------------------------------------
# 3. Standardize required columns
# --------------------------------------------------

experiment_col = find_col(
    df,
    ["experiment_id", "Experiment_ID", "Experiment", "Display_Name", "comparison_role"],
    required=True,
)

tcga_auc = get_numeric_col(df, ["TCGA_OOF_AUC", "tcga_oof_auc"], required=True)
external_auc = get_numeric_col(df, ["External_AUC", "external_auc", "cgga_auc"], required=True)

external_sens = get_numeric_col(
    df,
    ["External_Sensitivity", "external_sensitivity", "sensitivity"],
    required=True,
)

external_spec = get_numeric_col(
    df,
    ["External_Specificity", "external_specificity", "specificity"],
    required=True,
)

external_balacc = get_numeric_col(
    df,
    ["External_Balanced_Accuracy", "external_balanced_accuracy", "External_BalAcc", "balanced_accuracy"],
    required=True,
)


df_plot = pd.DataFrame(
    {
        "experiment_id": df[experiment_col].astype(str),
        "TCGA_OOF_AUC": tcga_auc,
        "External_AUC": external_auc,
        "External_Sensitivity": external_sens,
        "External_Specificity": external_spec,
        "External_Balanced_Accuracy": external_balacc,
    }
)


# --------------------------------------------------
# 4. Clean model labels and enforce manuscript order
# --------------------------------------------------

label_map = {
    "T1_GIBD_XGB_K100_CHAMPION_LOCKED": "GIBD-XGB\nK100",
    "T1_RAW_XGB_K100_TUNED": "Raw-XGB\nK100",
    "T2_RAW_RF_K100_TUNED": "Raw-RF\nK100",
    "T2_GIBD_RF_K100_TUNED": "GIBD-RF\nK100",
    "T2_RAW_LASSO_COX_K100_TUNED": "Raw LASSO-\nCox K100",
    "T3_GIBD_XGB_K120_COMPLEXITY_CONTROL": "GIBD-XGB\nK120",
}

order = list(label_map.keys())


def infer_experiment_key(text):
    text = str(text)
    for key in order:
        if key in text:
            return key

    low = text.lower()

    if "champion" in low:
        return "T1_GIBD_XGB_K100_CHAMPION_LOCKED"
    if "gibd" in low and ("xgb" in low or "xgboost" in low) and "120" in low:
        return "T3_GIBD_XGB_K120_COMPLEXITY_CONTROL"
    if "gibd" in low and ("xgb" in low or "xgboost" in low) and "100" in low:
        return "T1_GIBD_XGB_K100_CHAMPION_LOCKED"
    if "raw" in low and ("xgb" in low or "xgboost" in low):
        return "T1_RAW_XGB_K100_TUNED"
    if "raw" in low and ("rf" in low or "randomforest" in low or "random forest" in low):
        return "T2_RAW_RF_K100_TUNED"
    if "gibd" in low and ("rf" in low or "randomforest" in low or "random forest" in low):
        return "T2_GIBD_RF_K100_TUNED"
    if "lasso" in low or "cox" in low:
        return "T2_RAW_LASSO_COX_K100_TUNED"

    return text


df_plot["experiment_key"] = df_plot["experiment_id"].apply(infer_experiment_key)
df_plot["Model_Label"] = df_plot["experiment_key"].map(label_map).fillna(df_plot["experiment_id"])
df_plot["plot_order"] = df_plot["experiment_key"].apply(
    lambda x: order.index(x) if x in order else 999
)

df_plot = (
    df_plot[df_plot["experiment_key"].isin(order)]
    .sort_values("plot_order")
    .drop_duplicates(subset=["experiment_key"], keep="first")
    .reset_index(drop=True)
)

if len(df_plot) != 6:
    print("\nWARNING: Expected 6 final ablation rows, but found:", len(df_plot))
    print(df_plot[["experiment_id", "experiment_key", "Model_Label"]])


# --------------------------------------------------
# 5. Audit expected final numbers
# --------------------------------------------------
# This is a safety check only. It does not modify the figure.

expected = {
    "T1_GIBD_XGB_K100_CHAMPION_LOCKED": {
        "External_AUC": 0.609,
        "External_Sensitivity": 0.739,
        "External_Specificity": 0.506,
        "External_Balanced_Accuracy": 0.623,
    },
    "T1_RAW_XGB_K100_TUNED": {
        "External_AUC": 0.535,
        "External_Sensitivity": 0.696,
        "External_Specificity": 0.329,
        "External_Balanced_Accuracy": 0.513,
    },
    "T2_RAW_RF_K100_TUNED": {
        "External_AUC": 0.563,
        "External_Sensitivity": 0.826,
        "External_Specificity": 0.188,
        "External_Balanced_Accuracy": 0.507,
    },
    "T2_GIBD_RF_K100_TUNED": {
        "External_AUC": 0.583,
        "External_Sensitivity": 0.587,
        "External_Specificity": 0.541,
        "External_Balanced_Accuracy": 0.564,
    },
    "T2_RAW_LASSO_COX_K100_TUNED": {
        "External_AUC": 0.408,
        "External_Sensitivity": 0.457,
        "External_Specificity": 0.435,
        "External_Balanced_Accuracy": 0.446,
    },
    "T3_GIBD_XGB_K120_COMPLEXITY_CONTROL": {
        "External_AUC": 0.613,
        "External_Sensitivity": 0.543,
        "External_Specificity": 0.647,
        "External_Balanced_Accuracy": 0.595,
    },
}

print("\nAudit check against expected manuscript values:")
for _, row in df_plot.iterrows():
    key = row["experiment_key"]
    if key not in expected:
        continue
    for metric, expected_value in expected[key].items():
        observed = float(row[metric])
        if not np.isfinite(observed):
            print(f"  WARNING: {key} {metric} is missing.")
            continue
        if abs(observed - expected_value) > 0.015:
            print(
                f"  WARNING: {key} {metric}: "
                f"observed={observed:.3f}, expected≈{expected_value:.3f}"
            )

print("\nPlot data:")
print(
    df_plot[
        [
            "Model_Label",
            "TCGA_OOF_AUC",
            "External_AUC",
            "External_Sensitivity",
            "External_Specificity",
            "External_Balanced_Accuracy",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 6. Save plot data table
# --------------------------------------------------

plot_data_path = FIG_DIR / "Figure3_Ablation_Performance_v7_plot_data.csv"
df_plot.to_csv(plot_data_path, index=False)
print(f"\nSaved plot data:\n{plot_data_path}")


# --------------------------------------------------
# 7. Figure style
# --------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.7,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.8,
        "figure.titlesize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.85,
    }
)

# Color-blind-conscious muted palette.
color_tcga = "#B8B8B8"
color_external = "#2F6C99"
color_sens = "#3B7EA1"
color_spec = "#9C6B3E"
color_bal = "#4F8B4F"


# --------------------------------------------------
# 8. Build final composite with dedicated legend rows
# --------------------------------------------------

x = np.arange(len(df_plot))
labels = df_plot["Model_Label"].tolist()

fig = plt.figure(figsize=(6.9, 6.35))
gs = fig.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=[1.0, 1.12],
    hspace=0.38,
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

legend_kw = {
    "loc": "upper right",
    "frameon": False,
    "fontsize": 6.6,
    "handlelength": 1.25,
    "handletextpad": 0.45,
    "columnspacing": 0.85,
    "borderaxespad": 0.2,
}


# --------------------------------------------------
# Panel A: TCGA OOF AUC vs CGGA external AUC
# --------------------------------------------------

bar_width = 0.32

bars_tcga = ax1.bar(
    x - bar_width / 2,
    df_plot["TCGA_OOF_AUC"],
    width=bar_width,
    color=color_tcga,
    edgecolor="black",
    linewidth=0.6,
)

bars_cgga = ax1.bar(
    x + bar_width / 2,
    df_plot["External_AUC"],
    width=bar_width,
    color=color_external,
    edgecolor="black",
    linewidth=0.6,
)

ax1.axhline(0.50, color="black", linewidth=0.75, linestyle="--", alpha=0.75)
ax1.set_ylim(0.35, 0.72)
ax1.set_ylabel("AUC")
ax1.set_title("A. Discrimination", loc="left", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=0, ha="center")
ax1.tick_params(axis="x", pad=4)
ax1.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
ax1.get_xticklabels()[0].set_fontweight("bold")
ax1.legend(
    handles=[
        Patch(facecolor=color_tcga, edgecolor="black", label="TCGA OOF AUC"),
        Patch(facecolor=color_external, edgecolor="black", label="CGGA external AUC"),
    ],
    ncol=2,
    bbox_to_anchor=(0.985, 0.999),
    **legend_kw,
)

for bars in [bars_tcga, bars_cgga]:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax1.text(
                b.get_x() + b.get_width() / 2,
                h + 0.006,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                rotation=90,
            )


# --------------------------------------------------
# Panel B: External sensitivity, specificity, balanced accuracy
# --------------------------------------------------

sens_pct = df_plot["External_Sensitivity"] * 100.0
spec_pct = df_plot["External_Specificity"] * 100.0
bal_pct = df_plot["External_Balanced_Accuracy"] * 100.0

group_width = 0.66
w = group_width / 3.0

bars_sens = ax2.bar(
    x - w,
    sens_pct,
    width=w,
    color=color_sens,
    edgecolor="black",
    linewidth=0.6,
)

bars_spec = ax2.bar(
    x,
    spec_pct,
    width=w,
    color=color_spec,
    edgecolor="black",
    linewidth=0.6,
)

bars_bal = ax2.bar(
    x + w,
    bal_pct,
    width=w,
    color=color_bal,
    edgecolor="black",
    linewidth=0.6,
)

ax2.axhline(50, color="black", linewidth=0.75, linestyle="--", alpha=0.75)
ax2.set_ylim(0, 95)
ax2.set_ylabel("External CGGA metric (%)")
ax2.set_title("B. External operating characteristics", loc="left", fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=0, ha="center")
ax2.tick_params(axis="x", pad=4)
ax2.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.65)
ax2.get_xticklabels()[0].set_fontweight("bold")
ax2.legend(
    handles=[
        Patch(facecolor=color_sens, edgecolor="black", label="Sensitivity"),
        Patch(facecolor=color_spec, edgecolor="black", label="Specificity"),
        Patch(facecolor=color_bal, edgecolor="black", label="Balanced accuracy"),
    ],
    ncol=3,
    bbox_to_anchor=(0.985, 0.965),
    **legend_kw,
)

for bars in [bars_sens, bars_spec, bars_bal]:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax2.text(
                b.get_x() + b.get_width() / 2,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                rotation=90,
            )

# Avoid full-figure title/legend. The manuscript contains title and legend.
fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.085)


# --------------------------------------------------
# 9. Export final composite
# --------------------------------------------------

pdf_path = FIG_DIR / "Figure3_Ablation_Performance_final_composite_v7.pdf"
png_path = FIG_DIR / "Figure3_Ablation_Performance_final_composite_v7.png"
tiff_path = FIG_DIR / "Figure3_Ablation_Performance_final_composite_v7.tiff"
optional_pdf_path = FIG_DIR / "Figure3_Ablation_Performance_composite_optional.pdf"
optional_png_path = FIG_DIR / "Figure3_Ablation_Performance_composite_optional.png"
optional_tiff_path = FIG_DIR / "Figure3_Ablation_Performance_composite_optional.tiff"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(optional_pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(optional_png_path, dpi=600, bbox_inches="tight")
fig.savefig(
    tiff_path,
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(
    optional_tiff_path,
    dpi=600,
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},
)

plt.close(fig)

print("\nSaved final composite Figure 3 files:")
print(f"- {pdf_path}")
print(f"- {png_path}")
print(f"- {tiff_path}")
print(f"- {optional_pdf_path}")
print(f"- {optional_png_path}")
print(f"- {optional_tiff_path}")

print("\nManuscript figure title suggestion:")
print("Figure 3. Ablation and external validation.")

print("\nManuscript legend suggestion:")
print(
    "Ablation analysis compared the final locked GIBD-XGBoost K100 model with "
    "raw-expression XGBoost, raw-expression Random Forest, GIBD-Random Forest, "
    "raw LASSO-Cox, and a GIBD-XGBoost K120 complexity-control comparator. "
    "Panel A shows TCGA out-of-fold and post-lock CGGA external AUC values. "
    "Panel B shows external CGGA sensitivity, specificity, and balanced accuracy "
    "at the TCGA-derived locked thresholds. All non-champion comparators were "
    "selected using TCGA-only out-of-fold performance, and CGGA was used only "
    "after model/threshold locking for single-pass external evaluation. K120 is "
    "retained as a complexity-control comparator; GIBD-XGBoost K100 is the final "
    "locked model."
)



