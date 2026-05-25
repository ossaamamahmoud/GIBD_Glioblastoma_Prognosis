"""
GIBD analysis workflow — calibration and decision-curve analysis

This script generates calibration and decision-curve analysis outputs for the
locked GIBD-XGBoost K100 model. It uses TCGA out-of-fold predictions for
internal assessment and frozen post-lock CGGA predictions for external
clinical-context assessment.

Analysis guardrails:
- The locked model, locked feature set, and locked operating threshold are not changed.
- TCGA out-of-fold predictions are used for internal calibration and DCA when available.
- CGGA is used only for post-lock external reporting.
- Calibration and decision-curve outputs are clinical-context analyses, not evidence of
  clinical deployment readiness.

Expected project layout uses relative paths under Data/ and Data/Revision_Ablation/.
"""


import os
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation", "Clinical_Utility_MedianOS_K100"),

    # Locked champion artifacts.
    "LOCKED_JOBLIB": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_Model.joblib"),
    "LOCKED_METADATA": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_Model_metadata.json"),
    "LOCKED_FEATURES": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_Features.csv"),

    # Matrices.
    "TCGA_MATRIX": os.path.join("Data", "Revision_Ablation", "tcga_weighted_self_graph_cache.csv"),
    "CGGA_MATRIX": os.path.join("Data", "Revision_Ablation", "cgga_weighted_self_graph_cache.csv"),

    # Labels.
    "TCGA_LABELS_WITH_OS_EVENT": os.path.join("Data", "TCGA_survival_labels_with_os_event.csv"),
    "CGGA_LABELS_WITH_OS_EVENT": os.path.join("Data", "Revision_Ablation", "cgga_labels_tcga_median357_with_os_event.csv"),
    "CGGA_LABELS_FALLBACK": os.path.join("Data", "Revision_Ablation", "cgga_labels_tcga_median357_with_os_event.csv"),

    # True TCGA OOF file exported by patched 16U script.
    "TCGA_TRUE_OOF": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_oof_predictions.csv"),

    # Apparent training predictions are allowed only as a fallback and will be labeled as apparent.
    "TCGA_TRAINING_PREDICTIONS": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_training_predictions.csv"),

    # Expected sample counts.
    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,
    "EXPECTED_N_CGGA": 131,
    "EXPECTED_LOW_CGGA": 85,
    "EXPECTED_HIGH_CGGA": 46,

    # Calibration bins.
    "N_CALIBRATION_BINS": 10,

    # DCA thresholds.
    "DCA_THRESHOLD_MIN": 0.05,
    "DCA_THRESHOLD_MAX": 0.95,
    "DCA_N_THRESHOLDS": 181,

    # Check net benefit near locked classification threshold.
    "LOCKED_THRESHOLD_TOLERANCE": 0.0025,

    # Figure settings.
    "DPI": 300,
    "FIG_WIDTH_IN": 5.2,
    "FIG_HEIGHT_IN": 4.1,
}

os.makedirs(CONF["OUT_DIR"], exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def normalize_feature_name(name):
    text = str(name)
    text = re.sub(r"^(ENSG\d+)\.\d+(_.*)?$", r"\1\2", text)
    return text


def safe_float(x, default=np.nan):
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def clip_prob(scores, eps=1e-6):
    scores = np.asarray(scores, dtype=float)
    return np.clip(scores, eps, 1.0 - eps)


def load_feature_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found: {path}")

    X = pd.read_csv(path, index_col=0)
    X.index = X.index.astype(str)
    X.columns = [normalize_feature_name(c) for c in X.columns.astype(str)]
    X = X.loc[:, ~pd.Index(X.columns).duplicated()].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def load_labels(path, id_candidates=("Patient_ID", "CGGA_ID")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    y = pd.read_csv(path)

    id_col = None
    for c in id_candidates:
        if c in y.columns:
            id_col = c
            break

    if id_col is not None:
        y = y.set_index(id_col)
    elif "Unnamed: 0" in y.columns:
        y = y.set_index("Unnamed: 0")
    else:
        y = y.set_index(y.columns[0])

    y.index = y.index.astype(str)

    if "Risk_Label" not in y.columns:
        raise ValueError(f"Risk_Label missing from: {path}")

    y["Risk_Label"] = pd.to_numeric(y["Risk_Label"], errors="coerce")
    y = y.dropna(subset=["Risk_Label"]).copy()
    y["Risk_Label"] = y["Risk_Label"].astype(int)
    return y


def assert_label_counts(labels, label_name, expected_n, expected_low, expected_high):
    if len(labels) != int(expected_n):
        raise ValueError(f"{label_name}: expected N={expected_n}, got {len(labels)}")
    counts = labels["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))
    if low != int(expected_low) or high != int(expected_high):
        raise ValueError(
            f"{label_name}: expected low={expected_low}, high={expected_high}; "
            f"got low={low}, high={high}"
        )


def find_col(df, candidates):
    cols_lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    for cand in candidates:
        key = cand.lower()
        for c in df.columns:
            if key in str(c).lower():
                return c
    return None


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


# --------------------------------------------------
# 3. Locked model loading and prediction
# --------------------------------------------------

def load_locked_artifacts():
    for path in [CONF["LOCKED_JOBLIB"], CONF["LOCKED_METADATA"], CONF["LOCKED_FEATURES"]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing locked artifact: {path}")

    bundle = load(CONF["LOCKED_JOBLIB"])

    with open(CONF["LOCKED_METADATA"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feat_df = pd.read_csv(CONF["LOCKED_FEATURES"])
    if "feature" not in feat_df.columns:
        raise ValueError("16U_Final_Locked_Features.csv must contain a 'feature' column.")

    features = [normalize_feature_name(f) for f in feat_df["feature"].astype(str).tolist()]

    threshold = metadata.get("locked_threshold", metadata.get("ensemble_threshold"))
    if threshold is None:
        threshold = bundle.get("locked_threshold", bundle.get("ensemble_threshold"))
    if threshold is None:
        raise ValueError("Could not find locked threshold.")

    return bundle, metadata, features, float(threshold)


def model_score_from_fitted(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Model has neither predict_proba nor decision_function.")


def predict_locked_model(bundle, features, X):
    X_use = X.reindex(columns=features, fill_value=0.0)
    scaler = bundle.get("scaler", None)
    X_s = scaler.transform(X_use) if scaler is not None else X_use.values

    if "models" in bundle and bundle["models"] is not None:
        probs = [model_score_from_fitted(model, X_s) for model in bundle["models"]]
        return np.mean(np.column_stack(probs), axis=1)

    if "model" in bundle and bundle["model"] is not None:
        return model_score_from_fitted(bundle["model"], X_s)

    raise ValueError("Locked bundle contains no model(s).")


# --------------------------------------------------
# 4. TCGA OOF / apparent prediction loading
# --------------------------------------------------

def load_true_tcga_oof(tcga_labels):
    path = CONF["TCGA_TRUE_OOF"]
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)

    id_col = find_col(df, ["Patient_ID", "patient_id", "sample_id", "ID", "Unnamed: 0"])
    score_col = find_col(df, ["OOF_Probability", "oof_probability", "OOF_Prob", "oof_prob", "oof_score"])

    if id_col is None or score_col is None:
        raise ValueError(
            f"OOF file exists but required columns were not found: {path}\n"
            "Required: Patient_ID and OOF_Probability."
        )

    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)

    if df.index.duplicated().any():
        # If repeated rows exist, average OOF predictions per patient.
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        agg = df.groupby(df.index)[score_col].mean().to_frame(score_col)
        df = agg

    common = tcga_labels.index.intersection(df.index)
    if len(common) != len(tcga_labels):
        missing = tcga_labels.index.difference(df.index).tolist()
        raise ValueError(
            f"True TCGA OOF file does not cover all TCGA samples. "
            f"Expected {len(tcga_labels)}, found {len(common)}. "
            f"First missing IDs: {missing[:10]}"
        )

    scores = pd.to_numeric(df.loc[tcga_labels.index, score_col], errors="coerce")
    if scores.isna().any():
        bad = scores[scores.isna()].index.tolist()
        raise ValueError(f"OOF probabilities contain NaNs. First bad IDs: {bad[:10]}")

    arr = scores.astype(float).values
    if arr.min() < 0 or arr.max() > 1:
        raise ValueError("OOF probabilities must be within [0,1].")

    source = f"true_oof_file={path}; score_col={score_col}"
    return arr, source


def load_apparent_tcga_predictions(tcga_labels):
    path = CONF["TCGA_TRAINING_PREDICTIONS"]
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)
    id_col = find_col(df, ["Patient_ID", "patient_id", "sample_id", "ID", "Unnamed: 0"])
    score_col = find_col(df, ["Predicted_Probability", "probability", "prob", "score"])

    if id_col is None or score_col is None:
        return None, None

    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)

    common = tcga_labels.index.intersection(df.index)
    if len(common) != len(tcga_labels):
        return None, None

    scores = pd.to_numeric(df.loc[tcga_labels.index, score_col], errors="coerce")
    if scores.isna().any():
        return None, None

    arr = np.asarray(scores.astype(float).values, dtype=float)
    if arr.min() < 0 or arr.max() > 1:
        arr = 1.0 / (1.0 + np.exp(-arr))

    source = f"apparent_training_file={path}; score_col={score_col}"
    return arr, source


# --------------------------------------------------
# 5. Metrics, calibration, and DCA
# --------------------------------------------------

def binary_metrics(y_true, p, threshold):
    y_true = np.asarray(y_true).astype(int)
    p = clip_prob(p)
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    prevalence = float(np.mean(y_true))
    null_prob = np.full_like(p, fill_value=prevalence, dtype=float)
    null_brier = float(brier_score_loss(y_true, null_prob))
    model_brier = float(brier_score_loss(y_true, p))

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "n": int(len(y_true)),
        "prevalence": prevalence,
        "auc": safe_auc(y_true, p),
        "auprc": safe_auprc(y_true, p),
        "brier": model_brier,
        "null_brier_prevalence": null_brier,
        "brier_improvement_vs_null": float(null_brier - model_brier),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calibration_table(y_true, p, cohort, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    p = clip_prob(p)

    # Quantile calibration bins: more stable for N=133/147 than fixed-width bins.
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.unique(edges)

    if len(edges) <= 2:
        edges = np.linspace(0, 1, min(n_bins, len(p)) + 1)

    bin_ids = np.digitize(p, edges[1:-1], right=True)

    rows = []
    for b in sorted(np.unique(bin_ids)):
        idx = bin_ids == b
        n = int(idx.sum())
        if n == 0:
            continue

        mean_pred = float(np.mean(p[idx]))
        obs = float(np.mean(y_true[idx]))

        rows.append({
            "cohort": cohort,
            "bin_id": int(b) + 1,
            "n": n,
            "mean_predicted_risk": mean_pred,
            "observed_high_risk_fraction": obs,
            "absolute_calibration_error": abs(obs - mean_pred),
            "bin_min_score": float(np.min(p[idx])),
            "bin_max_score": float(np.max(p[idx])),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["bin_weight"] = out["n"] / out["n"].sum()
    ece = float((out["bin_weight"] * out["absolute_calibration_error"]).sum())
    out["expected_calibration_error"] = ece
    return out


def decision_curve(y_true, p, cohort, thresholds):
    y_true = np.asarray(y_true).astype(int)
    p = clip_prob(p)
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    rows = []
    for pt in thresholds:
        pt = float(pt)
        pred = p >= pt

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())

        nb_model = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        nb_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        nb_none = 0.0

        rows.append({
            "cohort": cohort,
            "threshold_probability": pt,
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
            "model_minus_treat_all": float(nb_model - nb_all),
            "model_minus_treat_none": float(nb_model - nb_none),
            "model_better_than_treat_all": bool(nb_model > nb_all),
            "model_better_than_treat_none": bool(nb_model > nb_none),
            "model_better_than_both": bool((nb_model > nb_all) and (nb_model > nb_none)),
            "n": int(n),
            "prevalence": prevalence,
            "tp": tp,
            "fp": fp,
        })

    return pd.DataFrame(rows)


def summarise_dca(dca_df, locked_threshold):
    rows = []

    for cohort, df in dca_df.groupby("cohort"):
        df = df.sort_values("threshold_probability").reset_index(drop=True)

        def ranges_for(mask_col):
            sub = df[df[mask_col].astype(bool)].copy()
            if sub.empty:
                return "", np.nan, np.nan

            # DCA thresholds are dense; give global min/max range for manuscript summary.
            return (
                f"{sub['threshold_probability'].min():.3f}-{sub['threshold_probability'].max():.3f}",
                float(sub["threshold_probability"].min()),
                float(sub["threshold_probability"].max()),
            )

        both_range, both_min, both_max = ranges_for("model_better_than_both")
        all_range, all_min, all_max = ranges_for("model_better_than_treat_all")
        none_range, none_min, none_max = ranges_for("model_better_than_treat_none")

        idx = (df["threshold_probability"] - float(locked_threshold)).abs().idxmin()
        row_locked = df.loc[idx]

        rows.append({
            "cohort": cohort,
            "locked_threshold": float(locked_threshold),
            "nearest_dca_threshold": float(row_locked["threshold_probability"]),
            "net_benefit_at_locked_threshold": float(row_locked["net_benefit_model"]),
            "treat_all_net_benefit_at_locked_threshold": float(row_locked["net_benefit_treat_all"]),
            "treat_none_net_benefit_at_locked_threshold": float(row_locked["net_benefit_treat_none"]),
            "model_minus_treat_all_at_locked_threshold": float(row_locked["model_minus_treat_all"]),
            "model_minus_treat_none_at_locked_threshold": float(row_locked["model_minus_treat_none"]),
            "model_better_than_treat_all_at_locked_threshold": bool(row_locked["model_better_than_treat_all"]),
            "model_better_than_treat_none_at_locked_threshold": bool(row_locked["model_better_than_treat_none"]),
            "model_better_than_both_at_locked_threshold": bool(row_locked["model_better_than_both"]),
            "threshold_range_model_better_than_treat_all": all_range,
            "threshold_range_model_better_than_treat_none": none_range,
            "threshold_range_model_better_than_both": both_range,
            "range_both_min": both_min,
            "range_both_max": both_max,
            "max_net_benefit_model": float(df["net_benefit_model"].max()),
            "threshold_at_max_net_benefit": float(df.loc[df["net_benefit_model"].idxmax(), "threshold_probability"]),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 6. Plotting
# --------------------------------------------------

def safe_name(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_calibration_plot(calib_df, cohort, path_stem):
    df = calib_df[calib_df["cohort"].eq(cohort)].copy()
    if df.empty:
        print(f"Skipping calibration plot for {cohort}: no calibration rows.")
        return

    fig, ax = plt.subplots(figsize=(CONF["FIG_WIDTH_IN"], CONF["FIG_HEIGHT_IN"]))

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Ideal calibration")
    ax.plot(
        df["mean_predicted_risk"],
        df["observed_high_risk_fraction"],
        marker="o",
        linewidth=1.8,
        label=cohort,
    )

    for _, row in df.iterrows():
        ax.text(
            row["mean_predicted_risk"],
            row["observed_high_risk_fraction"],
            str(int(row["n"])),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Mean predicted high-risk probability")
    ax.set_ylabel("Observed high-risk fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(path_stem + ".png", dpi=CONF["DPI"])
    fig.savefig(path_stem + ".pdf")
    plt.close(fig)


def save_dca_plot(dca_df, cohort, path_stem):
    df = dca_df[dca_df["cohort"].eq(cohort)].copy()
    if df.empty:
        print(f"Skipping DCA plot for {cohort}: no DCA rows.")
        return

    fig, ax = plt.subplots(figsize=(CONF["FIG_WIDTH_IN"], CONF["FIG_HEIGHT_IN"]))

    ax.plot(df["threshold_probability"], df["net_benefit_model"], linewidth=1.8, label="Locked GIBD-XGBoost")
    ax.plot(df["threshold_probability"], df["net_benefit_treat_all"], linestyle="--", linewidth=1.3, label="Treat all")
    ax.plot(df["threshold_probability"], df["net_benefit_treat_none"], linestyle=":", linewidth=1.3, label="Treat none")

    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(CONF["DCA_THRESHOLD_MIN"], CONF["DCA_THRESHOLD_MAX"])

    y_vals = pd.concat([
        df["net_benefit_model"],
        df["net_benefit_treat_all"],
        df["net_benefit_treat_none"],
    ]).replace([np.inf, -np.inf], np.nan).dropna()

    if len(y_vals) > 0:
        lo = float(np.quantile(y_vals, 0.02))
        hi = float(np.quantile(y_vals, 0.98))
        if hi > lo:
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(path_stem + ".png", dpi=CONF["DPI"])
    fig.savefig(path_stem + ".pdf")
    plt.close(fig)


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():
    print("=" * 80)
    print("Locked GIBD clinical utility: Median-OS K100 calibration + DCA")
    print("=" * 80)

    bundle, metadata, features, locked_threshold = load_locked_artifacts()

    print(f"Locked model features: {len(features)}")
    print(f"Locked threshold: {locked_threshold:.6f}")
    print("Expected TCGA labels: N=147, low=73, high=74")
    print("Expected CGGA357 labels: N=131, low=85, high=46")

    # ------------------------------
    # TCGA probabilities
    # ------------------------------
    y_tcga = load_labels(CONF["TCGA_LABELS_WITH_OS_EVENT"], id_candidates=("Patient_ID",))
    assert_label_counts(
        y_tcga,
        "TCGA median-OS labels",
        CONF["EXPECTED_N_TCGA"],
        CONF["EXPECTED_LOW_TCGA"],
        CONF["EXPECTED_HIGH_TCGA"],
    )

    tcga_scores, tcga_source = load_true_tcga_oof(y_tcga)

    if tcga_scores is not None:
        tcga_cohort = "TCGA_OOF"
        print(f"Using TRUE TCGA OOF predictions: {tcga_source}")
    else:
        print("\nWARNING: True TCGA OOF prediction file not found.")
        print(f"Expected: {CONF['TCGA_TRUE_OOF']}")
        print("Trying apparent training predictions only as fallback.")
        tcga_scores, tcga_source = load_apparent_tcga_predictions(y_tcga)

        if tcga_scores is None:
            # Last resort: locked model re-prediction on TCGA matrix.
            X_tcga = load_feature_matrix(CONF["TCGA_MATRIX"])
            common = X_tcga.index.intersection(y_tcga.index)
            X_tcga = X_tcga.loc[common].copy()
            y_tcga = y_tcga.loc[common].copy()
            tcga_scores = predict_locked_model(bundle, features, X_tcga)
            tcga_source = "apparent_locked_model_reprediction_on_full_TCGA"

        tcga_cohort = "TCGA_apparent_locked_model"
        print("IMPORTANT: TCGA output will be labeled as apparent, NOT OOF.")
        print("Do NOT use TCGA apparent calibration/DCA as internal validation evidence.")

    # ------------------------------
    # CGGA probabilities
    # ------------------------------
    X_cgga = load_feature_matrix(CONF["CGGA_MATRIX"])
    cgga_label_path = CONF["CGGA_LABELS_WITH_OS_EVENT"]
    if not os.path.exists(cgga_label_path):
        raise FileNotFoundError(
            "Median-OS CGGA357 censor-aware label file is required for this branch: "
            f"{cgga_label_path}"
        )
    y_cgga = load_labels(cgga_label_path, id_candidates=("CGGA_ID", "Patient_ID"))

    common_cgga = X_cgga.index.intersection(y_cgga.index)
    X_cgga = X_cgga.loc[common_cgga].copy()
    y_cgga = y_cgga.loc[common_cgga].copy()

    assert_label_counts(
        y_cgga,
        "CGGA357 median-OS censor-aware labels",
        CONF["EXPECTED_N_CGGA"],
        CONF["EXPECTED_LOW_CGGA"],
        CONF["EXPECTED_HIGH_CGGA"],
    )

    cgga_scores = predict_locked_model(bundle, features, X_cgga)
    cgga_cohort = "CGGA_external_locked"
    cgga_source = f"locked_model_post_lockdown; labels={cgga_label_path}"

    # ------------------------------
    # Build scores dataframe
    # ------------------------------
    scores_rows = []

    for pid, yval, prob in zip(y_tcga.index, y_tcga["Risk_Label"].astype(int).values, tcga_scores):
        scores_rows.append({
            "cohort": tcga_cohort,
            "sample_id": pid,
            "Risk_Label": int(yval),
            "predicted_probability": float(prob),
            "locked_threshold": float(locked_threshold),
            "prediction_source": tcga_source,
            "is_true_oof": bool(tcga_cohort == "TCGA_OOF"),
        })

    for pid, yval, prob in zip(y_cgga.index, y_cgga["Risk_Label"].astype(int).values, cgga_scores):
        scores_rows.append({
            "cohort": cgga_cohort,
            "sample_id": pid,
            "Risk_Label": int(yval),
            "predicted_probability": float(prob),
            "locked_threshold": float(locked_threshold),
            "prediction_source": cgga_source,
            "is_true_oof": False,
        })

    scores_df = pd.DataFrame(scores_rows)
    scores_path = os.path.join(CONF["OUT_DIR"], "locked_gibd_clinical_utility_scores_v2.csv")
    scores_df.to_csv(scores_path, index=False)

    # ------------------------------
    # Metrics, calibration, DCA
    # ------------------------------
    metric_rows = []
    calib_frames = []
    dca_frames = []

    cohorts_data = [
        (tcga_cohort, y_tcga["Risk_Label"].astype(int).values, tcga_scores, tcga_source),
        (cgga_cohort, y_cgga["Risk_Label"].astype(int).values, cgga_scores, cgga_source),
    ]

    thresholds = np.linspace(
        CONF["DCA_THRESHOLD_MIN"],
        CONF["DCA_THRESHOLD_MAX"],
        CONF["DCA_N_THRESHOLDS"],
    )

    for cohort_name, yvals, probs, source in cohorts_data:
        metrics = binary_metrics(yvals, probs, locked_threshold)
        calib = calibration_table(yvals, probs, cohort_name, CONF["N_CALIBRATION_BINS"])
        dca = decision_curve(yvals, probs, cohort_name, thresholds)

        ece = np.nan
        if not calib.empty and "expected_calibration_error" in calib.columns:
            ece = float(calib["expected_calibration_error"].iloc[0])

        metrics.update({
            "cohort": cohort_name,
            "prediction_source": source,
            "expected_calibration_error": ece,
        })

        metric_rows.append(metrics)
        calib_frames.append(calib)
        dca_frames.append(dca)

    metrics_df = pd.DataFrame(metric_rows)
    calib_df = pd.concat(calib_frames, ignore_index=True)
    dca_df = pd.concat(dca_frames, ignore_index=True)
    dca_summary_df = summarise_dca(dca_df, locked_threshold)

    metrics_path = os.path.join(CONF["OUT_DIR"], "locked_gibd_calibration_metrics_v2.csv")
    calib_path = os.path.join(CONF["OUT_DIR"], "locked_gibd_calibration_points_v2.csv")
    dca_path = os.path.join(CONF["OUT_DIR"], "locked_gibd_dca_curve_v2.csv")
    dca_summary_path = os.path.join(CONF["OUT_DIR"], "locked_gibd_dca_summary_v2.csv")

    metrics_df.to_csv(metrics_path, index=False)
    calib_df.to_csv(calib_path, index=False)
    dca_df.to_csv(dca_path, index=False)
    dca_summary_df.to_csv(dca_summary_path, index=False)

    # ------------------------------
    # Figures
    # ------------------------------
    for cohort_name in [tcga_cohort, cgga_cohort]:
        fig_label = "CGGA_External" if cohort_name == cgga_cohort else safe_name(cohort_name)

        save_calibration_plot(
            calib_df,
            cohort_name,
            os.path.join(CONF["OUT_DIR"], f"Figure_Calibration_{fig_label}"),
        )

        save_dca_plot(
            dca_df,
            cohort_name,
            os.path.join(CONF["OUT_DIR"], f"Figure_DCA_{fig_label}"),
        )

    # ------------------------------
    # Audit
    # ------------------------------
    audit = {
        "script": "12_calibration_dca_locked_model_median_os_k100.py",
        "locked_model_joblib": CONF["LOCKED_JOBLIB"],
        "locked_metadata": CONF["LOCKED_METADATA"],
        "locked_features": CONF["LOCKED_FEATURES"],
        "locked_threshold": locked_threshold,
        "tcga_cohort_label": tcga_cohort,
        "tcga_prediction_source": tcga_source,
        "tcga_true_oof_required_path": CONF["TCGA_TRUE_OOF"],
        "tcga_note": (
            "TCGA calibration/DCA is true OOF only if tcga_cohort_label == TCGA_OOF. "
            "If labeled TCGA_apparent_locked_model, do not present as OOF evidence."
        ),
        "cgga_cohort_label": cgga_cohort,
        "cgga_prediction_source": cgga_source,
        "cgga_use": "Post-lockdown external clinical-utility evaluation only.",
        "outputs": {
            "scores": scores_path,
            "metrics": metrics_path,
            "calibration_points": calib_path,
            "dca_curve": dca_path,
            "dca_summary": dca_summary_path,
            "figures_dir": CONF["OUT_DIR"],
        },
    }

    audit_path = os.path.join(CONF["OUT_DIR"], "locked_gibd_clinical_utility_audit_v2.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # ------------------------------
    # Console report
    # ------------------------------
    print("\n" + "=" * 80)
    print("CLINICAL UTILITY METRICS V2")
    print("=" * 80)

    show_cols = [
        "cohort",
        "n",
        "prevalence",
        "auc",
        "auprc",
        "brier",
        "null_brier_prevalence",
        "brier_improvement_vs_null",
        "expected_calibration_error",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "threshold",
    ]
    print(metrics_df[show_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("DCA SUMMARY V2")
    print("=" * 80)

    dca_show = [
        "cohort",
        "locked_threshold",
        "nearest_dca_threshold",
        "net_benefit_at_locked_threshold",
        "treat_all_net_benefit_at_locked_threshold",
        "treat_none_net_benefit_at_locked_threshold",
        "model_better_than_treat_all_at_locked_threshold",
        "model_better_than_treat_none_at_locked_threshold",
        "model_better_than_both_at_locked_threshold",
        "threshold_range_model_better_than_both",
    ]
    print(dca_summary_df[dca_show].to_string(index=False))

    # Warning at locked threshold.
    for _, row in dca_summary_df.iterrows():
        if not bool(row["model_better_than_both_at_locked_threshold"]):
            print("\nWARNING")
            print(
                f"{row['cohort']}: Model is NOT better than both treat-all and treat-none "
                f"at locked threshold {row['nearest_dca_threshold']:.3f}."
            )
            print(
                "Use DCA as range-based supportive analysis, not as proof of "
                "net benefit at the locked classification threshold."
            )

    if tcga_cohort != "TCGA_OOF":
        print("\nIMPORTANT TCGA WARNING")
        print("True TCGA OOF predictions were not found.")
        print("TCGA outputs are apparent and should not be used as OOF calibration/DCA evidence.")
    else:
        print("\nTCGA OOF status: TRUE OOF predictions were used.")

    print("\nSaved outputs:")
    print(f"- Scores: {scores_path}")
    print(f"- Metrics: {metrics_path}")
    print(f"- Calibration points: {calib_path}")
    print(f"- DCA curve: {dca_path}")
    print(f"- DCA summary: {dca_summary_path}")
    print(f"- Audit: {audit_path}")
    print(f"- Figures directory: {CONF['OUT_DIR']}")


if __name__ == "__main__":
    main()
