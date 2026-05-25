"""
GIBD analysis workflow — clinical-covariate benchmark

This script benchmarks the locked GIBD-XGBoost K100 score against simple
clinical covariate models based on available age and sex/gender variables. The
analysis is designed to contextualize the transcriptomics-only score without
replacing or modifying the final locked model.

Analysis guardrails:
- The locked GIBD model, features, parameters, and threshold are not changed.
- Clinical baselines are developed using TCGA-only internal resampling.
- CGGA is used only for post-lock external reporting.
- The exploratory GIBD-score-plus-clinical-covariate model is contextual only and
  is not treated as the final classifier.

Expected project layout uses relative paths under Data/ and Data/Revision_Ablation/.
"""


import os
import re
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation", "Clinical_Covariate_Benchmark_MedianOS_K100"),

    # Labels and locked GIBD scores.
    "TCGA_LABELS": os.path.join("Data", "TCGA_survival_labels_with_os_event.csv"),
    "TCGA_GIBD_OOF": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_oof_predictions.csv"),
    "CLINICAL_UTILITY_SCORES": os.path.join(
        "Data", "Revision_Ablation", "Clinical_Utility_MedianOS_K100", "locked_gibd_clinical_utility_scores_v2.csv"
    ),

    # CGGA labels for fallback matching.
    "CGGA_LABELS": os.path.join("Data", "Revision_Ablation", "cgga_labels_tcga_median357_with_os_event.csv"),

    # Clinical covariate candidate sources.
    "TCGA_COVARIATE_CACHES": [
        os.path.join("Data", "Revision_Ablation", "tcga_clinical_covariates_cache.csv"),
        os.path.join("Data", "tcga_clinical_covariates_cache.csv"),
    ],
    "TCGA_CLINICAL_JSON": os.path.join("Data", "clinical.json"),

    "CGGA_COVARIATE_CACHES": [
        os.path.join("Data", "Revision_Ablation", "cgga_clinical_covariates_cache.csv"),
        os.path.join("Data", "cgga_clinical_covariates_cache.csv"),
    ],
    "CGGA_CLINICAL_FILES": [
        os.path.join("Data", "CGGA_Data", "CGGA.mRNAseq_693_clinical.20200506.txt"),
        os.path.join("Data", "CGGA_Data", "CGGA.mRNAseq_325_clinical.20200506.txt"),
    ],

    # Expected sample counts after matching.
    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,
    "EXPECTED_N_CGGA": 131,
    "EXPECTED_LOW_CGGA": 85,
    "EXPECTED_HIGH_CGGA": 46,

    # TCGA-only internal resampling for clinical baselines.
    "N_SPLITS": 5,
    "N_REPEATS": 4,
    "SEED": 42,

    # Final GIBD threshold from frozen metadata.
    "GIBD_LOCKED_THRESHOLD": 0.53,

    # Threshold selection for clinical logistic baselines only.
    # This is TCGA-only, not CGGA-informed.
    "BASELINE_THRESHOLD_RULE": "youden",

    # Plotting.
    "DPI": 300,
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
# 2. General helpers
# --------------------------------------------------

def normalize_id(x):
    """Normalize sample/patient IDs for matching."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace('"', '').replace("'", "")
    return s


def tcga_key(x):
    """TCGA patient-level matching key; first 12 chars is standard patient barcode."""
    s = normalize_id(x)
    if s.startswith("TCGA") and len(s) >= 12:
        return s[:12]
    return s


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def find_col(columns, candidates):
    lower_map = {str(c).lower().strip(): c for c in columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]

    # relaxed contains search
    for cand in candidates:
        key = cand.lower().strip()
        for c in columns:
            low = str(c).lower().strip()
            if key in low:
                return c
    return None


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def clean_sex_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ["male", "m", "1"]:
        return "Male"
    if s in ["female", "f", "0"]:
        return "Female"
    if s in ["unknown", "not reported", "na", "nan", "", "none"]:
        return np.nan
    return str(x).strip()


def age_to_years(series):
    vals = pd.to_numeric(series, errors="coerce")
    # If values look like days, convert to years.
    # TCGA age_at_diagnosis is often in days.
    median_val = vals.dropna().median() if vals.notna().any() else np.nan
    if pd.notna(median_val) and median_val > 150:
        vals = vals / 365.25
    return vals


def load_labels(path, id_candidates=("Patient_ID", "CGGA_ID")):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    df = pd.read_csv(path)

    id_col = None
    for c in id_candidates:
        if c in df.columns:
            id_col = c
            break

    if id_col is not None:
        df = df.set_index(id_col)
    elif "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")
    else:
        df = df.set_index(df.columns[0])

    df.index = df.index.astype(str).map(normalize_id)

    if "Risk_Label" not in df.columns:
        raise ValueError(f"Risk_Label missing from {path}")

    df["Risk_Label"] = pd.to_numeric(df["Risk_Label"], errors="coerce")
    df = df.dropna(subset=["Risk_Label"]).copy()
    df["Risk_Label"] = df["Risk_Label"].astype(int)

    return df


def assert_label_counts(df, label_name, expected_n, expected_low, expected_high):
    if len(df) != int(expected_n):
        raise ValueError(f"{label_name}: expected N={expected_n}, got {len(df)}")
    counts = df["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))
    if low != int(expected_low) or high != int(expected_high):
        raise ValueError(
            f"{label_name}: expected low={expected_low}, high={expected_high}; "
            f"got low={low}, high={high}"
        )


# --------------------------------------------------
# 3. Load locked GIBD score predictions
# --------------------------------------------------

def load_tcga_gibd_oof(labels):
    path = CONF["TCGA_GIBD_OOF"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"True TCGA OOF predictions not found: {path}\n"
            "Run the final median-OS K100 16U lock script first."
        )

    df = pd.read_csv(path)
    id_col = find_col(df.columns, ["Patient_ID", "patient_id", "sample_id", "ID"])
    score_col = find_col(df.columns, ["OOF_Probability", "oof_probability", "oof_prob", "OOF_Prob"])

    if id_col is None or score_col is None:
        raise ValueError("TCGA OOF file must contain Patient_ID and OOF_Probability.")

    df[id_col] = df[id_col].astype(str).map(normalize_id)

    if df[id_col].duplicated().any():
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.groupby(id_col, as_index=False)[score_col].mean()

    df = df.set_index(id_col)

    common = labels.index.intersection(df.index)
    if len(common) != len(labels):
        raise ValueError(
            f"TCGA OOF predictions do not cover all labels. "
            f"Expected {len(labels)}, matched {len(common)}."
        )

    out = pd.DataFrame({
        "sample_id": labels.index,
        "Risk_Label": labels["Risk_Label"].values,
        "gibd_probability": pd.to_numeric(df.loc[labels.index, score_col], errors="coerce").values,
    }).set_index("sample_id")

    if out["gibd_probability"].isna().any():
        raise ValueError("TCGA OOF GIBD probabilities contain NaNs.")

    return out


def load_cgga_gibd_scores():
    path = CONF["CLINICAL_UTILITY_SCORES"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Clinical utility scores not found: {path}\n"
            "Run 12_calibration_dca_locked_model_median_os_k100_FIXED.py first."
        )

    scores = pd.read_csv(path)
    if "cohort" not in scores.columns or "sample_id" not in scores.columns:
        raise ValueError("Clinical utility scores file must contain cohort and sample_id.")

    sub = scores[scores["cohort"].astype(str).eq("CGGA_external_locked")].copy()
    if sub.empty:
        raise ValueError("No CGGA_external_locked rows found in clinical utility scores file.")

    if "Risk_Label" not in sub.columns or "predicted_probability" not in sub.columns:
        raise ValueError("Clinical utility scores file must contain Risk_Label and predicted_probability.")

    sub["sample_id"] = sub["sample_id"].astype(str).map(normalize_id)
    out = sub[["sample_id", "Risk_Label", "predicted_probability"]].copy()
    out = out.rename(columns={"predicted_probability": "gibd_probability"})
    out["Risk_Label"] = pd.to_numeric(out["Risk_Label"], errors="coerce").astype(int)
    out["gibd_probability"] = pd.to_numeric(out["gibd_probability"], errors="coerce")
    out = out.dropna(subset=["gibd_probability"]).set_index("sample_id")

    return out


# --------------------------------------------------
# 4. Clinical covariate extraction
# --------------------------------------------------

def standardize_covariate_table(df, cohort, id_candidates=None):
    if id_candidates is None:
        id_candidates = ["Patient_ID", "CGGA_ID", "sample_id", "Sample_ID", "ID", "patient_id"]

    id_col = find_col(df.columns, id_candidates)
    age_col = find_col(df.columns, [
        "age",
        "Age",
        "age_at_diagnosis",
        "Age_at_diagnosis",
        "age_at_index",
        "Age at diagnosis",
        "Age_at_index",
    ])
    sex_col = find_col(df.columns, [
        "sex",
        "Sex",
        "gender",
        "Gender",
        "demographic_gender",
    ])

    if id_col is None:
        raise ValueError(f"Could not identify ID column for {cohort} covariates.")
    if age_col is None and sex_col is None:
        raise ValueError(f"Could not identify age or sex/gender columns for {cohort} covariates.")

    out = pd.DataFrame()
    out["sample_id"] = df[id_col].astype(str).map(normalize_id)

    if cohort.upper() == "TCGA":
        out["match_key"] = out["sample_id"].map(tcga_key)
    else:
        out["match_key"] = out["sample_id"]

    if age_col is not None:
        out["age_years"] = age_to_years(df[age_col])
    else:
        out["age_years"] = np.nan

    if sex_col is not None:
        out["sex"] = df[sex_col].map(clean_sex_value)
    else:
        out["sex"] = np.nan

    out = out.drop_duplicates(subset=["match_key"], keep="first").set_index("match_key")
    return out, {"source_id_col": id_col, "age_col": age_col, "sex_col": sex_col}


def load_tcga_covariates_from_cache():
    for path in CONF["TCGA_COVARIATE_CACHES"]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cov, meta = standardize_covariate_table(df, "TCGA", id_candidates=["Patient_ID", "patient_id", "sample_id", "ID"])
        meta["source"] = path
        return cov, meta
    return None, None


def load_tcga_covariates_from_json():
    path = CONF["TCGA_CLINICAL_JSON"]
    if not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []

    for case in raw:
        case_id = case.get("submitter_id") or case.get("case_submitter_id") or case.get("case_id")
        case_id = normalize_id(case_id)

        demographic = case.get("demographic", {}) or {}
        gender = demographic.get("gender")
        age_at_index = demographic.get("age_at_index")

        age_days = np.nan
        diagnoses = case.get("diagnoses", []) or []
        if isinstance(diagnoses, list) and len(diagnoses) > 0:
            d0 = diagnoses[0] or {}
            age_days = d0.get("age_at_diagnosis", np.nan)

        age_value = age_at_index
        if age_value is None or pd.isna(age_value):
            age_value = age_days

        rows.append({
            "Patient_ID": tcga_key(case_id),
            "age_raw": age_value,
            "gender": gender,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None

    cov, meta = standardize_covariate_table(
        df,
        "TCGA",
        id_candidates=["Patient_ID"],
    )
    meta["source"] = path
    return cov, meta


def load_tcga_covariates():
    cov, meta = load_tcga_covariates_from_cache()
    if cov is not None:
        return cov, meta

    cov, meta = load_tcga_covariates_from_json()
    if cov is not None:
        return cov, meta

    raise FileNotFoundError(
        "Could not load TCGA age/sex covariates. Provide one of:\n"
        "- Data/Revision_Ablation/tcga_clinical_covariates_cache.csv\n"
        "- Data/tcga_clinical_covariates_cache.csv\n"
        "- Data/clinical.json"
    )


def load_cgga_covariates_from_cache():
    for path in CONF["CGGA_COVARIATE_CACHES"]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cov, meta = standardize_covariate_table(
            df,
            "CGGA",
            id_candidates=["CGGA_ID", "Patient_ID", "sample_id", "Sample_ID", "ID"]
        )
        meta["source"] = path
        return cov, meta
    return None, None


def load_cgga_covariates_from_labels():
    path = CONF["CGGA_LABELS"]
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)
    try:
        cov, meta = standardize_covariate_table(
            df,
            "CGGA",
            id_candidates=["CGGA_ID", "Patient_ID", "sample_id", "Sample_ID", "ID"]
        )
        meta["source"] = path
        return cov, meta
    except Exception:
        return None, None


def read_table_flexible(path):
    # Try tab first; then comma.
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path)


def load_cgga_covariates_from_clinical_files():
    frames = []

    for path in CONF["CGGA_CLINICAL_FILES"]:
        if not os.path.exists(path):
            continue

        df = read_table_flexible(path)
        if df.empty:
            continue

        # Try to standardize each file separately. If it fails, skip.
        try:
            cov, meta = standardize_covariate_table(
                df,
                "CGGA",
                id_candidates=[
                    "CGGA_ID",
                    "Sample_ID",
                    "sample_id",
                    "RNAseq_ID",
                    "RNAseq",
                    "Patient_ID",
                    "ID",
                ],
            )
            cov["source_file"] = path
            frames.append((cov, meta))
        except Exception as exc:
            print(f"Could not parse CGGA clinical file {path}: {exc}")

    if not frames:
        return None, None

    cov = pd.concat([x[0] for x in frames], axis=0)
    cov = cov[~cov.index.duplicated(keep="first")].copy()

    meta = {
        "source": ";".join([x[1].get("source", "") or x[0]["source_file"].iloc[0] for x in frames]),
        "note": "combined CGGA clinical files",
    }
    return cov.drop(columns=["source_file"], errors="ignore"), meta


def load_cgga_covariates():
    cov, meta = load_cgga_covariates_from_cache()
    if cov is not None:
        return cov, meta

    cov, meta = load_cgga_covariates_from_labels()
    if cov is not None:
        return cov, meta

    cov, meta = load_cgga_covariates_from_clinical_files()
    if cov is not None:
        return cov, meta

    raise FileNotFoundError(
        "Could not load CGGA age/sex covariates. Provide one of:\n"
        "- Data/Revision_Ablation/cgga_clinical_covariates_cache.csv\n"
        "- Data/Revision_Ablation/cgga_labels_tcga_median357_with_os_event.csv with age/gender columns\n"
        "- CGGA clinical txt files under Data/CGGA_Data/"
    )


# --------------------------------------------------
# 5. Dataset assembly
# --------------------------------------------------

def assemble_dataset():
    # Labels and GIBD scores.
    tcga_labels = load_labels(CONF["TCGA_LABELS"], id_candidates=("Patient_ID",))
    assert_label_counts(
        tcga_labels,
        "TCGA median-OS labels",
        CONF["EXPECTED_N_TCGA"],
        CONF["EXPECTED_LOW_TCGA"],
        CONF["EXPECTED_HIGH_TCGA"],
    )
    tcga_gibd = load_tcga_gibd_oof(tcga_labels)

    cgga_gibd = load_cgga_gibd_scores()
    assert_label_counts(
        cgga_gibd,
        "CGGA357 clinical-utility score labels",
        CONF["EXPECTED_N_CGGA"],
        CONF["EXPECTED_LOW_CGGA"],
        CONF["EXPECTED_HIGH_CGGA"],
    )

    # Clinical covariates.
    tcga_cov, tcga_cov_meta = load_tcga_covariates()
    cgga_cov, cgga_cov_meta = load_cgga_covariates()

    # Match TCGA by first 12 chars.
    tcga_df = tcga_gibd.copy()
    tcga_df["match_key"] = tcga_df.index.map(tcga_key)
    tcga_df = tcga_df.join(tcga_cov[["age_years", "sex"]], on="match_key")
    tcga_df["cohort"] = "TCGA_OOF"
    tcga_df["sample_id"] = tcga_df.index

    # Match CGGA by exact ID.
    cgga_df = cgga_gibd.copy()
    cgga_df["match_key"] = cgga_df.index.map(normalize_id)
    cgga_df = cgga_df.join(cgga_cov[["age_years", "sex"]], on="match_key")
    cgga_df["cohort"] = "CGGA_external_locked"
    cgga_df["sample_id"] = cgga_df.index

    # Diagnostics.
    for name, df in [("TCGA", tcga_df), ("CGGA", cgga_df)]:
        print(f"{name} matched dataset: {df.shape}")
        print(f"- Risk labels: {df['Risk_Label'].value_counts().sort_index().to_dict()}")
        print(f"- age missing: {int(df['age_years'].isna().sum())}/{len(df)}")
        print(f"- sex missing: {int(df['sex'].isna().sum())}/{len(df)}")
        print(f"- sex distribution: {df['sex'].value_counts(dropna=False).to_dict()}")

    assert_label_counts(
        tcga_df,
        "assembled TCGA benchmark dataset",
        CONF["EXPECTED_N_TCGA"],
        CONF["EXPECTED_LOW_TCGA"],
        CONF["EXPECTED_HIGH_TCGA"],
    )
    assert_label_counts(
        cgga_df,
        "assembled CGGA357 benchmark dataset",
        CONF["EXPECTED_N_CGGA"],
        CONF["EXPECTED_LOW_CGGA"],
        CONF["EXPECTED_HIGH_CGGA"],
    )

    audit_rows = []
    for cohort, df, meta in [
        ("TCGA", tcga_df, tcga_cov_meta),
        ("CGGA", cgga_df, cgga_cov_meta),
    ]:
        audit_rows.append({
            "cohort": cohort,
            "n": int(len(df)),
            "age_missing_n": int(df["age_years"].isna().sum()),
            "sex_missing_n": int(df["sex"].isna().sum()),
            "covariate_source": meta.get("source", ""),
            "source_id_col": meta.get("source_id_col", ""),
            "age_col": meta.get("age_col", ""),
            "sex_col": meta.get("sex_col", ""),
        })

    data_audit = pd.DataFrame(audit_rows)
    data_audit_path = os.path.join(CONF["OUT_DIR"], "clinical_covariate_benchmark_data_audit.csv")
    data_audit.to_csv(data_audit_path, index=False)

    return tcga_df, cgga_df, data_audit, tcga_cov_meta, cgga_cov_meta


# --------------------------------------------------
# 6. Modeling and metrics
# --------------------------------------------------

def make_clinical_pipeline(feature_cols):
    numeric_cols = [c for c in feature_cols if c in ["age_years", "gibd_probability"]]
    categorical_cols = [c for c in feature_cols if c in ["sex"]]

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
            ]),
            categorical_cols,
        ))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        random_state=CONF["SEED"],
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", clf),
    ])

    return pipe


def repeated_cv_oof_probabilities(X, y, feature_cols):
    y = np.asarray(y).astype(int)
    probs_sum = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)

    cv = RepeatedStratifiedKFold(
        n_splits=CONF["N_SPLITS"],
        n_repeats=CONF["N_REPEATS"],
        random_state=CONF["SEED"],
    )

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X[feature_cols], y), start=1):
        model = make_clinical_pipeline(feature_cols)
        model.fit(X.iloc[train_idx][feature_cols], y[train_idx])
        p = model.predict_proba(X.iloc[test_idx][feature_cols])[:, 1]

        probs_sum[test_idx] += p
        counts[test_idx] += 1

    if np.any(counts == 0):
        raise RuntimeError("Some TCGA samples did not receive OOF predictions.")

    return probs_sum / counts


def choose_threshold_youden(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)

    thresholds = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
    best_t = 0.5
    best_score = -np.inf

    for t in thresholds:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        score = sens + spec - 1.0
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t


def compute_metrics(y, p, threshold):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pred = (p >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    prevalence = float(np.mean(y))
    null_brier = float(brier_score_loss(y, np.full_like(p, prevalence, dtype=float)))
    brier = float(brier_score_loss(y, p))

    return {
        "n": int(len(y)),
        "prevalence": prevalence,
        "auc": safe_auc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": brier,
        "null_brier_prevalence": null_brier,
        "brier_improvement_vs_null": float(null_brier - brier),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_benchmark(tcga_df, cgga_df):
    models = {
        # Final frozen model score only. No training.
        "GIBD_score_only_final": {
            "type": "score_only",
            "feature_cols": ["gibd_probability"],
            "threshold": CONF["GIBD_LOCKED_THRESHOLD"],
            "role": "final frozen GIBD-XGBoost score; unchanged",
        },

        # Clinical covariate baselines.
        "Age_only_logistic": {
            "type": "logistic",
            "feature_cols": ["age_years"],
            "threshold": None,
            "role": "supplementary age-only clinical baseline",
        },
        "Age_sex_logistic": {
            "type": "logistic",
            "feature_cols": ["age_years", "sex"],
            "threshold": None,
            "role": "supplementary age+sex clinical baseline",
        },

        # Exploratory contextual extension, NOT final model.
        "GIBD_score_age_sex_exploratory": {
            "type": "logistic",
            "feature_cols": ["gibd_probability", "age_years", "sex"],
            "threshold": None,
            "role": "exploratory contextual model; not final GIBD model",
        },
    }

    summary_rows = []
    pred_rows = []

    y_tcga = tcga_df["Risk_Label"].astype(int).values
    y_cgga = cgga_df["Risk_Label"].astype(int).values

    for model_name, spec in models.items():
        print("\n" + "=" * 80)
        print(f"Benchmark model: {model_name}")
        print("=" * 80)

        feature_cols = spec["feature_cols"]

        if spec["type"] == "score_only":
            tcga_prob = tcga_df["gibd_probability"].values.astype(float)
            cgga_prob = cgga_df["gibd_probability"].values.astype(float)
            threshold = float(spec["threshold"])
            threshold_source = "final locked GIBD threshold"

        else:
            # TCGA-only OOF for internal estimate.
            tcga_prob = repeated_cv_oof_probabilities(tcga_df, y_tcga, feature_cols)

            # Threshold selected on TCGA OOF only for clinical baselines.
            threshold = choose_threshold_youden(y_tcga, tcga_prob)
            threshold_source = f"TCGA OOF {CONF['BASELINE_THRESHOLD_RULE']} threshold"

            # Fit on all TCGA and evaluate once on CGGA.
            final_model = make_clinical_pipeline(feature_cols)
            final_model.fit(tcga_df[feature_cols], y_tcga)
            cgga_prob = final_model.predict_proba(cgga_df[feature_cols])[:, 1]

        # Store predictions.
        for sample_id, yval, p in zip(tcga_df["sample_id"], y_tcga, tcga_prob):
            pred_rows.append({
                "model": model_name,
                "cohort": "TCGA_OOF",
                "sample_id": sample_id,
                "Risk_Label": int(yval),
                "predicted_probability": float(p),
                "threshold": float(threshold),
                "threshold_source": threshold_source,
                "role": spec["role"],
            })

        for sample_id, yval, p in zip(cgga_df["sample_id"], y_cgga, cgga_prob):
            pred_rows.append({
                "model": model_name,
                "cohort": "CGGA_external_locked",
                "sample_id": sample_id,
                "Risk_Label": int(yval),
                "predicted_probability": float(p),
                "threshold": float(threshold),
                "threshold_source": threshold_source,
                "role": spec["role"],
            })

        # Metrics.
        for cohort_name, yvals, probs in [
            ("TCGA_OOF", y_tcga, tcga_prob),
            ("CGGA_external_locked", y_cgga, cgga_prob),
        ]:
            m = compute_metrics(yvals, probs, threshold)
            m.update({
                "model": model_name,
                "cohort": cohort_name,
                "features_used": ",".join(feature_cols),
                "model_role": spec["role"],
                "threshold_source": threshold_source,
            })
            summary_rows.append(m)

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Order columns.
    front_cols = [
        "model",
        "cohort",
        "model_role",
        "features_used",
        "n",
        "prevalence",
        "auc",
        "auprc",
        "brier",
        "null_brier_prevalence",
        "brier_improvement_vs_null",
        "threshold",
        "threshold_source",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    summary_df = summary_df[front_cols]

    return pred_df, summary_df


# --------------------------------------------------
# 7. Plotting
# --------------------------------------------------

def plot_auc_summary(summary_df):
    out_stem = os.path.join(CONF["OUT_DIR"], "Figure_Clinical_Covariate_Benchmark_AUC")

    # Plot CGGA first because it is the external result.
    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["model"] + " | " + plot_df["cohort"]
    plot_df = plot_df.sort_values(["cohort", "auc"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(plot_df["label"], plot_df["auc"])
    ax.axvline(0.5, linestyle="--", linewidth=1.0)
    ax.set_xlabel("AUC")
    ax.set_ylabel("")
    ax.set_title("Supplementary clinical-covariate benchmark")
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    for ext in [".png", ".pdf", ".tiff"]:
        path = out_stem + ext
        if ext == ".tiff":
            fig.savefig(path, dpi=CONF["DPI"], pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=CONF["DPI"])

    plt.close(fig)


# --------------------------------------------------
# 8. Main
# --------------------------------------------------

def main():
    print("=" * 80)
    print("Supplementary clinical covariate benchmark: Median-OS K100 branch")
    print("=" * 80)
    print("Final GIBD-XGBoost K100 model is NOT retrained or changed.")
    print("Expected labels: TCGA N=147 low=73 high=74; CGGA357 N=131 low=85 high=46.")
    print("Required GIBD score source: Clinical_Utility_MedianOS_K100 outputs.")
    print("Age/sex models are supplementary baselines only.")
    print("=" * 80)

    tcga_df, cgga_df, data_audit, tcga_cov_meta, cgga_cov_meta = assemble_dataset()

    pred_df, summary_df = run_benchmark(tcga_df, cgga_df)

    pred_path = os.path.join(CONF["OUT_DIR"], "clinical_covariate_benchmark_predictions.csv")
    summary_path = os.path.join(CONF["OUT_DIR"], "clinical_covariate_benchmark_summary.csv")
    audit_path = os.path.join(CONF["OUT_DIR"], "clinical_covariate_benchmark_audit.json")

    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_auc_summary(summary_df)

    audit = {
        "script": "12b_clinical_covariate_benchmark_median_os_k100.py",
        "purpose": "Supplementary age/sex clinical covariate benchmark against the final frozen GIBD score.",
        "final_gibd_model_changed": False,
        "final_gibd_score_source": {
            "TCGA": CONF["TCGA_GIBD_OOF"],
            "CGGA": CONF["CLINICAL_UTILITY_SCORES"],
        },
        "clinical_covariate_sources": {
            "TCGA": tcga_cov_meta,
            "CGGA": cgga_cov_meta,
        },
        "methodology": {
            "gibd_score_only": "Uses final GIBD probabilities and locked threshold; no training.",
            "age_only_and_age_sex": "Logistic baselines trained with TCGA-only repeated OOF resampling; CGGA evaluated once.",
            "gibd_age_sex": "Exploratory contextual logistic model; not the final model.",
            "cgga_use": "CGGA not used for threshold selection, tuning, covariate selection, or model selection.",
        },
        "outputs": {
            "predictions": pred_path,
            "summary": summary_path,
            "data_audit": os.path.join(CONF["OUT_DIR"], "clinical_covariate_benchmark_data_audit.csv"),
            "figure_auc": os.path.join(CONF["OUT_DIR"], "Figure_Clinical_Covariate_Benchmark_AUC.png"),
        },
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    show_cols = [
        "model",
        "cohort",
        "auc",
        "auprc",
        "brier",
        "brier_improvement_vs_null",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "threshold",
        "threshold_source",
    ]
    print(summary_df[show_cols].to_string(index=False))

    print("\nSaved outputs:")
    print(f"- Predictions: {pred_path}")
    print(f"- Summary: {summary_path}")
    print(f"- Data audit: {os.path.join(CONF['OUT_DIR'], 'clinical_covariate_benchmark_data_audit.csv')}")
    print(f"- Audit: {audit_path}")
    print(f"- Figure: {os.path.join(CONF['OUT_DIR'], 'Figure_Clinical_Covariate_Benchmark_AUC.png')}")

    print("\nSafe manuscript interpretation:")
    print(
        "A supplementary age/sex benchmark was used to contextualize the final "
        "transcriptomics-only GIBD score. The final GIBD-XGBoost model remained "
        "unchanged; GIBD+age+sex was treated only as an exploratory contextual "
        "analysis and not as the final model."
    )


if __name__ == "__main__":
    main()
