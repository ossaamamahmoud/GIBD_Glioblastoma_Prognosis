"""
GIBD analysis workflow — ablation and comparator evaluation

This script runs the final ablation/comparator analysis for the locked median-OS
K100 GIBD model. It loads the frozen GIBD-XGBoost K100 champion artifacts,
selects non-champion comparator configurations using TCGA-only out-of-fold
performance, and then evaluates the locked comparator set on the post-lock CGGA
external-validation cohort.

Analysis guardrails:
- The final GIBD-XGBoost K100 champion is loaded from locked artifacts and is
  not recomputed inside this ablation script.
- Non-champion comparator selection uses TCGA-only out-of-fold information.
- CGGA is used only after comparator choices and operating thresholds have been
  fixed using TCGA-only information.
- This script reports ablation/comparator evidence and does not modify the
  locked champion model, locked feature set, or locked threshold.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import math
import re
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored

    SKSURV_AVAILABLE = True
except Exception:
    CoxnetSurvivalAnalysis = None
    Surv = None
    concordance_index_censored = None
    SKSURV_AVAILABLE = False

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RUN_NAME = "revision_ablation_master_v3_median_os_k100_g035_final"
CPU_COUNT = os.cpu_count() or 4

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation"),

    "TCGA_WEIGHTED_GRAPH": os.path.join("Data", "Revision_Ablation", "tcga_weighted_self_graph_cache.csv"),
    "TCGA_GRAPH_FALLBACK": os.path.join("Data", "TCGA_survival_expression_matrix_enhanced.csv"),

    "TCGA_RAW_EXPR": os.path.join("Data", "TCGA_survival_expression_matrix_protein_coding.csv"),

    # Keep this file for Cox because it contains OS_days and Event.
    "TCGA_SURVIVAL_LABELS": os.path.join("Data", "TCGA_survival_labels_with_os_event.csv"),

    "CGGA_WEIGHTED_GRAPH": os.path.join("Data", "Revision_Ablation", "cgga_weighted_self_graph_cache.csv"),
    "CGGA_GRAPH_FALLBACK": os.path.join("Data", "Revision_Ablation", "cgga_graph_informed_cache.csv"),

    "CGGA_LABELS": os.path.join(
        "Data",
        "Revision_Ablation",
        "cgga_labels_tcga_median357_with_os_event.csv",
    ),
    "CGGA_SURVIVAL_LABELS_OPTIONAL": os.path.join(
        "Data",
        "Revision_Ablation",
        "cgga_labels_tcga_median357_with_os_event.csv",
    ),

    "CHAMPION_LOCKED_JOBLIB": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_Model.joblib"),
    "CHAMPION_LOCKED_METADATA": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_Model_metadata.json"),
    "CHAMPION_LOCKED_FEATURES": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_Features.csv"),

    "SEED": 42,
    "N_SPLITS": 5,
    "N_REPEATS": 4,
    "MODEL_THREADS": 1,
    "TREE_METHOD": "hist",

    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,

    "EXPECTED_N_CGGA": 131,
    "EXPECTED_LOW_CGGA": 85,
    "EXPECTED_HIGH_CGGA": 46,

    "K_CHAMPION": 100,
    "K_COMPLEXITY": 120,
    "K_PARSIMONY_LOW": 80,
    "INCLUDE_K80_SENSITIVITY": False,

    "STABILITY_WEIGHT": 0.34,
    "GRAPH_BONUS": 0.035,
    "RF_ESTIMATORS_FEATURE_SELECTION": 375,
    "RF_SELECTION_REPEATS": 3,
    "RF_TOP_POOL_MULTIPLIER": 4,

    # Match final locked champion threshold policy.
    "THRESHOLD_MODE": "recall80_spec25",
    "THRESHOLD_MIN": 0.03,
    "THRESHOLD_MAX": 0.97,
    "N_THRESHOLDS": 189,

    # Final champion XGBoost parameters reused for the K120 complexity-control comparator.
    "CHAMPION_XGB_PARAMS": {
        "max_depth": 3,
        "learning_rate": 0.029,
        "n_estimators": 85,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "min_child_weight": 1.8,
        "gamma": 0.18,
        "reg_alpha": 0.30,
        "reg_lambda": 6.5,
        "scale_pos_weight": 2.40,
    },

    "COX_L1_RATIO_CANDIDATES": [1.0, 0.95, 0.90],
    "COX_ALPHA_MIN_RATIO_CANDIDATES": [0.10, 0.20, 0.30, 0.50],
    "COX_N_ALPHAS": 20,
    "COX_MAX_ITER": 200000,
    "COX_TOL": 1e-6,
    "COX_FINAL_ALPHA_MULTIPLIERS": [1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    "COX_INNER_SPLITS": 5,
}


os.makedirs(CONF["OUT_DIR"], exist_ok=True)

# Fair-fight bounded tuning grids.
RAW_XGB_GRID = []
for depth in [2, 3, 4]:
    for lr in [0.025, 0.035, 0.050]:
        for reg_lambda in [5.0, 8.0]:
            RAW_XGB_GRID.append({
                "max_depth": depth,
                "learning_rate": lr,
                "n_estimators": 105,
                "subsample": 0.70,
                "colsample_bytree": 0.65,
                "min_child_weight": 1.0,
                "gamma": 0.16,
                "reg_alpha": 0.50,
                "reg_lambda": reg_lambda,
                "scale_pos_weight": 2.2,
            })

RF_GRID = []
for n_estimators in [400, 600, 800]:
    for max_depth in [3, 4, None]:
        RF_GRID.append({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
        })

COX_CANDIDATES = [{"model_label": "lasso_cox_stable"}]


# --------------------------------------------------
# 2. Utility functions
# --------------------------------------------------

def to_builtin(obj):
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): to_builtin(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.DataFrame):
        return [{str(k): to_builtin(v) for k, v in row.items()} for row in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if hasattr(obj, "item"):
        try:
            return to_builtin(obj.item())
        except Exception:
            pass
    return str(obj)


def safe_json_dumps(obj):
    return json.dumps(to_builtin(obj), sort_keys=True)


def clean_id(text):
    text = str(text)
    for bad in [" ", "/", "\\", ":", ";", ",", "=", ".", "+", "(", ")", "[", "]", "{", "}", "|", "<", ">"]:
        text = text.replace(bad, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_feature_name(name):
    text = str(name)
    text = re.sub(r"^(ENSG\d+)\.\d+(_.*)?$", r"\1\2", text)
    return text


def is_graph_feature(feature_name):
    name = str(feature_name).lower()
    tokens = [
        "_wppi_self", "wppi_self",
        "_wppi_neighbor", "wppi_neighbor",
        "_nbr_mean", "nbr_mean",
        "neighbor_mean", "network_mean", "graph_mean",
    ]
    return any(token in name for token in tokens)


def graph_ratio(features):
    features = list(features)
    if not features:
        return 0.0
    return float(sum(is_graph_feature(f) for f in features) / len(features))


def raw_expression_columns(columns):
    return [c for c in columns if not is_graph_feature(c)]


def assert_tcga_only_sort_keys(sort_keys, context):
    keys = [sort_keys] if isinstance(sort_keys, str) else list(sort_keys)
    bad = []
    for key in keys:
        k = str(key).lower()
        tokens = re.split(r"[^a-z0-9]+", k)
        forbidden = (
                "cgga" in tokens or "external" in tokens or "monitor" in tokens
                or k.startswith("old_") or k.startswith("monitor_")
                or "oldcgga" in k or "old_cgga" in k
                or "new_external" in k or "retrospective" in k
                or "diagnostic" in k
                or k.startswith("gap_") or k.endswith("_gap")
                or "gap_compression" in k
        )
        if forbidden:
            bad.append(str(key))
    if bad:
        raise RuntimeError(f"Non-TCGA diagnostic key used in TCGA-only selection at {context}: {bad}")


# --------------------------------------------------
# 3. Loading functions
# --------------------------------------------------

def choose_matrix_path(preferred, fallback, label):
    if os.path.exists(preferred):
        print(f"Using {label} matrix: {preferred}")
        return preferred
    print(f"Preferred {label} matrix missing. Fallback: {fallback}")
    return fallback


def load_feature_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found: {path}")
    X = pd.read_csv(path, index_col=0)
    X.index = X.index.astype(str)
    X.columns = [normalize_feature_name(c) for c in X.columns.astype(str)]
    X = X.loc[:, ~pd.Index(X.columns).duplicated()].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def load_tcga_survival_labels():
    path = CONF["TCGA_SURVIVAL_LABELS"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required TCGA survival labels: {path}\n"
            "Run the median-OS TCGA survival-label generation script first."
        )

    labels = pd.read_csv(path)
    if "Patient_ID" in labels.columns:
        labels = labels.set_index("Patient_ID")
    else:
        labels = labels.set_index(labels.columns[0])

    labels.index = labels.index.astype(str)
    required = ["OS_days", "Event", "Risk_Label"]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"TCGA survival label file missing columns: {missing}")

    labels["OS_days"] = pd.to_numeric(labels["OS_days"], errors="coerce")
    labels["Event"] = pd.to_numeric(labels["Event"], errors="coerce")
    labels["Risk_Label"] = pd.to_numeric(labels["Risk_Label"], errors="coerce")
    labels = labels.dropna(subset=["OS_days", "Event", "Risk_Label"])
    labels = labels[labels["OS_days"] > 0].copy()
    labels["Event"] = labels["Event"].astype(int)
    labels["Risk_Label"] = labels["Risk_Label"].astype(int)

    if len(labels) != CONF["EXPECTED_N_TCGA"]:
        raise ValueError(f"TCGA survival labels expected N={CONF['EXPECTED_N_TCGA']}, got {len(labels)}")

    counts = labels["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))
    if low != CONF["EXPECTED_LOW_TCGA"] or high != CONF["EXPECTED_HIGH_TCGA"]:
        raise ValueError(f"TCGA Risk_Label mismatch: expected low=76 high=71; got low={low} high={high}")

    print(f"Loaded TCGA survival labels: {path}")
    print("- Event distribution:", labels["Event"].value_counts().sort_index().to_dict())
    print("- Risk_Label distribution:", labels["Risk_Label"].value_counts().sort_index().to_dict())
    return labels


def load_generic_labels(path, expected_n, expected_low, expected_high):
    labels = pd.read_csv(path)
    if "Patient_ID" in labels.columns:
        labels = labels.set_index("Patient_ID")
    elif "CGGA_ID" in labels.columns:
        labels = labels.set_index("CGGA_ID")
    elif "Unnamed: 0" in labels.columns:
        labels = labels.set_index("Unnamed: 0")
    else:
        labels = labels.set_index(labels.columns[0])

    labels.index = labels.index.astype(str)
    if "Risk_Label" not in labels.columns:
        raise ValueError(f"Risk_Label missing in: {path}")
    labels["Risk_Label"] = labels["Risk_Label"].astype(int)

    # Normalize optional survival names.
    if "OS_days" not in labels.columns:
        for c in ["OS", "OS_days", "OS_months", "OS.months", "OS_time", "survival_time", "survival_days", "time",
                  "days", "months"]:
            if c in labels.columns:
                if "month" in str(c).lower():
                    labels["OS_days"] = pd.to_numeric(labels[c], errors="coerce") * 30.4375
                else:
                    labels["OS_days"] = pd.to_numeric(labels[c], errors="coerce")
                break
    if "OS_days" not in labels.columns:
        labels["OS_days"] = np.nan

    if "Event" not in labels.columns:
        event_col = None
        for c in ["OS_event", "Event", "event", "Status", "status", "vital_status", "Censor", "censor", "censored",
                  "death_event", "Dead", "dead"]:
            if c in labels.columns:
                event_col = c
                break
        if event_col is None:
            labels["Event"] = np.nan
        else:
            raw = labels[event_col]
            if pd.api.types.is_numeric_dtype(raw):
                vals = pd.to_numeric(raw, errors="coerce")
                if "censor" in str(event_col).lower():
                    labels["Event"] = 1.0 - vals
                else:
                    labels["Event"] = vals
            else:
                txt = raw.astype(str).str.lower().str.strip()
                labels["Event"] = np.where(txt.isin(["dead", "deceased", "death", "event", "1", "true", "yes"]), 1,
                                           np.where(txt.isin(["alive", "living", "censored", "0", "false", "no"]), 0,
                                                    np.nan))

    labels["OS_days"] = pd.to_numeric(labels["OS_days"], errors="coerce")
    labels["Event"] = pd.to_numeric(labels["Event"], errors="coerce")

    if len(labels) != int(expected_n):
        raise ValueError(f"{path}: expected N={expected_n}, got {len(labels)}")

    counts = labels["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))
    if low != expected_low or high != expected_high:
        raise ValueError(f"{path}: expected low={expected_low}, high={expected_high}; got low={low}, high={high}")

    return labels


def load_tcga_phase1():
    print("\n" + "=" * 80)
    print("Phase 1: TCGA-only loading")
    print("=" * 80)

    tcga_path = choose_matrix_path(CONF["TCGA_WEIGHTED_GRAPH"], CONF["TCGA_GRAPH_FALLBACK"], "TCGA GIBD/WPPI")
    X_tcga = load_feature_matrix(tcga_path)
    y_tcga = load_tcga_survival_labels()

    common = X_tcga.index.intersection(y_tcga.index)
    X_tcga = X_tcga.loc[common].copy()
    y_tcga = y_tcga.loc[common].copy()

    if len(X_tcga) != CONF["EXPECTED_N_TCGA"]:
        raise ValueError(f"TCGA intersection expected {CONF['EXPECTED_N_TCGA']}, got {len(X_tcga)}")

    print("TCGA matrix:", X_tcga.shape)
    print("TCGA graph/WPPI features:", sum(is_graph_feature(c) for c in X_tcga.columns))
    print("TCGA raw-expression features:", sum(not is_graph_feature(c) for c in X_tcga.columns))
    return X_tcga, y_tcga, tcga_path


def load_cgga_phase2():
    print("\n" + "=" * 80)
    print("Phase 2: CGGA loading after all TCGA winners are locked")
    print("=" * 80)

    cgga_path = choose_matrix_path(CONF["CGGA_WEIGHTED_GRAPH"], CONF["CGGA_GRAPH_FALLBACK"], "CGGA GIBD/WPPI")
    X_cgga = load_feature_matrix(cgga_path)

    label_path = CONF["CGGA_SURVIVAL_LABELS_OPTIONAL"] if os.path.exists(CONF["CGGA_SURVIVAL_LABELS_OPTIONAL"]) else \
    CONF["CGGA_LABELS"]
    y_cgga = load_generic_labels(label_path, CONF["EXPECTED_N_CGGA"], CONF["EXPECTED_LOW_CGGA"],
                                 CONF["EXPECTED_HIGH_CGGA"])

    common = X_cgga.index.intersection(y_cgga.index)
    X_cgga = X_cgga.loc[common].copy()
    y_cgga = y_cgga.loc[common].copy()

    if len(X_cgga) != CONF["EXPECTED_N_CGGA"]:
        raise ValueError(f"CGGA intersection expected {CONF['EXPECTED_N_CGGA']}, got {len(X_cgga)}")

    print("CGGA matrix:", X_cgga.shape)
    print("CGGA labels:", y_cgga.shape)
    print("CGGA label file:", label_path)
    print("CGGA has OS/Event:", bool(y_cgga["OS_days"].notna().any() and y_cgga["Event"].notna().any()))
    return X_cgga, y_cgga, cgga_path, label_path


# --------------------------------------------------
# 4. Locked champion loading
# --------------------------------------------------

def load_locked_champion_artifacts():
    for key in ["CHAMPION_LOCKED_JOBLIB", "CHAMPION_LOCKED_METADATA", "CHAMPION_LOCKED_FEATURES"]:
        if not os.path.exists(CONF[key]):
            raise FileNotFoundError(f"Missing frozen champion artifact: {CONF[key]}")

    bundle = load(CONF["CHAMPION_LOCKED_JOBLIB"])
    with open(CONF["CHAMPION_LOCKED_METADATA"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feat_df = pd.read_csv(CONF["CHAMPION_LOCKED_FEATURES"])
    if "feature" not in feat_df.columns:
        raise ValueError("16U_Final_Locked_Features.csv must contain a 'feature' column")
    features = [normalize_feature_name(f) for f in feat_df["feature"].astype(str).tolist()]

    threshold = metadata.get("locked_threshold", metadata.get("ensemble_threshold", None))
    if threshold is None:
        threshold = bundle.get("locked_threshold", bundle.get("ensemble_threshold", None))
    if threshold is None:
        raise ValueError("Could not find locked threshold in 16U metadata/joblib")

    params = metadata.get("params", bundle.get("params", CONF["CHAMPION_XGB_PARAMS"]))
    oof = metadata.get("winner_tcga_oof_summary", {})

    print("\n" + "=" * 80)
    print("Locked Champion Mode")
    print("=" * 80)
    print("features:", len(features))
    print("graph ratio:", round(graph_ratio(features), 4))
    print("locked threshold:", threshold)
    print("params:", params)

    return {"bundle": bundle, "metadata": metadata, "features": features, "threshold": float(threshold),
            "params": params, "tcga_oof_summary": oof}


def predict_locked_champion(locked, X):
    bundle = locked["bundle"]
    features = locked["features"]
    X_use = X.reindex(columns=features, fill_value=0.0)
    scaler = bundle.get("scaler", None)
    X_s = scaler.transform(X_use) if scaler is not None else X_use.values

    if "models" in bundle and bundle["models"] is not None:
        probs = [m.predict_proba(X_s)[:, 1] for m in bundle["models"]]
        return np.mean(np.column_stack(probs), axis=1)
    if "model" in bundle and bundle["model"] is not None:
        return model_score_from_fitted(bundle["model"], X_s)
    raise ValueError("Locked champion bundle has no model(s)")


# --------------------------------------------------
# 5. CV and feature selection
# --------------------------------------------------

def build_cv_splits(X, y):
    cv = RepeatedStratifiedKFold(n_splits=CONF["N_SPLITS"], n_repeats=CONF["N_REPEATS"], random_state=CONF["SEED"])
    return [{"fold_id": i, "train_idx": tr, "test_idx": te} for i, (tr, te) in enumerate(cv.split(X, y), start=1)]


def rf_rank_features(X_train, y_train, candidate_columns, k, seed, graph_bonus_enabled):
    candidate_columns = list(candidate_columns)
    top_pool = min(len(candidate_columns), max(int(k), int(k * CONF["RF_TOP_POOL_MULTIPLIER"])))
    X_sub = X_train[candidate_columns]

    importance_sum = pd.Series(0.0, index=candidate_columns)
    stability_count = pd.Series(0.0, index=candidate_columns)

    for r in range(CONF["RF_SELECTION_REPEATS"]):
        rf = RandomForestClassifier(
            n_estimators=CONF["RF_ESTIMATORS_FEATURE_SELECTION"],
            random_state=int(seed) + 1009 * r,
            n_jobs=CONF["MODEL_THREADS"],
            max_features="sqrt",
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            bootstrap=True,
        )
        rf.fit(X_sub, y_train.values.astype(int))
        imp = pd.Series(rf.feature_importances_, index=candidate_columns)
        importance_sum += imp
        stability_count.loc[imp.nlargest(top_pool).index] += 1.0

    importance_mean = importance_sum / float(CONF["RF_SELECTION_REPEATS"])
    stability_rate = stability_count / float(CONF["RF_SELECTION_REPEATS"])
    max_imp = float(importance_mean.max())
    importance_scaled = importance_mean / max_imp if max_imp > 0 else importance_mean.copy()
    graph_flags = pd.Series([is_graph_feature(f) for f in importance_scaled.index], index=importance_scaled.index)
    graph_bonus = graph_flags.astype(float) * float(CONF["GRAPH_BONUS"]) if graph_bonus_enabled else graph_flags.astype(
        float) * 0.0

    final_score = (1.0 - CONF["STABILITY_WEIGHT"]) * importance_scaled + CONF[
        "STABILITY_WEIGHT"] * stability_rate + graph_bonus
    ranked = final_score.sort_values(ascending=False).index.tolist()
    diag = pd.DataFrame({
        "feature": final_score.index,
        "importance_mean": importance_mean.values,
        "importance_scaled": importance_scaled.values,
        "stability_rate": stability_rate.values,
        "is_graph_feature": graph_flags.values,
        "graph_bonus": graph_bonus.values,
        "final_score": final_score.values,
    }).sort_values("final_score", ascending=False)
    return ranked[: int(k)], diag


def compute_feature_sets_tcga_only(X_tcga, y_tcga, representation, k, cv_splits):
    y = y_tcga["Risk_Label"].astype(int)
    if representation == "gibd":
        candidate_columns = list(X_tcga.columns)
        graph_bonus_enabled = True
    elif representation == "raw":
        candidate_columns = raw_expression_columns(X_tcga.columns)
        graph_bonus_enabled = False
    else:
        raise ValueError(f"Unknown representation: {representation}")

    print("\n" + "=" * 80)
    print(f"TCGA-only fold-contained RF-Gini feature selection | {representation} | K={k}")
    print("=" * 80)

    feature_sets = {"cv": {}, "all_tcga": None}
    rows = []
    all_diag = []

    for split in cv_splits:
        fold_id = int(split["fold_id"])
        feats, diag = rf_rank_features(
            X_train=X_tcga.iloc[split["train_idx"]],
            y_train=y.iloc[split["train_idx"]],
            candidate_columns=candidate_columns,
            k=int(k),
            seed=CONF["SEED"] + 10000 + 97 * fold_id,
            graph_bonus_enabled=graph_bonus_enabled,
        )
        feature_sets["cv"][fold_id] = feats
        rows.append({
            "representation": representation,
            "scope": "cv_fold",
            "fold_id": fold_id,
            "k": int(k),
            "n_features": len(feats),
            "n_graph_features": int(sum(is_graph_feature(f) for f in feats)),
            "graph_ratio": graph_ratio(feats),
            "selected_features": ";".join(feats),
        })
        print(f"Fold {fold_id:02d}: graph={sum(is_graph_feature(f) for f in feats)}/{k} ({graph_ratio(feats):.3f})")

    feats_all, diag_all = rf_rank_features(
        X_train=X_tcga,
        y_train=y,
        candidate_columns=candidate_columns,
        k=int(k),
        seed=CONF["SEED"] + 50000,
        graph_bonus_enabled=graph_bonus_enabled,
    )
    feature_sets["all_tcga"] = feats_all
    all_diag.append(diag_all.head(int(k) * 4))
    rows.append({
        "representation": representation,
        "scope": "all_tcga",
        "fold_id": 0,
        "k": int(k),
        "n_features": len(feats_all),
        "n_graph_features": int(sum(is_graph_feature(f) for f in feats_all)),
        "graph_ratio": graph_ratio(feats_all),
        "selected_features": ";".join(feats_all),
    })

    pd.DataFrame(rows).to_csv(os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_features_{representation}_k{k}.csv"),
                              index=False)
    pd.concat(all_diag, axis=0, ignore_index=True).to_csv(
        os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_feature_ranking_{representation}_k{k}.csv"), index=False)
    return feature_sets


# --------------------------------------------------
# 6. Metrics and thresholding
# --------------------------------------------------

def threshold_grid():
    return np.linspace(CONF["THRESHOLD_MIN"], CONF["THRESHOLD_MAX"], CONF["N_THRESHOLDS"])


def threshold_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    bal = 0.5 * (sens + spec)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f2 = 5.0 * precision * sens / ((4.0 * precision) + sens) if ((4.0 * precision) + sens) > 0 else 0.0
    return sens, spec, bal, precision, f2, tn, fp, fn, tp


def select_threshold_from_tcga_oof(y_true, scores, mode):
    rows = []
    for t in threshold_grid():
        sens, spec, bal, precision, f2, tn, fp, fn, tp = threshold_metrics(y_true, scores, t)
        imbalance = abs(sens - spec)
        if mode == "recall70_spec30":
            if sens >= 0.70 and spec >= 0.30:
                objective = 3.0 + 0.50 * sens + 0.25 * bal + 0.15 * spec + 0.10 * f2 - 0.03 * imbalance
            else:
                objective = 0.52 * sens + 0.20 * bal + 0.12 * spec + 0.16 * f2 - 1.25 * max(0.0,
                                                                                            0.70 - sens) - 0.50 * max(
                    0.0, 0.30 - spec) - 0.03 * imbalance
        elif mode == "youden":
            objective = sens + spec - 1.0 - 0.02 * imbalance

        elif mode == "recall80_spec25":
            if sens >= 0.80 and spec >= 0.25:
                objective = (
                    3.0
                    + 0.50 * sens
                    + 0.24 * bal
                    + 0.18 * spec
                    + 0.08 * f2
                    - 0.03 * imbalance
                )
            else:
                objective = (
                    0.52 * sens
                    + 0.20 * bal
                    + 0.14 * spec
                    + 0.14 * f2
                    - 1.10 * max(0.0, 0.80 - sens)
                    - 0.45 * max(0.0, 0.25 - spec)
                    - 0.03 * imbalance
                )
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")
        rows.append((objective, sens, bal, spec, -abs(float(t) - 0.5), float(t)))
    return float(max(rows, key=lambda x: x)[-1])


def compute_binary_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan
    auprc = float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan
    brier = float(brier_score_loss(y_true, scores)) if np.all((scores >= 0) & (scores <= 1)) else np.nan
    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "min_sens_spec": float(min(sens, spec)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": brier,
        "threshold": float(threshold),
        "n_total": int(len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def safe_auc(y_binary, risk_scores):
    y_binary = np.asarray(y_binary).astype(int)
    risk_scores = np.asarray(risk_scores).astype(float)
    if len(np.unique(y_binary)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_binary, risk_scores))
    except Exception:
        return np.nan


def safe_cindex(event, time, risk_scores):
    if not SKSURV_AVAILABLE:
        return np.nan
    event = np.asarray(event)
    time = np.asarray(time).astype(float)
    risk_scores = np.asarray(risk_scores).astype(float)
    valid = np.isfinite(time) & np.isfinite(risk_scores) & pd.notna(event)
    event = event[valid].astype(bool)
    time = time[valid].astype(float)
    risk_scores = risk_scores[valid].astype(float)
    if len(time) < 10 or event.sum() < 3:
        return np.nan
    try:
        return float(concordance_index_censored(event, time, risk_scores)[0])
    except Exception:
        return np.nan


# --------------------------------------------------
# 7. Model factories
# --------------------------------------------------

def make_xgb(params, seed):
    p = dict(params)
    for key in ["random_state", "n_jobs", "objective", "eval_metric", "tree_method", "verbosity", "use_label_encoder"]:
        p.pop(key, None)
    p.update({"random_state": int(seed), "objective": "binary:logistic", "eval_metric": "logloss",
              "n_jobs": CONF["MODEL_THREADS"], "tree_method": CONF["TREE_METHOD"], "verbosity": 0})
    return XGBClassifier(**p)


def make_rf(params, seed):
    p = dict(params)
    p.update({"random_state": int(seed), "n_jobs": CONF["MODEL_THREADS"]})
    return RandomForestClassifier(**p)


def model_score_from_fitted(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Model has neither predict_proba nor decision_function")


# --------------------------------------------------
# 8. Binary model evaluation
# --------------------------------------------------

def evaluate_binary_candidate(candidate, X_tcga, y_tcga, cv_splits, feature_sets):
    y = y_tcga["Risk_Label"].astype(int).values
    n = len(y)
    oof_sum = np.zeros(n, dtype=float)
    oof_count = np.zeros(n, dtype=float)
    fold_rows = []

    try:
        for split in cv_splits:
            fold_id = int(split["fold_id"])
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            feats = feature_sets["cv"][fold_id]
            X_train = X_tcga.iloc[train_idx].reindex(columns=feats, fill_value=0.0)
            X_test = X_tcga.iloc[test_idx].reindex(columns=feats, fill_value=0.0)
            y_train = y[train_idx]
            y_test = y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if candidate["model_family"] == "xgb":
                model = make_xgb(candidate["params"], CONF["SEED"] + 1000 + fold_id)
            elif candidate["model_family"] == "rf":
                model = make_rf(candidate["params"], CONF["SEED"] + 1000 + fold_id)
            else:
                raise ValueError(f"Unsupported model: {candidate['model_family']}")

            model.fit(X_train_s, y_train)
            scores = model_score_from_fitted(model, X_test_s)
            oof_sum[test_idx] += scores
            oof_count[test_idx] += 1.0
            fold_rows.append({
                "experiment_id": candidate["experiment_id"],
                "candidate_id": candidate["candidate_id"],
                "fold_id": fold_id,
                "tcga_fold_auc": float(roc_auc_score(y_test, scores)),
                "n_valid": int(len(test_idx)),
                "valid_pos": int(y_test.sum()),
                "valid_neg": int((1 - y_test).sum()),
            })

        if np.any(oof_count == 0):
            raise RuntimeError(f"OOF missing predictions for {candidate['candidate_id']}")
        oof_scores = oof_sum / oof_count
        threshold = select_threshold_from_tcga_oof(y, oof_scores, CONF["THRESHOLD_MODE"])
        metrics = compute_binary_metrics(y, oof_scores, threshold)
        fold_df = pd.DataFrame(fold_rows)
        fold_auc_sd = float(fold_df["tcga_fold_auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0
        fold_auc_mean = float(fold_df["tcga_fold_auc"].mean())
        all_feats = feature_sets["all_tcga"]

        return {
            "status": "ok", "error": "",
            "experiment_id": candidate["experiment_id"],
            "tier": candidate["tier"],
            "comparison_role": candidate["comparison_role"],
            "candidate_id": candidate["candidate_id"],
            "model_family": candidate["model_family"],
            "representation": candidate["representation"],
            "k": int(candidate["k"]),
            "params_json": safe_json_dumps(candidate["params"]),
            "threshold_mode": CONF["THRESHOLD_MODE"],
            "selected_threshold_tcga_oof": threshold,
            "tcga_oof_auc": metrics["auc"],
            "tcga_oof_auprc": metrics["auprc"],
            "tcga_oof_balacc": metrics["balanced_accuracy"],
            "tcga_oof_sensitivity": metrics["sensitivity"],
            "tcga_oof_specificity": metrics["specificity"],
            "tcga_oof_min_sens_spec": metrics["min_sens_spec"],
            "tcga_oof_mcc": metrics["mcc"],
            "tcga_oof_brier": metrics["brier"],
            "tcga_oof_c_index": safe_cindex(y_tcga["Event"].values, y_tcga["OS_days"].values, oof_scores),
            "tcga_fold_auc_mean": fold_auc_mean,
            "tcga_fold_auc_sd": fold_auc_sd,
            "n_features": len(all_feats),
            "n_graph_features": int(sum(is_graph_feature(f) for f in all_feats)),
            "graph_ratio": graph_ratio(all_feats),
            "selected_features": ";".join(all_feats),
        }, fold_df

    except Exception as exc:
        return {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(),
                "experiment_id": candidate.get("experiment_id", ""),
                "candidate_id": candidate.get("candidate_id", "")}, pd.DataFrame()


# --------------------------------------------------
# 9. Coxnet utilities and evaluation
# --------------------------------------------------

def make_surv(labels_df):
    return Surv.from_arrays(event=labels_df["Event"].astype(bool).values,
                            time=labels_df["OS_days"].astype(float).values)


def remove_zero_variance_features(X_train, X_test):
    variances = X_train.var(axis=0)
    keep = variances[variances > 0].index.tolist()
    if len(keep) == 0:
        raise ValueError("No nonzero-variance features remain")
    return X_train[keep], X_test.reindex(columns=keep, fill_value=0.0), keep


def stable_build_alpha_grid(X_train_s, labels_train):
    y_surv = make_surv(labels_train)
    errors = []
    for l1_ratio in CONF["COX_L1_RATIO_CANDIDATES"]:
        for alpha_min_ratio in CONF["COX_ALPHA_MIN_RATIO_CANDIDATES"]:
            try:
                model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, n_alphas=CONF["COX_N_ALPHAS"],
                                               alpha_min_ratio=alpha_min_ratio, max_iter=CONF["COX_MAX_ITER"],
                                               tol=CONF["COX_TOL"])
                model.fit(X_train_s, y_surv)
                alphas = np.asarray(model.alphas_, dtype=float)
                alphas = alphas[np.isfinite(alphas)]
                alphas = np.unique(alphas)
                alphas = np.sort(alphas)[::-1]
                if len(alphas) == 0:
                    raise RuntimeError("Empty alpha path")
                return alphas, l1_ratio, alpha_min_ratio
            except Exception as exc:
                errors.append(f"l1_ratio={l1_ratio}, alpha_min_ratio={alpha_min_ratio}: {repr(exc)}")
    raise RuntimeError("Could not build stable Coxnet alpha grid: " + " | ".join(errors[:5]))


def fit_coxnet_fixed_alphas_stable(X_s, labels, alpha_grid, l1_ratio):
    y_surv = make_surv(labels)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
    alpha_grid = np.unique(alpha_grid)
    alpha_grid = np.sort(alpha_grid)[::-1]
    last_error = None
    for frac in [1.0, 0.75, 0.50, 0.35, 0.25, 0.15]:
        n_keep = max(1, int(np.ceil(len(alpha_grid) * frac)))
        try:
            model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=alpha_grid[:n_keep], max_iter=CONF["COX_MAX_ITER"],
                                           tol=CONF["COX_TOL"])
            model.fit(X_s, y_surv)
            return model
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Fixed-alpha Coxnet failed. Last error: {repr(last_error)}")


def select_alpha_inner_cv(X_train, labels_train, alpha_grid, l1_ratio, seed):
    y_strat = labels_train["Risk_Label"].astype(int).values
    min_class = pd.Series(y_strat).value_counts().min()
    n_inner = min(CONF["COX_INNER_SPLITS"], int(min_class))
    if n_inner < 2:
        return float(np.max(alpha_grid)), pd.DataFrame()

    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    rows = []
    for inner_id, (tr_idx, val_idx) in enumerate(inner_cv.split(X_train, y_strat), start=1):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        lab_tr = labels_train.iloc[tr_idx]
        lab_val = labels_train.iloc[val_idx]
        X_tr, X_val, _ = remove_zero_variance_features(X_tr, X_val)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        try:
            model = fit_coxnet_fixed_alphas_stable(X_tr_s, lab_tr, alpha_grid, l1_ratio)
        except Exception:
            continue
        for alpha in model.alphas_:
            try:
                risk_val = model.predict(X_val_s, alpha=float(alpha))
            except Exception:
                continue
            rows.append({
                "inner_fold": inner_id,
                "alpha": float(alpha),
                "cindex": safe_cindex(lab_val["Event"].values, lab_val["OS_days"].values, risk_val),
                "binary_auc": safe_auc(lab_val["Risk_Label"].values, risk_val),
            })
    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return float(np.max(alpha_grid)), score_df
    summary = score_df.groupby("alpha").agg(cindex_mean=("cindex", "mean"),
                                            binary_auc_mean=("binary_auc", "mean")).reset_index()
    summary = summary.sort_values(["cindex_mean", "binary_auc_mean"], ascending=[False, False])
    return float(summary.iloc[0]["alpha"]), score_df


def fit_final_coxnet_single_alpha_stable(X_train_s, labels_train, selected_alpha, l1_ratio):
    y_surv = make_surv(labels_train)
    last_error = None
    for multiplier in CONF["COX_FINAL_ALPHA_MULTIPLIERS"]:
        alpha_try = float(selected_alpha) * float(multiplier)
        try:
            model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[alpha_try], max_iter=CONF["COX_MAX_ITER"],
                                           tol=CONF["COX_TOL"])
            model.fit(X_train_s, y_surv)
            return model, alpha_try, multiplier, l1_ratio
        except Exception as exc:
            last_error = exc
    for fallback_l1 in [0.95, 0.90, 0.80]:
        for multiplier in CONF["COX_FINAL_ALPHA_MULTIPLIERS"]:
            alpha_try = float(selected_alpha) * float(multiplier)
            try:
                model = CoxnetSurvivalAnalysis(l1_ratio=fallback_l1, alphas=[alpha_try], max_iter=CONF["COX_MAX_ITER"],
                                               tol=CONF["COX_TOL"])
                model.fit(X_train_s, y_surv)
                return model, alpha_try, multiplier, fallback_l1
            except Exception as exc:
                last_error = exc
    raise RuntimeError(f"Final Coxnet failed. Last error: {repr(last_error)}")


def fit_lasso_cox_and_predict(X_train, labels_train, X_test, seed):
    X_train, X_test, kept_features = remove_zero_variance_features(X_train, X_test)
    scaler_path = StandardScaler()
    X_path_s = scaler_path.fit_transform(X_train)
    alpha_grid, path_l1_ratio, path_alpha_min_ratio = stable_build_alpha_grid(X_path_s, labels_train)
    selected_alpha, inner_scores = select_alpha_inner_cv(X_train, labels_train, alpha_grid, path_l1_ratio, seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model, final_alpha, multiplier, final_l1_ratio = fit_final_coxnet_single_alpha_stable(X_train_s, labels_train,
                                                                                          selected_alpha, path_l1_ratio)
    risk_test = model.predict(X_test_s, alpha=final_alpha)
    coef = np.asarray(model.coef_)
    if coef.ndim == 2:
        coef = coef[:, 0]
    coef_series = pd.Series(coef, index=kept_features)
    nonzero = coef_series[np.abs(coef_series) > 1e-8]
    diag = {
        "path_l1_ratio": path_l1_ratio,
        "path_alpha_min_ratio": path_alpha_min_ratio,
        "selected_alpha_inner_cv": selected_alpha,
        "final_alpha_used": final_alpha,
        "final_alpha_multiplier": multiplier,
        "final_l1_ratio_used": final_l1_ratio,
        "n_features_after_variance_filter": len(kept_features),
        "n_nonzero_features": int(len(nonzero)),
        "nonzero_features": ";".join(nonzero.index.tolist()),
        "kept_features": kept_features,
        "scaler": scaler,
        "model": model,
    }
    return risk_test, diag, inner_scores


def standardize_risk_to_01(train_risk, test_risk):
    lo = float(np.nanmin(train_risk))
    hi = float(np.nanmax(train_risk))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(test_risk, 0.5, dtype=float)
    return np.clip((test_risk - lo) / (hi - lo), 0.0, 1.0)


def evaluate_cox_candidate(candidate, X_tcga, y_tcga, cv_splits, feature_sets):
    if not SKSURV_AVAILABLE:
        return {"status": "failed", "error": "scikit-survival is not installed",
                "experiment_id": candidate["experiment_id"], "candidate_id": candidate["candidate_id"]}, pd.DataFrame()
    y_bin = y_tcga["Risk_Label"].astype(int).values
    n = len(y_bin)
    oof_risk = np.full(n, np.nan, dtype=float)
    fold_rows = []

    try:
        for split in cv_splits:
            fold_id = int(split["fold_id"])
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            feats = feature_sets["cv"][fold_id]
            X_train = X_tcga.iloc[train_idx].reindex(columns=feats, fill_value=0.0)
            X_test = X_tcga.iloc[test_idx].reindex(columns=feats, fill_value=0.0)
            labels_train = y_tcga.iloc[train_idx]
            labels_test = y_tcga.iloc[test_idx]
            risk_scores, diag, inner = fit_lasso_cox_and_predict(X_train, labels_train, X_test,
                                                                 seed=CONF["SEED"] + fold_id)
            oof_risk[test_idx] = risk_scores
            fold_rows.append({
                "experiment_id": candidate["experiment_id"],
                "candidate_id": candidate["candidate_id"],
                "fold_id": fold_id,
                "tcga_fold_auc": safe_auc(labels_test["Risk_Label"].values, risk_scores),
                "tcga_fold_c_index": safe_cindex(labels_test["Event"].values, labels_test["OS_days"].values,
                                                 risk_scores),
                "n_nonzero_features": diag.get("n_nonzero_features", np.nan),
                "n_valid": int(len(test_idx)),
            })
        valid = np.isfinite(oof_risk)
        if valid.sum() < 30:
            raise RuntimeError("Too few valid Cox OOF risk predictions")

        oof_score01 = standardize_risk_to_01(oof_risk[valid], oof_risk[valid])
        threshold = select_threshold_from_tcga_oof(y_bin[valid], oof_score01, CONF["THRESHOLD_MODE"])
        metrics = compute_binary_metrics(y_bin[valid], oof_score01, threshold)
        fold_df = pd.DataFrame(fold_rows)
        all_feats = feature_sets["all_tcga"]

        return {
            "status": "ok", "error": "",
            "experiment_id": candidate["experiment_id"],
            "tier": candidate["tier"],
            "comparison_role": candidate["comparison_role"],
            "candidate_id": candidate["candidate_id"],
            "model_family": "lasso_cox",
            "representation": candidate["representation"],
            "k": int(candidate["k"]),
            "params_json": safe_json_dumps(candidate["params"]),
            "threshold_mode": CONF["THRESHOLD_MODE"],
            "selected_threshold_tcga_oof": threshold,
            "tcga_oof_auc": metrics["auc"],
            "tcga_oof_auprc": metrics["auprc"],
            "tcga_oof_balacc": metrics["balanced_accuracy"],
            "tcga_oof_sensitivity": metrics["sensitivity"],
            "tcga_oof_specificity": metrics["specificity"],
            "tcga_oof_min_sens_spec": metrics["min_sens_spec"],
            "tcga_oof_mcc": metrics["mcc"],
            "tcga_oof_brier": np.nan,
            "tcga_oof_c_index": safe_cindex(y_tcga["Event"].values[valid], y_tcga["OS_days"].values[valid],
                                            oof_risk[valid]),
            "tcga_fold_auc_mean": float(fold_df["tcga_fold_auc"].mean()),
            "tcga_fold_auc_sd": float(fold_df["tcga_fold_auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
            "tcga_fold_c_index_mean": float(fold_df["tcga_fold_c_index"].mean()),
            "n_features": len(all_feats),
            "n_graph_features": int(sum(is_graph_feature(f) for f in all_feats)),
            "graph_ratio": graph_ratio(all_feats),
            "selected_features": ";".join(all_feats),
        }, fold_df
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(),
                "experiment_id": candidate.get("experiment_id", ""),
                "candidate_id": candidate.get("candidate_id", "")}, pd.DataFrame()


# --------------------------------------------------
# 10. Experiment definitions and TCGA-only selection
# --------------------------------------------------

def build_experiments():
    experiments = [
        {
            "experiment_id": "T1_GIBD_XGB_K100_CHAMPION_LOCKED",
            "tier": "Tier 1 Graph Impact / Champion Reference",
            "comparison_role": "Locked GIBD champion: WPPI-informed XGBoost K=100",
            "representation": "gibd",
            "model_family": "locked_champion",
            "k": CONF["K_CHAMPION"],
            "candidate_params": [],
            "locked_reference": True,
        },
        {
            "experiment_id": "T1_RAW_XGB_K100_TUNED",
            "tier": "Tier 1 Graph Impact",
            "comparison_role": "TCGA-tuned raw-expression XGBoost K=100",
            "representation": "raw",
            "model_family": "xgb",
            "k": CONF["K_CHAMPION"],
            "candidate_params": RAW_XGB_GRID,
            "locked_reference": False,
        },
        {
            "experiment_id": "T2_RAW_RF_K100_TUNED",
            "tier": "Tier 2 Model Impact",
            "comparison_role": "TCGA-tuned raw-expression RandomForest K=100",
            "representation": "raw",
            "model_family": "rf",
            "k": CONF["K_CHAMPION"],
            "candidate_params": RF_GRID,
            "locked_reference": False,
        },
        {
            "experiment_id": "T2_GIBD_RF_K100_TUNED",
            "tier": "Tier 2 Model Impact",
            "comparison_role": "TCGA-tuned GIBD RandomForest K=100",
            "representation": "gibd",
            "model_family": "rf",
            "k": CONF["K_CHAMPION"],
            "candidate_params": RF_GRID,
            "locked_reference": False,
        },
        {
            "experiment_id": "T2_RAW_LASSO_COX_K100_TUNED",
            "tier": "Tier 2 Model Impact",
            "comparison_role": "TCGA-tuned raw-expression LASSO-Cox K=100",
            "representation": "raw",
            "model_family": "lasso_cox",
            "k": CONF["K_CHAMPION"],
            "candidate_params": COX_CANDIDATES,
            "locked_reference": False,
        },
        {
            "experiment_id": "T3_GIBD_XGB_K120_COMPLEXITY_CONTROL",
            "tier": "Tier 3 Complexity Control",
            "comparison_role": "GIBD XGBoost K=120 complexity-control comparator",
            "representation": "gibd",
            "model_family": "xgb",
            "k": CONF["K_COMPLEXITY"],
            "candidate_params": [CONF["CHAMPION_XGB_PARAMS"]],
            "locked_reference": False,
        },
    ]

    if CONF.get("INCLUDE_K80_SENSITIVITY", False):
        experiments.append({
            "experiment_id": "T3_GIBD_XGB_K80_PARSIMONY_SENSITIVITY",
            "tier": "Tier 3 Optional Parsimony Sensitivity",
            "comparison_role": "Optional GIBD XGBoost K=80 parsimony-sensitivity comparator",
            "representation": "gibd",
            "model_family": "xgb",
            "k": CONF["K_PARSIMONY_LOW"],
            "candidate_params": [CONF["CHAMPION_XGB_PARAMS"]],
            "locked_reference": False,
        })

    return experiments


def expand_candidates(exp):
    rows = []
    for i, params in enumerate(exp["candidate_params"], start=1):
        rows.append({
            "experiment_id": exp["experiment_id"],
            "tier": exp["tier"],
            "comparison_role": exp["comparison_role"],
            "representation": exp["representation"],
            "model_family": exp["model_family"],
            "k": int(exp["k"]),
            "params": dict(params),
            "candidate_id": clean_id(f"{exp['experiment_id']}_cand{i}"),
        })
    return rows


def tcga_selection_score(row):
    auc = float(row.get("tcga_oof_auc", 0.0))
    sens = float(row.get("tcga_oof_sensitivity", 0.0))
    spec = float(row.get("tcga_oof_specificity", 0.0))
    bal = float(row.get("tcga_oof_balacc", 0.0))
    sd = float(row.get("tcga_fold_auc_sd", 0.15))
    cidx = row.get("tcga_oof_c_index", np.nan)
    cidx = 0.0 if pd.isna(cidx) else float(cidx)
    stability = 1.0 - min(max(sd, 0.0) / 0.15, 1.0)
    return float(
        0.30 * min(max(sens, 0.0) / 0.70, 1.0)
        + 0.24 * min(max(auc, 0.0) / 0.70, 1.0)
        + 0.20 * min(max(bal, 0.0) / 0.65, 1.0)
        + 0.12 * stability
        + 0.08 * min(max(spec, 0.0) / 0.50, 1.0)
        + 0.06 * min(max(cidx, 0.0) / 0.70, 1.0)
        - 0.04 * abs(sens - spec)
    )


def select_experiment_winner(cand_df):
    df = cand_df[cand_df["status"].eq("ok")].copy()
    if df.empty:
        return None, df
    df["tcga_selection_score"] = df.apply(tcga_selection_score, axis=1)
    sort_keys = ["tcga_selection_score", "tcga_oof_sensitivity", "tcga_oof_auc", "tcga_fold_auc_sd", "tcga_oof_balacc"]
    ascending = [False, False, False, True, False]
    assert_tcga_only_sort_keys(sort_keys, "ablation experiment winner selection")
    ranked = df.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)
    return ranked.iloc[0].to_dict(), ranked


def locked_champion_row(locked):
    meta = locked["metadata"]
    oof = locked["tcga_oof_summary"]
    features = locked["features"]

    def get_oof(*names, default=np.nan):
        for name in names:
            if name in oof:
                return oof[name]
        return default

    return {
        "status": "ok", "error": "",
        "experiment_id": "T1_GIBD_XGB_K100_CHAMPION_LOCKED",
        "tier": "Tier 1 Graph Impact / Champion Reference",
        "comparison_role": "Locked GIBD champion: WPPI-informed XGBoost K=100",
        "candidate_id": str(meta.get("candidate_id", "16U_Final_Locked_TCGA_Model")),
        "model_family": "locked_champion",
        "representation": "gibd",
        "k": int(meta.get("k", len(features))),
        "params_json": safe_json_dumps(locked["params"]),
        "threshold_mode": meta.get("threshold_mode", CONF["THRESHOLD_MODE"]),
        "selected_threshold_tcga_oof": locked["threshold"],
        "tcga_oof_auc": get_oof("tcga_oof_auc"),
        "tcga_oof_auprc": get_oof("tcga_oof_auprc"),
        "tcga_oof_balacc": get_oof("tcga_oof_balacc"),
        "tcga_oof_sensitivity": get_oof("tcga_oof_sensitivity"),
        "tcga_oof_specificity": get_oof("tcga_oof_specificity"),
        "tcga_oof_min_sens_spec": get_oof("tcga_oof_min_sens_spec"),
        "tcga_oof_mcc": get_oof("tcga_oof_mcc"),
        "tcga_oof_brier": get_oof("tcga_oof_brier"),
        "tcga_oof_c_index": get_oof("tcga_oof_c_index"),
        "tcga_fold_auc_mean": get_oof("tcga_fold_auc_mean"),
        "tcga_fold_auc_sd": get_oof("tcga_fold_auc_sd"),
        "n_features": len(features),
        "n_graph_features": int(sum(is_graph_feature(f) for f in features)),
        "graph_ratio": graph_ratio(features),
        "selected_features": ";".join(features),
        "locked_reference_source": "16U locked joblib + metadata + locked features",
    }


# --------------------------------------------------
# 11. External locked evaluation
# --------------------------------------------------

def fit_binary_full_tcga_and_external(winner, X_tcga, y_tcga, X_cgga, y_cgga, features):
    y_train = y_tcga["Risk_Label"].astype(int).values
    y_ext = y_cgga["Risk_Label"].astype(int).values
    threshold = float(winner["selected_threshold_tcga_oof"])
    params = json.loads(winner["params_json"])
    X_train = X_tcga.reindex(columns=features, fill_value=0.0)
    X_ext = X_cgga.reindex(columns=features, fill_value=0.0)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_ext_s = scaler.transform(X_ext)
    if winner["model_family"] == "xgb":
        model = make_xgb(params, CONF["SEED"])
    elif winner["model_family"] == "rf":
        model = make_rf(params, CONF["SEED"])
    else:
        raise ValueError(f"Unsupported binary family: {winner['model_family']}")
    model.fit(X_train_s, y_train)
    train_scores = model_score_from_fitted(model, X_train_s)
    ext_scores = model_score_from_fitted(model, X_ext_s)
    train_metrics = compute_binary_metrics(y_train, train_scores, threshold)
    ext_metrics = compute_binary_metrics(y_ext, ext_scores, threshold)
    ext_cidx = safe_cindex(y_cgga["Event"].values, y_cgga["OS_days"].values, ext_scores)
    bundle = {"experiment_id": winner["experiment_id"], "candidate_id": winner["candidate_id"],
              "model_family": winner["model_family"], "representation": winner["representation"], "k": int(winner["k"]),
              "features": list(features), "params": params, "threshold": threshold, "scaler": scaler, "model": model}
    return train_metrics, ext_metrics, ext_cidx, bundle


def fit_cox_full_tcga_and_external(winner, X_tcga, y_tcga, X_cgga, y_cgga, features):
    if not SKSURV_AVAILABLE:
        raise RuntimeError("scikit-survival is not installed")
    threshold = float(winner["selected_threshold_tcga_oof"])
    X_train = X_tcga.reindex(columns=features, fill_value=0.0)
    X_ext = X_cgga.reindex(columns=features, fill_value=0.0)
    risk_ext, diag, inner = fit_lasso_cox_and_predict(X_train, y_tcga, X_ext, seed=CONF["SEED"])
    risk_train, _, _ = fit_lasso_cox_and_predict(X_train, y_tcga, X_train, seed=CONF["SEED"])
    train_score01 = standardize_risk_to_01(risk_train, risk_train)
    ext_score01 = standardize_risk_to_01(risk_train, risk_ext)
    train_metrics = compute_binary_metrics(y_tcga["Risk_Label"].astype(int).values, train_score01, threshold)
    ext_metrics = compute_binary_metrics(y_cgga["Risk_Label"].astype(int).values, ext_score01, threshold)
    ext_cidx = safe_cindex(y_cgga["Event"].values, y_cgga["OS_days"].values, risk_ext)
    bundle = {"experiment_id": winner["experiment_id"], "candidate_id": winner["candidate_id"],
              "model_family": "lasso_cox", "representation": winner["representation"], "k": int(winner["k"]),
              "features": list(features), "threshold": threshold,
              "diagnostics": {k: v for k, v in diag.items() if k not in ["scaler", "model", "kept_features"]}}
    return train_metrics, ext_metrics, ext_cidx, bundle


# --------------------------------------------------
# 12. Main execution
# --------------------------------------------------

def main():
    print("=" * 80)
    print("Project GIBD: Master Revision Ablation Study V3")
    print("=" * 80)
    print("Locked Champion Mode + OS/Event Cox labels + post-lock external loading")
    print(f"scikit-survival available for LASSO-Cox: {SKSURV_AVAILABLE}")
    print("=" * 80)

    locked = load_locked_champion_artifacts()
    X_tcga, y_tcga, tcga_path = load_tcga_phase1()
    cv_splits = build_cv_splits(X_tcga, y_tcga["Risk_Label"].astype(int))

    experiments = build_experiments()

    # Build feature sets only for non-champion experiments. This keeps non-champion feature selection fold-contained.
    feature_key_to_sets = {}
    for exp in experiments:
        if exp["locked_reference"]:
            continue
        key = (exp["representation"], int(exp["k"]))
        if key not in feature_key_to_sets:
            feature_key_to_sets[key] = compute_feature_sets_tcga_only(X_tcga, y_tcga, exp["representation"],
                                                                      int(exp["k"]), cv_splits)

    all_candidate_tables = []
    all_fold_tables = []
    winners = []

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"TCGA-only locking experiment: {exp['experiment_id']}")
        print(exp["comparison_role"])
        print("=" * 80)

        if exp["locked_reference"]:
            winner = locked_champion_row(locked)
            ranked = pd.DataFrame([winner])
        else:
            feature_sets = feature_key_to_sets[(exp["representation"], int(exp["k"]))]
            summaries = []
            for cand in expand_candidates(exp):
                if cand["model_family"] in ["xgb", "rf"]:
                    summary, fold_df = evaluate_binary_candidate(cand, X_tcga, y_tcga, cv_splits, feature_sets)
                elif cand["model_family"] == "lasso_cox":
                    summary, fold_df = evaluate_cox_candidate(cand, X_tcga, y_tcga, cv_splits, feature_sets)
                else:
                    raise ValueError(f"Unknown model family: {cand['model_family']}")
                summaries.append(summary)
                if fold_df is not None and not fold_df.empty:
                    all_fold_tables.append(fold_df)
            cand_df = pd.DataFrame(summaries)
            winner, ranked = select_experiment_winner(cand_df)
            if winner is None:
                print(f"No successful candidate for {exp['experiment_id']}")
                all_candidate_tables.append(cand_df)
                continue

        all_candidate_tables.append(ranked)
        winners.append(winner)
        show_cols = ["candidate_id", "model_family", "representation", "k", "tcga_selection_score", "tcga_oof_auc",
                     "tcga_oof_sensitivity", "tcga_oof_specificity", "tcga_oof_balacc", "tcga_oof_c_index",
                     "tcga_fold_auc_sd"]
        show_cols = [c for c in show_cols if c in ranked.columns]
        print(ranked.head(10)[show_cols].to_string(index=False))

    # External data loaded only after all TCGA winners are selected.
    X_cgga, y_cgga, cgga_path, cgga_label_path = load_cgga_phase2()

    external_rows = []
    summary_rows = []
    feature_rows = []
    saved_model_paths = []

    for winner in winners:
        exp_id = winner["experiment_id"]
        features = [f for f in str(winner.get("selected_features", "")).split(";") if f]
        if winner["model_family"] == "locked_champion":
            features = locked["features"]

        for i, feat in enumerate(features, start=1):
            feature_rows.append({"experiment_id": exp_id, "feature_rank": i, "feature": feat,
                                 "is_graph_feature": is_graph_feature(feat), "representation": winner["representation"],
                                 "k": int(winner["k"])})

        try:
            if winner["model_family"] == "locked_champion":
                ext_scores = predict_locked_champion(locked, X_cgga)
                ext_metrics = compute_binary_metrics(y_cgga["Risk_Label"].astype(int).values, ext_scores,
                                                     locked["threshold"])
                ext_cidx = safe_cindex(y_cgga["Event"].values, y_cgga["OS_days"].values, ext_scores)
                model_path = CONF["CHAMPION_LOCKED_JOBLIB"]
            elif winner["model_family"] in ["xgb", "rf"]:
                train_metrics, ext_metrics, ext_cidx, bundle = fit_binary_full_tcga_and_external(winner, X_tcga, y_tcga,
                                                                                                 X_cgga, y_cgga,
                                                                                                 features)
                model_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_{exp_id}_locked_model.joblib")
                dump(bundle, model_path)
                saved_model_paths.append(model_path)
            elif winner["model_family"] == "lasso_cox":
                train_metrics, ext_metrics, ext_cidx, bundle = fit_cox_full_tcga_and_external(winner, X_tcga, y_tcga,
                                                                                              X_cgga, y_cgga, features)
                model_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_{exp_id}_locked_model.joblib")
                dump(bundle, model_path)
                saved_model_paths.append(model_path)
            else:
                raise ValueError(f"Unknown winner model family: {winner['model_family']}")

            external_rows.append({
                "experiment_id": exp_id,
                "tier": winner.get("tier", ""),
                "comparison_role": winner.get("comparison_role", ""),
                "winner_candidate_id": winner.get("candidate_id", ""),
                "model_family": winner.get("model_family", ""),
                "representation": winner.get("representation", ""),
                "k": int(winner.get("k", len(features))),
                "n_features": len(features),
                "n_graph_features": int(sum(is_graph_feature(f) for f in features)),
                "graph_ratio": graph_ratio(features),
                "locked_threshold": float(winner.get("selected_threshold_tcga_oof", locked["threshold"])),
                "params_json": winner.get("params_json", ""),
                "tcga_oof_auc": winner.get("tcga_oof_auc", np.nan),
                "tcga_oof_sensitivity": winner.get("tcga_oof_sensitivity", np.nan),
                "tcga_oof_specificity": winner.get("tcga_oof_specificity", np.nan),
                "tcga_oof_balacc": winner.get("tcga_oof_balacc", np.nan),
                "tcga_oof_c_index": winner.get("tcga_oof_c_index", np.nan),
                "external_auc": ext_metrics["auc"],
                "external_sensitivity": ext_metrics["sensitivity"],
                "external_specificity": ext_metrics["specificity"],
                "external_balanced_accuracy": ext_metrics["balanced_accuracy"],
                "external_mcc": ext_metrics["mcc"],
                "external_auprc": ext_metrics["auprc"],
                "external_brier": ext_metrics["brier"],
                "external_c_index": ext_cidx,
                "external_tp": ext_metrics["tp"],
                "external_fp": ext_metrics["fp"],
                "external_tn": ext_metrics["tn"],
                "external_fn": ext_metrics["fn"],
                "locked_model_path": model_path,
                "external_role": "single-pass post-lockdown evaluation only",
            })

            summary_rows.append({
                "Tier": winner.get("tier", ""),
                "Experiment": exp_id,
                "Comparison": winner.get("comparison_role", ""),
                "Representation": winner.get("representation", ""),
                "Model": winner.get("model_family", ""),
                "K": int(winner.get("k", len(features))),
                "Graph_Feature_Density": graph_ratio(features),
                "TCGA_OOF_AUC": winner.get("tcga_oof_auc", np.nan),
                "TCGA_OOF_Sensitivity": winner.get("tcga_oof_sensitivity", np.nan),
                "TCGA_OOF_Specificity": winner.get("tcga_oof_specificity", np.nan),
                "TCGA_OOF_Balanced_Accuracy": winner.get("tcga_oof_balacc", np.nan),
                "TCGA_OOF_C_index": winner.get("tcga_oof_c_index", np.nan),
                "External_AUC": ext_metrics["auc"],
                "External_Sensitivity": ext_metrics["sensitivity"],
                "External_Specificity": ext_metrics["specificity"],
                "External_Balanced_Accuracy": ext_metrics["balanced_accuracy"],
                "External_C_index": ext_cidx,
                "Locked_Threshold": float(winner.get("selected_threshold_tcga_oof", locked["threshold"])),
                "Winner_Candidate": winner.get("candidate_id", ""),
            })
        except Exception as exc:
            external_rows.append({"experiment_id": exp_id, "error": str(exc), "traceback": traceback.format_exc()})

    candidate_table = pd.concat(all_candidate_tables, axis=0,
                                ignore_index=True) if all_candidate_tables else pd.DataFrame()
    fold_table = pd.concat(all_fold_tables, axis=0, ignore_index=True) if all_fold_tables else pd.DataFrame()
    external_table = pd.DataFrame(external_rows)
    summary_table = pd.DataFrame(summary_rows)
    feature_table = pd.DataFrame(feature_rows)

    candidate_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_tcga_only_candidates.csv")
    folds_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_tcga_cv_folds.csv")
    external_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_external_locked_eval.csv")
    summary_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_summary.csv")
    features_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_selected_features.csv")
    audit_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_methodology_audit.json")

    candidate_table.to_csv(candidate_path, index=False)
    fold_table.to_csv(folds_path, index=False)
    external_table.to_csv(external_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    feature_table.to_csv(features_path, index=False)

    audit = {
        "run_name": RUN_NAME,
        "methodological_policy": {
            "median_tcga_survival_labels": "Uses Data/TCGA_survival_labels_with_os_event.csv generated under the TCGA empirical median-OS endpoint; labels are aligned to the final TCGA expression cohort.",
            "locked_champion_mode": "Champion row loaded from 16U locked joblib, locked features CSV, and locked metadata; no feature or threshold recomputation for champion.",
            "iron_curtain_external_load": "CGGA was loaded only after all TCGA-only experiment winners were selected and locked.",
            "feature_selection": "For non-champion ablations, RF-Gini feature ranking was recomputed within each TCGA training fold for OOF evaluation.",
            "raw_rf_baseline_added": "V3 includes a TCGA-tuned raw-expression Random Forest K=100 baseline to directly address the requested Random Forest-on-original-expression comparator.",
            "hyperparameter_selection": "Baseline candidate winners were selected using TCGA OOF metrics only.",
            "threshold_selection": "Baseline thresholds were selected using TCGA OOF scores only. Champion threshold was loaded from frozen 16U metadata.",
            "external_use": "CGGA was used only for single-pass post-lockdown evaluation.",
        },
        "model_size_policy": {"champion_k": CONF["K_CHAMPION"], "complexity_control_k": CONF["K_COMPLEXITY"], "optional_k80_sensitivity_enabled": CONF["INCLUDE_K80_SENSITIVITY"]},
        "cv": {"seed": CONF["SEED"], "n_splits": CONF["N_SPLITS"], "n_repeats": CONF["N_REPEATS"]},
        "dependencies": {"scikit_survival_available": SKSURV_AVAILABLE},
        "input_paths": {"tcga_matrix": tcga_path, "tcga_survival_labels": CONF["TCGA_SURVIVAL_LABELS"],
                        "cgga_matrix": cgga_path, "cgga_labels": cgga_label_path},
        "locked_champion_artifacts": {"joblib": CONF["CHAMPION_LOCKED_JOBLIB"],
                                      "metadata": CONF["CHAMPION_LOCKED_METADATA"],
                                      "features": CONF["CHAMPION_LOCKED_FEATURES"]},
        "outputs": {"candidate_table": candidate_path, "fold_table": folds_path, "external_locked_eval": external_path,
                    "summary_table": summary_path, "selected_features": features_path,
                    "saved_models": saved_model_paths},
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(audit), f, indent=2)

    print("\n" + "=" * 80)
    print("REVISION ABLATION V3 SUMMARY")
    print("=" * 80)
    if not summary_table.empty:
        display_cols = ["Tier", "Experiment", "Model", "Representation", "K", "Graph_Feature_Density", "TCGA_OOF_AUC",
                        "TCGA_OOF_Sensitivity", "TCGA_OOF_Specificity", "TCGA_OOF_Balanced_Accuracy",
                        "TCGA_OOF_C_index", "External_AUC", "External_Sensitivity", "External_Specificity",
                        "External_Balanced_Accuracy", "External_C_index"]
        display_cols = [c for c in display_cols if c in summary_table.columns]
        print(summary_table[display_cols].to_string(index=False))
    else:
        print("No successful summary rows generated.")

    print("\nSaved outputs:")
    print(f"- Summary table: {summary_path}")
    print(f"- TCGA-only candidates: {candidate_path}")
    print(f"- TCGA CV folds: {folds_path}")
    print(f"- External locked evaluation: {external_path}")
    print(f"- Selected features: {features_path}")
    print(f"- Methodology audit: {audit_path}")


if __name__ == "__main__":
    main()
