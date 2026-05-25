"""
GIBD analysis workflow — TCGA-only model locking

This script performs the TCGA-only model-development and locking step for the
locked GIBD-XGBoost K100 classifier. It builds fold-contained TCGA out-of-fold
predictions, selects the operating threshold from TCGA out-of-fold probabilities,
and freezes the final model, scaler, feature order, and locked feature artifacts.

Analysis guardrails:
- TCGA is the only cohort used for feature selection, model selection, scaler
  fitting, threshold selection, and final model locking.
- The final locked threshold is selected from TCGA out-of-fold probabilities
  using the pre-specified recall80_spec25 rule.
- The script does not perform post-lock external validation and does not alter
  the locked feature set, model parameters, or operating threshold after locking.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import ast
import json
import math
import re
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump

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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RUN_NAME = "dev16U_median_os_k100_g035_recall80_spec25_tcga_lockdown"
CPU_COUNT = os.cpu_count() or 4

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation"),

    "TCGA_WEIGHTED_GRAPH": os.path.join("Data", "Revision_Ablation", "tcga_weighted_self_graph_cache.csv"),
    "TCGA_GRAPH_FALLBACK": os.path.join("Data", "TCGA_survival_expression_matrix_enhanced.csv"),

    "TCGA_LABELS": os.path.join("Data", "TCGA_survival_labels_manuscript_aligned.csv"),

    "RANKED_16T_CANDIDATE_FILES": [],

    "FEATURE_CACHE_CANDIDATES": [],

    "K": 100,
    "RANK_PROFILE": "transport_s034_g035",
    "STABILITY_WEIGHT": 0.34,
    "GRAPH_BONUS": 0.035,
    "MIN_GRAPH_FEATURES_ALL_TCGA": 0,

    "SEED": 42,
    "N_SPLITS": 5,
    "N_REPEATS": 4,

    "N_TOP_16T_PARAM_ANCHORS": 1,
    "MAX_GRID_CANDIDATES": 1,
    "N_RANDOM_LOCAL_CANDIDATES": 0,

    "THRESHOLD_MODE": "recall80_spec25",
    "THRESHOLD_MIN": 0.03,
    "THRESHOLD_MAX": 0.97,
    "N_THRESHOLDS": 189,

    "MAX_DEPTH_OPTIONS": [3],
    "LEARNING_RATE_OPTIONS": [0.029],
    "N_ESTIMATORS_OPTIONS": [85],
    "REG_LAMBDA_OPTIONS": [6.5],
    "REG_ALPHA_OPTIONS": [0.30],
    "GAMMA_OPTIONS": [0.18],
    "SCALE_POS_WEIGHT_OPTIONS": [2.40],
    "SUBSAMPLE_OPTIONS": [0.80],
    "COLSAMPLE_BYTREE_OPTIONS": [0.75],
    "MIN_CHILD_WEIGHT_OPTIONS": [1.8],

    "FREEZE_SEEDS": [42],

    "MODEL_THREADS": 1,
    "N_JOBS": max(1, min(16, CPU_COUNT - 1)),
    "JOBLIB_BACKEND": "loky",
    "TREE_METHOD": "hist",

    "RF_ESTIMATORS": 375,
    "RF_SELECTION_REPEATS": 3,
    "RF_TOP_POOL_MULTIPLIER": 4,
    "FIXED_GAMMA": 0.18,

    "EXPORT_OOF_ONLY": False,

    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,
}

os.makedirs(CONF["OUT_DIR"], exist_ok=True)


# --------------------------------------------------
# 2. General utilities
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
        return [
            {str(k): to_builtin(v) for k, v in row.items()}
            for row in obj.to_dict(orient="records")
        ]
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


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def clean_id(text):
    text = str(text)
    for bad in [" ", "/", "\\", ":", ";", ",", "=", ".", "+", "(", ")", "[", "]", "{", "}"]:
        text = text.replace(bad, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def parse_params_json(value):
    if isinstance(value, dict):
        return dict(value)

    if value is None:
        return {}

    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


def is_graph_feature(feature_name):
    name = str(feature_name).lower()
    tokens = [
        "_wppi_self",
        "wppi_self",
        "_wppi_neighbor",
        "wppi_neighbor",
        "_nbr_mean",
        "nbr_mean",
        "neighbor_mean",
        "network_mean",
        "graph_mean",
    ]
    return any(token in name for token in tokens)


def graph_ratio(features):
    features = list(features)
    if not features:
        return 0.0
    return float(sum(is_graph_feature(f) for f in features) / len(features))


def coalesce_row(row, names, default=np.nan):
    for name in names:
        if name in row.index:
            val = safe_float(row[name], np.nan)
            if not math.isnan(val):
                return val
    return default


def assert_tcga_only_sort_keys(sort_keys, context):
    if sort_keys is None:
        return

    keys = [sort_keys] if isinstance(sort_keys, str) else list(sort_keys)

    bad = []
    for key in keys:
        k = str(key).lower()
        tokens = re.split(r"[^a-z0-9]+", k)

        forbidden = (
            "external" in tokens
            or k.startswith("old_")
            or "retrospective" in k
            or "diagnostic" in k
            or k.startswith("gap_")
            or k.endswith("_gap")
            or "gap_compression" in k
        )

        if forbidden:
            bad.append(str(key))

    if bad:
        raise RuntimeError(
            f"Non-TCGA key used in TCGA-only selection at {context}: {bad}"
        )


# --------------------------------------------------
# 3. Data loading
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
    X.columns = X.columns.astype(str).str.split(".").str[0]
    X = X.loc[:, ~X.columns.duplicated()].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def load_labels(path, expected_n, expected_low, expected_high):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    labels = pd.read_csv(path)

    if "Patient_ID" in labels.columns:
        labels = labels.set_index("Patient_ID")
    elif "Unnamed: 0" in labels.columns:
        labels = labels.set_index("Unnamed: 0")
    else:
        labels = labels.set_index(labels.columns[0])

    labels.index = labels.index.astype(str)

    if "Risk_Label" not in labels.columns:
        raise ValueError(f"Risk_Label missing in: {path}")

    labels["Risk_Label"] = labels["Risk_Label"].astype(int)

    if len(labels) != int(expected_n):
        raise ValueError(f"{path}: expected N={expected_n}, got {len(labels)}")

    counts = labels["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))

    if low != int(expected_low) or high != int(expected_high):
        raise ValueError(
            f"{path}: expected low={expected_low}, high={expected_high}; got low={low}, high={high}"
        )

    return labels


def load_data():
    print("\n" + "=" * 80)
    print("Loading TCGA matrix and labels")
    print("=" * 80)

    tcga_path = choose_matrix_path(
        CONF["TCGA_WEIGHTED_GRAPH"],
        CONF["TCGA_GRAPH_FALLBACK"],
        "TCGA",
    )

    X_tcga = load_feature_matrix(tcga_path)

    y_tcga_df = load_labels(
        CONF["TCGA_LABELS"],
        CONF["EXPECTED_N_TCGA"],
        CONF["EXPECTED_LOW_TCGA"],
        CONF["EXPECTED_HIGH_TCGA"],
    )

    common_tcga = X_tcga.index.intersection(y_tcga_df.index)
    X_tcga = X_tcga.loc[common_tcga].copy()
    y_tcga = y_tcga_df.loc[common_tcga, "Risk_Label"].astype(int).loc[X_tcga.index]

    if len(X_tcga) != CONF["EXPECTED_N_TCGA"]:
        raise ValueError(f"TCGA intersection expected {CONF['EXPECTED_N_TCGA']}, got {len(X_tcga)}")

    print(f"TCGA: {X_tcga.shape}; labels={y_tcga.value_counts().sort_index().to_dict()}")
    print(f"TCGA WPPI/graph-expanded features: {sum(is_graph_feature(c) for c in X_tcga.columns)}")

    return X_tcga, y_tcga, tcga_path


# --------------------------------------------------
# 4. CV and feature sets
# --------------------------------------------------

def build_cv_splits(X, y):
    cv = RepeatedStratifiedKFold(
        n_splits=CONF["N_SPLITS"],
        n_repeats=CONF["N_REPEATS"],
        random_state=CONF["SEED"],
    )

    return [
        {"fold_id": i, "train_idx": tr, "test_idx": te}
        for i, (tr, te) in enumerate(cv.split(X, y), start=1)
    ]


def parse_features_text(value):
    if value is None:
        return []

    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return []

    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]

    return [text]


def load_feature_sets_from_cache(cache_path, cv_splits):
    if not os.path.exists(cache_path):
        return None

    try:
        df = pd.read_csv(cache_path)
    except Exception as exc:
        print(f"Could not read feature cache {cache_path}: {exc}")
        return None

    required = {"scope", "fold_id", "rank_profile", "k", "selected_features"}
    if not required.issubset(set(df.columns)):
        return None

    profile = CONF["RANK_PROFILE"]
    k = int(CONF["K"])

    feature_sets = {"cv": {}, "all_tcga": {}}
    fold_ids = [int(s["fold_id"]) for s in cv_splits]

    for fold_id in fold_ids:
        row = df[
            (df["scope"].astype(str) == "cv_fold")
            & (df["fold_id"].astype(int) == fold_id)
            & (df["rank_profile"].astype(str) == profile)
            & (df["k"].astype(int) == k)
        ]

        if row.empty:
            print(f"Feature cache missing fold={fold_id}, profile={profile}, k={k}: {cache_path}")
            return None

        feats = parse_features_text(row.iloc[0]["selected_features"])
        if len(feats) < k:
            return None

        feature_sets["cv"][fold_id] = feats[:k]

    row = df[
        (df["scope"].astype(str) == "all_tcga")
        & (df["rank_profile"].astype(str) == profile)
        & (df["k"].astype(int) == k)
    ]

    if row.empty:
        print(f"Feature cache missing all_tcga profile={profile}, k={k}: {cache_path}")
        return None

    all_feats = parse_features_text(row.iloc[0]["selected_features"])
    if len(all_feats) < k:
        return None

    feature_sets["all_tcga"][profile] = all_feats[:k]

    out_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_reused_feature_cache.csv")
    df.to_csv(out_path, index=False)

    print(f"Reused feature cache: {cache_path}")
    print(f"Saved feature-cache audit copy: {out_path}")
    return feature_sets


def rf_rank_features(X_train, y_train, max_k, seed):
    top_pool = min(X_train.shape[1], max(max_k, int(max_k * CONF["RF_TOP_POOL_MULTIPLIER"])))

    importance_sum = pd.Series(0.0, index=X_train.columns)
    stability_count = pd.Series(0.0, index=X_train.columns)

    for r in range(CONF["RF_SELECTION_REPEATS"]):
        rf = RandomForestClassifier(
            n_estimators=CONF["RF_ESTIMATORS"],
            random_state=int(seed) + 1009 * r,
            n_jobs=CONF["MODEL_THREADS"],
            max_features="sqrt",
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            bootstrap=True,
        )
        rf.fit(X_train, y_train.values)

        imp = pd.Series(rf.feature_importances_, index=X_train.columns)
        importance_sum += imp
        stability_count.loc[imp.nlargest(top_pool).index] += 1.0

    importance_mean = importance_sum / float(CONF["RF_SELECTION_REPEATS"])
    stability_rate = stability_count / float(CONF["RF_SELECTION_REPEATS"])

    max_imp = float(importance_mean.max())
    importance_scaled = importance_mean / max_imp if max_imp > 0 else importance_mean.copy()

    graph_flags = pd.Series([is_graph_feature(f) for f in importance_scaled.index], index=importance_scaled.index)
    graph_bonus = graph_flags.astype(float) * float(CONF["GRAPH_BONUS"])

    final_score = (
        (1.0 - CONF["STABILITY_WEIGHT"]) * importance_scaled
        + CONF["STABILITY_WEIGHT"] * stability_rate
        + graph_bonus
    )

    ranked = final_score.sort_values(ascending=False).index.tolist()

    diagnostic = pd.DataFrame({
        "feature": final_score.index,
        "importance_mean": importance_mean.values,
        "importance_scaled": importance_scaled.values,
        "stability_rate": stability_rate.values,
        "is_graph_feature": graph_flags.values,
        "graph_bonus": graph_bonus.values,
        "rank_profile": CONF["RANK_PROFILE"],
        "final_score": final_score.values,
    }).sort_values("final_score", ascending=False)

    return ranked, diagnostic


def compute_feature_sets(X_tcga, y_tcga, cv_splits):
    print("\n" + "=" * 80)
    print("No reusable feature cache found. Recomputing fold-contained RF-Gini feature sets.")
    print(f"Fresh K{CONF['K']} graph-enriched feature selection is active.")
    print("=" * 80)

    k = int(CONF["K"])
    profile = CONF["RANK_PROFILE"]

    feature_sets = {"cv": {}, "all_tcga": {}}
    feature_rows = []
    diag_frames = []

    for split in cv_splits:
        fold_id = int(split["fold_id"])
        X_train = X_tcga.iloc[split["train_idx"]]
        y_train = y_tcga.iloc[split["train_idx"]]

        ranked, diag = rf_rank_features(
            X_train,
            y_train,
            max_k=k,
            seed=CONF["SEED"] + 10000 + fold_id * 97,
        )

        feats = ranked[:k]
        feature_sets["cv"][fold_id] = feats

        feature_rows.append({
            "scope": "cv_fold",
            "fold_id": fold_id,
            "rank_profile": profile,
            "k": k,
            "n_features": len(feats),
            "n_graph_features": int(sum(is_graph_feature(f) for f in feats)),
            "graph_ratio": graph_ratio(feats),
            "selected_features": ";".join(feats),
        })

        print(
            f"Fold {fold_id:02d}: graph={sum(is_graph_feature(f) for f in feats)}/{k} "
            f"({graph_ratio(feats):.3f})",
            flush=True,
        )

    ranked_all, diag_all = rf_rank_features(
        X_tcga,
        y_tcga,
        max_k=k,
        seed=CONF["SEED"] + 50000,
    )

    all_feats = ranked_all[:k]
    feature_sets["all_tcga"][profile] = all_feats

    feature_rows.append({
        "scope": "all_tcga",
        "fold_id": 0,
        "rank_profile": profile,
        "k": k,
        "n_features": len(all_feats),
        "n_graph_features": int(sum(is_graph_feature(f) for f in all_feats)),
        "graph_ratio": graph_ratio(all_feats),
        "selected_features": ";".join(all_feats),
    })

    diag_frames.append(diag_all.head(k * 3).copy())

    cache_df = pd.DataFrame(feature_rows)
    cache_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_precomputed_feature_sets.csv")
    diag_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_all_tcga_feature_ranking_diagnostic.csv")

    cache_df.to_csv(cache_path, index=False)
    pd.concat(diag_frames, axis=0, ignore_index=True).to_csv(diag_path, index=False)

    print(f"Saved feature cache: {cache_path}")
    print(f"Saved feature ranking diagnostic: {diag_path}")

    return feature_sets


def get_feature_sets(X_tcga, y_tcga, cv_splits):
    for path in CONF["FEATURE_CACHE_CANDIDATES"]:
        cached = load_feature_sets_from_cache(path, cv_splits)
        if cached is not None:
            return cached

    return compute_feature_sets(X_tcga, y_tcga, cv_splits)


# --------------------------------------------------
# 5. Thresholds and metrics
# --------------------------------------------------

def threshold_grid():
    return np.linspace(CONF["THRESHOLD_MIN"], CONF["THRESHOLD_MAX"], CONF["N_THRESHOLDS"])


def threshold_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    pred = (probs >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    balacc = 0.5 * (sens + spec)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f2 = 5.0 * precision * sens / ((4.0 * precision) + sens) if ((4.0 * precision) + sens) > 0 else 0.0

    return sens, spec, balacc, precision, f2, tn, fp, fn, tp


def select_threshold_from_probs(y_true, probs, mode):
    mode = str(mode).lower().strip()

    rows = []
    for t in threshold_grid():
        sens, spec, balacc, precision, f2, tn, fp, fn, tp = threshold_metrics(y_true, probs, t)
        imbalance = abs(sens - spec)

        if mode == "recall70_spec30":
            if sens >= 0.70 and spec >= 0.30:
                objective = (
                    3.0
                    + 0.50 * sens
                    + 0.25 * balacc
                    + 0.15 * spec
                    + 0.10 * f2
                    - 0.03 * imbalance
                )
            else:
                objective = (
                    0.52 * sens
                    + 0.20 * balacc
                    + 0.12 * spec
                    + 0.16 * f2
                    - 1.25 * max(0.0, 0.70 - sens)
                    - 0.50 * max(0.0, 0.30 - spec)
                    - 0.03 * imbalance
                )

        elif mode == "recall80_spec25":
            if sens >= 0.80 and spec >= 0.25:
                objective = (
                    3.0
                    + 0.50 * sens
                    + 0.24 * balacc
                    + 0.18 * spec
                    + 0.08 * f2
                    - 0.03 * imbalance
                )
            else:
                objective = (
                    0.52 * sens
                    + 0.20 * balacc
                    + 0.14 * spec
                    + 0.14 * f2
                    - 1.10 * max(0.0, 0.80 - sens)
                    - 0.45 * max(0.0, 0.25 - spec)
                    - 0.03 * imbalance
                )

        elif mode == "recall90_spec20":
            if sens >= 0.90 and spec >= 0.20:
                objective = (
                    3.0
                    + 0.55 * sens
                    + 0.20 * balacc
                    + 0.15 * spec
                    + 0.10 * f2
                    - 0.03 * imbalance
                )
            else:
                objective = (
                    0.55 * sens
                    + 0.18 * balacc
                    + 0.12 * spec
                    + 0.15 * f2
                    - 1.25 * max(0.0, 0.90 - sens)
                    - 0.35 * max(0.0, 0.20 - spec)
                    - 0.03 * imbalance
                )

        elif mode == "youden":
            objective = sens + spec - 1.0 - 0.02 * imbalance

        else:
            raise ValueError(f"Unknown threshold mode: {mode}")

        rows.append((
            objective,
            sens,
            balacc,
            spec,
            -abs(float(t) - 0.50),
            float(t),
        ))

    best = max(rows, key=lambda x: x)
    return float(best[-1])


def compute_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    pred = (probs >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    minss = min(sens, spec)

    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "min_sens_spec": float(minss),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, probs)),
        "threshold": float(threshold),
        "n_total": int(len(y_true)),
        "n_correct": int((pred == y_true).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# --------------------------------------------------
# 6. Model and candidate generation
# --------------------------------------------------

def clean_xgb_params(params):
    params = dict(params)

    remove = [
        "random_state",
        "n_jobs",
        "objective",
        "eval_metric",
        "tree_method",
        "verbosity",
        "use_label_encoder",
        "enable_categorical",
    ]

    for key in remove:
        params.pop(key, None)

    return params


def canonicalize_xgb_params(params):
    p = clean_xgb_params(params)

    out = {
        "max_depth": int(float(p.get("max_depth", 4))),
        "learning_rate": float(p.get("learning_rate", 0.035)),
        "n_estimators": int(float(p.get("n_estimators", 90))),
        "subsample": float(p.get("subsample", 0.65)),
        "colsample_bytree": float(p.get("colsample_bytree", 0.65)),
        "min_child_weight": float(p.get("min_child_weight", 1.0)),
        "gamma": float(p.get("gamma", CONF["FIXED_GAMMA"])),
        "reg_alpha": float(p.get("reg_alpha", 0.35)),
        "reg_lambda": float(p.get("reg_lambda", 5.0)),
        "scale_pos_weight": float(p.get("scale_pos_weight", 2.0)),
    }

    return out


def make_model(params, seed):
    p = clean_xgb_params(params)
    p.update({
        "random_state": int(seed),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": int(CONF["MODEL_THREADS"]),
        "tree_method": CONF["TREE_METHOD"],
        "verbosity": 0,
    })
    return XGBClassifier(**p)


def score_16t_anchor_tcga_only(row):
    auc = coalesce_row(row, ["tcga_cv_auc_mean", "tcga_oof_auc", "tcga_auc", "tcga_cv_auc"], 0.50)
    sd = coalesce_row(row, ["tcga_cv_auc_sd", "tcga_fold_auc_sd", "tcga_auc_sd"], 0.12)
    sens = coalesce_row(row, ["tcga_cv_sensitivity_mean", "tcga_oof_sensitivity", "tcga_sensitivity"], 0.50)
    bal = coalesce_row(row, ["tcga_cv_balacc_mean", "tcga_oof_balacc", "tcga_balacc"], 0.50)
    spec = coalesce_row(row, ["tcga_cv_specificity_mean", "tcga_oof_specificity", "tcga_specificity"], 0.50)

    stability = 1.0 - min(max(sd, 0.0) / 0.15, 1.0)
    sens_component = min(max(sens, 0.0) / 0.70, 1.0)
    auc_component = min(max(auc, 0.0) / 0.70, 1.0)
    spec_component = min(max(spec, 0.0) / 0.50, 1.0)

    overheat_penalty = 0.0
    if auc > 0.75:
        overheat_penalty += 0.75 * (auc - 0.75)
    if auc > 0.80:
        overheat_penalty += 1.50 * (auc - 0.80)

    return float(
        0.28 * auc_component
        + 0.28 * sens_component
        + 0.18 * bal
        + 0.16 * stability
        + 0.10 * spec_component
        - overheat_penalty
    )


def fallback_anchor_params():
    return [{
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
    }]


def read_16t_anchor_params():
    print("\n" + "=" * 80)
    print("Loading 16T parameter anchors using TCGA-only anchor ranking")
    print("=" * 80)

    frames = []

    for path in CONF["RANKED_16T_CANDIDATE_FILES"]:
        if not os.path.exists(path):
            print(f"16T ranked file not found, skipped: {path}")
            continue

        df = pd.read_csv(path)
        if df.empty or "params_json" not in df.columns:
            print(f"16T file unusable, skipped: {path}")
            continue

        df = df.copy()
        df["source_16t_file"] = path

        if "model" in df.columns:
            df = df[df["model"].astype(str).str.lower().eq("xgb")].copy()

        if "rank_profile" in df.columns:
            df = df[df["rank_profile"].astype(str).eq(CONF["RANK_PROFILE"])].copy()

        if "k" in df.columns:
            df = df[pd.to_numeric(df["k"], errors="coerce").fillna(-1).astype(int).eq(CONF["K"])].copy()

        if df.empty:
            continue

        df["anchor_tcga_only_score_16U"] = df.apply(score_16t_anchor_tcga_only, axis=1)
        frames.append(df)

        print(f"Loaded candidate anchors from: {path} | usable rows={len(df)}")

    if not frames:
        print("No usable 16T anchors found. Using fallback xgb_26-centered anchors.")
        return fallback_anchor_params()

    all_df = pd.concat(frames, axis=0, ignore_index=True)

    anchor_sort_keys = ["anchor_tcga_only_score_16U"]

    if "tcga_cv_sensitivity_mean" in all_df.columns:
        anchor_sort_keys.append("tcga_cv_sensitivity_mean")
    if "tcga_cv_auc_sd" in all_df.columns:
        anchor_sort_keys.append("tcga_cv_auc_sd")

    ascending = []
    for key in anchor_sort_keys:
        ascending.append(True if key.endswith("_sd") else False)

    assert_tcga_only_sort_keys(anchor_sort_keys, "16T anchor extraction")

    all_df = all_df.sort_values(anchor_sort_keys, ascending=ascending).reset_index(drop=True)

    anchors = []
    seen = set()

    for _, row in all_df.iterrows():
        params = canonicalize_xgb_params(parse_params_json(row["params_json"]))
        params["gamma"] = float(CONF["FIXED_GAMMA"])

        key = safe_json_dumps({k: v for k, v in params.items() if k != "scale_pos_weight"})
        if key in seen:
            continue

        seen.add(key)
        anchors.append(params)

        if len(anchors) >= CONF["N_TOP_16T_PARAM_ANCHORS"]:
            break

    if len(anchors) < CONF["N_TOP_16T_PARAM_ANCHORS"]:
        for p in fallback_anchor_params():
            p = canonicalize_xgb_params(p)
            p["gamma"] = float(CONF["FIXED_GAMMA"])

            key = safe_json_dumps({k: v for k, v in p.items() if k != "scale_pos_weight"})
            if key not in seen:
                anchors.append(p)
                seen.add(key)

            if len(anchors) >= CONF["N_TOP_16T_PARAM_ANCHORS"]:
                break

    anchor_audit_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_16T_anchor_tcga_only_audit.csv")
    all_df.head(50).to_csv(anchor_audit_path, index=False)

    print(f"Selected anchor parameter families: {len(anchors)}")
    print(f"Saved 16T anchor TCGA-only audit: {anchor_audit_path}")

    return anchors


def add_candidate(rows, seen, params, anchor_index, variant_label):
    params = canonicalize_xgb_params(params)
    key = safe_json_dumps(params)
    if key in seen:
        return
    seen.add(key)

    cid = clean_id(
        f"16Ugrid_anchor{anchor_index}"
        f"_{variant_label}"
        f"_k{CONF['K']}"
        f"_{CONF['RANK_PROFILE']}"
        f"_d{params['max_depth']}"
        f"_lr{params['learning_rate']}"
        f"_n{params['n_estimators']}"
        f"_l2{params['reg_lambda']}"
        f"_l1{params['reg_alpha']}"
        f"_g{params['gamma']}"
        f"_spw{params['scale_pos_weight']}"
        f"_ss{params['subsample']}"
        f"_cs{params['colsample_bytree']}"
        f"_mcw{params['min_child_weight']}"
    )

    rows.append({
        "candidate_id": cid,
        "model": "xgb",
        "rank_profile": CONF["RANK_PROFILE"],
        "k": int(CONF["K"]),
        "threshold_mode": CONF["THRESHOLD_MODE"],
        "params_json": safe_json_dumps(params),
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "min_child_weight": params["min_child_weight"],
        "gamma": params["gamma"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "scale_pos_weight": params["scale_pos_weight"],
        "variant_label": variant_label,
        "selection_rule": "TCGA-only OOF candidate ranking.",
    })


def build_candidates():
    anchors = read_16t_anchor_params()

    rows = []
    seen = set()
    max_candidates = int(CONF.get("MAX_GRID_CANDIDATES", 240))

    for i, base_params in enumerate(anchors, start=1):
        base = canonicalize_xgb_params(base_params)

        add_candidate(rows, seen, base, i, "base")

        for gamma in CONF["GAMMA_OPTIONS"]:
            for spw in CONF["SCALE_POS_WEIGHT_OPTIONS"]:
                p = dict(base)
                p["gamma"] = float(gamma)
                p["scale_pos_weight"] = float(spw)
                add_candidate(rows, seen, p, i, "gamma_spw")

        for alpha in CONF["REG_ALPHA_OPTIONS"]:
            for lamb in CONF["REG_LAMBDA_OPTIONS"]:
                p = dict(base)
                p["reg_alpha"] = float(alpha)
                p["reg_lambda"] = float(lamb)
                add_candidate(rows, seen, p, i, "reg")

        for lr in CONF["LEARNING_RATE_OPTIONS"]:
            for n_est in CONF["N_ESTIMATORS_OPTIONS"]:
                p = dict(base)
                p["learning_rate"] = float(lr)
                p["n_estimators"] = int(n_est)
                add_candidate(rows, seen, p, i, "lr_n")

        for subsample in CONF["SUBSAMPLE_OPTIONS"]:
            for colsample in CONF["COLSAMPLE_BYTREE_OPTIONS"]:
                p = dict(base)
                p["subsample"] = float(subsample)
                p["colsample_bytree"] = float(colsample)
                add_candidate(rows, seen, p, i, "sample")

        for depth in CONF["MAX_DEPTH_OPTIONS"]:
            for mcw in CONF["MIN_CHILD_WEIGHT_OPTIONS"]:
                p = dict(base)
                p["max_depth"] = int(depth)
                p["min_child_weight"] = float(mcw)
                add_candidate(rows, seen, p, i, "depth_mcw")

    rng = np.random.default_rng(int(CONF["SEED"]) + 160000)
    random_n = int(CONF.get("N_RANDOM_LOCAL_CANDIDATES", 0))
    base_pool = [canonicalize_xgb_params(p) for p in fallback_anchor_params()]

    for r in range(random_n):
        base = dict(base_pool[int(rng.integers(0, len(base_pool)))])
        base["gamma"] = float(rng.choice(CONF["GAMMA_OPTIONS"]))
        base["scale_pos_weight"] = float(rng.choice(CONF["SCALE_POS_WEIGHT_OPTIONS"]))
        base["reg_alpha"] = float(rng.choice(CONF["REG_ALPHA_OPTIONS"]))
        base["reg_lambda"] = float(rng.choice(CONF["REG_LAMBDA_OPTIONS"]))
        base["learning_rate"] = float(rng.choice(CONF["LEARNING_RATE_OPTIONS"]))
        base["n_estimators"] = int(rng.choice(CONF["N_ESTIMATORS_OPTIONS"]))
        base["subsample"] = float(rng.choice(CONF["SUBSAMPLE_OPTIONS"]))
        base["colsample_bytree"] = float(rng.choice(CONF["COLSAMPLE_BYTREE_OPTIONS"]))
        base["min_child_weight"] = float(rng.choice(CONF["MIN_CHILD_WEIGHT_OPTIONS"]))
        base["max_depth"] = int(rng.choice(CONF["MAX_DEPTH_OPTIONS"]))
        add_candidate(rows, seen, base, 900 + r, "random_local")

    candidates = pd.DataFrame(rows)

    if len(candidates) > max_candidates:
        candidates = candidates.head(max_candidates).copy()

    if int(CONF.get("MAX_GRID_CANDIDATES", 1)) == 1 and len(candidates) != 1:
        raise RuntimeError(f"Fixed-anchor run must contain exactly 1 candidate, got {len(candidates)}.")

    candidate_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_candidate_pool.csv")
    candidates.to_csv(candidate_path, index=False)

    print("\n" + "=" * 80)
    print("16U median-branch TCGA-only candidate pool")
    print("=" * 80)
    print(f"Candidates: {len(candidates)}")
    print(f"Candidate cap: {max_candidates}")
    print("Selection guardrail: TCGA-only OOF ranking.")
    print(f"Saved: {candidate_path}")

    if not candidates.empty:
        print(candidates[[
            "candidate_id",
            "variant_label",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "reg_lambda",
            "reg_alpha",
            "gamma",
            "scale_pos_weight",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
        ]].head(40).to_string(index=False))

    return candidates


# --------------------------------------------------
# 7. TCGA-only scoring
# --------------------------------------------------

def tcga_public_score(metrics, fold_auc_sd, graph_ratio_value):
    auc = metrics["auc"]
    sens = metrics["sensitivity"]
    spec = metrics["specificity"]
    bal = metrics["balanced_accuracy"]
    minss = metrics["min_sens_spec"]

    auc_component = min(max(auc, 0.0) / 0.70, 1.0)
    sens_component = min(max(sens, 0.0) / 0.70, 1.0)
    spec_component = min(max(spec, 0.0) / 0.50, 1.0)
    bal_component = min(max(bal, 0.0) / 0.65, 1.0)
    minss_component = min(max(minss, 0.0) / 0.40, 1.0)
    stability_component = 1.0 - min(max(fold_auc_sd, 0.0) / 0.15, 1.0)
    graph_component = min(max(graph_ratio_value, 0.0), 1.0)

    overheat_penalty = 0.0
    if auc > 0.75:
        overheat_penalty += 0.80 * (auc - 0.75)
    if auc > 0.82:
        overheat_penalty += 1.50 * (auc - 0.82)

    low_specificity_penalty = 0.25 * max(0.0, 0.20 - spec)
    imbalance_penalty = 0.04 * abs(sens - spec)

    score = (
        0.25 * sens_component
        + 0.20 * auc_component
        + 0.18 * bal_component
        + 0.14 * stability_component
        + 0.09 * minss_component
        + 0.07 * spec_component
        + 0.07 * graph_component
        - overheat_penalty
        - low_specificity_penalty
        - imbalance_penalty
    )

    return float(score)


def tcga_gate(metrics, fold_auc_sd, graph_ratio_value):
    auc = metrics["auc"]
    sens = metrics["sensitivity"]
    spec = metrics["specificity"]
    bal = metrics["balanced_accuracy"]

    if auc >= 0.60 and sens >= 0.70 and spec >= 0.30 and bal >= 0.50 and fold_auc_sd <= 0.12 and graph_ratio_value >= 0.50:
        return 4, "PLATINUM_TCGA_ONLY"

    if auc >= 0.60 and sens >= 0.65 and spec >= 0.25 and bal >= 0.48 and fold_auc_sd <= 0.15:
        return 3, "GOLD_TCGA_ONLY"

    if auc >= 0.58 and sens >= 0.60 and spec >= 0.20 and bal >= 0.45:
        return 2, "SILVER_TCGA_ONLY"

    if auc >= 0.56 and sens >= 0.55:
        return 1, "BRONZE_TCGA_ONLY"

    return 0, "NO_TCGA_GATE"


# --------------------------------------------------
# 8. Candidate evaluation
# --------------------------------------------------

def fit_full_tcga_ensemble(candidate, features, X_tcga, y_tcga, freeze_seeds):
    params = parse_params_json(candidate["params_json"])

    X_train = X_tcga.reindex(columns=features, fill_value=0.0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    models = []
    train_prob_list = []

    for seed in freeze_seeds:
        model = make_model(params, int(seed))
        model.fit(X_train_s, y_tcga.values.astype(int))

        train_prob_list.append(model.predict_proba(X_train_s)[:, 1])
        models.append(model)

    train_probs = np.mean(np.column_stack(train_prob_list), axis=1)

    bundle = {
        "run_name": RUN_NAME,
        "candidate_id": candidate["candidate_id"],
        "model_type": "xgb_seed_ensemble",
        "rank_profile": CONF["RANK_PROFILE"],
        "k": int(CONF["K"]),
        "threshold_mode": CONF["THRESHOLD_MODE"],
        "features": list(features),
        "selected_features": list(features),
        "feature_order": list(features),
        "params": params,
        "scaler": scaler,
        "models": models,
        "ensemble_model_count": len(models),
        "selection_policy": "Final winner selected using TCGA-only out-of-fold metrics.",
    }

    return train_probs, bundle


def evaluate_candidate(candidate, X_tcga, y_tcga, cv_splits, feature_sets):
    cid = candidate["candidate_id"]
    params = parse_params_json(candidate["params_json"])

    y = y_tcga.values.astype(int)
    n = len(y)

    oof_sum = np.zeros(n, dtype=float)
    oof_count = np.zeros(n, dtype=float)
    fold_rows = []
    oof_fold_rows = []

    try:
        for split in cv_splits:
            fold_id = int(split["fold_id"])
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]

            features = feature_sets["cv"][fold_id]

            X_train = X_tcga.iloc[train_idx].reindex(columns=features, fill_value=0.0)
            X_test = X_tcga.iloc[test_idx].reindex(columns=features, fill_value=0.0)

            y_train = y_tcga.iloc[train_idx].values.astype(int)
            y_test = y_tcga.iloc[test_idx].values.astype(int)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = make_model(params, CONF["SEED"] + 1000 + fold_id)
            model.fit(X_train_s, y_train)

            probs = model.predict_proba(X_test_s)[:, 1]

            oof_sum[test_idx] += probs
            oof_count[test_idx] += 1.0

            for local_i, sample_idx in enumerate(test_idx):
                oof_fold_rows.append({
                    "candidate_id": cid,
                    "Patient_ID": str(X_tcga.index[int(sample_idx)]),
                    "fold_id": fold_id,
                    "Risk_Label": int(y[int(sample_idx)]),
                    "OOF_Probability_Fold": float(probs[local_i]),
                })

            fold_auc = roc_auc_score(y_test, probs)

            fold_rows.append({
                "candidate_id": cid,
                "fold_id": fold_id,
                "tcga_fold_auc": float(fold_auc),
                "n_valid": int(len(test_idx)),
                "valid_pos": int(y_test.sum()),
                "valid_neg": int((1 - y_test).sum()),
            })

        if np.any(oof_count == 0):
            raise RuntimeError(f"OOF missing predictions for {cid}")

        oof_probs = oof_sum / oof_count

        threshold = select_threshold_from_probs(
            y_tcga.values,
            oof_probs,
            CONF["THRESHOLD_MODE"],
        )

        tcga_metrics = compute_metrics(y_tcga.values, oof_probs, threshold)

        fold_df = pd.DataFrame(fold_rows)
        fold_auc_mean = float(fold_df["tcga_fold_auc"].mean())
        fold_auc_sd = float(fold_df["tcga_fold_auc"].std(ddof=1)) if len(fold_df) > 1 else 0.0

        all_features = feature_sets["all_tcga"][CONF["RANK_PROFILE"]]
        g_ratio = graph_ratio(all_features)
        n_graph = int(sum(is_graph_feature(f) for f in all_features))

        score = tcga_public_score(tcga_metrics, fold_auc_sd, g_ratio)
        gate_priority, gate_tier = tcga_gate(tcga_metrics, fold_auc_sd, g_ratio)

        train_probs_full, _ = fit_full_tcga_ensemble(
            candidate,
            all_features,
            X_tcga,
            y_tcga,
            freeze_seeds=CONF["FREEZE_SEEDS"],
        )

        train_full_metrics = compute_metrics(y_tcga.values, train_probs_full, threshold)

        summary = {
            "status": "ok",
            "error": "",
            "traceback": "",
            "candidate_id": cid,
            "model": "xgb",
            "rank_profile": CONF["RANK_PROFILE"],
            "k": int(CONF["K"]),
            "threshold_mode": CONF["THRESHOLD_MODE"],
            "selected_threshold_oof": float(threshold),

            "max_depth": int(candidate["max_depth"]),
            "learning_rate": float(candidate["learning_rate"]),
            "n_estimators": int(candidate["n_estimators"]),
            "subsample": float(candidate["subsample"]),
            "colsample_bytree": float(candidate["colsample_bytree"]),
            "min_child_weight": float(candidate["min_child_weight"]),
            "gamma": float(candidate["gamma"]),
            "reg_alpha": float(candidate["reg_alpha"]),
            "reg_lambda": float(candidate["reg_lambda"]),
            "scale_pos_weight": float(candidate["scale_pos_weight"]),

            "tcga_oof_auc": tcga_metrics["auc"],
            "tcga_oof_auprc": tcga_metrics["auprc"],
            "tcga_oof_balacc": tcga_metrics["balanced_accuracy"],
            "tcga_oof_sensitivity": tcga_metrics["sensitivity"],
            "tcga_oof_specificity": tcga_metrics["specificity"],
            "tcga_oof_min_sens_spec": tcga_metrics["min_sens_spec"],
            "tcga_oof_mcc": tcga_metrics["mcc"],
            "tcga_oof_brier": tcga_metrics["brier"],
            "tcga_oof_tn": tcga_metrics["tn"],
            "tcga_oof_fp": tcga_metrics["fp"],
            "tcga_oof_fn": tcga_metrics["fn"],
            "tcga_oof_tp": tcga_metrics["tp"],
            "tcga_fold_auc_mean": fold_auc_mean,
            "tcga_fold_auc_sd": fold_auc_sd,

            "frozen_train_auc_diagnostic": train_full_metrics["auc"],
            "frozen_train_balacc_diagnostic": train_full_metrics["balanced_accuracy"],
            "frozen_train_sensitivity_diagnostic": train_full_metrics["sensitivity"],
            "frozen_train_specificity_diagnostic": train_full_metrics["specificity"],

            "n_graph_features": n_graph,
            "graph_ratio": g_ratio,
            "selected_features_all_tcga": ";".join(all_features),

            "gate_priority": int(gate_priority),
            "gate_tier": gate_tier,
            "tcga_public_score": score,

            "params_json": candidate["params_json"],
        }

        for r in fold_rows:
            r["threshold_mode"] = CONF["THRESHOLD_MODE"]

        oof_prediction_df = pd.DataFrame({
            "candidate_id": cid,
            "Patient_ID": X_tcga.index.astype(str),
            "Risk_Label": y.astype(int),
            "OOF_Probability": oof_probs.astype(float),
            "OOF_Count": oof_count.astype(int),
            "Selected_Threshold_OOF": float(threshold),
            "OOF_Predicted_Label": (oof_probs >= float(threshold)).astype(int),
            "threshold_mode": CONF["THRESHOLD_MODE"],
        })

        oof_fold_df = pd.DataFrame(oof_fold_rows)
        if not oof_fold_df.empty:
            oof_fold_df["threshold_mode"] = CONF["THRESHOLD_MODE"]

        return summary, pd.DataFrame(fold_rows), oof_prediction_df

    except Exception as exc:
        tb = traceback.format_exc()

        print("\n" + "=" * 80, flush=True)
        print(f"FAILED candidate: {cid}", flush=True)
        print(f"Error: {exc}", flush=True)
        print("=" * 80, flush=True)

        summary = {
            "status": "failed",
            "error": str(exc),
            "traceback": tb,
            "candidate_id": cid,
            "params_json": candidate.get("params_json", ""),
        }

        return summary, pd.DataFrame(), pd.DataFrame()


# --------------------------------------------------
# 9. Run and freeze
# --------------------------------------------------

def run_candidates(candidates, X_tcga, y_tcga, cv_splits, feature_sets):
    print("\n" + "=" * 80)
    print("Running 16U TCGA-only model selection")
    print("=" * 80)
    print(f"Candidates: {len(candidates)}")
    print(f"Parallel jobs: {CONF['N_JOBS']}")
    print("Final winner selection uses TCGA-only out-of-fold metrics.")
    print("=" * 80)

    results = Parallel(
        n_jobs=CONF["N_JOBS"],
        backend=CONF["JOBLIB_BACKEND"],
        verbose=10,
    )(
        delayed(evaluate_candidate)(
            row,
            X_tcga,
            y_tcga,
            cv_splits,
            feature_sets,
        )
        for row in candidates.to_dict(orient="records")
    )

    summaries = []
    folds = []
    oof_tables = []

    for summary, fold_df, oof_df in results:
        summaries.append(summary)
        if fold_df is not None and not fold_df.empty:
            folds.append(fold_df)
        if oof_df is not None and not oof_df.empty:
            oof_tables.append(oof_df)

    summary_df = pd.DataFrame(summaries)
    fold_df = pd.concat(folds, axis=0, ignore_index=True) if folds else pd.DataFrame()
    oof_df = pd.concat(oof_tables, axis=0, ignore_index=True) if oof_tables else pd.DataFrame()

    return summary_df, fold_df, oof_df


def choose_tcga_only_winner(results_df):
    ok = results_df[results_df["status"].eq("ok")].copy()

    if ok.empty:
        failed_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_failed_candidates_debug.csv")
        results_df.to_csv(failed_path, index=False)

        print("\n" + "=" * 80)
        print("No successful 16U candidates.")
        print("=" * 80)
        print(f"Saved failed-candidate debug file: {failed_path}")

        if "error" in results_df.columns:
            print("\nFirst errors:")
            print(results_df[["candidate_id", "error"]].head(20).to_string(index=False))

        raise RuntimeError("No successful 16U candidates. See failed_candidates_debug.csv.")

    sort_keys = [
        "gate_priority",
        "tcga_public_score",
        "tcga_oof_sensitivity",
        "tcga_fold_auc_sd",
        "tcga_oof_balacc",
        "tcga_oof_auc",
    ]

    ascending = [False, False, False, True, False, False]

    assert_tcga_only_sort_keys(sort_keys, "16U final winner selection")

    ranked = ok.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)
    winner = ranked.iloc[0]

    return winner, ranked


def freeze_final_model(winner, candidates, feature_sets, X_tcga, y_tcga, tcga_path):
    cid = str(winner["candidate_id"])
    match = candidates[candidates["candidate_id"].astype(str).eq(cid)]

    if match.empty:
        raise RuntimeError(f"Winner not found in candidate pool: {cid}")

    candidate = match.iloc[0].to_dict()
    features = feature_sets["all_tcga"][CONF["RANK_PROFILE"]]
    threshold = float(winner["selected_threshold_oof"])

    train_probs, bundle = fit_full_tcga_ensemble(
        candidate,
        features,
        X_tcga,
        y_tcga,
        freeze_seeds=CONF["FREEZE_SEEDS"],
    )

    train_metrics = compute_metrics(y_tcga.values, train_probs, threshold)

    bundle["winner_label"] = "16U_Final_Locked_TCGA_Model"
    bundle["locked_threshold"] = threshold
    bundle["ensemble_threshold"] = threshold
    bundle["threshold_source"] = f"TCGA OOF probabilities using {CONF['THRESHOLD_MODE']}."
    bundle["is_final_selection_tcga_only"] = True
    bundle["feature_locking"] = f"Exact all-TCGA RF-Gini K={CONF['K']} feature list stored in this joblib."

    model_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_Model.joblib")
    metadata_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_Model_metadata.json")
    train_pred_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_training_predictions.csv")
    features_csv_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_Features.csv")
    features_json_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_Features.json")

    train_pred = pd.DataFrame({
        "Patient_ID": X_tcga.index.astype(str),
        "Risk_Label": y_tcga.values.astype(int),
        "Predicted_Probability": train_probs,
        "Locked_Threshold": threshold,
        "Predicted_Label": (train_probs >= threshold).astype(int),
    })

    features_df = pd.DataFrame({
        "feature_rank": np.arange(1, len(features) + 1),
        "feature": features,
        "is_graph_feature": [is_graph_feature(f) for f in features],
    })

    metadata = {
        "run_name": RUN_NAME,
        "winner_label": "16U_Final_Locked_TCGA_Model",
        "candidate_id": cid,
        "selection_policy": "Strict TCGA-only winner selection using OOF metrics.",
        "is_final_selection_tcga_only": True,
        "threshold_mode": CONF["THRESHOLD_MODE"],
        "locked_threshold": threshold,
        "threshold_source": f"TCGA OOF probabilities using {CONF['THRESHOLD_MODE']}.",
        "rank_profile": CONF["RANK_PROFILE"],
        "k": int(CONF["K"]),
        "n_features": len(features),
        "n_graph_features": int(sum(is_graph_feature(f) for f in features)),
        "graph_ratio": graph_ratio(features),
        "features": features,
        "params": parse_params_json(candidate["params_json"]),
        "freeze_seeds": CONF["FREEZE_SEEDS"],
        "winner_tcga_oof_summary": to_builtin(dict(winner)),
        "tcga_training_metrics_locked_seed_ensemble": train_metrics,
        "tcga_matrix_path": tcga_path,
        "guardrail": "The final model was selected using TCGA-only OOF metrics.",
    }

    dump(bundle, model_path)
    train_pred.to_csv(train_pred_path, index=False)
    features_df.to_csv(features_csv_path, index=False)

    with open(features_json_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(features), f, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(metadata), f, indent=2)

    print("\n" + "=" * 80)
    print("Frozen 16U final model saved")
    print("=" * 80)
    print(f"- model: {model_path}")
    print(f"- metadata: {metadata_path}")
    print(f"- locked features CSV: {features_csv_path}")
    print(f"- locked features JSON: {features_json_path}")
    print(f"- TCGA training predictions: {train_pred_path}")

    return metadata


# --------------------------------------------------
# 10. Main
# --------------------------------------------------

def preflight_config_checks():
    if int(CONF["EXPECTED_N_TCGA"]) != 147 or int(CONF["EXPECTED_LOW_TCGA"]) != 73 or int(CONF["EXPECTED_HIGH_TCGA"]) != 74:
        raise ValueError("TCGA expected counts must be median branch: N=147, low=73, high=74.")

    forbidden_files = CONF.get("RANKED_16T_CANDIDATE_FILES", [])
    if forbidden_files:
        raise ValueError(
            "For this strict median refreeze script, RANKED_16T_CANDIDATE_FILES must remain empty. "
            "Do not import old anchored-label 16T/16g candidate ranks."
        )

    if int(CONF.get("K")) != 100:
        raise ValueError("Final fixed-anchor script must use K=100.")

    if CONF.get("RANK_PROFILE") != "transport_s034_g035":
        raise ValueError("Final fixed-anchor script must use RANK_PROFILE='transport_s034_g035'.")

    if float(CONF.get("GRAPH_BONUS")) != 0.035:
        raise ValueError("Final fixed-anchor script must use GRAPH_BONUS=0.035.")

    if CONF.get("THRESHOLD_MODE") != "recall80_spec25":
        raise ValueError("Final fixed-anchor script must use THRESHOLD_MODE='recall80_spec25'.")

    if int(CONF.get("MAX_GRID_CANDIDATES", 0)) != 1:
        raise ValueError("Final fixed-anchor script must be locked to exactly one candidate.")

    if int(CONF.get("N_RANDOM_LOCAL_CANDIDATES", -1)) != 0:
        raise ValueError("Final fixed-anchor script must have zero random local candidates.")


def main():
    preflight_config_checks()

    print("\n" + "=" * 80)
    print("16U TCGA-only median-OS K100 model locking")
    print("=" * 80)
    print(f"RUN_NAME: {RUN_NAME}")
    print(f"K: {CONF['K']}")
    print(f"Rank profile: {CONF['RANK_PROFILE']}")
    print(f"Graph bonus: {CONF['GRAPH_BONUS']}")
    print(f"Threshold mode: {CONF['THRESHOLD_MODE']}")
    print("=" * 80)

    X_tcga, y_tcga, tcga_path = load_data()

    cv_splits = build_cv_splits(X_tcga, y_tcga)
    feature_sets = get_feature_sets(X_tcga, y_tcga, cv_splits)

    all_features = feature_sets["all_tcga"][CONF["RANK_PROFILE"]]
    n_graph_all = int(sum(is_graph_feature(f) for f in all_features))

    print("\n" + "=" * 80)
    print("All-TCGA locked feature shell audit")
    print("=" * 80)
    print(f"Features: {len(all_features)}")
    print(f"Graph/WPPI-expanded features: {n_graph_all}")
    print(f"Graph ratio: {graph_ratio(all_features):.3f}")

    if int(CONF["MIN_GRAPH_FEATURES_ALL_TCGA"]) > 0 and n_graph_all < int(CONF["MIN_GRAPH_FEATURES_ALL_TCGA"]):
        raise RuntimeError(
            f"Configured graph-density audit failed: {n_graph_all}/{CONF['K']} graph features. "
            "This check should only be used if pre-specified before final model locking."
        )

    candidates = build_candidates()

    if candidates is None or candidates.empty:
        raise RuntimeError("Candidate pool is empty. Refreeze cannot proceed.")

    results_df, folds_df, oof_predictions_df = run_candidates(
        candidates,
        X_tcga,
        y_tcga,
        cv_splits,
        feature_sets,
    )

    raw_debug_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_raw_results_debug.csv")
    results_df.to_csv(raw_debug_path, index=False)

    print("\n" + "=" * 80)
    print("16U RAW STATUS COUNTS")
    print("=" * 80)
    print(results_df["status"].value_counts(dropna=False).to_string())
    print(f"Saved raw debug results: {raw_debug_path}")

    if "error" in results_df.columns:
        failed = results_df[results_df["status"].astype(str).eq("failed")].copy()
        if not failed.empty:
            print("\nFirst failed errors:")
            print(failed[["candidate_id", "error"]].head(10).to_string(index=False))

    winner, ranked_tcga = choose_tcga_only_winner(results_df)

    ranked_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_ranked_tcga_only.csv")
    folds_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_tcga_cv_folds.csv")
    metadata_run_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_run_metadata.json")
    all_candidate_oof_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_all_candidate_oof_predictions.csv")
    winner_oof_path = os.path.join(CONF["OUT_DIR"], f"{RUN_NAME}_winner_oof_predictions.csv")
    locked_oof_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_oof_predictions.csv")

    ranked_tcga.to_csv(ranked_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    if oof_predictions_df is None or oof_predictions_df.empty:
        raise RuntimeError("No OOF prediction table was generated. Cannot export TCGA OOF probabilities.")

    oof_predictions_df.to_csv(all_candidate_oof_path, index=False)
    winner_oof_df = oof_predictions_df[
        oof_predictions_df["candidate_id"].astype(str).eq(str(winner["candidate_id"]))
    ].copy()

    if len(winner_oof_df) != CONF["EXPECTED_N_TCGA"]:
        raise RuntimeError(
            f"Winner OOF table expected N={CONF['EXPECTED_N_TCGA']}, got {len(winner_oof_df)}"
        )

    winner_oof_df.to_csv(winner_oof_path, index=False)
    if not CONF.get("EXPORT_OOF_ONLY", False):
        winner_oof_df.to_csv(locked_oof_path, index=False)

    show_cols = [
        "candidate_id",
        "gate_priority",
        "gate_tier",
        "tcga_public_score",
        "selected_threshold_oof",
        "tcga_oof_auc",
        "tcga_fold_auc_sd",
        "tcga_oof_balacc",
        "tcga_oof_sensitivity",
        "tcga_oof_specificity",
        "tcga_oof_min_sens_spec",
        "graph_ratio",
        "n_graph_features",
        "max_depth",
        "learning_rate",
        "n_estimators",
        "reg_lambda",
        "reg_alpha",
        "gamma",
        "scale_pos_weight",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
    ]

    show_cols = [c for c in show_cols if c in ranked_tcga.columns]

    print("\n" + "=" * 80)
    print("TOP 16U TCGA-ONLY RANKING")
    print("=" * 80)
    print(ranked_tcga.head(20)[show_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("16U FINAL WINNER")
    print("=" * 80)
    print("Selected using TCGA-only sort keys.")
    print(winner[show_cols].to_string())

    if CONF.get("EXPORT_OOF_ONLY", False):
        print("\n" + "=" * 80)
        print("OOF export-only mode enabled")
        print("=" * 80)
        print("The existing locked 16U joblib/metadata/features were NOT overwritten.")
        print(f"- winner OOF predictions: {winner_oof_path}")
        print(f"- locked OOF predictions: {locked_oof_path}")

        existing_metadata_path = os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_Model_metadata.json")
        metadata = {
            "export_oof_only": True,
            "existing_locked_metadata_checked": False,
            "existing_locked_candidate_id": None,
            "rerun_winner_candidate_id": str(winner["candidate_id"]),
        }
        if os.path.exists(existing_metadata_path):
            try:
                with open(existing_metadata_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
                existing_cid = str(existing_meta.get("candidate_id", ""))
                metadata["existing_locked_metadata_checked"] = True
                metadata["existing_locked_candidate_id"] = existing_cid
                metadata["matches_existing_locked_candidate"] = existing_cid == str(winner["candidate_id"])
                if existing_cid != str(winner["candidate_id"]):
                    print("WARNING: rerun winner differs from existing locked metadata candidate_id.")
                    print(f"existing: {existing_cid}")
                    print(f"rerun:    {winner['candidate_id']}")
            except Exception as exc:
                metadata["existing_locked_metadata_check_error"] = str(exc)
    else:
        metadata = freeze_final_model(
            winner,
            candidates,
            feature_sets,
            X_tcga,
            y_tcga,
            tcga_path,
        )

    run_metadata = {
        "run_name": RUN_NAME,
        "selection_policy": "Strict TCGA-only model selection.",
        "is_final_selection_tcga_only": True,
        "candidate_count": int(len(candidates)),
        "winner_candidate_id": str(winner["candidate_id"]),
        "winner_gate_tier": str(winner["gate_tier"]),
        "winner_tcga_oof_auc": safe_float(winner["tcga_oof_auc"]),
        "winner_tcga_oof_sensitivity": safe_float(winner["tcga_oof_sensitivity"]),
        "winner_tcga_oof_specificity": safe_float(winner["tcga_oof_specificity"]),
        "winner_tcga_fold_auc_sd": safe_float(winner["tcga_fold_auc_sd"]),
        "winner_threshold": safe_float(winner["selected_threshold_oof"]),
        "outputs": {
            "ranked_tcga_only": ranked_path,
            "tcga_cv_folds": folds_path,
            "all_candidate_oof_predictions": all_candidate_oof_path,
            "winner_oof_predictions": winner_oof_path,
            "locked_tcga_oof_predictions": locked_oof_path,
            "raw_results_debug": raw_debug_path,
            "final_model_joblib": os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_Model.joblib"),
            "final_model_metadata": os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_TCGA_Model_metadata.json"),
            "locked_features_csv": os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_Features.csv"),
            "locked_features_json": os.path.join(CONF["OUT_DIR"], "16U_Final_Locked_Features.json"),
        },
        "locked_model_metadata_summary": metadata,
    }

    with open(metadata_run_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(run_metadata), f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL 16U OUTPUTS")
    print("=" * 80)
    print(f"- ranked_tcga_only: {ranked_path}")
    print(f"- tcga_cv_folds: {folds_path}")
    print(f"- all_candidate_oof_predictions: {all_candidate_oof_path}")
    print(f"- winner_oof_predictions: {winner_oof_path}")
    print(f"- locked_tcga_oof_predictions: {locked_oof_path}")
    print(f"- raw_results_debug: {raw_debug_path}")
    print(f"- run_metadata: {metadata_run_path}")
    print(f"- final_model_joblib: {os.path.join(CONF['OUT_DIR'], '16U_Final_Locked_TCGA_Model.joblib')}")

    print("\nNEXT STEP:")
    print(f"Review the locked K{CONF['K']} features and OOF predictions.")
    print("Proceed to the median-branch ablation only if the TCGA-only locked model is accepted on methodological grounds.")
    print("Do not alter features, params, threshold, threshold mode, or winner after locking.")


if __name__ == "__main__":
    main()

