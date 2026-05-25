"""
GIBD analysis workflow — CGGA LIME local explanations

This script generates post-lock LIME explanations for representative CGGA
external cases using the frozen GIBD-XGBoost K100 model. It uses TCGA as the
background data source for local explanation and reports selected TP, FP, TN,
and FN examples from the external cohort.

Analysis guardrails:
- The locked model, locked feature set, and locked threshold are not changed.
- LIME is used only for local post hoc explanation of already frozen predictions.
- LIME outputs are not used for biomarker selection, model revision, or threshold tuning.
- LIME intervals are local model-explanation intervals, not biological expression cutoffs.

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
from joblib import load

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import lime
    import lime.lime_tabular
except Exception as exc:
    raise ImportError(
        "The 'lime' package is required. Install it with: pip install lime"
    ) from exc

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation", "Explainability_LIME_MedianOS_K100_CGGA"),

    # Final frozen model artifacts.
    "LOCKED_JOBLIB": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_Model.joblib"),
    "LOCKED_METADATA": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_TCGA_Model_metadata.json"),
    "LOCKED_FEATURES": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_Features.csv"),

    # Matrices.
    "TCGA_MATRIX": os.path.join("Data", "Revision_Ablation", "tcga_weighted_self_graph_cache.csv"),
    "CGGA_MATRIX": os.path.join("Data", "Revision_Ablation", "cgga_weighted_self_graph_cache.csv"),

    # Labels.
    "TCGA_LABELS": os.path.join("Data", "TCGA_survival_labels_with_os_event.csv"),
    "CGGA_LABELS": os.path.join("Data", "Revision_Ablation", "cgga_labels_tcga_median357_with_os_event.csv"),

    # Optional gene map.
    "GENE_MAP": os.path.join("Data", "gene_type_map.csv"),

    # Median-OS K100 final-branch integrity checks.
    "EXPECTED_N_FEATURES": 100,
    "EXPECTED_N_GRAPH_FEATURES": 65,
    "EXPECTED_LOCKED_THRESHOLD": 0.53,
    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,
    "EXPECTED_N_CGGA": 131,
    "EXPECTED_LOW_CGGA": 85,
    "EXPECTED_HIGH_CGGA": 46,

    # LIME settings.
    "NUM_FEATURES": 15,
    "NUM_SAMPLES": 5000,
    "RANDOM_STATE": 42,

    # Locked threshold. Metadata value is preferred if available.
    "DEFAULT_LOCKED_THRESHOLD": 0.53,

    # Case-selection policy.
    # TP/TN = representative central correct cases.
    # FP/FN = near-threshold errors, because these are clinically informative.
    "CASE_TYPES": ["TP", "FP", "TN", "FN"],

    # Figure dimensions.
    "WIDTH_FULL_PAGE_INCH": 170 / 25.4,
    "HEIGHT_INCH": 5.3,
    "DPI": 300,
}

os.makedirs(CONF["OUT_DIR"], exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


# --------------------------------------------------
# 2. General helpers
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


def normalize_feature_name(name):
    text = str(name)
    text = re.sub(r"^(ENSG\d+)\.\d+(_.*)?$", r"\1\2", text)
    return text


def normalize_id(x):
    if pd.isna(x):
        return ""
    return str(x).strip().replace('"', '').replace("'", "")


def extract_base_gene_id(feature_name):
    text = normalize_feature_name(str(feature_name))
    suffix_patterns = [
        r"_wppi_self\d*.*$",
        r"_wppi_neighbor\d*.*$",
        r"_nbr_mean.*$",
        r"_neighbor_mean.*$",
        r"_network_mean.*$",
        r"_graph_mean.*$",
    ]

    out = text
    for pat in suffix_patterns:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)

    return out


def get_wppi_self_suffix(feature_name):
    match = re.search(r"_wppi_self(\d+)", str(feature_name), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def get_wppi_neighbor_suffix(feature_name):
    match = re.search(r"_wppi_neighbor(\d+)", str(feature_name), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def is_graph_feature(feature_name):
    lower = str(feature_name).lower()
    return any(token in lower for token in [
        "_wppi_self",
        "_wppi_neighbor",
        "_nbr_mean",
        "_neighbor_mean",
        "_network_mean",
        "_graph_mean",
    ])


def feature_family(feature_name):
    lower = str(feature_name).lower()
    if "_wppi_self" in lower:
        return "WPPI-self"
    if "_wppi_neighbor" in lower:
        return "WPPI-neighbor"
    if "_nbr_mean" in lower or "_neighbor_mean" in lower:
        return "NBR/neighbor"
    if "_network_mean" in lower or "_graph_mean" in lower:
        return "network"
    return "raw"


def load_gene_symbol_map(path):
    if not os.path.exists(path):
        print(f"Gene map not found. Using Ensembl/base IDs: {path}")
        return {}

    gene_map = pd.read_csv(path)

    id_col = None
    for c in ["clean_gene_id", "gene_id", "ensembl_gene_id", "Gene_ID", "gene"]:
        if c in gene_map.columns:
            id_col = c
            break

    sym_col = None
    for c in ["gene_name", "gene_symbol", "symbol", "external_gene_name", "Gene_Name"]:
        if c in gene_map.columns:
            sym_col = c
            break

    if id_col is None or sym_col is None:
        print(f"Could not identify gene map columns in {path}. Using IDs.")
        return {}

    gene_map[id_col] = gene_map[id_col].astype(str).map(normalize_feature_name)
    gene_map[sym_col] = gene_map[sym_col].astype(str)

    out = dict(zip(gene_map[id_col], gene_map[sym_col]))
    print(f"Loaded gene symbol map: {len(out)} entries.")
    return out


def feature_display_name(feature_name, gene_symbol_map):
    """
    Publication-safe display names.

    Important:
    - WPPI-self is not relabeled as Neighbor/NBR.
    - WPPI-self25/50/75 suffix is retained to avoid ambiguous duplicate labels.
    """
    base = extract_base_gene_id(feature_name)
    symbol = gene_symbol_map.get(base, base)
    lower = str(feature_name).lower()

    if "_wppi_self" in lower:
        suffix = get_wppi_self_suffix(feature_name)
        return f"{symbol} (WPPI-self{suffix})" if suffix else f"{symbol} (WPPI-self)"

    if "_wppi_neighbor" in lower:
        suffix = get_wppi_neighbor_suffix(feature_name)
        return f"{symbol} (WPPI-neighbor{suffix})" if suffix else f"{symbol} (WPPI-neighbor)"

    if "_nbr_mean" in lower or "_neighbor_mean" in lower:
        return f"{symbol} (NBR-mean)"

    if "_network_mean" in lower or "_graph_mean" in lower:
        return f"{symbol} (network)"

    return symbol


def make_unique(names):
    counts = {}
    unique = []
    for n in names:
        counts[n] = counts.get(n, 0) + 1
        if counts[n] == 1:
            unique.append(n)
        else:
            unique.append(f"{n} [duplicate {counts[n]}]")
    return unique


def load_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found: {path}")

    X = pd.read_csv(path, index_col=0)
    X.index = X.index.astype(str).map(normalize_id)
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

    y.index = y.index.astype(str).map(normalize_id)

    if "Risk_Label" not in y.columns:
        raise ValueError(f"Risk_Label missing from {path}")

    y["Risk_Label"] = pd.to_numeric(y["Risk_Label"], errors="coerce")
    y = y.dropna(subset=["Risk_Label"]).copy()
    y["Risk_Label"] = y["Risk_Label"].astype(int)
    return y


def assert_label_counts(label_df, cohort_name, expected_n, expected_low, expected_high):
    if len(label_df) != int(expected_n):
        raise ValueError(f"{cohort_name}: expected N={expected_n}, got {len(label_df)}")
    counts = label_df["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))
    if low != int(expected_low) or high != int(expected_high):
        raise ValueError(
            f"{cohort_name}: expected low={expected_low}, high={expected_high}; "
            f"got low={low}, high={high}"
        )


# --------------------------------------------------
# 3. Load final frozen model
# --------------------------------------------------

def load_locked_bundle():
    for path in [CONF["LOCKED_JOBLIB"], CONF["LOCKED_METADATA"], CONF["LOCKED_FEATURES"]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing locked artifact: {path}")

    bundle = load(CONF["LOCKED_JOBLIB"])

    with open(CONF["LOCKED_METADATA"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    features_df = pd.read_csv(CONF["LOCKED_FEATURES"])
    if "feature" not in features_df.columns:
        raise ValueError("Locked features CSV must contain a 'feature' column.")

    features = [normalize_feature_name(x) for x in features_df["feature"].astype(str).tolist()]

    if "models" in bundle and bundle["models"] is not None:
        models = list(bundle["models"])
    elif "model" in bundle and bundle["model"] is not None:
        models = [bundle["model"]]
    else:
        raise ValueError("Locked bundle contains no model(s).")

    scaler = bundle.get("scaler", None)

    threshold = metadata.get("locked_threshold", metadata.get("ensemble_threshold", None))
    if threshold is None:
        threshold = bundle.get("locked_threshold", bundle.get("ensemble_threshold", CONF["DEFAULT_LOCKED_THRESHOLD"]))
    threshold = float(threshold)

    n_graph = int(sum(is_graph_feature(f) for f in features))

    if len(features) != int(CONF["EXPECTED_N_FEATURES"]):
        raise ValueError(
            f"This script is for the final K100 branch. Expected "
            f"{CONF['EXPECTED_N_FEATURES']} locked features, got {len(features)}."
        )
    if n_graph != int(CONF["EXPECTED_N_GRAPH_FEATURES"]):
        raise ValueError(
            f"Expected {CONF['EXPECTED_N_GRAPH_FEATURES']} WPPI/graph features, got {n_graph}."
        )
    if abs(float(threshold) - float(CONF["EXPECTED_LOCKED_THRESHOLD"])) > 1e-9:
        raise ValueError(
            f"Expected locked threshold {CONF['EXPECTED_LOCKED_THRESHOLD']}, got {threshold}."
        )

    print("Loaded final frozen median-OS K100 GIBD-XGBoost bundle.")
    print(f"- models: {len(models)}")
    print(f"- locked features: {len(features)}")
    print(f"- WPPI/graph features: {n_graph}")
    print(f"- graph density: {n_graph / len(features):.3f}")
    print(f"- locked threshold: {threshold:.6f}")

    return bundle, metadata, models, scaler, features, threshold


def prepare_locked_matrix(matrix_path, label_path, features, cohort_name, id_candidates):
    X_all = load_matrix(matrix_path)
    y = load_labels(label_path, id_candidates=id_candidates)

    common = X_all.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError(f"No matching sample IDs between {matrix_path} and {label_path}.")

    X_all = X_all.loc[common].copy()
    y = y.loc[common].copy()

    X_locked = X_all.reindex(columns=features, fill_value=0.0)

    if cohort_name.upper().startswith("TCGA"):
        assert_label_counts(
            y, "TCGA median-OS LIME background labels",
            CONF["EXPECTED_N_TCGA"], CONF["EXPECTED_LOW_TCGA"], CONF["EXPECTED_HIGH_TCGA"]
        )
    elif cohort_name.upper().startswith("CGGA"):
        assert_label_counts(
            y, "CGGA357 median-OS LIME labels",
            CONF["EXPECTED_N_CGGA"], CONF["EXPECTED_LOW_CGGA"], CONF["EXPECTED_HIGH_CGGA"]
        )

    print(f"{cohort_name}: X={X_locked.shape}; labels={y['Risk_Label'].value_counts().sort_index().to_dict()}")

    return X_locked, y


def transform_for_model(X_locked, scaler):
    if scaler is not None:
        X_scaled = scaler.transform(X_locked)
    else:
        X_scaled = X_locked.values

    return pd.DataFrame(X_scaled, index=X_locked.index, columns=X_locked.columns)


def ensemble_predict_proba_scaled_array(arr, models, features):
    """
    LIME predict_fn.

    arr is already in the scaled feature space because the LIME background data
    are scaled TCGA features. We convert it to a DataFrame with original locked
    feature names so XGBoost sees the expected columns.
    """
    X_df = pd.DataFrame(arr, columns=features)

    probs = []
    for model in models:
        p = model.predict_proba(X_df)[:, 1]
        probs.append(p)

    p_mean = np.mean(np.vstack(probs), axis=0)
    p_mean = np.clip(p_mean, 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p_mean, p_mean])


# --------------------------------------------------
# 4. Case selection
# --------------------------------------------------

def compute_prediction_table(X_scaled, y, models, features, threshold, cohort):
    proba = ensemble_predict_proba_scaled_array(
        X_scaled.values,
        models=models,
        features=features,
    )[:, 1]

    out = pd.DataFrame({
        "sample_id": X_scaled.index,
        "cohort": cohort,
        "Risk_Label": y.loc[X_scaled.index, "Risk_Label"].astype(int).values,
        "predicted_probability": proba,
    }).set_index("sample_id")

    out["predicted_label"] = (out["predicted_probability"] >= threshold).astype(int)

    def case_type(row):
        if row["Risk_Label"] == 1 and row["predicted_label"] == 1:
            return "TP"
        if row["Risk_Label"] == 0 and row["predicted_label"] == 1:
            return "FP"
        if row["Risk_Label"] == 0 and row["predicted_label"] == 0:
            return "TN"
        if row["Risk_Label"] == 1 and row["predicted_label"] == 0:
            return "FN"
        return "UNK"

    out["case_type"] = out.apply(case_type, axis=1)
    out["distance_to_threshold"] = np.abs(out["predicted_probability"] - threshold)
    return out


def choose_representative_cases(pred_table, threshold):
    """
    Selection policy:
    - TP: closest to median TP probability, representative correct high-risk case.
    - TN: closest to median TN probability, representative correct low-risk case.
    - FP: closest above threshold, clinically informative borderline false alarm.
    - FN: closest below threshold, clinically informative missed high-risk case.
    """
    selected = []

    for ct in CONF["CASE_TYPES"]:
        sub = pred_table[pred_table["case_type"] == ct].copy()

        if sub.empty:
            print(f"WARNING: No {ct} cases found. Skipping.")
            continue

        if ct in ["TP", "TN"]:
            median_prob = sub["predicted_probability"].median()
            sub["selection_distance"] = np.abs(sub["predicted_probability"] - median_prob)
            policy = f"closest to median {ct} probability"
        elif ct in ["FP", "FN"]:
            sub["selection_distance"] = sub["distance_to_threshold"]
            policy = "closest to locked threshold"
        else:
            sub["selection_distance"] = sub["distance_to_threshold"]
            policy = "closest to locked threshold"

        chosen_id = sub.sort_values(["selection_distance", "predicted_probability"], ascending=[True, False]).index[0]
        chosen = pred_table.loc[chosen_id].copy()
        chosen["sample_id"] = chosen_id
        chosen["selection_policy"] = policy
        selected.append(chosen)

    selected_df = pd.DataFrame(selected).reset_index(drop=True)
    return selected_df


# --------------------------------------------------
# 5. LIME explanation and plotting
# --------------------------------------------------

def create_lime_explainer(X_tcga_scaled_display, categorical_features=None):
    if categorical_features is None:
        categorical_features = []

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_tcga_scaled_display.values,
        feature_names=X_tcga_scaled_display.columns.tolist(),
        class_names=["Low Risk", "High Risk"],
        mode="classification",
        discretize_continuous=True,
        random_state=CONF["RANDOM_STATE"],
        categorical_features=categorical_features,
        verbose=False,
    )

    return explainer


def explain_one_case(explainer, sample_id, X_scaled_original, X_scaled_display, models, features):
    row_display = X_scaled_display.loc[sample_id].values

    def predict_fn(arr_display):
        # Display and original feature order are identical; only names differ.
        return ensemble_predict_proba_scaled_array(
            arr_display,
            models=models,
            features=features,
        )

    exp = explainer.explain_instance(
        data_row=row_display,
        predict_fn=predict_fn,
        num_features=CONF["NUM_FEATURES"],
        num_samples=CONF["NUM_SAMPLES"],
        labels=(1,),
    )

    return exp


def explanation_to_dataframe(exp, sample_id, case_type, risk_label, pred_label, prob):
    rows = []
    for rank, (condition, weight) in enumerate(exp.as_list(label=1), start=1):
        rows.append({
            "sample_id": sample_id,
            "case_type": case_type,
            "Risk_Label": int(risk_label),
            "predicted_label": int(pred_label),
            "predicted_probability": float(prob),
            "rank": rank,
            "lime_condition": condition,
            "lime_weight_for_high_risk": float(weight),
            "direction": "pushes_high_risk" if weight > 0 else "pushes_low_risk",
            "abs_weight": float(abs(weight)),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def plot_lime_tornado(explanation_df, sample_id, case_type, risk_label, pred_label, prob, out_stem):
    plot_df = explanation_df.copy()
    plot_df = plot_df.sort_values("abs_weight", ascending=True)

    labels = plot_df["lime_condition"].tolist()
    weights = plot_df["lime_weight_for_high_risk"].tolist()
    colors = ["firebrick" if w > 0 else "steelblue" for w in weights]

    fig, ax = plt.subplots(figsize=(CONF["WIDTH_FULL_PAGE_INCH"], CONF["HEIGHT_INCH"] + 0.4))

    ax.barh(labels, weights, color=colors, edgecolor="black", linewidth=0.55)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", linestyle=":", alpha=0.45)

    ax.set_xlabel("Contribution to high-risk prediction (LIME weight)", fontweight="bold", labelpad=8)
    ax.set_ylabel("")

    label_name = "High-risk" if int(risk_label) == 1 else "Low-risk"
    pred_name = "High-risk" if int(pred_label) == 1 else "Low-risk"

    title = (
        f"{case_type} case | {sample_id} | "
        f"True: {label_name} | Predicted: {pred_name} | P(high-risk)={prob:.3f}"
    )
    ax.set_title(title, fontsize=9)

    legend_elements = [
        Patch(facecolor="firebrick", edgecolor="black", linewidth=0.5, label="Pushes toward high-risk"),
        Patch(facecolor="steelblue", edgecolor="black", linewidth=0.5, label="Pushes toward low-risk"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    for ext in [".png", ".pdf", ".tiff"]:
        path = out_stem + ext
        if ext == ".tiff":
            fig.savefig(path, dpi=CONF["DPI"], bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=CONF["DPI"], bbox_inches="tight")

    plt.close(fig)


def plot_composite_lime(all_expl_df, selected_cases, out_stem):
    """
    Creates a 2x2 compact composite for reviewer-facing supplementary figure.
    """
    if selected_cases.empty:
        return

    case_order = [ct for ct in ["TP", "FP", "TN", "FN"] if ct in selected_cases["case_type"].values]
    n_panels = len(case_order)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    axes = axes.ravel()

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_panels:
            ax.axis("off")
            continue

        ct = case_order[ax_idx]
        row = selected_cases[selected_cases["case_type"] == ct].iloc[0]
        sample_id = row["sample_id"]

        sub = all_expl_df[all_expl_df["sample_id"] == sample_id].copy()
        sub = sub.sort_values("abs_weight", ascending=False).head(10)
        sub = sub.sort_values("abs_weight", ascending=True)

        weights = sub["lime_weight_for_high_risk"].values
        labels = sub["lime_condition"].values
        colors = ["firebrick" if w > 0 else "steelblue" for w in weights]

        ax.barh(labels, weights, color=colors, edgecolor="black", linewidth=0.45)
        ax.axvline(0, color="black", linewidth=0.75, linestyle="--")
        ax.grid(axis="x", linestyle=":", alpha=0.35)

        true_name = "High" if int(row["Risk_Label"]) == 1 else "Low"
        pred_name = "High" if int(row["predicted_label"]) == 1 else "Low"
        ax.set_title(
            f"{ct}: {sample_id}\nTrue={true_name}, Pred={pred_name}, P={row['predicted_probability']:.3f}",
            fontsize=8,
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)

    fig.text(0.5, 0.03, "Contribution to high-risk prediction (LIME weight)", ha="center", fontsize=9)

    legend_elements = [
        Patch(facecolor="firebrick", edgecolor="black", linewidth=0.5, label="Pushes high-risk"),
        Patch(facecolor="steelblue", edgecolor="black", linewidth=0.5, label="Pushes low-risk"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in [".png", ".pdf", ".tiff"]:
        path = out_stem + ext
        if ext == ".tiff":
            fig.savefig(path, dpi=CONF["DPI"], bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=CONF["DPI"], bbox_inches="tight")

    plt.close(fig)


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

def main():
    print("=" * 80)
    print("Final external CGGA LIME local explainability for frozen median-OS K100 GIBD-XGBoost")
    print("=" * 80)
    print("No retraining, tuning, feature reselection, or threshold modification is performed.")
    print("=" * 80)

    bundle, metadata, models, scaler, features, threshold = load_locked_bundle()
    gene_symbol_map = load_gene_symbol_map(CONF["GENE_MAP"])

    display_names = [feature_display_name(f, gene_symbol_map) for f in features]
    display_names = make_unique(display_names)

    # TCGA background for LIME.
    X_tcga, y_tcga = prepare_locked_matrix(
        CONF["TCGA_MATRIX"],
        CONF["TCGA_LABELS"],
        features,
        "TCGA background",
        id_candidates=("Patient_ID",),
    )

    # CGGA external cases for patient-level explanations.
    X_cgga, y_cgga = prepare_locked_matrix(
        CONF["CGGA_MATRIX"],
        CONF["CGGA_LABELS"],
        features,
        "CGGA external",
        id_candidates=("CGGA_ID", "Patient_ID"),
    )

    X_tcga_scaled = transform_for_model(X_tcga, scaler)
    X_cgga_scaled = transform_for_model(X_cgga, scaler)

    X_tcga_scaled_display = X_tcga_scaled.copy()
    X_tcga_scaled_display.columns = display_names

    X_cgga_scaled_display = X_cgga_scaled.copy()
    X_cgga_scaled_display.columns = display_names

    pred_table = compute_prediction_table(
        X_cgga_scaled,
        y_cgga,
        models=models,
        features=features,
        threshold=threshold,
        cohort="CGGA_external_locked",
    )

    pred_table_path = os.path.join(CONF["OUT_DIR"], "lime_case_selection_pool_cgga.csv")
    pred_table.reset_index().to_csv(pred_table_path, index=False)

    print("\nCGGA external confusion by case type:")
    print(pred_table["case_type"].value_counts().to_string())

    selected_cases = choose_representative_cases(pred_table, threshold)
    selected_path = os.path.join(CONF["OUT_DIR"], "lime_selected_cases.csv")
    selected_cases.to_csv(selected_path, index=False)

    print("\nSelected LIME cases:")
    print(selected_cases[[
        "sample_id",
        "case_type",
        "Risk_Label",
        "predicted_label",
        "predicted_probability",
        "selection_policy",
    ]].to_string(index=False))

    explainer = create_lime_explainer(X_tcga_scaled_display)

    all_explanations = []

    for _, case in selected_cases.iterrows():
        sample_id = case["sample_id"]
        case_type = case["case_type"]
        y_true = int(case["Risk_Label"])
        y_pred = int(case["predicted_label"])
        prob = float(case["predicted_probability"])

        print("\n" + "-" * 80)
        print(f"Explaining {case_type} case: {sample_id} | true={y_true} pred={y_pred} prob={prob:.4f}")
        print("-" * 80)

        exp = explain_one_case(
            explainer=explainer,
            sample_id=sample_id,
            X_scaled_original=X_cgga_scaled,
            X_scaled_display=X_cgga_scaled_display,
            models=models,
            features=features,
        )

        expl_df = explanation_to_dataframe(
            exp=exp,
            sample_id=sample_id,
            case_type=case_type,
            risk_label=y_true,
            pred_label=y_pred,
            prob=prob,
        )

        all_explanations.append(expl_df)

        case_stem = os.path.join(CONF["OUT_DIR"], f"Figure_LIME_{case_type}_{sample_id}")
        safe_stem = re.sub(r"[^A-Za-z0-9_\-\\/.:]", "_", case_stem)

        plot_lime_tornado(
            explanation_df=expl_df,
            sample_id=sample_id,
            case_type=case_type,
            risk_label=y_true,
            pred_label=y_pred,
            prob=prob,
            out_stem=safe_stem,
        )

    if all_explanations:
        all_expl_df = pd.concat(all_explanations, axis=0).reset_index(drop=True)
    else:
        all_expl_df = pd.DataFrame()

    explanations_path = os.path.join(CONF["OUT_DIR"], "lime_local_explanations.csv")
    all_expl_df.to_csv(explanations_path, index=False)

    composite_stem = os.path.join(CONF["OUT_DIR"], "Figure_LIME_Composite_CGGA_TP_FP_TN_FN")
    plot_composite_lime(all_expl_df, selected_cases, composite_stem)

    audit = {
        "script": "14_lime_locked_gibd_cases_median_os_k100.py",
        "analysis_role": "Primary post-lockdown external CGGA local explanation for final frozen median-OS K100 GIBD-XGBoost model.",
        "no_retraining": True,
        "no_tuning": True,
        "no_feature_reselection": True,
        "no_threshold_change": True,
        "lime_not_used_for_biomarker_selection": True,
        "locked_joblib": CONF["LOCKED_JOBLIB"],
        "locked_metadata": CONF["LOCKED_METADATA"],
        "locked_features": CONF["LOCKED_FEATURES"],
        "locked_threshold": threshold,
        "n_locked_features": int(len(features)),
        "n_graph_features": int(sum(is_graph_feature(f) for f in features)),
        "graph_feature_density": float(sum(is_graph_feature(f) for f in features) / len(features)),
        "tcga_background_n": int(len(X_tcga_scaled)),
        "cgga_external_n": int(len(X_cgga_scaled)),
        "lime_settings": {
            "num_features": CONF["NUM_FEATURES"],
            "num_samples": CONF["NUM_SAMPLES"],
            "random_state": CONF["RANDOM_STATE"],
            "background": "Scaled TCGA locked-feature matrix",
            "explained_cohort": "Scaled CGGA locked-feature matrix",
        },
        "case_selection_policy": {
            "TP": "closest to median TP probability",
            "TN": "closest to median TN probability",
            "FP": "closest to locked threshold",
            "FN": "closest to locked threshold",
        },
        "selected_cases": selected_cases.to_dict(orient="records"),
        "outputs": {
            "case_selection_pool": pred_table_path,
            "selected_cases": selected_path,
            "local_explanations": explanations_path,
            "composite_figure": composite_stem + ".png",
        },
        "safe_interpretation": (
            "LIME provides local approximate explanations of individual model decisions. "
            "It does not validate biomarkers and was not used to select features or change the model."
        ),
    }

    audit_path = os.path.join(CONF["OUT_DIR"], "lime_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(audit), f, indent=2)

    print("\n" + "=" * 80)
    print("LIME analysis complete")
    print("=" * 80)
    print(f"Outputs directory: {CONF['OUT_DIR']}")
    print(f"- Case pool: {pred_table_path}")
    print(f"- Selected cases: {selected_path}")
    print(f"- Local explanations: {explanations_path}")
    print(f"- Audit: {audit_path}")
    print(f"- Composite figure: {composite_stem}.png")

    print("\nRecommended manuscript wording:")
    print(
        "LIME was applied post hoc to representative external CGGA cases using the final frozen "
        "GIBD-XGBoost model and locked K=100 feature set. The explanations illustrate local "
        "feature contributions for true-positive, false-positive, true-negative, and false-negative "
        "predictions. LIME was used only for local interpretation and was not used for biomarker "
        "selection, model tuning, or threshold modification."
    )


if __name__ == "__main__":
    main()
