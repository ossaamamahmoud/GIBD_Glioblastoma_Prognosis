"""
GIBD analysis workflow — SHAP interpretability

This script performs post-lock SHAP analysis for the frozen GIBD-XGBoost K100
model. It generates global feature-attribution summaries, optional model-level
interaction summaries, targeted dependence plots, and audit outputs for the
locked model.

Analysis guardrails:
- SHAP is applied after feature selection, model locking, and threshold selection.
- SHAP outputs are not used to train, tune, refit, reselect features, or change thresholds.
- Feature labels preserve WPPI-self notation and do not relabel WPPI features as
  neighbor-mean or NBR features.
- SHAP interaction outputs are interpreted as model-level dependencies, not
  biochemical interactions or causal mechanisms.

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
import shap

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation", "Explainability_SHAP_MedianOS_K100"),

    # Final frozen GIBD-XGBoost artifacts.
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

    # Median-OS K100 final branch integrity checks.
    "EXPECTED_N_FEATURES": 100,
    "EXPECTED_N_GRAPH_FEATURES": 65,
    "EXPECTED_LOCKED_THRESHOLD": 0.53,
    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,
    "EXPECTED_N_CGGA": 131,
    "EXPECTED_LOW_CGGA": 85,
    "EXPECTED_HIGH_CGGA": 46,

    # Keep TCGA as primary explainability. CGGA is optional supplementary inspection.
    "RUN_CGGA_SHAP_IF_AVAILABLE": False,

    # Interaction values can be computationally heavier but are useful for revised Figure 8.
    "RUN_INTERACTIONS": True,

    # Main manuscript figure settings.
    "TOP_N_GLOBAL": 15,
    "TOP_N_INTERACTIONS": 15,
    "TOP_N_DEPENDENCE": 4,

    # Targeted dependence pairs are selected dynamically from the current
    # K100 median-OS SHAP interaction ranking. Do not hard-code old-branch pairs.
    "TARGETED_DEPENDENCE_PAIRS": [],

    # Figure settings.
    "FIG_DPI": 300,
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
    """
    Extract the numeric WPPI-self suffix, e.g.
    ENSG..._wppi_self25 -> 25.
    """
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
    Publication display names.

    Important:
    - Do NOT relabel WPPI-self as Neighbor/NBR.
    - WPPI-self is not a simple neighbor mean. It is a self-preserving
      WPPI-weighted graph-informed feature.
    - Include numeric suffix to avoid duplicate "[2]" labels.
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
    """
    Keep only as last safety net. With explicit WPPI suffixes, this should
    rarely add [2].
    """
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
    X.index = X.index.astype(str)
    X.columns = [normalize_feature_name(c) for c in X.columns.astype(str)]
    X = X.loc[:, ~pd.Index(X.columns).duplicated()].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def load_label_table(path, id_candidates=("Patient_ID", "CGGA_ID")):
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


def save_current_figure(path_stem):
    plt.tight_layout()
    for ext in [".png", ".pdf", ".tiff"]:
        out = path_stem + ext
        if ext == ".tiff":
            plt.savefig(out, dpi=CONF["FIG_DPI"], pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(out, dpi=CONF["FIG_DPI"])
    plt.close()


def save_figure_object(fig, path_stem):
    fig.tight_layout()
    for ext in [".png", ".pdf", ".tiff"]:
        out = path_stem + ext
        if ext == ".tiff":
            fig.savefig(out, dpi=CONF["FIG_DPI"], pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(out, dpi=CONF["FIG_DPI"])
    plt.close(fig)


# --------------------------------------------------
# 3. Load final frozen model and data
# --------------------------------------------------

def load_locked_model_artifacts():
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

    threshold = metadata.get("locked_threshold", metadata.get("ensemble_threshold"))
    if threshold is None:
        threshold = bundle.get("locked_threshold", bundle.get("ensemble_threshold"))
    if threshold is None:
        raise ValueError("Could not find locked threshold.")

    if "models" in bundle and bundle["models"] is not None:
        models = list(bundle["models"])
    elif "model" in bundle and bundle["model"] is not None:
        models = [bundle["model"]]
    else:
        raise ValueError("Locked bundle contains no model(s).")

    scaler = bundle.get("scaler", None)

    n_graph = int(sum(is_graph_feature(f) for f in features))

    if len(features) != int(CONF["EXPECTED_N_FEATURES"]):
        raise ValueError(
            f"This script is for the final K100 branch. Expected "
            f"{CONF['EXPECTED_N_FEATURES']} locked features, got {len(features)}."
        )

    if n_graph != int(CONF["EXPECTED_N_GRAPH_FEATURES"]):
        raise ValueError(
            f"Expected {CONF['EXPECTED_N_GRAPH_FEATURES']} WPPI/graph features in the locked "
            f"K100 branch, got {n_graph}. Check that the correct locked feature file is loaded."
        )

    if abs(float(threshold) - float(CONF["EXPECTED_LOCKED_THRESHOLD"])) > 1e-9:
        raise ValueError(
            f"Expected locked threshold {CONF['EXPECTED_LOCKED_THRESHOLD']}, got {threshold}."
        )

    print("Loaded final frozen GIBD-XGBoost K100 median-OS model bundle.")
    print(f"- models: {len(models)}")
    print(f"- locked features: {len(features)}")
    print(f"- graph/WPPI features: {n_graph}")
    print(f"- graph density: {n_graph / len(features):.3f}")
    print(f"- locked threshold: {threshold}")

    return bundle, metadata, models, scaler, features, float(threshold)


def prepare_locked_matrix(matrix_path, label_path, features, cohort_name):
    X_all = load_matrix(matrix_path)
    y = load_label_table(label_path)

    common = X_all.index.intersection(y.index)
    X_all = X_all.loc[common].copy()
    y = y.loc[common].copy()

    X_locked = X_all.reindex(columns=features, fill_value=0.0)

    if cohort_name.upper() == "TCGA":
        assert_label_counts(
            y,
            "TCGA median-OS SHAP labels",
            CONF["EXPECTED_N_TCGA"],
            CONF["EXPECTED_LOW_TCGA"],
            CONF["EXPECTED_HIGH_TCGA"],
        )
    elif cohort_name.upper() == "CGGA":
        assert_label_counts(
            y,
            "CGGA357 median-OS SHAP labels",
            CONF["EXPECTED_N_CGGA"],
            CONF["EXPECTED_LOW_CGGA"],
            CONF["EXPECTED_HIGH_CGGA"],
        )

    print(f"{cohort_name}: X={X_locked.shape}; labels={y['Risk_Label'].value_counts().sort_index().to_dict()}")

    return X_locked, y


def transform_for_model(X_locked, scaler):
    if scaler is not None:
        X_scaled = scaler.transform(X_locked)
    else:
        X_scaled = X_locked.values

    return pd.DataFrame(X_scaled, index=X_locked.index, columns=X_locked.columns)


# --------------------------------------------------
# 4. SHAP computation
# --------------------------------------------------

def compute_shap_for_models(models, X_scaled_df):
    shap_values_list = []
    interaction_values_list = []
    expected_values = []

    for idx, model in enumerate(models, start=1):
        print(f"Computing SHAP values for model {idx}/{len(models)}...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)
        shap_values_list.append(shap_values)

        ev = explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            ev = np.asarray(ev).ravel()[-1]
        expected_values.append(float(ev))

        if CONF["RUN_INTERACTIONS"]:
            print(f"Computing SHAP interaction values for model {idx}/{len(models)}...")
            inter = explainer.shap_interaction_values(X_scaled_df)
            if isinstance(inter, list):
                inter = inter[-1]
            interaction_values_list.append(np.asarray(inter))

    shap_values_mean = np.mean(np.stack(shap_values_list, axis=0), axis=0)
    expected_value_mean = float(np.mean(expected_values))

    if CONF["RUN_INTERACTIONS"] and len(interaction_values_list) > 0:
        interaction_mean = np.mean(np.stack(interaction_values_list, axis=0), axis=0)
    else:
        interaction_mean = None

    return shap_values_mean, interaction_mean, expected_value_mean


# --------------------------------------------------
# 5. Rankings and exports
# --------------------------------------------------

def make_feature_ranking(shap_values, X_locked, display_names, gene_symbol_map, cohort):
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    std_abs = np.abs(shap_values).std(axis=0)

    rows = []
    for i, feat in enumerate(X_locked.columns):
        base = extract_base_gene_id(feat)
        symbol = gene_symbol_map.get(base, base)

        rows.append({
            "cohort": cohort,
            "rank": 0,
            "feature": feat,
            "display_name": display_names[i],
            "base_gene_id": base,
            "gene_symbol": symbol,
            "feature_family": feature_family(feat),
            "is_graph_feature": bool(is_graph_feature(feat)),
            "mean_abs_shap": float(mean_abs[i]),
            "mean_signed_shap": float(mean_signed[i]),
            "std_abs_shap": float(std_abs[i]),
        })

    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def make_gene_level_ranking(feature_ranking, cohort):
    grp = feature_ranking.groupby(["base_gene_id", "gene_symbol"], dropna=False)

    out = grp.agg(
        n_features=("feature", "count"),
        n_graph_features=("is_graph_feature", "sum"),
        total_mean_abs_shap=("mean_abs_shap", "sum"),
        max_mean_abs_shap=("mean_abs_shap", "max"),
        mean_signed_shap=("mean_signed_shap", "mean"),
    ).reset_index()

    out["graph_feature_fraction"] = out["n_graph_features"] / out["n_features"]
    out = out.sort_values("total_mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(0, "cohort", cohort)
    out.insert(1, "gene_rank", np.arange(1, len(out) + 1))
    return out


def save_shap_values(shap_values, X_locked, cohort):
    df = pd.DataFrame(shap_values, index=X_locked.index, columns=X_locked.columns)
    df.insert(0, "sample_id", df.index)
    path = os.path.join(CONF["OUT_DIR"], f"shap_values_{cohort.lower()}.csv")
    df.to_csv(path, index=False)
    return path


def make_interaction_ranking(shap_interactions, X_locked, display_names, gene_symbol_map, cohort):
    if shap_interactions is None:
        return pd.DataFrame()

    mean_abs_interaction = np.abs(shap_interactions).mean(axis=0)
    np.fill_diagonal(mean_abs_interaction, 0.0)

    rows = []
    n = mean_abs_interaction.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            f1 = X_locked.columns[i]
            f2 = X_locked.columns[j]
            b1 = extract_base_gene_id(f1)
            b2 = extract_base_gene_id(f2)

            rows.append({
                "cohort": cohort,
                "rank": 0,
                "feature_1": f1,
                "feature_2": f2,
                "display_1": display_names[i],
                "display_2": display_names[j],
                "pair_display": f"{display_names[i]} — {display_names[j]}",
                "base_gene_id_1": b1,
                "base_gene_id_2": b2,
                "gene_symbol_1": gene_symbol_map.get(b1, b1),
                "gene_symbol_2": gene_symbol_map.get(b2, b2),
                "feature_1_family": feature_family(f1),
                "feature_2_family": feature_family(f2),
                "feature_1_is_graph": bool(is_graph_feature(f1)),
                "feature_2_is_graph": bool(is_graph_feature(f2)),
                "mean_abs_interaction": float(mean_abs_interaction[i, j]),
            })

    out = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------
# 6. Plotting
# --------------------------------------------------

def plot_global_beeswarm(shap_values, X_scaled_display, cohort, path_stem):
    print(f"Plotting global SHAP beeswarm: {cohort}")

    plt.figure(figsize=(8.0, 5.8))
    shap.summary_plot(
        shap_values,
        X_scaled_display,
        plot_type="dot",
        max_display=CONF["TOP_N_GLOBAL"],
        show=False,
    )
    plt.xlabel("SHAP value (impact on high-risk prediction)")
    save_current_figure(path_stem)


def plot_global_bar(feature_ranking, cohort, path_stem, title_suffix=""):
    print(f"Plotting global SHAP bar: {cohort}")

    top = feature_ranking.head(CONF["TOP_N_GLOBAL"]).iloc[::-1].copy()

    plt.figure(figsize=(7.2, 5.4))
    plt.barh(top["display_name"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("")
    title = f"Global SHAP importance ({cohort})"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)
    save_current_figure(path_stem)


def plot_top_interactions(interaction_ranking, cohort, path_stem):
    if interaction_ranking.empty:
        return

    print(f"Plotting top SHAP interactions: {cohort}")

    top = interaction_ranking.head(CONF["TOP_N_INTERACTIONS"]).iloc[::-1].copy()

    plt.figure(figsize=(8.4, 5.8))
    plt.barh(top["pair_display"], top["mean_abs_interaction"])
    plt.xlabel("Mean |SHAP interaction value|")
    plt.ylabel("")
    plt.title(f"Exploratory SHAP interactions ({cohort})")
    save_current_figure(path_stem)


def find_feature_by_gene_and_preference(X_scaled_display, display_names, target_gene):
    """
    Find the most appropriate displayed feature for a target gene.

    Preference:
      1. Exact raw gene symbol.
      2. WPPI-self feature.
      3. Any feature starting with target_gene.
      4. Any feature containing target_gene.
    """
    target = str(target_gene).upper()

    # 1. exact raw display name.
    for name in display_names:
        if str(name).upper() == target:
            return name

    # 2. WPPI-self display name.
    for name in display_names:
        upper = str(name).upper()
        if upper.startswith(target + " ") and "WPPI-SELF" in upper:
            return name

    # 3. any display starting with gene.
    for name in display_names:
        if str(name).upper().startswith(target + " "):
            return name

    # 4. contains.
    for name in display_names:
        if target in str(name).upper():
            return name

    return None


def make_targeted_dependence_pairs(display_names, interaction_ranking):
    """
    Select targeted dependence plots from the current K100 median-OS interaction
    ranking. This avoids reusing old hard-coded biological stories from a
    superseded model.
    """
    if interaction_ranking is None or interaction_ranking.empty:
        return []

    resolved = []
    top = interaction_ranking.head(CONF["TOP_N_DEPENDENCE"]).copy()

    for _, row in top.iterrows():
        main_feature = str(row["display_1"])
        interaction_feature = str(row["display_2"])

        if main_feature not in display_names or interaction_feature not in display_names:
            continue

        resolved.append({
            "main_gene": str(row.get("gene_symbol_1", "")),
            "interaction_gene": str(row.get("gene_symbol_2", "")),
            "main_feature": main_feature,
            "interaction_feature": interaction_feature,
            "source_interaction_rank": int(row.get("rank", 0)),
            "source_mean_abs_interaction": float(row.get("mean_abs_interaction", np.nan)),
        })

    return resolved


def plot_targeted_dependence(shap_values, X_scaled_display, display_names, interaction_ranking, cohort, path_stem):
    pairs = make_targeted_dependence_pairs(display_names, interaction_ranking)

    if not pairs:
        print("No targeted dependence pairs were resolved from current interaction ranking. Skipping targeted dependence figure.")
        return [], None

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    axes = axes.ravel()

    plotted_pairs = []

    for i, ax in enumerate(axes):
        if i >= len(pairs):
            ax.axis("off")
            continue

        main_feature = pairs[i]["main_feature"]
        interaction_feature = pairs[i]["interaction_feature"]

        try:
            shap.dependence_plot(
                main_feature,
                shap_values,
                X_scaled_display,
                interaction_index=interaction_feature,
                ax=ax,
                show=False,
                dot_size=18,
                alpha=0.72,
            )
            ax.set_title(f"{main_feature} × {interaction_feature}", fontsize=9)
            ax.set_ylabel(f"SHAP value for\n{main_feature}")
            plotted_pairs.append(pairs[i])

        except Exception as exc:
            ax.axis("off")
            ax.text(
                0.05,
                0.5,
                f"Could not plot:\n{main_feature}\nvs\n{interaction_feature}\n{exc}",
                fontsize=8,
            )

    save_figure_object(fig, path_stem)
    return plotted_pairs, path_stem


# --------------------------------------------------
# 7. Cohort runner
# --------------------------------------------------

def run_shap_for_cohort(
    cohort,
    X_locked,
    y,
    models,
    scaler,
    display_names,
    gene_symbol_map,
    is_primary=True,
):
    print("\n" + "=" * 80)
    print(f"Running SHAP for {cohort}")
    print("=" * 80)

    X_scaled = transform_for_model(X_locked, scaler)
    X_display = X_scaled.copy()
    X_display.columns = display_names

    shap_values, shap_interactions, expected_value = compute_shap_for_models(models, X_scaled)

    shap_path = save_shap_values(shap_values, X_locked, cohort)

    feature_ranking = make_feature_ranking(
        shap_values=shap_values,
        X_locked=X_locked,
        display_names=display_names,
        gene_symbol_map=gene_symbol_map,
        cohort=cohort,
    )
    gene_ranking = make_gene_level_ranking(feature_ranking, cohort)

    feature_ranking_path = os.path.join(CONF["OUT_DIR"], f"shap_global_feature_ranking_{cohort.lower()}.csv")
    gene_ranking_path = os.path.join(CONF["OUT_DIR"], f"shap_global_gene_level_ranking_{cohort.lower()}.csv")

    feature_ranking.to_csv(feature_ranking_path, index=False)
    gene_ranking.to_csv(gene_ranking_path, index=False)

    interaction_ranking = pd.DataFrame()
    interaction_ranking_path = None

    if shap_interactions is not None:
        interaction_ranking = make_interaction_ranking(
            shap_interactions=shap_interactions,
            X_locked=X_locked,
            display_names=display_names,
            gene_symbol_map=gene_symbol_map,
            cohort=cohort,
        )
        interaction_ranking_path = os.path.join(CONF["OUT_DIR"], f"shap_interaction_ranking_{cohort.lower()}.csv")
        interaction_ranking.to_csv(interaction_ranking_path, index=False)

    prefix = "TCGA" if cohort.upper() == "TCGA" else cohort.upper()

    plot_global_beeswarm(
        shap_values,
        X_display,
        cohort,
        os.path.join(CONF["OUT_DIR"], f"Figure_SHAP_01_Global_Beeswarm_{prefix}"),
    )

    plot_global_bar(
        feature_ranking,
        cohort,
        os.path.join(CONF["OUT_DIR"], f"Figure_SHAP_02_Global_Bar_{prefix}"),
    )

    plotted_pairs = []
    targeted_path = None

    # Only TCGA gets main interaction/dependence figures.
    # CGGA, if enabled, remains supplementary qualitative inspection.
    if is_primary:
        if not interaction_ranking.empty:
            plot_top_interactions(
                interaction_ranking,
                cohort,
                os.path.join(CONF["OUT_DIR"], f"Figure_SHAP_03_Top_Interactions_{prefix}"),
            )

        plotted_pairs, targeted_path = plot_targeted_dependence(
            shap_values,
            X_display,
            display_names,
            interaction_ranking,
            cohort,
            os.path.join(CONF["OUT_DIR"], f"Figure_SHAP_04_Targeted_Dependence_{prefix}"),
        )

    print(f"\nTop {CONF['TOP_N_GLOBAL']} SHAP features for {cohort}:")
    print(feature_ranking.head(CONF["TOP_N_GLOBAL"])[[
        "rank",
        "display_name",
        "feature",
        "gene_symbol",
        "feature_family",
        "mean_abs_shap",
    ]].to_string(index=False))

    if not interaction_ranking.empty:
        print(f"\nTop {CONF['TOP_N_INTERACTIONS']} SHAP interactions for {cohort}:")
        print(interaction_ranking.head(CONF["TOP_N_INTERACTIONS"])[[
            "rank",
            "pair_display",
            "mean_abs_interaction",
        ]].to_string(index=False))

    return {
        "cohort": cohort,
        "n_samples": int(len(X_locked)),
        "n_features": int(X_locked.shape[1]),
        "expected_value": float(expected_value),
        "shap_values": shap_path,
        "feature_ranking": feature_ranking_path,
        "gene_ranking": gene_ranking_path,
        "interaction_ranking": interaction_ranking_path,
        "targeted_dependence_pairs": plotted_pairs,
        "targeted_dependence_figure_stem": targeted_path,
        "top_features": feature_ranking.head(20).to_dict(orient="records"),
        "top_genes": gene_ranking.head(20).to_dict(orient="records"),
        "top_interactions": interaction_ranking.head(20).to_dict(orient="records") if not interaction_ranking.empty else [],
    }


# --------------------------------------------------
# 8. Main
# --------------------------------------------------

def main():
    print("=" * 80)
    print("FINAL SHAP analysis for the final frozen median-OS K100 GIBD-XGBoost model")
    print("=" * 80)
    print("No training, tuning, feature selection, or threshold modification is performed.")
    print("Primary SHAP cohort: TCGA only. CGGA SHAP remains disabled unless explicitly enabled as qualitative inspection.")
    print("=" * 80)

    bundle, metadata, models, scaler, features, threshold = load_locked_model_artifacts()
    gene_symbol_map = load_gene_symbol_map(CONF["GENE_MAP"])

    display_names = [feature_display_name(f, gene_symbol_map) for f in features]
    display_names = make_unique(display_names)

    X_tcga, y_tcga = prepare_locked_matrix(
        CONF["TCGA_MATRIX"],
        CONF["TCGA_LABELS"],
        features,
        "TCGA",
    )

    audit = {
        "script": "13_shap_locked_gibd_model_median_os_k100_FINAL.py",
        "analysis_role": "Post-lockdown biological explainability for the final frozen median-OS K100 GIBD-XGBoost model.",
        "no_training_or_tuning_performed": True,
        "locked_joblib": CONF["LOCKED_JOBLIB"],
        "locked_metadata": CONF["LOCKED_METADATA"],
        "locked_features": CONF["LOCKED_FEATURES"],
        "locked_threshold": threshold,
        "n_locked_features": int(len(features)),
        "n_graph_features": int(sum(is_graph_feature(f) for f in features)),
        "graph_feature_density": float(sum(is_graph_feature(f) for f in features) / len(features)),
        "models_in_bundle": int(len(models)),
        "metadata_candidate_id": metadata.get("candidate_id"),
        "metadata_selection_policy": metadata.get("selection_policy"),
        "display_label_policy": {
            "wppi_self_label": "WPPI-selfXX suffix retained, e.g. WPPI-self25/50/75",
            "reason": "Avoids ambiguous duplicate labels and avoids mislabeling WPPI-self as Neighbor/NBR.",
            "neighbor_label_policy": "Only actual neighbor/NBR features are labeled as NBR-mean.",
        },
        "interaction_axis_label": "SHAP interaction value (impact on high-risk prediction)",
        "top_n_global_main": CONF["TOP_N_GLOBAL"],
        "top_n_interactions_main": CONF["TOP_N_INTERACTIONS"],
        "cgga_role_if_enabled": (
            "Post-lockdown external qualitative inspection only; not used for biomarker discovery, "
            "model selection, feature selection, or threshold adjustment."
        ),
        "cohorts": {},
    }

    tcga_result = run_shap_for_cohort(
        cohort="TCGA",
        X_locked=X_tcga,
        y=y_tcga,
        models=models,
        scaler=scaler,
        display_names=display_names,
        gene_symbol_map=gene_symbol_map,
        is_primary=True,
    )
    audit["cohorts"]["TCGA"] = tcga_result

    if CONF["RUN_CGGA_SHAP_IF_AVAILABLE"] and os.path.exists(CONF["CGGA_MATRIX"]) and os.path.exists(CONF["CGGA_LABELS"]):
        X_cgga, y_cgga = prepare_locked_matrix(
            CONF["CGGA_MATRIX"],
            CONF["CGGA_LABELS"],
            features,
            "CGGA",
        )

        cgga_result = run_shap_for_cohort(
            cohort="CGGA",
            X_locked=X_cgga,
            y=y_cgga,
            models=models,
            scaler=scaler,
            display_names=display_names,
            gene_symbol_map=gene_symbol_map,
            is_primary=False,
        )
        audit["cohorts"]["CGGA"] = cgga_result

    expected_path = os.path.join(CONF["OUT_DIR"], "shap_expected_value.json")
    with open(expected_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "expected_value_tcga": audit["cohorts"]["TCGA"]["expected_value"],
                "note": "Expected value is in the XGBoost log-odds SHAP space for the explained model.",
            },
            f,
            indent=2,
        )

    audit["expected_value_file"] = expected_path
    audit["outputs_dir"] = CONF["OUT_DIR"]

    audit_path = os.path.join(CONF["OUT_DIR"], "shap_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(audit), f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SHAP analysis complete")
    print("=" * 80)
    print(f"Outputs directory: {CONF['OUT_DIR']}")
    print(f"Audit: {audit_path}")
    print("\nRecommended manuscript wording:")
    print(
        "SHAP analysis was performed post hoc on the final frozen GIBD-XGBoost model "
        "without retraining, feature reselection, or threshold modification. Global SHAP "
        "rankings were used to interpret locked WPPI-informed biomarkers, while SHAP "
        "interaction values were treated as exploratory model-level nonlinear dependency "
        "patterns rather than validated biochemical interactions."
    )


if __name__ == "__main__":
    main()
