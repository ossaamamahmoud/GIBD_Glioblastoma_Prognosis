"""
GIBD analysis workflow — locked-signature pathway over-representation analysis.

This script performs post-lock pathway contextualization for the final frozen
GIBD-XGBoost median-OS K100 signature. It loads the locked feature list, maps
features to gene symbols, builds locked and SHAP-ranked query gene sets, and
runs local-GMT over-representation analysis using a transcriptomic/WPPI feature
background where available.

Analysis guardrails:
- The locked feature set, model parameters, and operating threshold are not changed.
- No CGGA outcome labels or external-validation metrics are used for pathway analysis.
- Enrichment results are used for biological plausibility and pathway context only.
- ORA/Enrichr-style results are supplementary to full-transcriptome TCGA GSEA.

Expected local resources include the locked K100 feature list, gene map, final
SHAP ranking outputs if available, and one or more GMT gene-set files.
"""
import os
import re
import json
import math
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import hypergeom
except Exception as exc:
    raise ImportError("scipy is required for hypergeometric ORA. Install scipy.") from exc

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation", "GSEA_LOCKED_SIGNATURE_MedianOS_K100_ORA_ONLY_v2"),

    # Locked final artifacts.
    "LOCKED_FEATURES": os.path.join("Data", "Revision_Ablation", "16U_Final_Locked_Features.csv"),
    "GENE_MAP": os.path.join("Data", "gene_type_map.csv"),

    # Median-OS K100 final-branch integrity checks.
    "EXPECTED_N_FEATURES": 100,
    "EXPECTED_N_GRAPH_FEATURES": 65,
    "EXPECTED_LOCKED_THRESHOLD": 0.53,

    # Full TCGA feature matrix used only to derive a transcriptomic/WPPI background universe.
    # This is not used for model fitting here.
    "TCGA_FEATURE_BACKGROUND_MATRIX": os.path.join(
        "Data", "Revision_Ablation", "tcga_weighted_self_graph_cache.csv"
    ),

    # Candidate locations for final SHAP ranking outputs.
    "SHAP_GENE_RANKING_CANDIDATES": [
        os.path.join("Data", "Revision_Ablation", "Explainability_SHAP_MedianOS_K100", "shap_global_gene_level_ranking_tcga.csv"),
    ],
    "SHAP_FEATURE_RANKING_CANDIDATES": [
        os.path.join("Data", "Revision_Ablation", "Explainability_SHAP_MedianOS_K100", "shap_global_feature_ranking_tcga.csv"),
    ],

    # GMT search locations. Add custom files manually here if needed.
    "GMT_DIRS": [
        os.path.join("Data", "GSEA", "gmt"),
        os.path.join("Data", "GSEA"),
        os.path.join("Data", "MSigDB"),
        os.path.join("Data", "GMT"),
    ],
    "GMT_FILES_MANUAL": [],

    # ORA query gene sets.
    "SHAP_TOP_N_LIST": [15, 25, 50, 100],

    # ORA filters.
    "MIN_GENE_SET_SIZE": 5,
    "MAX_GENE_SET_SIZE": 500,
    "MIN_OVERLAP": 2,
    "FDR_CUTOFF_REPORT": 0.25,

    # Optional preranked GSEA via gseapy.
    "RUN_GSEAPY_PRERANK_IF_AVAILABLE": False,
    "GSEAPY_PERMUTATIONS": 1000,
    "GSEAPY_MIN_SIZE": 5,
    "GSEAPY_MAX_SIZE": 500,
    "SEED": 42,

    # Plotting.
    "TOP_TERMS_TO_PLOT": 20,
    "DPI": 300,
}

os.makedirs(CONF["OUT_DIR"], exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 7
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


def is_graph_feature(feature_name):
    return feature_family(feature_name) != "raw"


def clean_symbol(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["", "nan", "none", "na"]:
        return ""
    return s


def symbol_key(x):
    return clean_symbol(x).upper()


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    out = np.full(n, np.nan)
    good = np.isfinite(pvals)
    if good.sum() == 0:
        return out
    idx = np.where(good)[0]
    p = pvals[idx]
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[idx[order]] = q
    return out


# --------------------------------------------------
# 3. Load gene symbols and locked signature
# --------------------------------------------------

def load_gene_symbol_map(path):
    if not os.path.exists(path):
        print(f"Gene map not found: {path}. Using base IDs where necessary.")
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
        print("Could not identify gene map columns. Using IDs.")
        return {}

    gene_map[id_col] = gene_map[id_col].astype(str).map(normalize_feature_name)
    gene_map[sym_col] = gene_map[sym_col].map(clean_symbol)
    gene_map = gene_map[gene_map[sym_col] != ""].copy()

    out = dict(zip(gene_map[id_col], gene_map[sym_col]))
    print(f"Loaded gene map: {len(out)} entries")
    return out


def load_locked_signature(gene_map):
    path = CONF["LOCKED_FEATURES"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Locked features file not found: {path}")

    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError("Locked features CSV must contain a 'feature' column.")

    df["feature"] = df["feature"].astype(str).map(normalize_feature_name)
    df["base_gene_id"] = df["feature"].map(extract_base_gene_id)
    df["gene_symbol"] = df["base_gene_id"].map(lambda x: gene_map.get(x, x))
    df["gene_symbol"] = df["gene_symbol"].map(clean_symbol)
    df["gene_symbol_key"] = df["gene_symbol"].map(symbol_key)
    df["feature_family"] = df["feature"].map(feature_family)
    df["is_graph_feature"] = df["feature"].map(is_graph_feature)

    df = df[df["gene_symbol"] != ""].copy()

    n_features = int(len(df))
    n_graph_features = int(df["is_graph_feature"].sum())
    if n_features != int(CONF["EXPECTED_N_FEATURES"]):
        raise ValueError(
            f"This script is for the final median-OS K100 branch. "
            f"Expected {CONF['EXPECTED_N_FEATURES']} locked features, got {n_features}."
        )
    if n_graph_features != int(CONF["EXPECTED_N_GRAPH_FEATURES"]):
        raise ValueError(
            f"Expected {CONF['EXPECTED_N_GRAPH_FEATURES']} WPPI/graph locked features, "
            f"got {n_graph_features}. Check that the correct K100 feature file is loaded."
        )

    gene_summary = (
        df.groupby(["base_gene_id", "gene_symbol", "gene_symbol_key"], as_index=False)
        .agg(
            n_locked_features=("feature", "count"),
            n_graph_features=("is_graph_feature", "sum"),
            feature_families=("feature_family", lambda x: ";".join(sorted(set(map(str, x))))),
            example_features=("feature", lambda x: ";".join(list(map(str, x))[:5])),
        )
    )
    gene_summary["has_graph_feature"] = gene_summary["n_graph_features"] > 0

    print(f"Locked feature count: {len(df)}")
    print(f"Unique locked genes: {gene_summary['gene_symbol_key'].nunique()}")
    print(f"Locked genes with graph features: {int(gene_summary['has_graph_feature'].sum())}")

    df.to_csv(os.path.join(CONF["OUT_DIR"], "locked_signature_feature_list.csv"), index=False)
    gene_summary.to_csv(os.path.join(CONF["OUT_DIR"], "locked_signature_gene_list.csv"), index=False)

    return df, gene_summary


def load_background_genes(gene_map):
    path = CONF["TCGA_FEATURE_BACKGROUND_MATRIX"]
    if not os.path.exists(path):
        print(f"Background matrix not found: {path}. Background will be inferred from GMT genes.")
        return set(), "gmt_only"

    print(f"Loading feature background columns from: {path}")
    # Read header only for speed.
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # First column is usually sample ID/index after saving with index.
    cols = [normalize_feature_name(c) for c in header[1:]] if len(header) > 1 else []

    symbols = []
    for feat in cols:
        base = extract_base_gene_id(feat)
        sym = gene_map.get(base, base)
        key = symbol_key(sym)
        if key:
            symbols.append(key)

    bg = set(symbols)
    print(f"Background unique genes from TCGA matrix: {len(bg)}")
    return bg, "tcga_feature_matrix"


# --------------------------------------------------
# 4. Load SHAP ranking
# --------------------------------------------------

def load_shap_gene_ranking(gene_map):
    path = first_existing(CONF["SHAP_GENE_RANKING_CANDIDATES"])

    if path is not None:
        df = pd.read_csv(path)
        required = {"gene_symbol"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"SHAP gene ranking found but missing gene_symbol: {path}")

        df["gene_symbol"] = df["gene_symbol"].map(clean_symbol)
        df = df[df["gene_symbol"] != ""].copy()
        df["gene_symbol_key"] = df["gene_symbol"].map(symbol_key)

        # Harmonize scores.
        if "total_mean_abs_shap" not in df.columns:
            if "max_mean_abs_shap" in df.columns:
                df["total_mean_abs_shap"] = pd.to_numeric(df["max_mean_abs_shap"], errors="coerce")
            else:
                raise ValueError("SHAP gene ranking must contain total_mean_abs_shap or max_mean_abs_shap.")
        df["total_mean_abs_shap"] = pd.to_numeric(df["total_mean_abs_shap"], errors="coerce").fillna(0.0)

        if "mean_signed_shap" not in df.columns:
            df["mean_signed_shap"] = 0.0
        df["mean_signed_shap"] = pd.to_numeric(df["mean_signed_shap"], errors="coerce").fillna(0.0)

        df = df.sort_values("total_mean_abs_shap", ascending=False).drop_duplicates("gene_symbol_key", keep="first")
        df["rank_abs_shap"] = np.arange(1, len(df) + 1)
        print(f"Loaded SHAP gene ranking: {path} | genes={len(df)}")
        return df, path

    # Fallback: derive from feature ranking.
    fpath = first_existing(CONF["SHAP_FEATURE_RANKING_CANDIDATES"])
    if fpath is None:
        print("No SHAP ranking found. Using locked feature order only.")
        return pd.DataFrame(), None

    feat = pd.read_csv(fpath)
    if "feature" not in feat.columns:
        raise ValueError(f"SHAP feature ranking found but missing feature column: {fpath}")

    feat["feature"] = feat["feature"].astype(str).map(normalize_feature_name)
    feat["base_gene_id"] = feat["feature"].map(extract_base_gene_id)
    feat["gene_symbol"] = feat["base_gene_id"].map(lambda x: gene_map.get(x, x))
    feat["gene_symbol"] = feat["gene_symbol"].map(clean_symbol)
    feat = feat[feat["gene_symbol"] != ""].copy()
    feat["gene_symbol_key"] = feat["gene_symbol"].map(symbol_key)

    if "mean_abs_shap" not in feat.columns:
        raise ValueError("SHAP feature ranking must contain mean_abs_shap.")
    feat["mean_abs_shap"] = pd.to_numeric(feat["mean_abs_shap"], errors="coerce").fillna(0.0)
    if "mean_signed_shap" not in feat.columns:
        feat["mean_signed_shap"] = 0.0
    feat["mean_signed_shap"] = pd.to_numeric(feat["mean_signed_shap"], errors="coerce").fillna(0.0)

    agg = (
        feat.groupby(["base_gene_id", "gene_symbol", "gene_symbol_key"], as_index=False)
        .agg(
            total_mean_abs_shap=("mean_abs_shap", "sum"),
            max_mean_abs_shap=("mean_abs_shap", "max"),
            mean_signed_shap=("mean_signed_shap", "mean"),
        )
    )
    agg = agg.sort_values("total_mean_abs_shap", ascending=False)
    agg["rank_abs_shap"] = np.arange(1, len(agg) + 1)
    print(f"Derived SHAP gene ranking from feature ranking: {fpath} | genes={len(agg)}")
    return agg, fpath


def export_ranked_lists(shap_df):
    if shap_df.empty:
        return None, None

    out_csv = os.path.join(CONF["OUT_DIR"], "shap_ranked_gene_list_tcga.csv")
    shap_df.to_csv(out_csv, index=False)

    # GSEA .rnk file uses signed SHAP. If all signs are zero, use absolute SHAP.
    rnk = shap_df[["gene_symbol", "mean_signed_shap", "total_mean_abs_shap"]].copy()
    if np.isclose(rnk["mean_signed_shap"].abs().sum(), 0.0):
        print("mean_signed_shap is unavailable/zero; using total_mean_abs_shap for .rnk export.")
        rnk["rank_score"] = rnk["total_mean_abs_shap"]
    else:
        rnk["rank_score"] = rnk["mean_signed_shap"]

    rnk = rnk[["gene_symbol", "rank_score"]].dropna()
    rnk = rnk.groupby("gene_symbol", as_index=False)["rank_score"].max()
    rnk = rnk.sort_values("rank_score", ascending=False)

    out_rnk = os.path.join(CONF["OUT_DIR"], "shap_ranked_gene_list_tcga.rnk")
    rnk.to_csv(out_rnk, sep="\t", index=False, header=False)

    return out_csv, out_rnk


# --------------------------------------------------
# 5. GMT parsing and Enrichr-style ORA
# --------------------------------------------------

def discover_gmt_files():
    files = []
    for p in CONF["GMT_FILES_MANUAL"]:
        if os.path.exists(p):
            files.append(p)

    for d in CONF["GMT_DIRS"]:
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "*.gmt")))

    files = sorted(set(files))
    return files


def parse_gmt(path):
    gene_sets = []
    collection = Path(path).stem

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0].strip()
            desc = parts[1].strip()
            genes = {symbol_key(g) for g in parts[2:] if symbol_key(g)}
            if len(genes) == 0:
                continue
            gene_sets.append({
                "collection": collection,
                "term": term,
                "description": desc,
                "genes": genes,
                "source_file": path,
            })

    return gene_sets


def load_all_gene_sets(gmt_files):
    all_sets = []
    for path in gmt_files:
        parsed = parse_gmt(path)
        print(f"Loaded GMT: {path} | gene sets={len(parsed)}")
        all_sets.extend(parsed)
    return all_sets


def run_ora_for_query(query_name, query_genes, background_genes, gene_sets):
    query_genes = {symbol_key(g) for g in query_genes if symbol_key(g)}
    background_genes = {symbol_key(g) for g in background_genes if symbol_key(g)}

    if not background_genes:
        # fallback: use all genes appearing in GMT files
        for gs in gene_sets:
            background_genes.update(gs["genes"])

    query_genes = query_genes & background_genes
    N = len(background_genes)
    n = len(query_genes)

    rows = []
    if N == 0 or n == 0:
        return pd.DataFrame()

    for gs in gene_sets:
        term_genes = gs["genes"] & background_genes
        K = len(term_genes)
        if K < CONF["MIN_GENE_SET_SIZE"] or K > CONF["MAX_GENE_SET_SIZE"]:
            continue
        overlap = sorted(query_genes & term_genes)
        k = len(overlap)
        if k < CONF["MIN_OVERLAP"]:
            continue

        pval = float(hypergeom.sf(k - 1, N, K, n))
        enrichment_ratio = (k / n) / (K / N) if K > 0 and n > 0 else np.nan

        rows.append({
            "query_name": query_name,
            "collection": gs["collection"],
            "term": gs["term"],
            "description": gs["description"],
            "source_file": gs["source_file"],
            "background_size_N": N,
            "query_size_n": n,
            "gene_set_size_K": K,
            "overlap_k": k,
            "p_value": pval,
            "enrichment_ratio": enrichment_ratio,
            "overlap_genes": ";".join(overlap),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fdr_bh"] = bh_fdr(out["p_value"].values)
        out["minus_log10_fdr"] = -np.log10(np.clip(out["fdr_bh"].astype(float), 1e-300, 1.0))
        out = out.sort_values(["fdr_bh", "p_value", "overlap_k"], ascending=[True, True, False])

    return out


def make_query_gene_sets(locked_gene_summary, shap_df):
    queries = {}

    locked_all = sorted(set(locked_gene_summary["gene_symbol_key"].dropna()))
    queries["LOCKED_ALL_UNIQUE_GENES"] = locked_all

    locked_wppi = sorted(set(locked_gene_summary.loc[locked_gene_summary["has_graph_feature"], "gene_symbol_key"].dropna()))
    if locked_wppi:
        queries["LOCKED_WPPI_ASSOCIATED_GENES"] = locked_wppi

    if not shap_df.empty:
        ranked = shap_df.sort_values("total_mean_abs_shap", ascending=False)
        for n in CONF["SHAP_TOP_N_LIST"]:
            genes = ranked.head(n)["gene_symbol_key"].dropna().tolist()
            if genes:
                queries[f"SHAP_TOP_{n}_GENES"] = sorted(set(genes))

    query_rows = []
    for q, genes in queries.items():
        for g in sorted(set(genes)):
            query_rows.append({"query_name": q, "gene_symbol_key": g})
    pd.DataFrame(query_rows).to_csv(os.path.join(CONF["OUT_DIR"], "gsea_query_gene_sets.csv"), index=False)

    return queries


def run_all_ora(queries, background_genes, gene_sets):
    all_results = []
    for name, genes in queries.items():
        print(f"Running ORA for {name}: n={len(set(genes))}")
        res = run_ora_for_query(name, genes, background_genes, gene_sets)
        if not res.empty:
            all_results.append(res)

    if not all_results:
        return pd.DataFrame()

    out = pd.concat(all_results, axis=0).reset_index(drop=True)
    # BH was per query. Add global BH across all tests as a conservative extra column.
    out["fdr_bh_global"] = bh_fdr(out["p_value"].values)
    out["minus_log10_fdr_global"] = -np.log10(np.clip(out["fdr_bh_global"].astype(float), 1e-300, 1.0))
    return out


# --------------------------------------------------
# 6. Optional gseapy prerank
# --------------------------------------------------

def run_optional_gseapy_prerank(rnk_file, gmt_files):
    if not CONF["RUN_GSEAPY_PRERANK_IF_AVAILABLE"]:
        return {"attempted": False, "reason": "disabled"}
    if rnk_file is None or not os.path.exists(rnk_file):
        return {"attempted": False, "reason": "no rnk file"}

    try:
        import gseapy as gp
    except Exception:
        return {"attempted": False, "reason": "gseapy not installed"}

    prerank_dir = os.path.join(CONF["OUT_DIR"], "gseapy_prerank")
    os.makedirs(prerank_dir, exist_ok=True)

    summaries = []
    for gmt in gmt_files:
        collection = Path(gmt).stem
        outdir = os.path.join(prerank_dir, collection)
        os.makedirs(outdir, exist_ok=True)
        print(f"Running optional gseapy prerank: {collection}")
        try:
            pre_res = gp.prerank(
                rnk=rnk_file,
                gene_sets=gmt,
                outdir=outdir,
                min_size=CONF["GSEAPY_MIN_SIZE"],
                max_size=CONF["GSEAPY_MAX_SIZE"],
                permutation_num=CONF["GSEAPY_PERMUTATIONS"],
                seed=CONF["SEED"],
                no_plot=True,
                verbose=False,
            )
            res = pre_res.res2d.copy()
            res["collection"] = collection
            out_csv = os.path.join(outdir, "prerank_results.csv")
            res.to_csv(out_csv, index=False)
            summaries.append(res)
        except Exception as exc:
            print(f"WARNING: gseapy prerank failed for {gmt}: {exc}")

    if summaries:
        combined = pd.concat(summaries, axis=0, ignore_index=True)
        combined_path = os.path.join(CONF["OUT_DIR"], "prerank_results_all.csv")
        combined.to_csv(combined_path, index=False)
        return {"attempted": True, "status": "completed", "combined_results": combined_path}

    return {"attempted": True, "status": "failed_or_empty"}


# --------------------------------------------------
# 7. Plotting
# --------------------------------------------------

def _shorten_term(term, max_len=82):
    term = str(term)
    for prefix in ["HALLMARK_", "REACTOME_", "GOBP_", "WP_", "KEGG_", "BIOCARTA_"]:
        if term.startswith(prefix):
            term = term[len(prefix):]
    term = term.replace("_", " ")
    if len(term) > max_len:
        term = term[: max_len - 3] + "..."
    return term


def _plot_one_ora_dotplot(plot_df, out_stem, title):
    if plot_df.empty:
        return None

    plot_df = plot_df.copy()
    plot_df = plot_df.dropna(subset=["fdr_bh", "p_value", "term", "overlap_k"])
    plot_df = plot_df[plot_df["fdr_bh"] <= CONF["FDR_CUTOFF_REPORT"]].copy()

    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(["fdr_bh", "p_value", "overlap_k"], ascending=[True, True, False])
    plot_df = plot_df.head(CONF["TOP_TERMS_TO_PLOT"]).copy()
    plot_df["label"] = plot_df["term"].map(_shorten_term)
    plot_df["score"] = -np.log10(np.clip(plot_df["fdr_bh"].astype(float), 1e-300, 1.0))
    plot_df = plot_df.sort_values("score", ascending=True).copy()

    fig_height = max(5.0, 0.38 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))

    sizes = 35 + 22 * plot_df["overlap_k"].astype(float)
    ax.scatter(
        plot_df["score"],
        plot_df["label"],
        s=sizes,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.axvline(-np.log10(CONF["FDR_CUTOFF_REPORT"]), linestyle="--", linewidth=0.9)
    ax.set_xlabel("-log10(BH-FDR)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    x_max = max(plot_df["score"].max(), -np.log10(CONF["FDR_CUTOFF_REPORT"])) + 0.35
    ax.set_xlim(0, x_max)

    for _, row in plot_df.iterrows():
        ax.text(
            row["score"] + 0.03,
            row["label"],
            f"k={int(row['overlap_k'])}",
            va="center",
            fontsize=6.5,
        )

    fig.tight_layout()
    for ext in [".png", ".pdf", ".tiff"]:
        path = out_stem + ext
        if ext == ".tiff":
            fig.savefig(path, dpi=CONF["DPI"], bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(path, dpi=CONF["DPI"], bbox_inches="tight")
    plt.close(fig)
    return out_stem + ".png"


def plot_ora_dotplot(ora_df):
    """
    Create separate clean dotplots for locked and SHAP query sets.
    This avoids one cramped mixed figure and prevents optional prerank warnings
    from being mistaken for locked-signature ORA failure.
    """
    if ora_df.empty:
        print("No ORA results to plot.")
        return None

    generated = []
    query_order = [
        "LOCKED_ALL_UNIQUE_GENES",
        "LOCKED_WPPI_ASSOCIATED_GENES",
        "SHAP_TOP_15_GENES",
        "SHAP_TOP_25_GENES",
        "SHAP_TOP_50_GENES",
        "SHAP_TOP_100_GENES",
    ]

    for q in query_order:
        sub = ora_df[ora_df["query_name"].astype(str).eq(q)].copy()
        if sub.empty:
            continue

        out_stem = os.path.join(CONF["OUT_DIR"], f"Figure_ORA_Dotplot_{q}")
        fig_path = _plot_one_ora_dotplot(
            sub,
            out_stem,
            title=f"Enrichr-style ORA: {q.replace('_', ' ')}",
        )
        if fig_path is not None:
            generated.append(fig_path)

    overall = ora_df.sort_values(["fdr_bh", "p_value", "overlap_k"], ascending=[True, True, False]).copy()
    out_stem = os.path.join(CONF["OUT_DIR"], "Figure_ORA_Dotplot_Overall_TopTerms")
    fig_path = _plot_one_ora_dotplot(
        overall,
        out_stem,
        title="Enrichr-style ORA: overall top terms across locked/SHAP query sets",
    )
    if fig_path is not None:
        generated.append(fig_path)

    if generated:
        try:
            import shutil
            preferred = generated[0]
            for ext in [".png", ".pdf", ".tiff"]:
                src_path = preferred.replace(".png", ext)
                if os.path.exists(src_path):
                    shutil.copyfile(
                        src_path,
                        os.path.join(CONF["OUT_DIR"], "Figure_GSEA_ORA_Dotplot_TopTerms" + ext),
                    )
        except Exception:
            pass

        pd.DataFrame({"figure_path": generated}).to_csv(
            os.path.join(CONF["OUT_DIR"], "ora_figure_manifest.csv"),
            index=False,
        )
        print("Generated ORA figures:")
        for p in generated:
            print(f"- {p}")
        return generated[0]

    print(f"No ORA terms at FDR <= {CONF['FDR_CUTOFF_REPORT']} for plotting.")
    return None


# --------------------------------------------------
# 8. Main
# --------------------------------------------------

def main():
    print("=" * 80)
    print("Locked-signature Enrichr-style ORA / GSEA-style pathway analysis")
    print("=" * 80)
    print("No model retraining, no feature reselection, no threshold changes, no CGGA label use.")
    print("Locked-signature analysis uses Enrichr-style ORA only; full-transcriptome preranked GSEA is handled by Script 15b.")
    print("=" * 80)

    gene_map = load_gene_symbol_map(CONF["GENE_MAP"])
    locked_features_df, locked_gene_summary = load_locked_signature(gene_map)
    background_genes, background_source = load_background_genes(gene_map)

    shap_df, shap_source = load_shap_gene_ranking(gene_map)
    ranked_csv, ranked_rnk = export_ranked_lists(shap_df)

    queries = make_query_gene_sets(locked_gene_summary, shap_df)

    gmt_files = discover_gmt_files()
    gmt_manifest = pd.DataFrame({"gmt_file": gmt_files})
    gmt_manifest.to_csv(os.path.join(CONF["OUT_DIR"], "gsea_gmt_files_used.csv"), index=False)

    audit = {
        "script": "15_gsea_locked_signature_median_os_k100_ORA_ONLY_v2.py",
        "analysis_role": "Post-lockdown biological plausibility analysis for final locked GIBD signature.",
        "no_retraining": True,
        "no_tuning": True,
        "no_feature_reselection": True,
        "no_threshold_change": True,
        "no_cgga_label_use": True,
        "pathways_not_used_for_model_revision": True,
        "locked_features": CONF["LOCKED_FEATURES"],
        "gene_map": CONF["GENE_MAP"],
        "shap_ranking_source": shap_source,
        "ranked_gene_list_csv": ranked_csv,
        "ranked_gene_list_rnk": ranked_rnk,
        "background_source": background_source,
        "background_n_genes": len(background_genes),
        "n_locked_features": int(len(locked_features_df)),
        "n_locked_unique_genes": int(locked_gene_summary["gene_symbol_key"].nunique()),
        "n_locked_graph_genes": int(locked_gene_summary["has_graph_feature"].sum()),
        "gmt_files_found": gmt_files,
        "query_sets": {k: len(set(v)) for k, v in queries.items()},
        "interpretation_guardrail": "Pathway results support biological plausibility only, not causal or experimental validation.",
    }

    if not gmt_files:
        print("\nWARNING: No GMT files were found.")
        print("The script exported locked gene lists and SHAP-ranked lists, but no enrichment was run.")
        print("Place GMT files under Data/GSEA/gmt/ and rerun.")
        audit["status"] = "stopped_no_gmt_files"
        audit_path = os.path.join(CONF["OUT_DIR"], "gsea_locked_signature_audit.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(audit), f, indent=2)
        print(f"Audit saved: {audit_path}")
        return

    gene_sets = load_all_gene_sets(gmt_files)
    if not gene_sets:
        print("No valid gene sets parsed from GMT files.")
        audit["status"] = "stopped_no_valid_gene_sets"
        audit_path = os.path.join(CONF["OUT_DIR"], "gsea_locked_signature_audit.json")
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(audit), f, indent=2)
        return

    # If TCGA background unavailable, use all GMT genes as background.
    if not background_genes:
        bg = set()
        for gs in gene_sets:
            bg.update(gs["genes"])
        background_genes = bg
        background_source = "all_genes_in_gmt_files"
        audit["background_source"] = background_source
        audit["background_n_genes"] = len(background_genes)

    ora_df = run_all_ora(queries, background_genes, gene_sets)
    ora_all_path = os.path.join(CONF["OUT_DIR"], "ora_results_all.csv")
    ora_sig_path = os.path.join(CONF["OUT_DIR"], "ora_results_significant.csv")

    if ora_df.empty:
        print("No ORA results passed overlap filters.")
        pd.DataFrame().to_csv(ora_all_path, index=False)
        pd.DataFrame().to_csv(ora_sig_path, index=False)
        figure_path = None
    else:
        ora_df.to_csv(ora_all_path, index=False)
        sig = ora_df[ora_df["fdr_bh"] <= CONF["FDR_CUTOFF_REPORT"]].copy()
        sig.to_csv(ora_sig_path, index=False)
        figure_path = plot_ora_dotplot(ora_df)
        print("\nTop ORA terms:")
        show_cols = ["query_name", "collection", "term", "overlap_k", "p_value", "fdr_bh", "enrichment_ratio"]
        print(ora_df[show_cols].head(20).to_string(index=False))

    prerank_status = run_optional_gseapy_prerank(ranked_rnk, gmt_files)

    audit.update({
        "status": "completed",
        "n_gmt_files": len(gmt_files),
        "n_gene_sets_loaded": len(gene_sets),
        "ora_results_all": ora_all_path,
        "ora_results_significant": ora_sig_path,
        "ora_figure": figure_path,
        "optional_prerank_status": prerank_status,
    })

    audit_path = os.path.join(CONF["OUT_DIR"], "gsea_locked_signature_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(audit), f, indent=2)

    print("\n" + "=" * 80)
    print("GSEA/pathway enrichment complete")
    print("=" * 80)
    print(f"Output directory: {CONF['OUT_DIR']}")
    print(f"- Locked genes: {os.path.join(CONF['OUT_DIR'], 'locked_signature_gene_list.csv')}")
    print(f"- SHAP ranked genes: {ranked_csv}")
    print(f"- ORA all: {ora_all_path}")
    print(f"- ORA significant: {ora_sig_path}")
    print(f"- Audit: {audit_path}")
    if figure_path:
        print(f"- Figure: {figure_path}")

    print("\nRecommended manuscript wording:")
    print(
        "Pathway enrichment was performed post hoc on the final locked GIBD signature and "
        "SHAP-ranked genes using local gene-set GMT files. This analysis was used only to "
        "assess biological plausibility and was not used for feature selection, model tuning, "
        "or threshold modification."
    )


if __name__ == "__main__":
    main()
