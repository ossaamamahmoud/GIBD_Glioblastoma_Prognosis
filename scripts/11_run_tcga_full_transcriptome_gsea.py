"""
GIBD analysis workflow — TCGA full-transcriptome preranked GSEA

This script performs TCGA-only full-transcriptome preranked GSEA for the locked
median-OS K100 analysis. Genes are ranked by signed Welch t-statistics comparing
TCGA high-risk and low-risk groups, where positive scores indicate higher
expression in high-risk patients and negative scores indicate higher expression
in low-risk patients.

Analysis guardrails:
- The analysis uses TCGA expression and TCGA median-OS risk labels only.
- No CGGA labels or external cohort outcomes are used.
- No model training, feature selection, model tuning, threshold selection, or model
  revision is performed.
- GSEA is used for post-lock pathway contextualization only.

Expected project layout uses relative paths under Data/ and Data/Revision_Ablation/.
"""


from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import ttest_ind
except Exception as e:
    raise ImportError(
        "scipy is required. Install with: pip install scipy"
    ) from e

try:
    import gseapy as gp
except Exception as e:
    raise ImportError(
        "gseapy is required. Install with: pip install gseapy"
    ) from e


# =============================================================================
# 0. Configuration
# =============================================================================

CONF = {
    # TCGA expression matrix used for transcriptome-wide ranking.
    # Rows should be patients; columns should be genes/Ensembl IDs.
    "TCGA_EXPR": Path("Data/TCGA_survival_expression_matrix_protein_coding.csv"),

    # TCGA labels generated under the empirical median-OS endpoint for the final K100 branch.
    # Must contain Risk_Label. OS_days/Event are optional but will be audited if present.
    "TCGA_LABELS": Path("Data/TCGA_survival_labels_with_os_event.csv"),

    # Median-OS K100 final-branch label integrity checks.
    "EXPECTED_N_TCGA": 147,
    "EXPECTED_LOW_TCGA": 73,
    "EXPECTED_HIGH_TCGA": 74,

    # Gene map used to convert ENSG IDs to HGNC symbols.
    "GENE_MAP": Path("Data/gene_type_map.csv"),

    # Local MSigDB GMT files downloaded as Gene Symbols.
    "GMT_DIR": Path("Data/GSEA/gmt"),

    # Output directory.
    "OUT_DIR": Path("Data/Revision_Ablation/GSEA_FULL_TRANSCRIPTOME_MedianOS_K100"),

    # GSEA settings.
    "MIN_SIZE": 5,
    "MAX_SIZE": 500,
    "PERMUTATION_NUM": 1000,   # For a quick test, temporarily set to 100. For manuscript, use 1000.
    "SEED": 42,
    "THREADS": 4,

    # FDR thresholds.
    "FDR_SIGNIFICANT": 0.05,
    "FDR_EXPLORATORY": 0.25,

    # Deterministic negligible tie-breaker for identical ranking scores.
    # This prevents arbitrary ranking of exact ties in gseapy without materially changing the ranking.
    "TIE_BREAK_EPS": 1e-10,
}

OUT_DIR = CONF["OUT_DIR"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

GSEA_OUT_DIR = OUT_DIR / "gseapy_prerank"
GSEA_OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TCGA full-transcriptome preranked GSEA — median-OS K100 final branch")
print("=" * 80)
print("No model training, no feature selection, no threshold modification, no CGGA use.")
print("Ranking: signed Welch t-statistic, high-risk vs low-risk TCGA.")
print("Positive NES = pathway enriched toward high-risk/upregulated genes.")
print("Negative NES = pathway enriched toward low-risk/upregulated genes.")
print("=" * 80)


# =============================================================================
# 1. Helper functions
# =============================================================================

def clean_gene_id(x):
    """Remove Ensembl version suffix if present."""
    if pd.isna(x):
        return x
    s = str(x).strip()
    if "." in s and s.upper().startswith("ENSG"):
        return s.split(".")[0]
    return s


def find_first_existing_column(df, candidates):
    """Return first matching column name, case-insensitive where useful."""
    exact = [c for c in candidates if c in df.columns]
    if exact:
        return exact[0]

    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_gene_symbol_map(path):
    """Load ENSG/clean gene ID -> HGNC symbol mapping."""
    if not path.exists():
        print(f"WARNING: gene map not found: {path}. Gene IDs will be used as-is.")
        return {}

    gm = pd.read_csv(path)

    id_col = find_first_existing_column(
        gm,
        ["clean_gene_id", "gene_id", "ensembl_gene_id", "ensembl_id", "Gene_ID", "id"],
    )
    symbol_col = find_first_existing_column(
        gm,
        ["gene_name", "gene_symbol", "symbol", "hgnc_symbol", "Gene_Symbol"],
    )

    if id_col is None or symbol_col is None:
        print("WARNING: gene map columns not detected. Gene IDs will be used as-is.")
        print(f"Gene map columns: {list(gm.columns)[:20]}")
        return {}

    gm = gm[[id_col, symbol_col]].dropna()
    gm[id_col] = gm[id_col].map(clean_gene_id)
    gm[symbol_col] = gm[symbol_col].astype(str).str.strip()
    gm = gm[(gm[id_col] != "") & (gm[symbol_col] != "")]
    gm = gm.drop_duplicates(subset=[id_col], keep="first")

    mapping = dict(zip(gm[id_col], gm[symbol_col]))
    print(f"Loaded gene symbol map: {len(mapping)} entries from {path}")
    return mapping


def load_labels(path):
    """Load TCGA labels and standardize index."""
    if not path.exists():
        raise FileNotFoundError(f"TCGA labels file not found: {path}")

    ydf = pd.read_csv(path)

    id_col = find_first_existing_column(
        ydf,
        ["Patient_ID", "patient_id", "Sample_ID", "sample_id", "case_id", "Case_ID"],
    )
    if id_col is not None:
        ydf[id_col] = ydf[id_col].astype(str)
        ydf = ydf.set_index(id_col)
    else:
        # If no explicit ID column, assume first column was saved as index.
        ydf = pd.read_csv(path, index_col=0)
        ydf.index = ydf.index.astype(str)

    if "Risk_Label" not in ydf.columns:
        raise ValueError(
            f"Risk_Label column not found in {path}. Columns: {list(ydf.columns)}"
        )

    ydf["Risk_Label"] = pd.to_numeric(ydf["Risk_Label"], errors="coerce")
    ydf = ydf.dropna(subset=["Risk_Label"])
    ydf["Risk_Label"] = ydf["Risk_Label"].astype(int)

    if not set(ydf["Risk_Label"].unique()).issubset({0, 1}):
        raise ValueError("Risk_Label must contain only 0/1 labels.")

    return ydf


def assert_tcga_label_counts(labels_df):
    expected_n = int(CONF["EXPECTED_N_TCGA"])
    expected_low = int(CONF["EXPECTED_LOW_TCGA"])
    expected_high = int(CONF["EXPECTED_HIGH_TCGA"])

    if len(labels_df) != expected_n:
        raise ValueError(f"TCGA median-OS labels: expected N={expected_n}, got {len(labels_df)}")

    counts = labels_df["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))

    if low != expected_low or high != expected_high:
        raise ValueError(
            f"TCGA median-OS labels: expected low={expected_low}, high={expected_high}; "
            f"got low={low}, high={high}"
        )


def load_expression_and_align(expr_path, labels_df):
    """Load expression matrix, detect orientation, align rows to labels."""
    if not expr_path.exists():
        raise FileNotFoundError(f"TCGA expression matrix not found: {expr_path}")

    X = pd.read_csv(expr_path, index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)

    label_ids = set(labels_df.index.astype(str))
    row_overlap = len(set(X.index) & label_ids)
    col_overlap = len(set(X.columns) & label_ids)

    print(f"Expression raw shape: {X.shape}")
    print(f"Label overlap with expression rows: {row_overlap}")
    print(f"Label overlap with expression columns: {col_overlap}")

    if row_overlap >= max(20, col_overlap):
        print("Detected orientation: rows = patients, columns = genes")
    elif col_overlap >= 20 and col_overlap > row_overlap:
        print("Detected orientation: rows = genes, columns = patients. Transposing matrix.")
        X = X.T
        X.index = X.index.astype(str)
        X.columns = X.columns.astype(str)
    else:
        raise ValueError(
            "Could not align expression matrix with labels. "
            f"row_overlap={row_overlap}, col_overlap={col_overlap}"
        )

    common = [idx for idx in X.index if idx in labels_df.index]
    if len(common) < 20:
        raise ValueError(f"Too few matched samples after alignment: {len(common)}")

    X = X.loc[common].copy()
    y = labels_df.loc[common, "Risk_Label"].copy()

    # Numeric conversion.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove genes that are fully missing.
    X = X.dropna(axis=1, how="all")

    # Fill occasional missing values by per-gene median. This should normally be unused.
    missing_total = int(X.isna().sum().sum())
    if missing_total > 0:
        print(f"WARNING: filling {missing_total} missing expression values by gene median.")
        X = X.fillna(X.median(axis=0))

    print(f"Aligned TCGA expression matrix: {X.shape}")
    print(f"Risk_Label distribution: {y.value_counts().sort_index().to_dict()}")

    return X, y


def build_signed_t_ranking(X, y, symbol_map):
    """Compute signed Welch t-statistic high-risk vs low-risk for every gene."""
    high_mask = y.astype(int).values == 1
    low_mask = y.astype(int).values == 0

    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    if n_high < 5 or n_low < 5:
        raise ValueError(f"Too few samples per group: high={n_high}, low={n_low}")

    X_high = X.loc[high_mask].values
    X_low = X.loc[low_mask].values

    print("Computing Welch t-statistics for all genes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        t_stat, p_val = ttest_ind(
            X_high,
            X_low,
            axis=0,
            equal_var=False,
            nan_policy="omit",
        )

    mean_high = np.nanmean(X_high, axis=0)
    mean_low = np.nanmean(X_low, axis=0)
    mean_diff = mean_high - mean_low

    rank = pd.DataFrame(
        {
            "feature": X.columns,
            "clean_gene_id": [clean_gene_id(c) for c in X.columns],
            "mean_high_risk": mean_high,
            "mean_low_risk": mean_low,
            "mean_diff_high_minus_low": mean_diff,
            "signed_t_stat": t_stat,
            "p_value_welch": p_val,
        }
    )

    rank["gene_symbol"] = rank["clean_gene_id"].map(symbol_map)

    # Fallback: if the column already looks like a gene symbol, keep it.
    # Avoid using ENSG fallback for GSEA because MSigDB GMT uses symbols.
    rank["gene_symbol"] = rank["gene_symbol"].fillna(rank["clean_gene_id"])
    rank["gene_symbol"] = rank["gene_symbol"].astype(str).str.strip()

    # Drop genes that remain non-finite or empty.
    rank = rank.replace([np.inf, -np.inf], np.nan)
    rank = rank.dropna(subset=["signed_t_stat"])
    rank = rank[rank["gene_symbol"] != ""]

    # Remove identifiers unlikely to match symbol GMT files if no mapping happened.
    # Keep non-ENSG names as symbols.
    before_symbol_filter = len(rank)
    rank = rank[~rank["gene_symbol"].str.upper().str.startswith("ENSG")].copy()
    removed_unmapped = before_symbol_filter - len(rank)

    # Collapse duplicate symbols by the strongest absolute signed t-statistic.
    rank["abs_t"] = rank["signed_t_stat"].abs()
    rank = rank.sort_values(["gene_symbol", "abs_t"], ascending=[True, False])
    duplicate_rows = rank[rank.duplicated(subset=["gene_symbol"], keep="first")].copy()
    collapsed = rank.drop_duplicates(subset=["gene_symbol"], keep="first").copy()

    # Deterministic tie-breaking jitter, negligible relative to t-statistics.
    collapsed = collapsed.sort_values(["signed_t_stat", "gene_symbol"], ascending=[False, True]).copy()
    n = len(collapsed)
    if n == 0:
        raise ValueError("No genes remained after mapping/collapsing. Check gene map and expression columns.")

    jitter = np.linspace(CONF["TIE_BREAK_EPS"], 0.0, n)
    collapsed["rank_score_for_gsea"] = collapsed["signed_t_stat"].astype(float).values + jitter

    collapsed = collapsed.sort_values("rank_score_for_gsea", ascending=False).copy()

    print(f"Genes before symbol filter: {before_symbol_filter}")
    print(f"Removed unmapped ENSG-like genes: {removed_unmapped}")
    print(f"Unique gene symbols for GSEA ranking: {len(collapsed)}")
    print(f"Duplicate symbol rows collapsed: {len(duplicate_rows)}")

    return rank, collapsed, duplicate_rows


def normalize_gsea_results(res_df, collection_name):
    """Normalize gseapy result columns into a stable schema."""
    df = res_df.copy()

    if "Term" not in df.columns:
        # Some versions keep terms in the index.
        df = df.reset_index().rename(columns={"index": "Term"})

    rename_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ["nes"]:
            rename_map[c] = "NES"
        elif cl in ["es"]:
            rename_map[c] = "ES"
        elif cl in ["nom p-val", "nom p-val", "nominal p-value", "pval", "p-value"]:
            rename_map[c] = "NOM_pval"
        elif cl in ["fdr q-val", "fdr q-value", "fdr", "fdr_q_val", "qval"]:
            rename_map[c] = "FDR_qval"
        elif cl in ["fwer p-val", "fwer"]:
            rename_map[c] = "FWER_pval"
        elif cl in ["lead_genes", "leading edge", "ledge_genes", "genes"]:
            rename_map[c] = "Lead_genes"
        elif cl == "term":
            rename_map[c] = "Term"
        elif cl == "name":
            rename_map[c] = "Name"

    df = df.rename(columns=rename_map)
    df["collection"] = collection_name

    for c in ["NES", "ES", "NOM_pval", "FDR_qval", "FWER_pval"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    if "Lead_genes" not in df.columns:
        df["Lead_genes"] = ""

    # Keep all original columns too, but put core columns first.
    core = ["collection", "Term", "ES", "NES", "NOM_pval", "FDR_qval", "FWER_pval", "Lead_genes"]
    existing_core = [c for c in core if c in df.columns]
    other = [c for c in df.columns if c not in existing_core]
    df = df[existing_core + other]

    return df


def safe_collection_name(path):
    return path.stem.replace(".", "_").replace("-", "_")


def make_dotplot(df, out_path, title, top_n=20, fdr_limit=0.25):
    """Create a compact NES dotplot for significant/exploratory GSEA terms."""
    if df.empty:
        return False

    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=["FDR_qval", "NES", "Term"])
    plot_df = plot_df[plot_df["FDR_qval"] <= fdr_limit].copy()
    if plot_df.empty:
        return False

    plot_df = plot_df.sort_values("FDR_qval", ascending=True).head(top_n).copy()
    plot_df = plot_df.sort_values("NES", ascending=True).copy()

    # Shorten term labels for readability.
    def shorten_term(t):
        t = str(t)
        for prefix in ["HALLMARK_", "REACTOME_", "GOBP_", "WP_", "KEGG_", "BIOCARTA_"]:
            if t.startswith(prefix):
                t = t[len(prefix):]
        return t[:85]

    plot_df["term_short"] = plot_df["Term"].map(shorten_term)
    plot_df["minus_log10_fdr"] = -np.log10(plot_df["FDR_qval"].clip(lower=1e-300))

    plt.figure(figsize=(10, max(6, 0.35 * len(plot_df) + 2)))
    sizes = 40 + 25 * plot_df["minus_log10_fdr"].values
    plt.scatter(plot_df["NES"], plot_df["term_short"], s=sizes, alpha=0.8)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Normalized enrichment score (NES)\nPositive = enriched toward high-risk; negative = enriched toward low-risk")
    plt.ylabel("")
    plt.title(title)
    plt.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


# =============================================================================
# 2. Load inputs
# =============================================================================

labels_df = load_labels(CONF["TCGA_LABELS"])
assert_tcga_label_counts(labels_df)
X, y = load_expression_and_align(CONF["TCGA_EXPR"], labels_df)
symbol_map = load_gene_symbol_map(CONF["GENE_MAP"])

gmt_files = sorted(CONF["GMT_DIR"].glob("*.gmt"))
if not gmt_files:
    raise FileNotFoundError(
        f"No GMT files found in {CONF['GMT_DIR']}. Put MSigDB .symbols.gmt files there."
    )

print("\nGMT files detected:")
for g in gmt_files:
    print(f"- {g}")


# =============================================================================
# 3. Build full-transcriptome ranking
# =============================================================================

rank_all, rank_unique, duplicate_rows = build_signed_t_ranking(X, y, symbol_map)

rank_all_path = OUT_DIR / "tcga_full_transcriptome_gene_level_statistics_all_features.csv"
rank_unique_path = OUT_DIR / "tcga_full_transcriptome_gene_level_statistics_unique_symbols.csv"
duplicate_path = OUT_DIR / "tcga_full_transcriptome_duplicate_symbol_rows_collapsed.csv"
rnk_path = OUT_DIR / "tcga_high_vs_low_full_transcriptome_signed_t.rnk"

rank_all.to_csv(rank_all_path, index=False)
rank_unique.to_csv(rank_unique_path, index=False)
duplicate_rows.to_csv(duplicate_path, index=False)

# RNK format: two tab-separated columns, no header.
rnk_df = rank_unique[["gene_symbol", "rank_score_for_gsea"]].copy()
rnk_df.to_csv(rnk_path, sep="\t", index=False, header=False)

print("\nSaved ranking files:")
print(f"- {rank_all_path}")
print(f"- {rank_unique_path}")
print(f"- {duplicate_path}")
print(f"- {rnk_path}")

print("\nTop high-risk-associated genes by signed t-statistic:")
print(
    rank_unique[["gene_symbol", "feature", "signed_t_stat", "mean_diff_high_minus_low", "p_value_welch"]]
    .head(15)
    .to_string(index=False)
)

print("\nTop low-risk-associated genes by signed t-statistic:")
print(
    rank_unique[["gene_symbol", "feature", "signed_t_stat", "mean_diff_high_minus_low", "p_value_welch"]]
    .tail(15)
    .sort_values("signed_t_stat", ascending=True)
    .to_string(index=False)
)


# =============================================================================
# 4. Run preranked GSEA for each GMT file
# =============================================================================

all_results = []
run_errors = []
start_time = time.time()

for gmt in gmt_files:
    collection = safe_collection_name(gmt)
    collection_out = GSEA_OUT_DIR / collection
    collection_out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Running preranked GSEA: {gmt.name}")
    print("=" * 80)

    try:
        pre_res = gp.prerank(
            rnk=str(rnk_path),
            gene_sets=str(gmt),
            outdir=str(collection_out),
            min_size=CONF["MIN_SIZE"],
            max_size=CONF["MAX_SIZE"],
            permutation_num=CONF["PERMUTATION_NUM"],
            seed=CONF["SEED"],
            threads=CONF["THREADS"],
            verbose=True,
            no_plot=False,
            format="png",
            graph_num=20,
        )

        res_df = normalize_gsea_results(pre_res.res2d, collection)
        res_df["gmt_file"] = str(gmt)
        res_df["gseapy_outdir"] = str(collection_out)

        collection_result_path = OUT_DIR / f"gsea_full_transcriptome_results_{collection}.csv"
        res_df.to_csv(collection_result_path, index=False)
        print(f"Saved collection result: {collection_result_path}")

        all_results.append(res_df)

    except Exception as e:
        msg = f"FAILED {gmt.name}: {repr(e)}"
        print(msg)
        run_errors.append({"gmt_file": str(gmt), "error": repr(e)})

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
else:
    raise RuntimeError("No GSEA result tables were generated.")

# Stable sort.
combined = combined.sort_values(["FDR_qval", "NOM_pval", "collection", "Term"], ascending=[True, True, True, True])

combined_path = OUT_DIR / "gsea_full_transcriptome_results_all.csv"
sig_path = OUT_DIR / "gsea_full_transcriptome_results_fdr_lt_0_05.csv"
exploratory_path = OUT_DIR / "gsea_full_transcriptome_results_fdr_lt_0_25.csv"

combined.to_csv(combined_path, index=False)
combined[combined["FDR_qval"] < CONF["FDR_SIGNIFICANT"]].to_csv(sig_path, index=False)
combined[combined["FDR_qval"] < CONF["FDR_EXPLORATORY"]].to_csv(exploratory_path, index=False)

print("\n" + "=" * 80)
print("FULL-TRANSCRIPTOME GSEA SUMMARY")
print("=" * 80)
print(f"Total GSEA terms tested after size filters: {len(combined)}")
print(f"FDR < 0.05 terms: {(combined['FDR_qval'] < CONF['FDR_SIGNIFICANT']).sum()}")
print(f"FDR < 0.25 terms: {(combined['FDR_qval'] < CONF['FDR_EXPLORATORY']).sum()}")

print("\nTop GSEA terms by FDR:")
cols_to_show = [c for c in ["collection", "Term", "NES", "NOM_pval", "FDR_qval", "Lead_genes"] if c in combined.columns]
print(combined[cols_to_show].head(25).to_string(index=False))


# =============================================================================
# 5. Figures
# =============================================================================

fig_all = FIG_DIR / "Figure_GSEA_FullTranscriptome_TopTerms_FDR025.png"
made_all = make_dotplot(
    combined,
    fig_all,
    title="Full-transcriptome preranked GSEA: TCGA high-risk vs low-risk",
    top_n=25,
    fdr_limit=CONF["FDR_EXPLORATORY"],
)

hallmark_mask = combined["collection"].astype(str).str.contains("h_all", case=False, na=False)
fig_hallmark = FIG_DIR / "Figure_GSEA_FullTranscriptome_Hallmark_FDR025.png"
made_hallmark = make_dotplot(
    combined[hallmark_mask].copy(),
    fig_hallmark,
    title="Hallmark preranked GSEA: TCGA high-risk vs low-risk",
    top_n=20,
    fdr_limit=CONF["FDR_EXPLORATORY"],
)

print("\nFigures:")
print(f"- {fig_all if made_all else 'No all-term figure generated; no terms at FDR < 0.25.'}")
print(f"- {fig_hallmark if made_hallmark else 'No Hallmark figure generated; no Hallmark terms at FDR < 0.25.'}")


# =============================================================================
# 6. Audit
# =============================================================================

elapsed = time.time() - start_time

audit = {
    "script": "15b_preranked_gsea_tcga_full_transcriptome_median_os_k100.py",
    "purpose": "Strict TCGA full-transcriptome preranked GSEA for revised GIBD manuscript.",
    "methodological_guardrails": {
        "uses_tcga_expression": True,
        "uses_tcga_risk_labels": True,
        "uses_cgga_or_external_labels": False,
        "trains_or_retrains_model": False,
        "performs_feature_selection": False,
        "changes_locked_features": False,
        "changes_locked_threshold": False,
        "changes_hyperparameters": False,
        "used_for_model_optimization": False,
        "analysis_role": "post hoc transcriptome-level pathway interpretation",
    },
    "inputs": {
        "tcga_expression": str(CONF["TCGA_EXPR"]),
        "tcga_labels": str(CONF["TCGA_LABELS"]),
        "gene_map": str(CONF["GENE_MAP"]),
        "gmt_dir": str(CONF["GMT_DIR"]),
        "gmt_files": [str(x) for x in gmt_files],
    },
    "tcga_samples": {
        "n_total_aligned": int(len(y)),
        "risk_label_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()},
        "os_days_available": "OS_days" in labels_df.columns,
        "event_available": "Event" in labels_df.columns,
    },
    "ranking": {
        "ranking_statistic": "signed Welch t-statistic: high-risk expression minus low-risk expression",
        "positive_score_interpretation": "higher expression in TCGA high-risk group",
        "negative_score_interpretation": "higher expression in TCGA low-risk group",
        "n_expression_features_after_numeric_filter": int(X.shape[1]),
        "n_unique_gene_symbols_ranked": int(len(rank_unique)),
        "duplicate_symbol_rows_collapsed": int(len(duplicate_rows)),
        "duplicate_symbol_collapse_rule": "keep feature with largest absolute signed t-statistic per gene symbol",
        "tie_breaking": f"deterministic negligible jitter <= {CONF['TIE_BREAK_EPS']} added only to rank_score_for_gsea",
    },
    "gsea_parameters": {
        "min_size": CONF["MIN_SIZE"],
        "max_size": CONF["MAX_SIZE"],
        "permutation_num": CONF["PERMUTATION_NUM"],
        "seed": CONF["SEED"],
        "threads": CONF["THREADS"],
        "fdr_significant_threshold": CONF["FDR_SIGNIFICANT"],
        "fdr_exploratory_threshold": CONF["FDR_EXPLORATORY"],
    },
    "outputs": {
        "rank_all_features": str(rank_all_path),
        "rank_unique_symbols": str(rank_unique_path),
        "duplicate_symbols_collapsed": str(duplicate_path),
        "rnk_file": str(rnk_path),
        "combined_gsea_results": str(combined_path),
        "fdr_lt_0_05_results": str(sig_path),
        "fdr_lt_0_25_results": str(exploratory_path),
        "gseapy_prerank_dir": str(GSEA_OUT_DIR),
        "figures_dir": str(FIG_DIR),
    },
    "result_counts": {
        "total_terms_in_combined_results": int(len(combined)),
        "terms_fdr_lt_0_05": int((combined["FDR_qval"] < CONF["FDR_SIGNIFICANT"]).sum()),
        "terms_fdr_lt_0_25": int((combined["FDR_qval"] < CONF["FDR_EXPLORATORY"]).sum()),
    },
    "run_errors": run_errors,
    "elapsed_seconds": elapsed,
}

audit_path = OUT_DIR / "gsea_full_transcriptome_audit.json"
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("\nSaved outputs:")
print(f"- Combined GSEA results: {combined_path}")
print(f"- Significant results FDR < 0.05: {sig_path}")
print(f"- Exploratory results FDR < 0.25: {exploratory_path}")
print(f"- Audit: {audit_path}")
print(f"- gseapy reports: {GSEA_OUT_DIR}")

print("\nRecommended interpretation:")
print(
    "Preranked GSEA was performed on a TCGA full-transcriptome ranking comparing "
    "OS-derived high-risk and low-risk groups. Positive NES indicates enrichment among genes "
    "upregulated in the high-risk group; negative NES indicates enrichment among genes upregulated "
    "in the low-risk group. This analysis was post hoc and did not affect the locked GIBD model."
)

print("\nDone.")
