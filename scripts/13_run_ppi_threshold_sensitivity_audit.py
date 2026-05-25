"""
GIBD analysis workflow — STRING PPI threshold sensitivity audit.

This script performs a descriptive graph-coverage audit for STRING confidence
thresholds used in WPPI feature construction. It quantifies retained STRING
edges/nodes, graph-eligible TCGA expression genes, locked-gene coverage, locked
WPPI-associated gene coverage, and Jaccard overlap across thresholds.

Analysis guardrails:
- The final locked GIBD model is not rebuilt, refit, or retuned.
- The locked feature set and operating threshold are not changed.
- CGGA labels and external-validation performance are not read or used.
- The audit is a supplementary graph-construction sensitivity analysis only.

The default threshold set is 400, 700, and 900, with 700 representing the
high-confidence, coverage-preserving cutoff used in the locked WPPI workflow.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration / CLI
# -----------------------------------------------------------------------------

DEFAULTS = {
    "tcga_matrix_candidates": [
        "Data/Revision_Ablation/tcga_weighted_self_graph_cache.csv",
        "Data/TCGA_survival_expression_matrix_enhanced.csv",
        "Data/TCGA_survival_expression_matrix_protein_coding.csv",
    ],
    "locked_features": "Data/Revision_Ablation/16U_Final_Locked_Features.csv",
    "gene_map": "Data/gene_type_map.csv",
    "mapping_cache": "Data/Revision_Ablation/ensembl_gene_protein_map_cache.csv",
    "string_db": "Data/9606.protein.links.full.v12.0.txt.gz",
    "out_dir": "Data/Revision_Ablation/PPI_THRESHOLD_SENSITIVITY_MedianOS_K100",
    "thresholds": [400, 700, 900],
    "min_valid_neighbors": 2,
    "expected_n_features": 100,
    "expected_n_graph_features": 65,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supplementary PPI threshold coverage sensitivity audit for GIBD.")
    p.add_argument("--tcga-matrix", default=None, help="TCGA matrix CSV. If omitted, common project paths are tried.")
    p.add_argument("--locked-features", default=DEFAULTS["locked_features"])
    p.add_argument("--gene-map", default=DEFAULTS["gene_map"])
    p.add_argument("--mapping-cache", default=DEFAULTS["mapping_cache"], help="Cached ENSG-to-ENSP mapping from Script 03b.")
    p.add_argument("--string-db", default=DEFAULTS["string_db"], help="STRING full links file, e.g. 9606.protein.links.full.v12.0.txt.gz")
    p.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    p.add_argument("--thresholds", default=",".join(map(str, DEFAULTS["thresholds"])), help="Comma-separated thresholds, e.g. 400,700,900")
    p.add_argument("--min-valid-neighbors", type=int, default=DEFAULTS["min_valid_neighbors"])
    p.add_argument("--expected-n-features", type=int, default=DEFAULTS["expected_n_features"])
    p.add_argument("--expected-n-graph-features", type=int, default=DEFAULTS["expected_n_graph_features"])
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers copied/adapted from final GIBD graph-feature workflow
# -----------------------------------------------------------------------------


def normalize_feature_name(name: str) -> str:
    text = str(name)
    text = re.sub(r"^(ENSG\d+)\.\d+(_.*)?$", r"\1\2", text)
    return text


def extract_base_gene_id(feature_name: str) -> str:
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


def feature_family(feature_name: str) -> str:
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


def is_graph_feature_name(feature_name: str) -> bool:
    return feature_family(feature_name) != "raw"


def clean_ensg(x: str) -> str:
    s = str(x).strip()
    if "." in s and s.upper().startswith("ENSG"):
        s = s.split(".")[0]
    return s


def resolve_first_existing(candidates: Iterable[str]) -> str:
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("None of these candidate files exists: " + "; ".join(map(str, candidates)))


def load_gene_symbols(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        print(f"WARNING: gene map not found: {path}. Symbols will fall back to ENSG IDs.")
        return {}
    gm = pd.read_csv(path)
    id_col = next((c for c in ["clean_gene_id", "gene_id", "ensembl_gene_id", "Gene_ID", "id"] if c in gm.columns), None)
    sym_col = next((c for c in ["gene_name", "gene_symbol", "symbol", "external_gene_name", "Gene_Name"] if c in gm.columns), None)
    if id_col is None or sym_col is None:
        print(f"WARNING: could not detect gene map columns in {path}. Columns: {list(gm.columns)}")
        return {}
    gm = gm[[id_col, sym_col]].dropna()
    gm[id_col] = gm[id_col].astype(str).map(clean_ensg)
    gm[sym_col] = gm[sym_col].astype(str).str.strip()
    gm = gm.drop_duplicates(subset=[id_col], keep="first")
    return dict(zip(gm[id_col], gm[sym_col]))


def load_expression_gene_universe(matrix_path: str) -> Set[str]:
    print(f"Loading TCGA expression gene universe from header: {matrix_path}")
    header = pd.read_csv(matrix_path, nrows=0).columns.astype(str).tolist()
    genes = []
    for col in header:
        base = extract_base_gene_id(col)
        base = clean_ensg(base)
        if base.upper().startswith("ENSG"):
            # Strip old graph-derived columns automatically.
            if not is_graph_feature_name(col):
                genes.append(base)
    genes = sorted(set(genes))
    if not genes:
        raise ValueError(
            f"No ENSG-like base expression columns detected in {matrix_path}. "
            "Check orientation and whether this is the correct TCGA matrix."
        )
    print(f"Expression base genes detected: {len(genes):,}")
    return set(genes)


def load_locked_features(path: str, gene_symbols: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading locked features: {path}")
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError("Locked features file must contain a 'feature' column.")
    df["feature"] = df["feature"].astype(str).map(normalize_feature_name)
    df["base_gene_id"] = df["feature"].map(extract_base_gene_id).map(clean_ensg)
    df["feature_family"] = df["feature"].map(feature_family)
    if "is_graph_feature" not in df.columns:
        df["is_graph_feature"] = df["feature"].map(is_graph_feature_name)
    else:
        df["is_graph_feature"] = df["is_graph_feature"].astype(bool) | df["feature"].map(is_graph_feature_name)
    df["gene_symbol"] = df["base_gene_id"].map(lambda g: gene_symbols.get(g, g))

    summary = (
        df.groupby(["base_gene_id", "gene_symbol"], as_index=False)
        .agg(
            n_locked_features=("feature", "count"),
            n_locked_graph_features=("is_graph_feature", "sum"),
            feature_families=("feature_family", lambda x: ";".join(sorted(set(map(str, x))))),
            example_features=("feature", lambda x: ";".join(list(map(str, x))[:5])),
        )
    )
    summary["has_locked_wppi_feature"] = summary["n_locked_graph_features"] > 0
    print(f"Locked features: {len(df):,}")
    print(f"Locked unique genes: {summary['base_gene_id'].nunique():,}")
    print(f"Locked genes with WPPI/graph selected feature: {summary['has_locked_wppi_feature'].sum():,}")
    return df, summary


def load_ensg_ensp_mapping(path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mapping cache not found: {path}\n"
            "Run 03b_weighted_self_graph_features.py once to create this cache, or pass --mapping-cache."
        )
    print(f"Loading ENSG-to-ENSP mapping cache: {path}")
    m = pd.read_csv(path).dropna(subset=["ENSG", "ENSP"])
    ensg_to_ensps: Dict[str, List[str]] = defaultdict(list)
    ensp_to_ensgs: Dict[str, List[str]] = defaultdict(list)
    for _, row in m.iterrows():
        ensg = clean_ensg(row["ENSG"])
        ensp = str(row["ENSP"]).replace("9606.", "").strip()
        if not ensg or not ensp or ensp.lower() == "nan":
            continue
        ensg_to_ensps[ensg].append(ensp)
        ensp_to_ensgs[ensp].append(ensg)
    # de-duplicate lists
    ensg_to_ensps = {k: sorted(set(v)) for k, v in ensg_to_ensps.items()}
    ensp_to_ensgs = {k: sorted(set(v)) for k, v in ensp_to_ensgs.items()}
    print(f"Genes with protein mapping: {len(ensg_to_ensps):,}")
    print(f"Proteins with gene mapping: {len(ensp_to_ensgs):,}")
    return ensg_to_ensps, ensp_to_ensgs


def load_string_links(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"STRING file not found: {path}")
    print(f"Loading STRING links once: {path}")
    sdf = pd.read_csv(path, sep=" ", usecols=["protein1", "protein2", "combined_score"])
    sdf["protein1"] = sdf["protein1"].astype(str).str.replace("9606.", "", regex=False)
    sdf["protein2"] = sdf["protein2"].astype(str).str.replace("9606.", "", regex=False)
    sdf["combined_score"] = pd.to_numeric(sdf["combined_score"], errors="coerce").fillna(0).astype(int)
    print(f"STRING links loaded: {len(sdf):,}")
    return sdf


def build_gene_neighbor_counts(
    sdf: pd.DataFrame,
    threshold: int,
    expression_genes: Set[str],
    ensp_to_ensgs: Dict[str, List[str]],
) -> Tuple[Dict[str, int], int, int]:
    """Return gene -> number of valid expression-neighbor genes at a STRING threshold."""
    sub = sdf[sdf["combined_score"] >= int(threshold)]
    edge_count = int(len(sub))
    node_count = int(len(set(sub["protein1"]) | set(sub["protein2"]))) if edge_count else 0

    neighbors: Dict[str, Set[str]] = defaultdict(set)

    # Pre-filter protein mappings to expression genes only.
    def expr_genes_for_protein(p: str) -> List[str]:
        return [g for g in ensp_to_ensgs.get(p, []) if g in expression_genes]

    for p1, p2 in zip(sub["protein1"].values, sub["protein2"].values):
        genes1 = expr_genes_for_protein(str(p1))
        genes2 = expr_genes_for_protein(str(p2))
        if not genes1 or not genes2:
            continue
        for g1 in genes1:
            for g2 in genes2:
                if g1 != g2:
                    neighbors[g1].add(g2)
                    neighbors[g2].add(g1)

    counts = {g: len(neighbors.get(g, set())) for g in expression_genes}
    return counts, edge_count, node_count


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    return float(len(a & b) / denom) if denom else 0.0


# -----------------------------------------------------------------------------
# Main audit
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    thresholds = [int(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    thresholds = sorted(set(thresholds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tcga_matrix = args.tcga_matrix or resolve_first_existing(DEFAULTS["tcga_matrix_candidates"])

    gene_symbols = load_gene_symbols(args.gene_map)
    expression_genes = load_expression_gene_universe(tcga_matrix)
    locked_features, locked_gene_summary = load_locked_features(args.locked_features, gene_symbols)
    if len(locked_features) != int(args.expected_n_features):
        raise ValueError(f"Expected locked K100 features={args.expected_n_features}, got {len(locked_features)}")
    n_graph_locked = int(locked_features["is_graph_feature"].sum())
    if n_graph_locked != int(args.expected_n_graph_features):
        raise ValueError(f"Expected locked graph/WPPI features={args.expected_n_graph_features}, got {n_graph_locked}")
    ensg_to_ensps, ensp_to_ensgs = load_ensg_ensp_mapping(args.mapping_cache)
    sdf = load_string_links(args.string_db)

    locked_genes = set(locked_gene_summary["base_gene_id"])
    locked_wppi_genes = set(locked_gene_summary.loc[locked_gene_summary["has_locked_wppi_feature"], "base_gene_id"])

    summary_rows = []
    coverage_rows = []
    eligible_sets = {
        "all_expression_genes": {},
        "locked_unique_genes": {},
        "locked_wppi_associated_genes": {},
    }

    for thr in thresholds:
        print("\n" + "=" * 80)
        print(f"Auditing STRING threshold >= {thr}")
        print("=" * 80)
        counts, edge_count, node_count = build_gene_neighbor_counts(sdf, thr, expression_genes, ensp_to_ensgs)

        expr_with_1 = {g for g, n in counts.items() if n >= 1}
        expr_eligible = {g for g, n in counts.items() if n >= args.min_valid_neighbors}
        locked_eligible = locked_genes & expr_eligible
        locked_wppi_eligible = locked_wppi_genes & expr_eligible

        eligible_sets["all_expression_genes"][thr] = expr_eligible
        eligible_sets["locked_unique_genes"][thr] = locked_eligible
        eligible_sets["locked_wppi_associated_genes"][thr] = locked_wppi_eligible

        summary_rows.append({
            "string_threshold": thr,
            "string_edges_retained": edge_count,
            "string_protein_nodes_retained": node_count,
            "expression_genes_total": len(expression_genes),
            "expression_genes_with_at_least_1_neighbor": len(expr_with_1),
            "expression_genes_graph_eligible_min_neighbors": len(expr_eligible),
            "expression_gene_graph_eligibility_fraction": len(expr_eligible) / len(expression_genes),
            "locked_unique_genes_total": len(locked_genes),
            "locked_unique_genes_graph_eligible": len(locked_eligible),
            "locked_unique_gene_graph_eligibility_fraction": len(locked_eligible) / len(locked_genes) if locked_genes else np.nan,
            "locked_wppi_associated_genes_total": len(locked_wppi_genes),
            "locked_wppi_associated_genes_graph_eligible": len(locked_wppi_eligible),
            "locked_wppi_gene_graph_eligibility_fraction": len(locked_wppi_eligible) / len(locked_wppi_genes) if locked_wppi_genes else np.nan,
            "min_valid_neighbors": args.min_valid_neighbors,
        })

        for _, row in locked_gene_summary.iterrows():
            gene = row["base_gene_id"]
            n_neighbors = int(counts.get(gene, 0))
            coverage_rows.append({
                "string_threshold": thr,
                "base_gene_id": gene,
                "gene_symbol": row["gene_symbol"],
                "n_locked_features": int(row["n_locked_features"]),
                "n_locked_graph_features": int(row["n_locked_graph_features"]),
                "has_locked_wppi_feature": bool(row["has_locked_wppi_feature"]),
                "n_valid_expression_neighbors": n_neighbors,
                "graph_eligible_min_neighbors": n_neighbors >= args.min_valid_neighbors,
                "feature_families": row["feature_families"],
                "example_features": row["example_features"],
            })

        print(f"STRING edges retained: {edge_count:,}")
        print(f"Expression genes graph-eligible: {len(expr_eligible):,}/{len(expression_genes):,}")
        print(f"Locked unique genes graph-eligible: {len(locked_eligible):,}/{len(locked_genes):,}")
        print(f"Locked WPPI-associated genes graph-eligible: {len(locked_wppi_eligible):,}/{len(locked_wppi_genes):,}")

    # Pairwise Jaccard overlap matrix for graph-eligible sets.
    overlap_rows = []
    for set_name, by_thr in eligible_sets.items():
        for t1 in thresholds:
            for t2 in thresholds:
                a = by_thr.get(t1, set())
                b = by_thr.get(t2, set())
                overlap_rows.append({
                    "set_name": set_name,
                    "threshold_1": t1,
                    "threshold_2": t2,
                    "n_threshold_1": len(a),
                    "n_threshold_2": len(b),
                    "n_intersection": len(a & b),
                    "n_union": len(a | b),
                    "jaccard": jaccard(a, b),
                })

    summary_df = pd.DataFrame(summary_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    summary_path = out_dir / "ppi_threshold_sensitivity_summary.csv"
    coverage_path = out_dir / "ppi_threshold_sensitivity_locked_gene_coverage.csv"
    overlap_path = out_dir / "ppi_threshold_sensitivity_overlap_matrix.csv"
    audit_path = out_dir / "ppi_threshold_sensitivity_audit.json"

    summary_df.to_csv(summary_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    audit = {
        "script": "17_ppi_threshold_sensitivity_audit_median_os_k100.py",
        "analysis_role": "Supplementary descriptive graph-construction sensitivity audit only.",
        "no_model_retraining": True,
        "no_feature_reselection": True,
        "no_threshold_change_to_locked_model": True,
        "no_cgga_label_use": True,
        "not_used_for_model_selection": True,
        "inputs": {
            "tcga_matrix": str(tcga_matrix),
            "locked_features": str(args.locked_features),
            "gene_map": str(args.gene_map),
            "mapping_cache": str(args.mapping_cache),
            "string_db": str(args.string_db),
        },
        "thresholds": thresholds,
        "min_valid_neighbors": args.min_valid_neighbors,
        "outputs": {
            "summary": str(summary_path),
            "locked_gene_coverage": str(coverage_path),
            "overlap_matrix": str(overlap_path),
        },
        "recommended_interpretation": (
            "The audit quantifies graph coverage at STRING confidence thresholds 400/700/900. "
            "It supports the use of 700 as a high-confidence threshold balancing coverage and noise reduction, "
            "but it does not modify the frozen model or external validation results."
        ),
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\nSaved outputs:")
    print(f"- {summary_path}")
    print(f"- {coverage_path}")
    print(f"- {overlap_path}")
    print(f"- {audit_path}")

    print("\nSummary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
