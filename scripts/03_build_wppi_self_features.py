"""
GIBD analysis workflow — WPPI-self feature construction

This script builds weighted protein-protein interaction (WPPI) self-preserving
features from log-CPM-normalized expression matrices and STRING v12 topology.
For each eligible target gene, measured non-self network neighbors are combined
using STRING confidence weights, and alpha-specific self-preserving features are
created by mixing target-gene expression with the weighted neighbor signal.

Analysis guardrails:
- STRING topology is used as an external graph prior, not as patient-specific
  proteomic evidence.
- Unmapped or unmeasured neighbors are excluded rather than zero-filled.
- WPPI-self features are generated only when at least two measured non-self
  neighbors are available.
- This script constructs feature matrices only; it does not train models, select
  thresholds, or evaluate performance.
"""

import os
import sys
import urllib.parse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CONF = {
    "OUT_DIR": os.path.join("Data", "Revision_Ablation"),

    # Input matrices may contain prior graph-derived columns; these are stripped before rebuilding WPPI-self features.
    "TCGA_INPUT_MATRIX": os.path.join("Data", "TCGA_survival_expression_matrix_enhanced.csv"),
    "CGGA_INPUT_MATRIX": os.path.join("Data", "Revision_Ablation", "cgga_graph_informed_cache.csv"),

    "STRING_DB": os.path.join("Data", "9606.protein.links.full.v12.0.txt.gz"),
    "MAPPING_CACHE": os.path.join("Data", "Revision_Ablation", "ensembl_gene_protein_map_cache.csv"),

    "TCGA_OUTPUT_MATRIX": os.path.join("Data", "Revision_Ablation", "tcga_weighted_self_graph_cache.csv"),
    "CGGA_OUTPUT_MATRIX": os.path.join("Data", "Revision_Ablation", "cgga_weighted_self_graph_cache.csv"),
    "DIAGNOSTIC_OUTPUT": os.path.join("Data", "Revision_Ablation", "weighted_self_graph_feature_diagnostic.csv"),

    "CONFIDENCE_THRESHOLD": 700,
    "MIN_VALID_NEIGHBORS": 2,

    # Three pre-specified self-loop strengths. 0.50 is the main biologically balanced option.
    "SELF_ALPHA_VALUES": [0.25, 0.50, 0.75],

    # Keep False unless you explicitly want pure weighted-neighbor features as an additional ablation.
    "MAKE_PURE_WEIGHTED_NEIGHBOR": False,
}

os.makedirs(CONF["OUT_DIR"], exist_ok=True)


def is_graph_feature(name: str) -> bool:
    name = str(name).lower()
    tokens = [
        "neighbor_mean", "_nbr_mean", "nbr_mean", "network_mean", "graph_mean",
        "_wppi_self", "_wppi_neighbor",
    ]
    return any(t in name for t in tokens)


def load_matrix_base_only(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input matrix not found: {path}")

    print(f"Loading matrix: {path}")
    X = pd.read_csv(path, index_col=0)
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str).str.split(".").str[0]
    X = X.loc[:, ~X.columns.duplicated()].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    base_cols = [c for c in X.columns if not is_graph_feature(c)]
    if not base_cols:
        raise ValueError(f"No base expression columns found after stripping graph columns: {path}")

    Xb = X[base_cols].copy()
    print(f"  Original columns: {X.shape[1]}")
    print(f"  Base expression columns retained: {Xb.shape[1]}")
    return Xb


def fetch_or_load_mapping() -> pd.DataFrame:
    cache = CONF["MAPPING_CACHE"]
    if os.path.exists(cache):
        print(f"Loading cached Ensembl gene-protein mapping: {cache}")
        m = pd.read_csv(cache)
        m = m.dropna(subset=["ENSG", "ENSP"])
        return m

    print("Fetching Ensembl ENSG-to-ENSP mapping from BioMart.")
    xml_query = """<?xml version='1.0' encoding='UTF-8'?>
    <!DOCTYPE Query>
    <Query virtualSchemaName='default' formatter='TSV' header='0' uniqueRows='0' count='' datasetConfigVersion='0.6'>
        <Dataset name='hsapiens_gene_ensembl' interface='default'>
            <Attribute name='ensembl_gene_id'/>
            <Attribute name='ensembl_peptide_id'/>
        </Dataset>
    </Query>"""
    url = "https://www.ensembl.org/biomart/martservice?query=" + urllib.parse.quote_plus(xml_query)

    try:
        m = pd.read_csv(url, sep="\t", names=["ENSG", "ENSP"])
        m = m.dropna(subset=["ENSG", "ENSP"])
        m["ENSG"] = m["ENSG"].astype(str).str.split(".").str[0]
        m["ENSP"] = m["ENSP"].astype(str).str.replace("9606.", "", regex=False)
        m = m.drop_duplicates()
        m.to_csv(cache, index=False)
        print(f"Saved mapping cache: {cache}")
        return m
    except Exception as exc:
        raise RuntimeError(
            "BioMart mapping failed and no cached mapping file was found. "
            f"Expected cache path: {cache}. Original error: {exc}"
        )


def build_mapping_dicts(mapping_df: pd.DataFrame):
    ensg_to_ensps = defaultdict(list)
    ensp_to_ensgs = defaultdict(list)

    for _, row in mapping_df.iterrows():
        ensg = str(row["ENSG"]).split(".")[0]
        ensp = str(row["ENSP"]).replace("9606.", "")
        if not ensg or not ensp or ensp == "nan":
            continue
        ensg_to_ensps[ensg].append(ensp)
        ensp_to_ensgs[ensp].append(ensg)

    return dict(ensg_to_ensps), dict(ensp_to_ensgs)


def load_string_edges() -> dict:
    path = CONF["STRING_DB"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"STRING file not found: {path}")

    print(f"Loading STRING links: {path}")
    sdf = pd.read_csv(path, sep=" ")
    sdf["protein1"] = sdf["protein1"].astype(str).str.replace("9606.", "", regex=False)
    sdf["protein2"] = sdf["protein2"].astype(str).str.replace("9606.", "", regex=False)
    sdf = sdf[sdf["combined_score"] >= int(CONF["CONFIDENCE_THRESHOLD"])].copy()

    adjacency = defaultdict(list)
    for p1, p2, score in zip(sdf["protein1"], sdf["protein2"], sdf["combined_score"]):
        w = float(score) / 1000.0
        adjacency[p1].append((p2, w))
        adjacency[p2].append((p1, w))

    print(f"High-confidence STRING edges retained: {len(sdf)}")
    print(f"STRING nodes with adjacency: {len(adjacency)}")
    return dict(adjacency)


def build_weighted_self_features(X_base: pd.DataFrame, adjacency: dict, ensg_to_ensps: dict, ensp_to_ensgs: dict):
    genes = list(X_base.columns)
    gene_set = set(genes)

    new_arrays = []
    new_names = []
    diagnostic_rows = []

    X_values = X_base  # keep as DataFrame for clear column access

    for i, gene in enumerate(genes, start=1):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(genes)} genes", flush=True)

        proteins = ensg_to_ensps.get(gene, [])
        neighbor_weight = defaultdict(float)

        for prot in proteins:
            for nbr_prot, w in adjacency.get(prot, []):
                for nbr_gene in ensp_to_ensgs.get(nbr_prot, []):
                    nbr_gene = str(nbr_gene).split(".")[0]
                    if nbr_gene in gene_set and nbr_gene != gene:
                        # If multiple protein mappings lead to the same gene, accumulate their STRING confidence weights.
                        neighbor_weight[nbr_gene] += float(w)

        valid_neighbors = [g for g, w in neighbor_weight.items() if w > 0 and g in gene_set]
        if len(valid_neighbors) < int(CONF["MIN_VALID_NEIGHBORS"]):
            diagnostic_rows.append({
                "gene": gene,
                "n_valid_neighbors": len(valid_neighbors),
                "feature_created": False,
                "total_weight": 0.0,
            })
            continue

        weights = np.array([neighbor_weight[g] for g in valid_neighbors], dtype=float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        weights = weights / weight_sum

        neighbor_matrix = X_values[valid_neighbors].values.astype(float)
        weighted_neighbor = np.dot(neighbor_matrix, weights)
        self_values = X_values[gene].values.astype(float)

        if CONF["MAKE_PURE_WEIGHTED_NEIGHBOR"]:
            new_arrays.append(weighted_neighbor)
            new_names.append(f"{gene}_wppi_neighbor")

        for alpha in CONF["SELF_ALPHA_VALUES"]:
            alpha = float(alpha)
            vals = (alpha * self_values) + ((1.0 - alpha) * weighted_neighbor)
            suffix = str(int(round(alpha * 100))).zfill(2)
            new_arrays.append(vals)
            new_names.append(f"{gene}_wppi_self{suffix}")

        diagnostic_rows.append({
            "gene": gene,
            "n_valid_neighbors": len(valid_neighbors),
            "feature_created": True,
            "total_weight": weight_sum,
        })

    if not new_arrays:
        raise RuntimeError("No weighted graph features were created. Check mapping and STRING paths.")

    graph_df = pd.DataFrame(
        np.column_stack(new_arrays),
        index=X_base.index,
        columns=new_names,
    )

    out = pd.concat([X_base, graph_df], axis=1)
    diag = pd.DataFrame(diagnostic_rows)

    print(f"  Base features: {X_base.shape[1]}")
    print(f"  Weighted/self graph features: {graph_df.shape[1]}")
    print(f"  Final matrix shape: {out.shape}")
    return out, diag


def main():
    mapping_df = fetch_or_load_mapping()
    ensg_to_ensps, ensp_to_ensgs = build_mapping_dicts(mapping_df)
    adjacency = load_string_edges()

    all_diagnostics = []

    for label, input_path, output_path in [
        ("TCGA", CONF["TCGA_INPUT_MATRIX"], CONF["TCGA_OUTPUT_MATRIX"]),
        ("OLD_CGGA", CONF["CGGA_INPUT_MATRIX"], CONF["CGGA_OUTPUT_MATRIX"]),
    ]:
        print("\n" + "=" * 72)
        print(f"Building weighted/self graph matrix for {label}")
        print("=" * 72)

        X_base = load_matrix_base_only(input_path)
        X_new, diag = build_weighted_self_features(X_base, adjacency, ensg_to_ensps, ensp_to_ensgs)
        X_new.to_csv(output_path)

        diag.insert(0, "cohort", label)
        all_diagnostics.append(diag)
        print(f"Saved: {output_path}")

    diagnostic = pd.concat(all_diagnostics, axis=0, ignore_index=True)
    diagnostic.to_csv(CONF["DIAGNOSTIC_OUTPUT"], index=False)
    print("\nSaved diagnostic:", CONF["DIAGNOSTIC_OUTPUT"])
    print("\nNEXT STEP:")
    print("Run 16f_multiroute_generalization_boost.py. It will automatically use these weighted/self graph matrices if present.")


if __name__ == "__main__":
    main()
