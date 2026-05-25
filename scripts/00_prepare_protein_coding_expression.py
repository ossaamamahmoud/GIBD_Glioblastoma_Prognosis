"""
GIBD analysis workflow — protein-coding gene filtering

This script filters the TCGA expression matrix to retain protein-coding genes
and exports a gene annotation map for downstream feature construction. The
output protein-coding expression matrix is used as the measured transcriptomic
input for subsequent WPPI feature construction and model-development steps.

Inputs and outputs are configured relative to the project directory under Data/.
No model fitting, feature selection, threshold selection, or external-validation
analysis is performed in this script.
"""

import os
import pandas as pd

# --- Configuration ---
CONF = {
    'DATA_DIR': 'Data',
    'DOWNLOAD_DIR': os.path.join('Data', 'GDC_Downloads'),
    'MANIFEST_FILE': os.path.join('Data', 'gdc_manifest.txt'),
    'FULL_MATRIX': os.path.join('Data', 'TCGA_survival_expression_matrix.csv'),
    # Outputs
    'OUTPUT_MAP': os.path.join('Data', 'gene_type_map.csv'),
    'OUTPUT_MATRIX': os.path.join('Data', 'TCGA_survival_expression_matrix_protein_coding.csv')
}


def get_gene_metadata(manifest_path, download_dir):
    """Extracts gene type and name information from a representative GDC sample file."""
    print("Extracting gene metadata from raw GDC files...")
    manifest = pd.read_csv(manifest_path, sep='\t')

    # Find the first available sample folder to read metadata from
    sample_path = None
    for _, row in manifest.iterrows():
        path = os.path.join(download_dir, row['id'])
        if os.path.isdir(path):
            sample_path = path
            break

    if not sample_path:
        raise FileNotFoundError("No valid sample directories found in GDC_Downloads.")

    # Find the TSV file within that folder
    tsv_file = next((f for f in os.listdir(sample_path) if f.endswith(".tsv")), None)
    if not tsv_file:
        raise FileNotFoundError(f"No TSV file found in {sample_path}")

    # Load and process the file to get gene info
    df = pd.read_csv(os.path.join(sample_path, tsv_file), sep='\t', comment='#')

    # Filter for Ensembl IDs only
    df = df[df['gene_id'].str.startswith('ENSG')]
    df['clean_gene_id'] = df['gene_id'].str.split('.').str[0]

    # Return mapping DataFrame (indexed by clean_gene_id)
    # Keeping 'gene_name' and 'gene_type'
    return df[['clean_gene_id', 'gene_name', 'gene_type']].drop_duplicates().set_index('clean_gene_id')


def main():
    # 1. Generate Gene Map
    gene_map = get_gene_metadata(CONF['MANIFEST_FILE'], CONF['DOWNLOAD_DIR'])
    gene_map.to_csv(CONF['OUTPUT_MAP'])
    print(f"Gene map generated and saved. Total gene entries: {len(gene_map)}")

    # 2. Load Full Expression Matrix
    if not os.path.exists(CONF['FULL_MATRIX']):
        raise FileNotFoundError(f"Input matrix not found: {CONF['FULL_MATRIX']}")

    print(f"Loading full expression matrix from {CONF['FULL_MATRIX']}...")
    full_matrix = pd.read_csv(CONF['FULL_MATRIX'], index_col=0)

    # 3. Filter for Protein Coding Genes
    print("Filtering for protein-coding genes...")
    coding_genes = gene_map[gene_map['gene_type'] == 'protein_coding'].index

    # Intersect with matrix columns to ensure we only keep genes present in the data
    valid_genes = full_matrix.columns.intersection(coding_genes)
    filtered_matrix = full_matrix[valid_genes]

    # 4. Save Filtered Matrix
    filtered_matrix.to_csv(CONF['OUTPUT_MATRIX'])

    print("Filtering complete.")
    print(f"Original features: {full_matrix.shape[1]}")
    print(f"Protein-coding features: {filtered_matrix.shape[1]}")
    print(f"Output saved to: {CONF['OUTPUT_MATRIX']}")


if __name__ == "__main__":
    main()