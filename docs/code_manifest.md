# GIBD code manifest

This manifest lists the public-facing analysis scripts in the recommended execution order. The numbering is intended to make the workflow readable and reproducible; it does not alter the underlying analysis logic.

| Step | Script | Inputs required before running | Principal outputs |
|---:|---|---|---|
| 00 | `00_prepare_protein_coding_expression.py` | TCGA expression matrix, GDC manifest/downloads | Protein-coding TCGA matrix, gene map |
| 01 | `01_generate_tcga_median_os_labels.py` | TCGA clinical JSON, protein-coding TCGA matrix | TCGA OS/Event/Risk_Label files, median audit |
| 02 | `02_generate_cgga_tcga_cutoff_labels.py` | CGGA cohort cache, CGGA clinical file, TCGA median audit | Binary-eligible CGGA labels and full audit |
| 03 | `03_build_wppi_self_features.py` | TCGA/CGGA matrices, STRING v12 links, gene-protein mapping | WPPI-self feature matrices and diagnostics |
| 04 | `04_lock_tcga_gibd_xgboost_k100.py` | TCGA WPPI matrix, TCGA labels | Locked K100 model artifacts, locked features, OOF predictions |
| 05 | `05_run_ablation_comparators.py` | Locked artifacts, TCGA/CGGA matrices and labels | Ablation summary, external validation metrics, audit JSON |
| 06 | `06_plot_figure3_ablation_performance.py` | Ablation result CSV files | Figure 3 plot data and composite figure |
| 07 | `07_run_calibration_and_dca.py` | Locked artifacts, OOF predictions, CGGA matrix/labels | Calibration/DCA metrics and figures |
| 08 | `08_run_clinical_covariate_benchmark.py` | Clinical covariates, locked GIBD scores | Clinical benchmark summary and figure |
| 09 | `09_run_shap_interpretability.py` | Locked model, TCGA matrix/labels | SHAP rankings, values, figures, audit JSON |
| 10 | `10_run_cgga_lime_explanations.py` | Locked model, TCGA background, CGGA matrix/labels | LIME case explanations and figures |
| 11 | `11_run_tcga_full_transcriptome_gsea.py` | TCGA expression, TCGA labels, GMT files | Full-transcriptome GSEA tables and figures |
| 12 | `12_run_locked_signature_ora.py` | Locked features, gene map, SHAP ranking, GMT files | ORA tables, dotplots, audit JSON |
| 13 | `13_run_ppi_threshold_sensitivity_audit.py` | STRING links, mapping cache, locked features | Threshold sensitivity tables and audit JSON |
