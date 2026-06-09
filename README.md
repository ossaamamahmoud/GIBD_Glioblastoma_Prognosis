# GIBD Glioblastoma Prognosis

Public-facing analysis workflow for:

**GIBD: Graph-Informed Biomarker Discovery-Driven Machine Learning Framework for Glioblastoma Prognosis**

This repository provides scripts supporting a locked, transcriptomics-only, graph-informed glioblastoma prognosis workflow. The workflow is intended for reproducibility, methodological transparency, and code review. Patient-level TCGA/CGGA-derived matrices, clinical files, prediction tables, fitted model objects, and local intermediate artifacts are not redistributed; users must obtain authorized input data from the original data sources.

## Study summary

GIBD integrates RNA-seq expression with high-confidence STRING topology through weighted protein-protein interaction (WPPI) self-preserving feature construction. Model development, feature selection, scaler fitting, threshold selection, and final model locking were performed using TCGA only. CGGA was reserved for post-lock external validation.

The locked model is interpreted as a transcriptomic risk-prioritization signal, not as a clinically validated decision tool, treatment-allocation rule, causal biological model, or patient-specific PPI activity assay.

## Key reported locked-workflow results

- Final model: GIBD-XGBoost K100
- Locked feature count: 100 features
  - 65 WPPI-derived features
  - 35 raw-expression features
- Locked classification threshold: 0.53
- TCGA development cohort: 147 patients
- TCGA empirical median OS cutoff: 357 days
- CGGA binary-evaluable external validation cohort: 131 patients
- TCGA out-of-fold AUC: 0.617
- CGGA external AUC: 0.609
- CGGA external sensitivity: 73.9%
- CGGA external specificity: 50.6%
- CGGA external balanced accuracy: 62.3%
- CGGA C-index: 0.537

## Repository layout

```text
GIBD_Glioblastoma_Prognosis/
├── README.md
├── requirements_freeze.txt
├── .gitignore
├── scripts/
│   ├── 00_prepare_protein_coding_expression.py
│   ├── 01_generate_tcga_median_os_labels.py
│   ├── 02_generate_cgga_tcga_cutoff_labels.py
│   ├── 03_build_wppi_self_features.py
│   ├── 04_lock_tcga_gibd_xgboost_k100.py
│   ├── 05_run_ablation_comparators.py
│   ├── 06_plot_figure3_ablation_performance.py
│   ├── 07_run_calibration_and_dca.py
│   ├── 08_run_clinical_covariate_benchmark.py
│   ├── 09_run_shap_interpretability.py
│   ├── 10_run_cgga_lime_explanations.py
│   ├── 11_run_tcga_full_transcriptome_gsea.py
│   ├── 12_run_locked_signature_ora.py
│   └── 13_run_ppi_threshold_sensitivity_audit.py
├── docs/
│   ├── code_manifest.md
│   └── repository_structure.md
└── figures/
    ├── main/
    └── supplementary/
```

## Script run order

| Step | Script | Purpose |
|---:|---|---|
| 00 | `00_prepare_protein_coding_expression.py` | Prepare protein-coding TCGA expression features. |
| 01 | `01_generate_tcga_median_os_labels.py` | Generate TCGA OS/Event/Risk_Label using the empirical TCGA median OS cutoff. |
| 02 | `02_generate_cgga_tcga_cutoff_labels.py` | Generate censor-aware CGGA binary labels using the TCGA-derived cutoff. |
| 03 | `03_build_wppi_self_features.py` | Build STRING-confidence-weighted WPPI-self features. |
| 04 | `04_lock_tcga_gibd_xgboost_k100.py` | Perform TCGA-only model development and lock the final GIBD-XGBoost K100 model. |
| 05 | `05_run_ablation_comparators.py` | Run ablation/comparator models and post-lock CGGA evaluation. |
| 06 | `06_plot_figure3_ablation_performance.py` | Generate the ablation/external-validation performance figure. |
| 07 | `07_run_calibration_and_dca.py` | Run calibration and decision-curve analyses. |
| 08 | `08_run_clinical_covariate_benchmark.py` | Run clinical-covariate benchmark analyses. |
| 09 | `09_run_shap_interpretability.py` | Run post-lock SHAP global and interaction analyses. |
| 10 | `10_run_cgga_lime_explanations.py` | Run post-lock LIME explanations for representative CGGA cases. |
| 11 | `11_run_tcga_full_transcriptome_gsea.py` | Run TCGA-only full-transcriptome preranked GSEA. |
| 12 | `12_run_locked_signature_ora.py` | Run locked-signature ORA and pathway cross-checks. |
| 13 | `13_run_ppi_threshold_sensitivity_audit.py` | Run supplementary PPI threshold sensitivity auditing. |

## Data requirements

This repository does not redistribute patient-level input data. To execute the workflow, users must obtain authorized input data from the original resources and organize them locally.

Expected local data resources include:

- TCGA transcriptomic and clinical data
- CGGA transcriptomic and clinical data
- STRING v12 protein-protein interaction files
- local GMT pathway collections for GSEA/ORA analyses

A typical local-only input structure is:

```text
Data/
├── clinical.json
├── gdc_manifest.txt
├── TCGA_survival_expression_matrix.csv
├── TCGA_survival_expression_matrix_protein_coding.csv
├── gene_type_map.csv
├── 9606.protein.links.full.v12.0.txt.gz
├── GSEA/
│   └── gmt/
│       ├── h.all.*.Hs.symbols.gmt
│       ├── c2.cp.*.Hs.symbols.gmt
│       └── c5.go.bp.*.Hs.symbols.gmt
└── CGGA_Data/
    ├── CGGA.mRNAseq_693_clinical.20200506.txt
    └── CGGA expression input files
```

## Data access and redistribution policy

This repository does not redistribute patient-level TCGA/CGGA-derived matrices, clinical files, prediction tables, fitted model objects, or local intermediate artifacts. Users must obtain authorized input data from the original data sources and follow the relevant data-use terms for TCGA, CGGA, STRING, MSigDB, Reactome, and related resources.

## Reproducibility notes

The repository provides public-facing scripts and documentation for the final locked workflow. Full re-execution requires access to the authorized source data and third-party resources listed above. The manuscript and Supplementary Information describe the locked artifacts, model hyperparameters, stochastic seeds, major package versions, and fixed script run order.

## License, citation, and authorship

This repository is shared for academic review, methodological transparency, and
non-commercial reproducibility of the associated GIBD study. It is not released
as unrestricted open-source software.

The code, documentation, figures, workflow design, reported results, and
analysis structure remain the intellectual work of the author unless otherwise
stated. Use of this repository requires proper citation of the associated
manuscript and this repository. Redistribution, commercial use, sublicensing, or
presentation of this work as another person's original research is not permitted
without prior written permission.

See `LICENSE` for the full research-use and citation terms, and `CITATION.cff`
for citation metadata.

## Methodological guardrails

- TCGA was used for model development and locking.
- CGGA was reserved for post-lock external validation.
- CGGA was not used for feature selection, hyperparameter selection, threshold selection, scaling-parameter fitting, model refitting, or recalibration.
- SHAP, LIME, GSEA, ORA, Enrichr-style analyses, and PPI threshold auditing were post-lock interpretive or contextual analyses.
- STRING topology was used as an external graph prior, not as patient-specific proteomic evidence.

## Interpretation limits

The final GIBD model should be interpreted as a transcriptomic risk-prioritization signal requiring further prospective recalibration, larger multicenter validation, multimodal integration, and clinical decision-analytic evaluation before translational use.

The repository does not support claims of clinical deployment readiness, treatment allocation, causal biology, patient-specific PPI activity, or validated biomarker mechanism.

## License, citation, and authorship

This repository is released under the MIT License for the public code and documentation. The MIT License applies only to the repository code and documentation. Patient level TCGA and CGGA derived matrices, clinical files, prediction tables, fitted model objects, and local intermediate artifacts are not redistributed or licensed by this repository. Users must obtain all third party data from the original authorized sources and comply with the relevant data use terms for TCGA, CGGA, STRING, MSigDB, Reactome, and related resources.

Use of this repository should cite the associated manuscript after publication and the archived Zenodo software record for the exact release used. The repository is intended to support academic review, methodological transparency, and reproducibility of the final locked GIBD workflow. It is not a clinically validated device, treatment allocation tool, or patient specific decision system.

## Suggested citation

If using this workflow, please cite the associated manuscript after publication and the Zenodo archived software release corresponding to the version used.

## Contact

For questions about the analysis workflow, contact the corresponding author listed in the manuscript.
