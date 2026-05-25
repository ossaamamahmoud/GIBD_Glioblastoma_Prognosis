# Repository Structure

This document describes the recommended GitHub repository layout for the GIBD analysis workflow.

The repository is intended to provide a clean, public-facing analysis workflow supporting the manuscript:

**GIBD: Graph-Informed Biomarker Discovery-Driven Machine Learning Framework for Glioblastoma Prognosis**

The repository should contain reproducibility scripts, environment information, manuscript figures, and documentation. Raw patient-level data and protected clinical files should not be uploaded unless redistribution is explicitly permitted.

---

## Recommended repository layout

```text
GIBD_Glioblastoma_Prognosis/
├── README.md
├── requirements_freeze.txt
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
├── figures/
│   ├── main/
│   │   ├── Mahmoud_Fig1.tiff
│   │   ├── Mahmoud_Fig2.tiff
│   │   ├── Mahmoud_Fig3.tiff
│   │   ├── Mahmoud_Fig4.tiff
│   │   └── Mahmoud_Fig5.tiff
│   └── supplementary/
│       ├── Supplementary_Figure_S1_Calibration.png
│       ├── Supplementary_Figure_S2_DCA.png
│       ├── Supplementary_Figure_S3_Clinical_Covariate_Benchmark.png
│       ├── Supplementary_Figure_S4_CGGA_LIME.png
│       ├── Supplementary_Figure_S5_SHAP_Interactions.png
│       ├── Supplementary_Figure_S6_SHAP_Dependence.png
│       ├── Supplementary_Figure_S7_Locked_Signature_ORA.png
│       └── Supplementary_Figure_S8_Extended_GSEA.png
├── docs/
│   ├── code_manifest.md
│   └── repository_structure.md
└── Data/
    └── Not included by default
```

---

## Root-level files

### `README.md`

The main repository landing page. It should summarize:

- the study objective;
- the locked GIBD median-OS K100 workflow;
- key reported results;
- methodological guardrails;
- the run order;
- data requirements;
- environment setup;
- interpretation limits.

### `requirements_freeze.txt`

Python package versions used in the final analysis environment.

This file supports reproducibility but does not guarantee full re-execution unless the required data inputs and licensed resources are also available.

---

## `scripts/`

This folder contains the numbered analysis scripts in their intended run order.

The numbering should be kept stable because the manuscript workflow depends on this order.

### Script order

| Step | Script | Role |
|---:|---|---|
| 00 | `00_prepare_protein_coding_expression.py` | Filters the TCGA expression matrix to protein-coding genes. |
| 01 | `01_generate_tcga_median_os_labels.py` | Generates TCGA OS/Event/Risk_Label files using the empirical TCGA median OS cutoff. |
| 02 | `02_generate_cgga_tcga_cutoff_labels.py` | Generates censor-aware CGGA binary labels using the TCGA-derived OS cutoff. |
| 03 | `03_build_wppi_self_features.py` | Builds STRING-confidence-weighted WPPI-self features. |
| 04 | `04_lock_tcga_gibd_xgboost_k100.py` | Performs TCGA-only model development and locks the final K100 model. |
| 05 | `05_run_ablation_comparators.py` | Runs the ablation/comparator framework and post-lock CGGA evaluation. |
| 06 | `06_plot_figure3_ablation_performance.py` | Generates the ablation/external-validation performance figure. |
| 07 | `07_run_calibration_and_dca.py` | Generates calibration and decision-curve analysis outputs. |
| 08 | `08_run_clinical_covariate_benchmark.py` | Runs the clinical-covariate benchmark analyses. |
| 09 | `09_run_shap_interpretability.py` | Performs post-lock SHAP global and interaction interpretation. |
| 10 | `10_run_cgga_lime_explanations.py` | Performs post-lock LIME explanations for representative CGGA cases. |
| 11 | `11_run_tcga_full_transcriptome_gsea.py` | Runs TCGA-only full-transcriptome preranked GSEA. |
| 12 | `12_run_locked_signature_ora.py` | Runs locked-signature ORA and pathway cross-checks. |
| 13 | `13_run_ppi_threshold_sensitivity_audit.py` | Runs supplementary PPI threshold sensitivity auditing. |

---

## `figures/main/`

This folder should contain the five main manuscript figures as separate files.

Recommended file names:

```text
Mahmoud_Fig1.tiff
Mahmoud_Fig2.tiff
Mahmoud_Fig3.tiff
Mahmoud_Fig4.tiff
Mahmoud_Fig5.tiff
```

Each complete multi-panel figure should remain one file. Do not split Figure 3 into Figure 3A and Figure 3B files. Do not split Figure 4 into separate SHAP panel files if the manuscript uses one composite Figure 4.

---

## `figures/supplementary/`

This folder should contain supplementary figure outputs used in the Supplementary Information file.

Recommended file names:

```text
Supplementary_Figure_S1_Calibration.png
Supplementary_Figure_S2_DCA.png
Supplementary_Figure_S3_Clinical_Covariate_Benchmark.png
Supplementary_Figure_S4_CGGA_LIME.png
Supplementary_Figure_S5_SHAP_Interactions.png
Supplementary_Figure_S6_SHAP_Dependence.png
Supplementary_Figure_S7_Locked_Signature_ORA.png
Supplementary_Figure_S8_Extended_GSEA.png
```

TIFF can also be used if journal-resolution figure archives are preferred. PNG is usually acceptable for GitHub display.
### Actual supplementary figure files in this repository

```text
Supplementary_Figure_S1_Calibration_Analysis.tiff
Supplementary_Figure_S2_Decision_Curve_Analysis.tiff
Supplementary_Figure_S3_Clinical_Covariate_Benchmark.tiff
Supplementary_Figure_S4_CGGA_LIME_Local_Explanations.tiff
Supplementary_Figure_S5_SHAP_Interaction_Analysis.tiff
Supplementary_Figure_S6_SHAP_Targeted_Dependence_Analysis.tiff
Supplementary_Figure_S7_Locked_Signature_ORA_Dotplots.tiff
Supplementary_Figure_S8_Extended_Hallmark_GSEA_Results.tiff
```


---

## `docs/`

### `docs/code_manifest.md`

Explains the purpose of each numbered script and summarizes the full run order.

### `docs/repository_structure.md`

This file. It documents the recommended repository organization and upload policy.

---

## `Data/`

The scripts expect a project-local `Data/` directory when they are executed.

However, the `Data/` directory should generally **not** be uploaded to GitHub because it may contain:

- patient-level expression matrices;
- clinical tables;
- survival labels;
- model predictions;
- protected sample identifiers;
- large third-party resources;
- files governed by TCGA, CGGA, STRING, MSigDB, or other data-use terms.

Instead, document expected input locations in the README and scripts.

### Expected local data structure for execution

A user who has legitimate access to the required datasets can organize local inputs as:

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
├── CGGA_Data/
│   ├── CGGA.mRNAseq_693_clinical.20200506.txt
│   └── CGGA expression input files
└── Revision_Ablation/
    └── generated analysis outputs
```

Only upload small, non-sensitive example files if they are fully anonymized and redistribution is allowed.

---

## Files not recommended for public upload

Avoid uploading the following unless data-use permissions are clear and the files contain no sensitive identifiers:

```text
Data/
*.joblib
*_predictions.csv
*_oof_predictions.csv
*_training_predictions.csv
*_clinical*.csv
*_labels*.csv
TCGA_*expression*.csv
CGGA_*expression*.csv
clinical.json
*.gmt
9606.protein.links.full.v12.0.txt.gz
```

`*.joblib` model files should also be reviewed before upload because they may embed fitted preprocessing objects, feature names, and project-specific metadata.

---

## Recommended `.gitignore`

A conservative `.gitignore` can include:

```text
Data/
*.joblib
*.pkl
*.pickle
__pycache__/
.ipynb_checkpoints/
.DS_Store
*.log
*.tmp
```

If selected derived CSV outputs are intentionally shared, add exceptions explicitly and review them first.

---

## Suggested GitHub upload workflow

1. Create a new repository, for example:

   ```text
   GIBD_Glioblastoma_Prognosis
   ```

2. Upload root files:

   ```text
   README.md
   requirements_freeze.txt
   ```

3. Upload all numbered scripts into:

   ```text
   scripts/
   ```

4. Upload documentation into:

   ```text
   docs/
   ```

5. Create figure folders:

   ```text
   figures/main/
   figures/supplementary/
   ```

6. Upload main and supplementary figures.

7. Do not upload `Data/` unless a specific file has been checked for redistribution permissions and patient-level privacy.

8. Add a short repository description:

   ```text
   Leakage-controlled graph-informed transcriptomic modeling workflow for glioblastoma median-OS risk prioritization.
   ```

---

## Public-facing interpretation statement

The repository should describe GIBD as a reproducible, leakage-controlled, graph-informed transcriptomic risk-prioritization workflow.

Avoid language implying that the repository proves clinical utility, treatment-allocation readiness, causal biology, or patient-specific PPI activity.

Recommended wording:

> This repository provides scripts supporting a locked, transcriptomics-only, graph-informed glioblastoma prognosis workflow. The final model is interpreted as a risk-prioritization signal requiring further prospective recalibration, multimodal validation, and clinical decision-analytic evaluation before translational use.
