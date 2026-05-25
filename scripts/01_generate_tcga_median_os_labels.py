#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GIBD analysis workflow — TCGA median-OS label generation

This script generates TCGA overall-survival labels for the final aligned
protein-coding expression cohort. It extracts OS_days and Event from the TCGA
clinical JSON file, anchors the cohort to the final expression matrix, computes
the empirical TCGA median OS cutoff, and exports the binary median-OS Risk_Label
used by downstream model-development scripts.

Analysis guardrails:
- The cutoff is computed from the final TCGA development cohort only.
- The output label files are used for TCGA-only model development and locking.
- External CGGA labels, predictions, and performance metrics are not used here.
- Changing these labels requires rerunning all downstream model-development,
  validation, interpretability, enrichment, and sensitivity analyses.
"""

import os
import json
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLINICAL_JSON = "Data/clinical.json"
MASTER_MATRIX = "Data/TCGA_survival_expression_matrix_protein_coding.csv"

OUTPUT_FILE = "Data/TCGA_survival_labels_with_os_event.csv"
OUTPUT_BINARY_LABELS = "Data/TCGA_survival_labels_manuscript_aligned.csv"
OUTPUT_MEDIAN_AUDIT = "Data/TCGA_survival_median_cutoff_audit.csv"

EXPECTED_N = 147


# --------------------------------------------------
# 1. Load master expression cohort
# --------------------------------------------------

if not os.path.exists(MASTER_MATRIX):
    raise FileNotFoundError(f"Master expression matrix not found: {MASTER_MATRIX}")

master_matrix = pd.read_csv(MASTER_MATRIX, index_col=0)
master_ids = master_matrix.index.astype(str).tolist()
master_set = set(master_ids)

print("Master expression matrix shape:", master_matrix.shape)
print("Master cohort size from expression matrix:", len(master_ids))

if len(master_ids) != EXPECTED_N:
    raise ValueError(
        f"Master matrix cohort size mismatch: expected {EXPECTED_N}, got {len(master_ids)}"
    )


# --------------------------------------------------
# 2. Load clinical JSON
# --------------------------------------------------

if not os.path.exists(CLINICAL_JSON):
    raise FileNotFoundError(f"Clinical JSON not found: {CLINICAL_JSON}")

with open(CLINICAL_JSON, "r", encoding="utf-8") as f:
    clinical_data = json.load(f)


# --------------------------------------------------
# 3. Extract OS_days and Event, anchored to master IDs
# --------------------------------------------------

rows = []

for patient in clinical_data:
    pid = patient.get("submitter_id")

    if pid not in master_set:
        continue

    demographic = patient.get("demographic", {}) or {}
    vital_status = demographic.get("vital_status")

    if vital_status == "Dead":
        event = 1
        os_days = demographic.get("days_to_death")

    elif vital_status == "Alive":
        event = 0
        diagnoses = patient.get("diagnoses", []) or []
        os_days = diagnoses[0].get("days_to_last_follow_up") if diagnoses else None

    else:
        continue

    os_days = pd.to_numeric(os_days, errors="coerce")

    if pd.isna(os_days) or os_days <= 0:
        continue

    rows.append({
        "Patient_ID": str(pid),
        "OS_days": float(os_days),
        "Event": int(event),
    })


df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No anchored clinical records were recovered. Check Patient_ID format.")

df = df.drop_duplicates(subset=["Patient_ID"], keep="first")


# --------------------------------------------------
# 4. Integrity check before labeling
# --------------------------------------------------

recovered_ids = set(df["Patient_ID"].astype(str))
missing_ids = [pid for pid in master_ids if pid not in recovered_ids]

if missing_ids:
    print("\nERROR: Some expression-matrix patients lack reconstructed survival labels:")
    print(missing_ids)
    raise ValueError(f"Missing clinical survival labels for {len(missing_ids)} master patients.")

if len(df) != EXPECTED_N:
    raise ValueError(f"Recovered anchored cohort mismatch: expected {EXPECTED_N}, got {len(df)}")


# --------------------------------------------------
# 5. Automatic median-OS label generation
# --------------------------------------------------

median_os = float(df["OS_days"].median())

# Primary median branch rule.
# Using <= places patients with OS exactly at the median in the high-risk group.
df["Risk_Label"] = (df["OS_days"] <= median_os).astype(int)

# Diagnostic ranks only; these are not used to assign Risk_Label.
df_ranked = df.sort_values(
    by=["OS_days", "Patient_ID"],
    ascending=[True, True]
).reset_index(drop=True)
df_ranked["Survival_Rank"] = range(1, len(df_ranked) + 1)

# Count labels.
risk_counts = df["Risk_Label"].value_counts().sort_index()
n_low = int(risk_counts.get(0, 0))
n_high = int(risk_counts.get(1, 0))

# Median-boundary diagnostics.
max_high_os = float(df.loc[df["Risk_Label"] == 1, "OS_days"].max())
min_low_os = float(df.loc[df["Risk_Label"] == 0, "OS_days"].min()) if n_low > 0 else float("nan")

print("\nMedian-OS label-generation diagnostics:")
print(f"Empirical median OS_days: {median_os}")
print(f"Rule: Risk_Label=1 if OS_days <= {median_os}; otherwise 0")
print(f"High-risk count: {n_high}")
print(f"Low-risk count: {n_low}")
print(f"Max OS_days among median-defined high-risk patients: {max_high_os}")
print(f"Min OS_days among median-defined low-risk patients: {min_low_os}")

if n_high == 0 or n_low == 0:
    raise ValueError("Median labeling produced an invalid one-class endpoint.")


# --------------------------------------------------
# 6. Restore original expression-matrix order
# --------------------------------------------------

df_final = df.set_index("Patient_ID").loc[master_ids].reset_index()
df_final = df_final[["Patient_ID", "OS_days", "Event", "Risk_Label"]]


# --------------------------------------------------
# 7. Final integrity checks
# --------------------------------------------------

risk_counts_final = df_final["Risk_Label"].value_counts().sort_index()
event_counts_final = df_final["Event"].value_counts().sort_index()

n_total = len(df_final)
n_low_final = int(risk_counts_final.get(0, 0))
n_high_final = int(risk_counts_final.get(1, 0))

print("\nFinal median-based survival-label file:")
print(df_final.head())
print("\nFinal shape:", df_final.shape)
print("\nEvent distribution:")
print(event_counts_final)
print("\nRisk_Label distribution:")
print(risk_counts_final)

if n_total != EXPECTED_N:
    raise ValueError(f"Count mismatch: expected N={EXPECTED_N}, got N={n_total}")


# --------------------------------------------------
# 8. Save outputs
# --------------------------------------------------

df_final.to_csv(OUTPUT_FILE, index=False)

df_final[["Patient_ID", "Risk_Label"]].to_csv(OUTPUT_BINARY_LABELS, index=False)

# Audit file: keep ranked view and median diagnostics.
audit_df = df_ranked[["Patient_ID", "OS_days", "Event", "Survival_Rank", "Risk_Label"]].copy()
audit_df["Median_OS_days"] = median_os
audit_df["Median_Rule"] = f"Risk_Label=1 if OS_days <= {median_os}; else 0"
audit_df["Median_High_Risk_Count"] = n_high_final
audit_df["Median_Low_Risk_Count"] = n_low_final
audit_df["Max_OS_Median_High_Risk"] = max_high_os
audit_df["Min_OS_Median_Low_Risk"] = min_low_os
audit_df.to_csv(OUTPUT_MEDIAN_AUDIT, index=False)

print(f"\nSUCCESS: Median-based survival labels saved to: {OUTPUT_FILE}")
print(f"SUCCESS: Median-based binary labels saved to: {OUTPUT_BINARY_LABELS}")
print(f"SUCCESS: Median cutoff audit saved to: {OUTPUT_MEDIAN_AUDIT}")
