#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GIBD analysis workflow — CGGA external endpoint alignment

This script generates censor-aware CGGA binary labels aligned to the TCGA-derived
median-OS cutoff. The existing CGGA cohort cache is used only to define the
external cohort IDs. The new external binary endpoint is assigned from CGGA
OS_days and Event using the fixed TCGA cutoff, while early censored cases at or
before the cutoff are retained in the audit file and excluded from binary
classification metrics.

Analysis guardrails:
- The OS cutoff is inherited from TCGA and is not selected using CGGA performance.
- CGGA predictions, AUC, C-index, and model outputs are not read by this script.
- The binary-eligible output supports post-lock external validation only.
- Ambiguous early censored cases are documented for auditability.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CONF = {
    # Existing CGGA labels are used only to define the exact external cohort IDs
    # already present in the revision pipeline. Their old Risk_Label is retained
    # in the audit file as Old_Risk_Label but is NOT used as the new endpoint.
    "EXISTING_CGGA_LABELS": Path("Data/Revision_Ablation/cgga_labels_cache.csv"),

    "CGGA_CLINICAL": Path("Data/CGGA_Data/CGGA.mRNAseq_693_clinical.20200506.txt"),

    # Optional TCGA median audit file from:
    # 00b_generate_tcga_survival_labels_with_os_event_MEDIAN.py
    "TCGA_MEDIAN_AUDIT": Path("Data/TCGA_survival_median_cutoff_audit.csv"),

    # Fallback if the TCGA audit file is missing.
    "FALLBACK_TCGA_MEDIAN_OS": 357.0,

    "OUT_BINARY_ELIGIBLE": Path("Data/Revision_Ablation/cgga_labels_tcga_median357_with_os_event.csv"),
    "OUT_FULL_AUDIT": Path("Data/Revision_Ablation/cgga_labels_tcga_median357_with_os_event_full_audit.csv"),
    "OUT_REPORT": Path("Data/Revision_Ablation/cgga_labels_tcga_median357_with_os_event_audit.txt"),

    "EXPECTED_INPUT_N": 133,

    # Keep this True for the methodologically cleaner external binary endpoint.
    "CENSOR_AWARE_BINARY_LABELS": True,
}


# --------------------------------------------------
# Loading helpers
# --------------------------------------------------

def load_existing_cgga_ids(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Existing CGGA label cache not found: {path}")

    labels = pd.read_csv(path)

    if "CGGA_ID" in labels.columns:
        labels = labels.set_index("CGGA_ID")
    elif "Patient_ID" in labels.columns:
        labels = labels.set_index("Patient_ID")
    elif "Unnamed: 0" in labels.columns:
        labels = labels.set_index("Unnamed: 0")
    else:
        labels = labels.set_index(labels.columns[0])

    labels.index = labels.index.astype(str)

    if "Risk_Label" in labels.columns:
        labels["Old_Risk_Label"] = pd.to_numeric(labels["Risk_Label"], errors="coerce")
        labels = labels.drop(columns=["Risk_Label"])
    elif "Old_Risk_Label" not in labels.columns:
        labels["Old_Risk_Label"] = np.nan

    return labels


def load_cgga_clinical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CGGA clinical file not found: {path}\n"
            "Update CONF['CGGA_CLINICAL'] to the correct clinical file path."
        )

    clin = pd.read_csv(path, sep="\t")

    required = [
        "CGGA_ID",
        "PRS_type",
        "Histology",
        "OS",
        "Censor (alive=0; dead=1)",
    ]
    missing = [c for c in required if c not in clin.columns]
    if missing:
        raise ValueError(f"CGGA clinical file missing required columns: {missing}")

    clin["CGGA_ID"] = clin["CGGA_ID"].astype(str)
    clin["OS_days"] = pd.to_numeric(clin["OS"], errors="coerce")
    clin["Event"] = pd.to_numeric(clin["Censor (alive=0; dead=1)"], errors="coerce")

    clin = clin.dropna(subset=["CGGA_ID", "OS_days", "Event"]).copy()
    clin = clin[clin["OS_days"] > 0].copy()
    clin["Event"] = clin["Event"].astype(int)

    clin = clin.drop_duplicates(subset=["CGGA_ID"], keep="first")
    return clin.set_index("CGGA_ID")


def get_tcga_cutoff() -> float:
    audit = CONF["TCGA_MEDIAN_AUDIT"]

    if audit.exists():
        df = pd.read_csv(audit)
        for col in ["Median_OS_days", "Empirical_Median_OS_days", "median_os", "Median_OS"]:
            if col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(val) > 0:
                    return float(val.iloc[0])

    return float(CONF["FALLBACK_TCGA_MEDIAN_OS"])


def assign_labels(out: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    out = out.copy()

    out["TCGA_Derived_OS_Cutoff"] = float(cutoff)
    out["Early_Censored_Before_Or_At_Cutoff"] = (
        (out["Event"].astype(int) == 0) & (out["OS_days"].astype(float) <= cutoff)
    ).astype(int)

    if CONF["CENSOR_AWARE_BINARY_LABELS"]:
        # Start with missing/ambiguous.
        out["Risk_Label"] = np.nan

        # Observed death before or at the TCGA-derived cutoff.
        out.loc[(out["Event"] == 1) & (out["OS_days"] <= cutoff), "Risk_Label"] = 1

        # Known survival beyond cutoff.
        out.loc[out["OS_days"] > cutoff, "Risk_Label"] = 0

        out["Binary_Label_Status"] = "eligible"
        out.loc[out["Risk_Label"].isna(), "Binary_Label_Status"] = (
            "ambiguous_early_censored_excluded_from_binary_metrics"
        )

    else:
        # Simpler but less censoring-rigorous policy.
        out["Risk_Label"] = (out["OS_days"] <= cutoff).astype(int)
        out["Binary_Label_Status"] = "eligible_simple_observed_os_cutoff"

    return out


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Generate CGGA labels using TCGA-derived median OS cutoff")
    print("=" * 80)

    labels = load_existing_cgga_ids(CONF["EXISTING_CGGA_LABELS"])
    clin = load_cgga_clinical(CONF["CGGA_CLINICAL"])
    cutoff = get_tcga_cutoff()

    print(f"Existing CGGA cohort label cache shape: {labels.shape}")
    print(f"Clinical table with usable OS/Event shape: {clin.shape}")
    print(f"TCGA-derived OS cutoff used for CGGA endpoint: {cutoff}")
    print(f"Censor-aware binary labels: {CONF['CENSOR_AWARE_BINARY_LABELS']}")

    if len(labels) != CONF["EXPECTED_INPUT_N"]:
        raise ValueError(
            f"Expected existing CGGA cohort N={CONF['EXPECTED_INPUT_N']}, got {len(labels)}"
        )

    common = labels.index.intersection(clin.index)
    missing_from_clinical = labels.index.difference(clin.index).tolist()

    if len(common) != len(labels):
        print("\nERROR: Some CGGA cohort IDs lack matching clinical OS/Event rows.")
        print(f"Missing count: {len(missing_from_clinical)}")
        print("First missing IDs:", missing_from_clinical[:20])
        raise ValueError("Cannot generate complete CGGA OS/Event audit file.")

    out = labels.loc[common].copy()
    out["OS_days"] = clin.loc[common, "OS_days"].astype(float)
    out["Event"] = clin.loc[common, "Event"].astype(int)

    for col in [
        "PRS_type",
        "Histology",
        "Grade",
        "Gender",
        "Age",
        "IDH_mutation_status",
        "1p19q_codeletion_status",
        "MGMTp_methylation_status",
    ]:
        if col in clin.columns:
            out[col] = clin.loc[common, col]

    out = assign_labels(out, cutoff=cutoff)

    out = out.reset_index().rename(columns={"index": "CGGA_ID"})
    if "CGGA_ID" not in out.columns:
        out = out.rename(columns={out.columns[0]: "CGGA_ID"})

    eligible = out[out["Risk_Label"].notna()].copy()
    eligible["Risk_Label"] = eligible["Risk_Label"].astype(int)

    full_n = len(out)
    eligible_n = len(eligible)
    ambiguous_n = int(out["Risk_Label"].isna().sum())

    counts = eligible["Risk_Label"].value_counts().sort_index()
    low = int(counts.get(0, 0))
    high = int(counts.get(1, 0))
    event_counts = out["Event"].value_counts().sort_index()

    old_counts = {}
    if "Old_Risk_Label" in out.columns:
        old_counts = out["Old_Risk_Label"].value_counts(dropna=False).sort_index().to_dict()

    print("\nFinal diagnostics:")
    print(f"Full CGGA cohort N: {full_n}")
    print(f"Binary-eligible N: {eligible_n}")
    print(f"Ambiguous early-censored excluded from binary metrics: {ambiguous_n}")
    print(f"New Risk_Label counts among eligible: low={low}, high={high}")
    print(f"Event counts in full cohort: {event_counts.to_dict()}")
    print(f"Old Risk_Label counts from cache: {old_counts}")
    print(f"OS_days range: {out['OS_days'].min()} to {out['OS_days'].max()}")

    if eligible_n == 0 or low == 0 or high == 0:
        raise ValueError("Invalid external binary label distribution after censoring-aware assignment.")

    if not set(out["Event"].dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError("Event must be binary: 0=alive/censored, 1=dead/event.")

    CONF["OUT_BINARY_ELIGIBLE"].parent.mkdir(parents=True, exist_ok=True)

    # Save full audit and binary-eligible files.
    out.to_csv(CONF["OUT_FULL_AUDIT"], index=False)
    eligible.to_csv(CONF["OUT_BINARY_ELIGIBLE"], index=False)

    report_lines = [
        "CGGA TCGA-derived median OS endpoint audit",
        "==========================================",
        "",
        f"Existing CGGA cohort file: {CONF['EXISTING_CGGA_LABELS']}",
        f"CGGA clinical file: {CONF['CGGA_CLINICAL']}",
        f"TCGA-derived OS cutoff: {cutoff}",
        f"Censor-aware binary labels: {CONF['CENSOR_AWARE_BINARY_LABELS']}",
        "",
        f"Full CGGA cohort N: {full_n}",
        f"Binary-eligible N: {eligible_n}",
        f"Ambiguous early-censored excluded from binary metrics: {ambiguous_n}",
        f"New eligible low-risk count: {low}",
        f"New eligible high-risk count: {high}",
        f"Event counts full cohort: {event_counts.to_dict()}",
        f"Old Risk_Label counts from cache: {old_counts}",
        "",
        "Binary endpoint rule:",
        f"  Risk_Label=1 if Event=1 and OS_days <= {cutoff}",
        f"  Risk_Label=0 if OS_days > {cutoff}",
        f"  Event=0 and OS_days <= {cutoff}: ambiguous; excluded from binary metrics; retained in full audit/C-index context",
        "",
        f"Saved binary-eligible labels: {CONF['OUT_BINARY_ELIGIBLE']}",
        f"Saved full audit: {CONF['OUT_FULL_AUDIT']}",
    ]

    CONF["OUT_REPORT"].write_text("\n".join(report_lines), encoding="utf-8")

    print("\nSUCCESS")
    print(f"Saved binary-eligible labels: {CONF['OUT_BINARY_ELIGIBLE']}")
    print(f"Saved full audit: {CONF['OUT_FULL_AUDIT']}")
    print(f"Saved report: {CONF['OUT_REPORT']}")
    print("\nIMPORTANT:")
    print("Before running ablation with this external endpoint, patch the ablation script")
    print("to use the binary-eligible CGGA file and its printed expected counts.")


if __name__ == "__main__":
    main()
