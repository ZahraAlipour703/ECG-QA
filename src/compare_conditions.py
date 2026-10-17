"""Paired ECG preprocessing/noise experiment for ECG-QA.

Evaluates the existing R-peak detector under identical settings on:
- clean raw ECG and clean 5-20 Hz band-passed ECG;
- synthetic baseline wander, high-frequency noise, and their combination;
- each perturbed signal before and after band-pass filtering.

Synthetic noise amplitudes are relative to each record's robust signal scale.
This is a controlled stress test, not a clinical noise model or diagnostic study.
Run from the ECG-QA repository root with: python -m src.compare_conditions
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.run_analysis import bandpass_ecg, detect_r_peaks, load_record, match_peaks

RECORDS = [
    "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124", "200", "201", "202", "203", "205",
    "207", "208", "209", "210", "212", "213", "214", "215", "217",
    "219", "220", "221", "222", "223", "228", "230", "231", "232",
    "233", "234",
]

# Standard MIT-BIH beat annotation symbols included by the baseline evaluation.
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e",
    "j", "n", "E", "/", "f", "Q", "?",
}
CONDITIONS = [
    "clean_raw",
    "clean_filtered",
    "baseline_wander_raw",
    "baseline_wander_filtered",
    "high_frequency_noise_raw",
    "high_frequency_noise_filtered",
    "combined_noise_raw",
    "combined_noise_filtered",
]

RESULTS_DIR = Path("results")
PER_RECORD_PATH = RESULTS_DIR / "condition_comparison_per_record.csv"
SUMMARY_PATH = RESULTS_DIR / "condition_comparison_summary.csv"
FAILURES_PATH = RESULTS_DIR / "condition_experiment_failures.csv"


def beat_annotations(samples: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    mask = np.asarray([symbol in BEAT_SYMBOLS for symbol in symbols], dtype=bool)
    return np.asarray(samples, dtype=int)[mask]


def summarize_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    # Existing CSV rows load as strings; coerce aggregate fields before summing.
    frame = frame.copy()
    numeric_columns = [
        "reference_beats", "TP", "FP", "FN", "sensitivity", "PPV", "F1",
        "mean_absolute_timing_error_ms",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        group = frame[frame["condition"] == condition]
        if group.empty:
            continue
        tp = int(group["TP"].sum())
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        pooled_f1 = (
            2 * sensitivity * ppv / (sensitivity + ppv)
            if sensitivity + ppv else float("nan")
        )
        rows.append({
            "condition": condition,
            "n_records": int(group["record"].nunique()),
            "reference_beats": int(group["reference_beats"].sum()),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "pooled_sensitivity": sensitivity,
            "pooled_PPV": ppv,
            "pooled_F1": pooled_f1,
            "mean_record_F1": float(group["F1"].mean()),
            "median_record_F1": float(group["F1"].median()),