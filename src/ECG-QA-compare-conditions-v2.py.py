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
from scipy.signal import butter, sosfiltfilt

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
            "mean_record_absolute_timing_error_ms": float(
                group["mean_absolute_timing_error_ms"].mean()
            ),
        })
    return pd.DataFrame(rows)


def run_one_record(record: str, seconds: float, tolerance_ms: float) -> list[dict[str, Any]]:
    signal, fs, ref_samples, ref_symbols = load_record(record, seconds)
    refs = beat_annotations(ref_samples, ref_symbols)
    if len(refs) == 0:
        raise ValueError("No recognized beat annotations in this segment.")

    median = np.median(signal)
    robust_scale = 1.4826 * float(np.median(np.abs(signal - median)))
    if not np.isfinite(robust_scale) or robust_scale <= np.finfo(float).eps:
        robust_scale = float(np.std(signal))
    if not np.isfinite(robust_scale) or robust_scale <= np.finfo(float).eps:
        raise ValueError("Signal has no usable variation for noise generation.")

    t = np.arange(len(signal), dtype=float) / fs
    # Pre-specified synthetic stressors, relative to robust signal scale.
    baseline_wander = (0.5 * robust_scale) * np.sin(2.0 * np.pi * 0.33 * t)
    rng = np.random.default_rng(20261017 + int(record))
    # Generate noise concentrated above the detector's 5-20 Hz passband.
    # Start with Gaussian noise, high-pass it at 25 Hz, then normalize its
    # realized standard deviation to 0.15 x robust signal scale.
    white_noise = rng.normal(loc=0.0, scale=1.0, size=len(signal))
    noise_sos = butter(4, 25.0, btype="highpass", fs=fs, output="sos")
    high_frequency_noise = sosfiltfilt(noise_sos, white_noise)
    high_frequency_noise -= np.mean(high_frequency_noise)
    noise_sd = float(np.std(high_frequency_noise))
    if not np.isfinite(noise_sd) or noise_sd <= np.finfo(float).eps:
        raise ValueError("Could not generate valid high-frequency noise.")
    high_frequency_noise *= (0.15 * robust_scale) / noise_sd

    noisy_baseline = signal + baseline_wander
    noisy_high_frequency = signal + high_frequency_noise
    noisy_combined = signal + baseline_wander + high_frequency_noise

    prepared = {
        "clean_raw": (signal, "No added noise; no band-pass filtering"),
        "clean_filtered": (
            bandpass_ecg(signal, fs), "No added noise; 3rd-order 5-20 Hz zero-phase band-pass"
        ),
        "baseline_wander_raw": (
            noisy_baseline, "0.33 Hz sinusoid; peak amplitude 0.5 x robust signal scale"
        ),
        "baseline_wander_filtered": (
            bandpass_ecg(noisy_baseline, fs),
            "0.33 Hz sinusoid; peak amplitude 0.5 x robust signal scale; then 5-20 Hz filter",
        ),
        "high_frequency_noise_raw": (
            noisy_high_frequency, "Gaussian noise high-pass filtered at 25 Hz; realized SD 0.15 x robust signal scale"
        ),
        "high_frequency_noise_filtered": (
            bandpass_ecg(noisy_high_frequency, fs),
            "Gaussian noise high-pass filtered at 25 Hz; realized SD 0.15 x robust signal scale; then 5-20 Hz filter",
        ),
        "combined_noise_raw": (
            noisy_combined,
            "Baseline wander (0.5 x scale) plus Gaussian noise high-pass filtered at 25 Hz (SD 0.15 x scale)",
        ),
        "combined_noise_filtered": (
            bandpass_ecg(noisy_combined, fs),
            "Combined baseline wander plus 25 Hz high-pass Gaussian noise; then 5-20 Hz filter",
        ),
    }

    record_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        processed_signal, noise_definition = prepared[condition]
        detected = detect_r_peaks(processed_signal, fs)
        metrics = match_peaks(refs, detected, fs, tolerance_ms)
        row: dict[str, Any] = {
            "record": str(record),
            "condition": condition,
            "noise_definition": noise_definition,
            "sampling_rate_hz": fs,
            "segment_seconds": len(signal) / fs,
            "robust_signal_scale": robust_scale,
        }
        row.update(metrics)
        record_rows.append(row)
    return record_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw/filtered ECG detection and controlled synthetic noise."
    )
    parser.add_argument(
        "--records", nargs="+", default=None,
        help="Record IDs to evaluate. Default: all 47 validation records (excluding development record 100).",
    )
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    records = args.records if args.records else RECORDS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    # If resuming after an interruption, existing rows from records not selected
    # in this run are retained. Rows for newly rerun records are replaced.
    if PER_RECORD_PATH.exists():
        try:
            existing = pd.read_csv(PER_RECORD_PATH)
            all_rows = existing.to_dict(orient="records")
        except Exception:
            all_rows = []

    for record in records:
        print(f"\n========== RECORD {record} ==========")
        new_rows: list[dict[str, Any]] | None = None
        last_error = "unknown error"
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                new_rows = run_one_record(record, args.seconds, args.tolerance_ms)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"Attempt {attempt} failed: {last_error[:220]}")
                if attempt < max(1, args.retries):
                    time.sleep(5 * attempt)

        if new_rows is None:
            failures.append({"record": str(record), "error": last_error})
            pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
            print(f"FAILED record {record}; continuing.")
            continue

        # Replace stale rows for this record, then checkpoint.
        all_rows = [r for r in all_rows if str(r.get("record")) != str(record)]
        all_rows.extend(new_rows)
        pd.DataFrame(all_rows).to_csv(PER_RECORD_PATH, index=False)
        print(f"SUCCESS record {record}: all {len(new_rows)} conditions evaluated.")

    if all_rows:
        frame = pd.DataFrame(all_rows)
        # Restrict final output to selected records and known condition order.
        selected = {str(x) for x in records}
        frame = frame[frame["record"].astype(str).isin(selected)].copy()
        frame.to_csv(PER_RECORD_PATH, index=False)
        summary = summarize_conditions(frame)
        summary.to_csv(SUMMARY_PATH, index=False)
        print("\n=== Condition comparison summary ===")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nPer-record results: {PER_RECORD_PATH}")
        print(f"Summary: {SUMMARY_PATH}")

    if failures:
        print(f"\nFailed records: {', '.join(item['record'] for item in failures)}")
        print(f"Failure log: {FAILURES_PATH}")


if __name__ == "__main__":
    main()
