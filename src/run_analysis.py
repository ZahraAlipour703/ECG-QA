from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "results" / "figures"
RESULTS_DIR = ROOT / "results"


def load_record(record_name: str, seconds: float | None):
    """Download one PhysioNet record and annotations, then load locally."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # wfdb downloads the record files into the project data/raw directory.
    record = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb",
        sampfrom=0,
        sampto=None,
    )
    annotation = wfdb.rdann(record_name, "atr", pn_dir="mitdb")
    signal = record.p_signal[:, 0].astype(float)
    fs = float(record.fs)
    limit = len(signal) if seconds is None else min(len(signal), int(seconds * fs))
    signal = signal[:limit]
    ref_samples = np.asarray(annotation.sample)
    ref_symbols = np.asarray(annotation.symbol)
    keep = ref_samples < limit
    ref_samples = ref_samples[keep]
    ref_symbols = ref_symbols[keep]
    return signal, fs, ref_samples, ref_symbols


def bandpass_ecg(x: np.ndarray, fs: float, low_hz: float = 5.0, high_hz: float = 20.0):
    """Offline zero-phase Butterworth bandpass. Requires enough samples."""
    nyquist = fs / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(f"Invalid filter cutoffs for sampling rate {fs} Hz.")
    sos = butter(3, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return filtfilt(sos, x)


def detect_r_peaks(x: np.ndarray, fs: float):
    """
    Transparent baseline detector:
    - use absolute filtered amplitude to tolerate inverted QRS complexes;
    - minimum distance prevents detections clustered within one QRS;
    - prominence is estimated from robust signal scale.
    This is deliberately not a clinically validated detector.
    """
    centered = x - np.median(x)
    robust_scale = 1.4826 * np.median(np.abs(centered - np.median(centered)))
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        return np.array([], dtype=int)
    min_distance = max(1, int(0.25 * fs))
    prominence = max(0.5 * robust_scale, np.finfo(float).eps)
    peaks, _ = find_peaks(
        np.abs(centered),
        distance=min_distance,
        prominence=prominence,
    )
    return peaks.astype(int)


def match_peaks(reference: np.ndarray, detected: np.ndarray, fs: float, tolerance_ms: float):
    """Greedy one-to-one matching in chronological order."""
    tolerance = int(round(tolerance_ms * fs / 1000.0))
    ref = np.asarray(reference, dtype=int)
    det = np.asarray(detected, dtype=int)
    used = np.zeros(len(det), dtype=bool)
    errors_samples = []
    tp = 0

    for r in ref:
        left = np.searchsorted(det, r - tolerance, side="left")
        right = np.searchsorted(det, r + tolerance, side="right")
        candidates = [i for i in range(left, right) if not used[i]]
        if candidates:
            best = min(candidates, key=lambda i: abs(int(det[i]) - int(r)))
            used[best] = True
            tp += 1
            errors_samples.append(abs(int(det[best]) - int(r)))

    fp = int(len(det) - tp)
    fn = int(len(ref) - tp)
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) else float("nan")
    timing_error_ms = float(np.mean(errors_samples) * 1000.0 / fs) if errors_samples else float("nan")
    return {
        "reference_beats": int(len(ref)),
        "detected_peaks": int(len(det)),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "sensitivity": sensitivity,
        "PPV": ppv,
        "F1": f1,
        "mean_absolute_timing_error_ms": timing_error_ms,
        "timing_tolerance_ms": float(tolerance_ms),
    }


def plot_signal(t: np.ndarray, raw: np.ndarray, filtered: np.ndarray,
                reference: np.ndarray, detected: np.ndarray, fs: float, record: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Show a short excerpt for visual inspection, regardless of total duration.
    show_n = min(len(raw), int(10 * fs))
    tt = t[:show_n]
    ref = reference[reference < show_n]
    det = detected[detected < show_n]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(tt, raw[:show_n], linewidth=0.7)
    axes[0].set_title(f"MIT-BIH record {record}: raw ECG (first {tt[-1] if len(tt) else 0:.1f} s)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(tt, filtered[:show_n], linewidth=0.7, label="Filtered ECG")
    if len(ref):
        axes[1].scatter(ref / fs, filtered[ref], s=18, marker="x", label="Reference annotations")
    if len(det):
        axes[1].scatter(det / fs, filtered[det], s=18, marker="o", facecolors="none",
                         label="Detected peaks")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Filtered amplitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    path = FIG_DIR / f"record_{record}_baseline.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="ECG-QA baseline R-peak detection evaluation.")
    parser.add_argument("--record", default="100", help="MIT-BIH record identifier, e.g. 100 or 101")
    parser.add_argument("--seconds", type=float, default=120,
                        help="Number of seconds to analyze; use 0 for the entire record")
    parser.add_argument("--tolerance-ms", type=float, default=100,
                        help="Maximum reference-to-detection matching tolerance in milliseconds")
    args = parser.parse_args()

    seconds = None if args.seconds == 0 else args.seconds
    signal, fs, ref_samples, ref_symbols = load_record(args.record, seconds)
    filtered = bandpass_ecg(signal, fs)
    detected = detect_r_peaks(filtered, fs)

    # MIT-BIH beat annotations include non-beat and rhythm-related symbols.
    # For this first baseline, exclude known non-beat annotations; document this
    # operational choice and refine it against the database documentation.
    nonbeat_symbols = {"+", "~", "|", "!", "[", "]", "x", "(", ")", "p", "t", "u", "`", "'"}
    beat_mask = np.array([symbol not in nonbeat_symbols for symbol in ref_symbols], dtype=bool)
    reference_beats = ref_samples[beat_mask]

    metrics = match_peaks(reference_beats, detected, fs, args.tolerance_ms)
    metrics.update({
        "record": args.record,
        "sampling_rate_hz": fs,
        "analyzed_seconds": len(signal) / fs,
        "filter": "3rd-order Butterworth 5-20 Hz, zero-phase",
        "annotation_policy": "excluded selected non-beat symbols; review before final study",
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"record_{args.record}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    t = np.arange(len(signal)) / fs
    fig_path = plot_signal(t, signal, filtered, reference_beats, detected, fs, args.record)

    print("\nECG-QA baseline results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nMetrics CSV: {out_csv}")
    print(f"Figure: {fig_path}")
    print("\nInterpret results as a baseline only. Inspect annotations and plots before drawing conclusions.")


if __name__ == "__main__":
    main()
