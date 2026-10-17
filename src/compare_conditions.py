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
            noisy_high_frequency, "Gaussian noise SD 0.15 x robust signal scale"
        ),
        "high_frequency_noise_filtered": (
            bandpass_ecg(noisy_high_frequency, fs),
            "Gaussian noise SD 0.15 x robust signal scale; then 5-20 Hz filter",
        ),
        "combined_noise_raw": (
            noisy_combined,
            "Baseline wander (0.5 x scale) plus Gaussian noise (SD 0.15 x scale)",
        ),
        "combined_noise_filtered": (
            bandpass_ecg(noisy_combined, fs),
            "Combined synthetic noise; then 5-20 Hz filter",
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
