# ECG-QA: ECG Signal Quality and R-Peak Detection

A reproducible, non-diagnostic study of how filtering affects R-peak detection on annotated ECG recordings.

## Research question
How do common ECG noise conditions affect R-peak detection, and how much does a basic filtering pipeline improve agreement with reference annotations?

## Scope and limitations
- Uses the MIT-BIH Arrhythmia Database and its reference beat annotations.
- Starts with a transparent baseline detector, not a clinical algorithm.
- Reports sensitivity, positive predictive value (PPV), F1, and absolute timing error.
- This is a research/learning project, **not** a diagnostic system and not clinically validated.
- Initial version evaluates the first record (`100`) as a smoke test. Extend to multiple records before making general claims.

## Setup (Windows PowerShell)
Use Python 3.10–3.12 in a virtual environment.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
Download one record and its annotation, then run the baseline:

```powershell
python -m src.run_analysis --record 100 --seconds 120
```

The script loads record `100` and its annotation from PhysioNet through `wfdb`, processes a 120-second segment, and writes figures and a metrics CSV under `results/`. An internet connection is required for the first run.

To analyze a different record:
```powershell
python -m src.run_analysis --record 101 --seconds 120
```

Do not compare a record's metrics to another record as if the sample were representative. For a proper experiment, evaluate several records and report per-record and aggregate results.

## Method
1. Load the ECG channel and reference annotations.
2. Apply a 5–20 Hz Butterworth bandpass filter (zero-phase filtering for offline analysis).
3. Detect candidate peaks using a simple amplitude/prominence heuristic.
4. Match detections to reference annotations within a configurable timing tolerance (default 100 ms), one-to-one.
5. Compute TP, FP, FN, sensitivity, PPV, F1, and absolute timing error.

The baseline detector is intentionally simple and may fail on some morphologies. Results must be interpreted as a baseline, not as evidence of clinical suitability.

## Dataset and citation
MIT-BIH Arrhythmia Database, PhysioNet:
https://physionet.org/content/mitdb/1.0.0/

Please read the dataset documentation and cite the database in any report or public release. Respect the dataset's terms of use.

## Repository structure
```text
ECG-QA/
  data/raw/             # downloaded files (gitignored)
  results/figures/      # generated plots
  src/run_analysis.py
  requirements.txt
  .gitignore
```

## Planned experiments after the smoke test
- Run a pre-specified set of multiple records (not just one).
- Compare raw and filtered signals with the same detector settings.
- Add controlled baseline-wander and high-frequency-noise perturbations with fixed random seeds.
- Report per-record results and confidence intervals or variability; do not treat every beat as an independent subject.
- Add error-case plots and discuss limitations.
