# Bus Captain Risk Review

**Version 1.0 — Minimum core functionality met.**

A local Python and Streamlit research project for reviewing bus captains using recorded driving behaviour, fatigue alerts and collision warnings. It prepares the supplied Excel data, trains and evaluates candidate models, and displays a ranked queue with explanations.

**Status:** the software pipeline works, but reliable accident prediction has not been demonstrated. The dashboard shows provisional review suggestions. A high priority score does not establish that a captain is unsafe.

## Versions and release history

`VERSION` is the source of truth for the project version. Version `1.0` marks the minimum core software baseline: data preparation, model training/evaluation, captain ranking, explanations, dashboard and automated tests. It does not certify predictive reliability or completion of the separate assessment submissions.

For each future completed edit or improvement, increment the second number by one: `1.0` → `1.1` → `1.2`, continuing to `1.10` after `1.9`. Treat each number as an integer, not a decimal. Increment once per completed logical update, rather than once per edited file or keystroke. Update `VERSION`, the version line above and this history together; check the change, then tag its commit `v<version>`. Preserve existing tags.

| Version | Milestone |
|---|---|
| 1.0 | Minimum core functionality met; baseline checked with 19 passing tests, Ruff and dependency checks. Predictive reliability remains unproven. |

## Requirements

- Windows and PowerShell for the supplied shortcuts and VS Code configurations.
- Python 3.12; the checked environment uses Python 3.12.14.
- VS Code with the Microsoft Python extension. Jupyter is optional.
- Authorized access to the original course workbooks through the team's approved channel. Data and fitted models are excluded from Git.

The Python source uses project-relative paths. Other operating systems require equivalent virtual-environment and launch commands; they have not been checked.

## First setup

To get the source from GitHub:

```powershell
git clone https://github.com/lushiqiS7/CEG2003_Insturment-Data.git
cd CEG2003_Insturment-Data
```

Open this project folder in VS Code and use its integrated PowerShell terminal. Run these commands from the folder containing `config.json`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
New-Item -ItemType Directory -Force -Path data/raw,data/processed,models,outputs | Out-Null
```

If `py` is unavailable, use the full path to a Python 3.12 executable for the first command. Do not copy another computer's `.venv`. If pinned installation fails, diagnose the error before changing versions; `requirements.txt` lists direct dependencies but does not reproduce the checked environment.

Copy the two authorized input files to:

```text
data/raw/Mar24_part1.xlsx
data/raw/Traffic Accident Database 2024.xlsx
```

The separately supplied `Mar24_part1 (1).xlsx` is an exact duplicate and must not be ingested again. Keep the briefing, final reports and user manual outside this coding folder. This README is the repository entry point.

For a continuation of the existing team's experiment, obtain the matching private `driver_map.parquet` and `partition_registry.parquet` through the approved channel and place them in `data/processed/` before preparation. Preserve them on reruns. Without them, new pseudonyms and partition membership can differ, so regenerated results are a new run rather than an exact continuation. These files must stay out of Git.

In VS Code, run **Python: Select Interpreter** and select `.venv\Scripts\python.exe`.

## Prepare and train

```powershell
.\.venv\Scripts\python.exe -m src.prepare
.\.venv\Scripts\python.exe -m src.train
```

Run preparation before training. Preparation reads all supplied alarm sheets, removes exact duplicates and produces Parquet tables. Training builds captain features, evaluates four candidates using nested development cross-validation, applies sigmoid calibration and produces scores and grouped Shapley explanations. The saved original test membership stays outside further model selection.

Both commands overwrite current processed/model/output files. Preserve the current matching artifacts privately before experiments. Keep the identity map and partition registry; do not change partitions to chase better metrics. Do not launch concurrent preparation or training jobs.

The current implementation expects `models/` to exist; the setup command above creates it.

## Launch the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501/>. To view it inside VS Code, open the Command Palette, choose **Simple Browser: Show**, and enter that URL. Keep the server terminal running; press **Ctrl+C** there to stop it.

If port 8501 is already in use, first open the existing dashboard. For a deliberate second instance, use `--server.port 8502` and open the matching URL.

VS Code also provides **Terminal > Run Task** entries for preparation, training, the full pipeline, tests and dashboard. `run.ps1` supports `prepare`, `train`, `all`, `test` and `dashboard` if local PowerShell policy permits scripts. The direct Python commands do not require activation or a policy change.

## Read the dashboard

- **Captains scored and provisional flags:** the scored population and estimates above their corresponding review thresholds.
- **Evidence notice:** the model's development validation status and limitations.
- **Queue and search:** switch between flagged/all captains and find a pseudonymous ID.
- **Priority 0–100:** relative percentile against development cross-fitted scores, not incident probability.
- **Top three factors:** grouped Shapley effects on the model estimate. Positive percentage points raise it; negative values lower it. They are not causal findings.
- **Recent alerts:** compares fatigue-plus-collision counts on 9–15 March against 2–8 March. It is not a driving-exposure-adjusted risk trend.
- **Selected captain:** click a row to see counts, explanations, model provenance, the separate probability estimate and its threshold, and fatigue/collision charts.
- **Verify selected score:** repeats saved-model inference without retraining.
- **Validation and score definitions:** expands the metrics, method and unresolved limitations.

Filtering does not retrain the model. The app rejects mismatched preparation/configuration/model/score/chart files; rebuild a consistent set instead of bypassing this check.

## Test and check changes

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m ruff check src dashboard tests
.\.venv\Scripts\python.exe -m pip check
```

The checked local build passed **19 tests** on 29 October 2026. Some integration/dashboard tests skip when trained outputs are absent; a pass with skips does not verify the complete pipeline. Assertions for current counts and class totals are specific to the supplied data. Review them deliberately when the dataset or approved label protocol changes.

Use VS Code's Testing beaker to run/debug individual tests. Use F5 with the supplied training or dashboard launch configuration for breakpoints.

## Project structure

| Path | Purpose |
|---|---|
| `src/common.py`, `src/excel_stream.py` | Configuration, logs, Excel streaming, identifier/date parsing and file hashes |
| `src/prepare.py` | Clean, deduplicate, pseudonymize and aggregate the original inputs |
| `src/features.py` | Captain features and later-incident labels |
| `src/validation.py` | Partitions, thresholds, metrics and evidence checks |
| `src/risk_training.py`, `src/train.py` | Candidate selection, calibration, scoring and training entry point |
| `src/explain.py` | Exact Shapley contributions for six feature groups |
| `dashboard/` | Streamlit application, CSS, local font and font licence |
| `tests/` | Pipeline, risk-feature, explanation, integrity and dashboard tests |
| `data/raw/`, `data/processed/` | Private original and prepared data; excluded from Git |
| `models/` | Private fitted models and metadata; excluded from Git |
| `outputs/` | Private scores, metrics, logs and experiment history; excluded from Git |
| `config.json` | Source paths, periods, exclusions, review fraction and seed |
| `requirements*.txt`, `pyproject.toml` | Dependencies, exact environment versions and test/lint configuration |
| `.vscode/`, `.streamlit/`, `run.ps1` | Editor tasks, server/theme settings and command shortcuts |

## Current evidence and limitations

Features use alarms from **1–15 March 2024**; provisional positive labels use included incident reports from **16–31 March 2024**. Source preparation found 2,266,768 raw alarm rows, 153 exact duplicates and 2,266,615 cleaned events. There are 1,509 identified captains; 1,508 have usable provisional labels, including 20 positives.

Nested development validation used 1,206 captains and 16 positives. The previously inspected 302-person test cohort remains excluded from further selection and is not a fresh independent evaluation.

| Development metric | Checked result |
|---|---:|
| ROC-AUC | 0.4602 |
| Average precision | 0.01457 |
| Precision in top 10 percent | 2/121 = 1.65% |
| Recall in top 10 percent | 2/16 = 12.5% |

The final selected candidate is a sigmoid-calibrated random forest. Calibration is a fitting procedure here; independent calibration quality and useful predictive discrimination have not been established. Software tests do not establish those properties.

Incident eligibility and complete negative follow-up remain unconfirmed. Hours, mileage, trips, shifts, severity and demographic attributes are not supplied, and only 15 days of alarms are available. Do not invent exposure rates, 30/90-day trends or fairness claims from these inputs.

## Team improvements

1. Confirm incident definitions, exclusions, captain matching and completeness of follow-up.
2. Request authorized additional months and driving exposure; define a genuinely later evaluation period before experiments.
3. Preserve the existing private identity map and partition registry. Develop on the development population and keep preprocessing/calibration inside training folds.
4. Compare small, defensible feature/model changes against simple baselines using average precision, precision/recall at review capacity, ROC-AUC and calibration measures. Record the protocol and uncertainty outside the coding folder.
5. Add meaningful regression tests, rerun checks, rebuild consistent artifacts when needed, and keep the queue focused on who needs review and why.

## Confidentiality and GitHub publication

The course data are confidential. Commit source, configuration and this README only. `.gitignore` excludes `data/`, `models/`, `outputs/`, `.venv/`, caches, common data/model formats and local environment files. Do not commit identity maps, original IDs, exports, screenshots with private rows, credentials or real-data fixtures. Inspect staged file contents as well as filenames before any commit or push.

Ignore rules do not remove files already tracked or scrub previous Git history. This is a public source repository; operator data and private artifacts must remain outside Git. Access to confidential inputs must be arranged separately through the team's approved channel.

No project-wide licence has been selected. The bundled font retains its own licence in `dashboard/assets/OFL.txt`.
