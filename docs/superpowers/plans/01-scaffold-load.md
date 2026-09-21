# Stage 01: Scaffold + Event Loading

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the Python environment, git hygiene, test scaffolding, and a tested raw-event loader that reads all five CSVs into one typed DataFrame/parquet.

**Architecture:** A `src/` package with small single-responsibility modules, tested with pytest on tiny synthetic fixtures (never on the 2.3 GB raw data). Notebooks and scripts import from `src/` only. Raw CSVs stay gitignored; a processed-events parquet is the intermediate artifact.

**Tech Stack:** Python 3.12, uv, polars, pytest, `pyproject.toml` + pinned `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`

---

## Global Constraints

Applies to every task in every stage. Copied from the spec:

- Repo: `dimassaa/ecommerce-funnel-analysis` (push planned after completion; local for now). Branch `main`, already initialized.
- Analysis is on the E-commerce Events History in Cosmetics Shop dataset (Kaggle mkechinov), Oct 2019 – Feb 2020, 5 CSVs in `data/` (~2.3 GB, 20.7 M events). Raw CSVs and `data/processed/` artifacts are gitignored and never committed.
- Funnel unit of analysis: `user_session` (primary); user-level robustness check later.
- Funnel steps (per-session indicators): `view` = ≥1 view, `cart` = ≥1 cart, `purchase` = ≥1 purchase. Rates: `view→cart`, `cart→purchase`, `view→purchase`, each with Wilson 95% CI.
- Segmentation: fixed business price tiers (<5 / 5–30 / 30–150 / 150+ EUR), visit index (new vs returning), weekday/weekend, hour dayparts, month (exploratory only). Price tiers are pinned and committed in `reports/pre_registration.md` BEFORE any confirmatory test runs.
- Confirmatory statistics: chi-square independence + pairwise two-proportion z-tests, α = 0.05, two-sided, per-family Bonferroni (+ FDR as sensitivity). Sparse rule: for proportional tests, if min(n·p, n·(1−p)) < 5 in any cell → return NaN z/p/CI and set `sparse=True` flag (reported descriptively).
- Statistical significance (p, CI) is never the verdict alone; practical significance (p.p., relative uplift, business volume) is reported in the same table.
- All reusable logic lives in tested `src/` functions; notebooks and scripts call `src/` and never re-implement analysis inline. Cap reuse across modules; no duplicated logic.
- Every plot answers one question and is saved to `assets/` with a descriptive filename; matplotlib only via `src/plots.py`.
- Randomness: fix `random_state` wherever any exists (e.g., bootstrap). Keep `binary` metrics sanity-checked at integrity stage (already integer 0/1 by construction here — still assert).
- Code comments and commit messages: English, "why not what". Learning notes in `notes/` are Russian (personal). Notebook/report narrative in English (repo ships for resume).
- `pytest` must stay green after every commit.
- Pre-registration commit must precede confirmatory (segmentation/hypothesis) notebook outputs in git history.

---

### Task 1: Environment + test scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Create: `src/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: venv at `.venv`; `src` importable as a package; `pytest` discovers `src`/`tests` with no config beyond `pyproject.toml`.

- [ ] **Step 1: Create the virtual environment**

Run: `uv venv .venv` then `uv pip install --python .venv/bin/python ipykernel pytest polars numpy scipy statsmodels matplotlib seaborn nbformat nbclient nbconvert`
Verify with: `.venv/bin/python -m pytest --version` and `.venv/bin/python -c "import polars, scipy, statsmodels, matplotlib, seaborn, nbformat, nbclient, nbconvert"`.

- [ ] **Step 2: Freeze pinned dependencies**

Inside the active `.venv`, write `requirements.txt` with exact `==` pins for every installed package (record via `uv pip freeze`). Add a header comment listing Python version (3.12) and generation date. This is the reproducibility contract for all stages.

- [ ] **Step 3: Write `pyproject.toml`**

Add pytest configuration: tool section setting `pythonpath = ["."]` and `testpaths = ["tests"]`, plus project metadata (name `ecommerce-funnel-analysis`, version 0.1.0, requires-python `>=3.12`). Do not duplicate dependency lists here — `requirements.txt` is the single source for pins.

- [ ] **Step 4: Add package init and smoke test**

Create `src/__init__.py` exposing `__version__ = "0.1.0"`. Create `tests/test_smoke.py` asserting `import src` works and `src.__version__` equals "0.1.0". Add a `tests/conftest.py` that (a) makes a `data/` fixture path helper for future synthetic-data tests and (b) nothing else for now.

- [ ] **Step 5: Run and verify**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 1 passed, 0 warnings beyond noise. Also run `git status --short` and confirm `data/`, `.venv/`, `AGENTS.md`, `The Ultimate README Guide.md`, `docs/superpowers/` are all already ignored or tracked as intended.

- [ ] **Step 6: Register the notebook kernel (needed Stage 05+)**

Run: `.venv/bin/python -m ipykernel install --user --name ecommerce-funnel --display-name "Python 3.12 (ecommerce-funnel)"`
Verify: `jupyter kernelspec list` shows `ecommerce-funnel`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt src/__init__.py tests/conftest.py tests/test_smoke.py
git commit -m "build: scaffold python 3.12 env, pinned deps, pytest baseline"
```

---

### Task 2: Raw-event loader (`src/data_io.py`)

**Files:**
- Create: `src/data_io.py`
- Test: `tests/test_data_io.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `event_files(raw_dir: Path) -> list[Path]` — sorted list of the 5 monthly CSVs from `raw_dir`.
  - `load_events_lazy(paths: Sequence[Path]) -> pl.LazyFrame` — lazy concatenation with a fixed schema and `event_time` parsed to `pl.Datetime` (the raw strings end in `" UTC"`; strip that suffix before parsing and treat the result as UTC).
  - `write_events_parquet(raw_dir: Path, out_file: Path) -> None` — collects a streamed scan to parquet, `sink`/`sink_parquet` with deterministic dtype widths.

- [ ] **Step 1: Write the failing test**

Fixture: create a tiny temp CSV (a few rows each of `view/cart/remove_from_purchase`) with the exact raw header, plus a second temp CSV as a second month, in a pytest `tmp_path`. The test asserts: `event_files` returns the two paths sorted; `load_events_lazy(paths)` read via `.collect()` yields the concatenated row count and `event_time` as `pl.Datetime`; a known `event_time` string round-trips to the expected UTC timestamp. Run it; verify it fails with "function not defined".

- [ ] **Step 2: Implement `event_files` and `load_events_lazy`**

Implement the sorted glob, the fixed schema dict (event_type string, product_id/category_id/user_id int64, category_code/brand string nullable, price float32, user_session string, event_time datetime), and the lazy concat with time parsing. Keep the read cheap: `low_memory`, no materialization until `.collect()`.

- [ ] **Step 3: Implement `write_events_parquet`**

Implement streaming write: from the lazy frame produce `out_file` (parquet) without loading the full frame into memory. Use `overwrite_mode="drop"`.

- [ ] **Step 4: Extend the test for the writer**

Assert `write_events_parquet` creates the expected file and that reading it back with `pl.read_parquet` matches the concatenated row count and datetime type.

- [ ] **Step 5: Run and verify**

Run: `.venv/bin/python -m pytest tests/test_data_io.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/data_io.py tests/test_data_io.py
git commit -m "feat: add typed lazy loader for raw event CSVs"
```

---

### Task 3: Integrity report + processed-events artifact

**Files:**
- Create: `src/eda.py`
- Create: `scripts/etl_events.py`
- Test: `tests/test_eda.py`

**Interfaces:**
- Consumes: `src/data_io.load_events_lazy`, `synthetic events fixture` from Task 2.
- Produces:
  - `integrity_report(events: pl.DataFrame) -> dict` — structured stats (rows, distinct users, distinct sessions, per-event_type counts, missing % for `brand` and `category_code`, price min/max/share-of-zero, sessions with purchase-without-cart, sessions with cart-without-view, sessions with only-view, duplicate-session check).
  - `scripts/etl_events.py` — CLI-free idempotent script: builds `data/processed/events.parquet` from `data/`; prints row count and elapsed seconds.

- [ ] **Step 1: Write the failing test**

Using the synthetic fixture, assert `integrity_report` returns the exact expected dict: correct row count, per-type counts, known NaN shares, and correct purchase-without-cart / cart-without-view session counts for the crafted fixture. Run; verify it fails (no module yet).

- [ ] **Step 2: Implement `integrity_report`**

Implement the report over the aggregated small frame. All session-level counts derive from grouping `user_session`. This is the data-integrity gate (golden standard 1). Add docstring listing returned keys and meanings.

- [ ] **Step 3: Write `scripts/etl_events.py`**

Implement the ETL script: glob `data/2019-*.csv` + `data/2020-*.csv`, load lazily, print integrity summary via `integrity_report`, `write_events_parquet` to `data/processed/events.parquet`. Make it safe to re-run (overwrite).

- [ ] **Step 4: Run the ETL once on the real data**

Run: `.venv/bin/python scripts/etl_events.py`
Expected: finishes in ~1–2 minutes; prints the integrity report; `data/processed/events.parquet` exists. Record the printed numbers into the stage report (they are the ground truth: 20,692,840 events; 1,639,358 users; ~4,535,941 sessions; 98.3%/42.3% missing — confirm against these).

- [ ] **Step 5: Verify gitignore + tests + commit**

Confirm `data/processed/events.parquet` is gitignored (`git check-ignore data/processed/events.parquet`). Run full suite (all PASS).

```bash
git add src/eda.py scripts/etl_events.py tests/test_eda.py
git commit -m "feat: add integrity report and events parquet ETL"
```

---

### Task 4: Stage 01 report + learning note

**Files:**
- Create: `notes/00_scope_and_plan.md`

**Interfaces:**
- Consumes: the ground-truth numbers from Task 3 Step 4.
- Produces: a Russian learning note documenting the scope, why data integrity is the non-negotiable first step, key terms (data integrity, unit of analysis, missing data share) with English equivalents, what was done, conclusions, and pitfalls (category_code 98% missing ⇒ cannot segment by product category; step order does not hold within sessions).

- [ ] **Step 1: Write the note**

Russian narrative per AGENTS.md structure: цель этапа, разбор «зачем это важно», ключевые термины (+ англ.), что сделано, выводы, подводные камни.

- [ ] **Step 2: Commit**

```bash
git add notes/00_scope_and_plan.md
git commit -m "docs: add stage 01 learning note (RU)"
```

---

**Stage verification gate:** `pytest` green; `data/processed/events.parquet` reproducible from `scripts/etl_events.py`; commit log contains the three commits above. Report to the user: ground-truth integrity numbers + elapsed time.