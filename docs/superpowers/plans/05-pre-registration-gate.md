# Stage 05: Pre-registration Gate (nb 01 + report + commit)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute notebook 01 (descriptive data integrity + EDA + price distribution), pin the price-tier bands, write `reports/pre_registration.md` fixing ALL confirmatory decisions, and commit it **before** any segment-conversion outcome exists (notebooks 03/04 run in Stage 06, after this commit). This is the golden-standards gate: nothing below is tuned to a found result.

**Architecture:** A shared `scripts/build_notebook.py` runner turns `notebooks/build_01_*.py` scripts into executed `.ipynb` files binding the kernel `ecommerce-funnel`. Notebook 01 imports `src/` functions only; it never re-implements analysis.

**Tech Stack:** nbformat, nbclient, ipykernel, polars.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`

---

### Task 1: Notebook build runner (`scripts/build_notebook.py`)

**Files:**
- Create: `scripts/build_notebook.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `build_and_execute(name: str, cells: list[dict], out_dir: Path) -> Path` — where each cell dict is `{"kind": "markdown"|"code", "source": str}`; builds an nbformat notebook (title Metadata markdown included), executes it via nbclient with `kernel_name="ecommerce-funnel"` and a generous cell timeout, saves `out_dir/<name>.ipynb` with outputs embedded, and raises if any cell erroring (execution error) — errors must never be silently embedded.

- [ ] **Step 1: Implement the runner**

Implement completion of cell dicts via `nbformat.v4.new_markdown_cell/new_code_cell`, assemble & execute with `nbclient.NotebookClient`, write out, and validate the executed notebook has no `error` outputs (assert with a clear message listing the offending cell index).

- [ ] **Step 2: Smoke-test the runner**

Create a throwaway `scripts/_smoke_nb.py` calling `build_and_execute("_smoke", [markdown cell, code cell printing "1+1"])` into `docs/superpowers/_scratch/`. Run it; verify the printed `2` is embedded in the executed notebook outputs. Delete the scratch notebook afterwards; keep `scripts/_smoke_nb.py` out of git (listed in `.gitignore` as `scripts/_smoke_nb.py` — add the line) OR delete it. Decide: add the ignore line (developer utility).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_notebook.py .gitignore
git commit -m "build: add reproducible notebook build-execute runner"
```

---

### Task 2: Notebook 01 — data integrity + EDA

**Files:**
- Create: `notebooks/build_01_data_integrity.py`
- Create: `notebooks/01_data_integrity.ipynb` (generated, committed)
- Create: `assets/integrity_events_mix.png`, `assets/integrity_price_distribution.png` (generated, committed)

**Interfaces:**
- Consumes: `src.eda.integrity_report`, `src.data_io`, `data/processed/events.parquet`; Stage-01 ground-truth numbers.
- Produces: executed notebook 01 covering, in order:
  1. Markdown: purpose of this stage; the unit-of-analysis decision (session vs user) and why (references `notes/00_*`, `notes/01_*`).
  2. Code: load events.parquet; run `integrity_report`; present the report as a markdown table.
  3. Code: event-type mix bar chart (saved to `integrity_events_mix.png`); short markdown interpreting view/cart/remove/purchase volumes.
  4. Code: session-level diagnostics — sessions with purchase-without-cart (≈19%), cart-without-view (≈20%), and their share of the funnel (from `scripts/etl_pipeline.py` outputs re-derived here via `src`); markdown: implications for funnel definition (reached-step indicators).
  5. Code: price distribution (histogram + decile table) saved to `integrity_price_distribution.png`; markdown stating this is a *feature* view used only to pin tiers — outcomes are not inspected here.
  6. Markdown: explicit "nothing here is a significance claim" note (descriptive stage).

- [ ] **Step 1: Write the build script**

Implement `build_01_data_integrity.py` per spec above, using `build_and_execute`. Code cells import from `src`, read the parquet, call `src` functions, and print/save figures. Keep the EDA strictly descriptive.

- [ ] **Step 2: Build and execute**

Run: `.venv/bin/python notebooks/build_01_data_integrity.py`
Expected: notebook executed without errors; renders `assets/integrity_events_mix.png` and `assets/integrity_price_distribution.png`; integrity table matches Stage-01 ground truth (20,692,840 events, 1,639,358 users, ~4.54 M sessions, 98.3%/42.3% missing).

- [ ] **Step 3: Inspect outputs honestly**

Open the generated `notebooks/01_data_integrity.ipynb` fresh (read the JSON outputs) and check: numbers match ground truth; every cell saved an output or expected empty; no warning-polluted figures. Flag any mismatch in the stage report — do not gloss over.

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_01_data_integrity.py notebooks/01_data_integrity.ipynb assets/integrity_events_mix.png assets/integrity_price_distribution.png
git commit -m "feat: add executed data integrity notebook (descriptive EDA)"
```

---

### Task 3: Pin price tiers + write `reports/pre_registration.md` + commit gate

**Files:**
- Create: `reports/pre_registration.md`

**Interfaces:**
- Consumes: notebook-01 price distribution (deciles); Stage-02 funnel definition; approved segments.
- Produces: a pre-registration document that fixes, before any confirmatory output:
  1. **Scope**: dataset, months, unit of analysis (session-primary, user-robustness), funnel steps and the three rates; explicit statement that overall funnel figures are descriptive and NOT confirmatory.
  2. **Price tiers**: the fixed bands `<5 / [5,30) / [30,150) / 150+ EUR`; the tier-validation clause: if any band has < 500 sessions, it is merged upward with the band below and the change recorded in this document *before* any test runs (transparency rule; adjustment at planning time is documented, not post-hoc).
  3. **Confirmatory comparisons** (the complete list — everything else is exploratory):
     - family A: price tier × view→cart (chi-square over 4 tiers; then pairwise z-tests among tiers with cart≠0);
     - family B: visit_kind (new vs returning) × cart→purchase (single z-test);
     - family C: weekday/weekend × cart→purchase (single z-test).
  4. **Parameters**: α = 0.05 two-sided for every test; per-family Bonferroni applied to the pairwise z-tests in family A; FDR (BH) reported as sensitivity; sparse rule (n·p < 5 → no test, `sparse=True`).
  5. **Practical-significance rule**: a difference is decision-relevant only if its 95% CI excludes 0 AND the relative uplift is ≥ 10% AND the affected volume is ≥ 1% of sessions (pre-set thresholds, not chosen after seeing results). Motivates thresholds in "business terms".
  6. **Boundary statement**: no hypothesis or threshold below may be changed after notebook 03/04 run; any analysis adjustment becomes an explicitly-labelled exploratory post hoc (documented in notebook 06).

- [ ] **Step 1: Verify tier feasibility**

From notebook-01 price deciles, confirm each of the four bands is non-empty; record session counts per band (from `data/processed/segment_price_tier.csv` if available, or a quick `src` call). If any band < 500 sessions, apply the merge clause and record it in the document.

- [ ] **Step 2: Write the document**

Markdown with: H₀/H₁ in words and in math per family, the comparison table, parameter section, practical-significance rule, sparse rule, and the boundary statement.

- [ ] **Step 3: Commit the gate**

```bash
git add reports/pre_registration.md
git commit -m "docs: pre-register confirmatory segment comparisons (planning gate)"
```

- [ ] **Step 4: Verify the gate ordering**

Run `git log --oneline | head -8` and record the SHA of the pre-registration commit. Assert there are NO commits for notebooks 03/04 or any `segment_*csv` outcome analysis with an earlier date. This git ordering is the verifiable evidence of the gate.

- [ ] **Step 5: Stage report + learning note**

Create `notes/01_data_integrity.md` and `notes/05_pre_registration.md` (RU): why pre-registration exists (no tuning to found results / hysteresis), what α/β/MDE mean here, why thresholds are fixed before results, and the honest trap of "validating a tier post hoc". Commit both in one `docs:` commit.

---

**Stage verification gate:** notebook 01 executed `errors=none`, integrity numbers match ground truth; `reports/pre_registration.md` committed; git history proves pre-registration precedes any confirmatory output. Report to the user: the pinned price-band counts and the pre-registration decision summary, and my honest read of whether any pre-registered parameter is too aggressive/relaxed.