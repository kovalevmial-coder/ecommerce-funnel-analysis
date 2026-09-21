# Stage 06: Confirmatory Analysis + Business + Robustness (notebooks 02–06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute notebooks 02 (funnel), 03 (segmentation), 04 (confirmatory hypothesis tests), 05 (business/bottleneck + executive summary), 06 (robustness). Notebooks 04–06 are confirmatory and run **after** the Stage-05 pre-registration commit. Every notebook runs `errors=none`.

**Architecture:** Each notebook is a `notebooks/build_XX_*.py` + `scripts/build_notebook.py` pair; analysis calls `src/` only. Confirmatory notebook 04 runs exactly the pre-registered comparisons — no additional tests, no post-hoc tuning.

**Tech Stack:** nbformat/nbclient, polars, scipy, statsmodels (z-test cross-check only), matplotlib/seaborn via `src/plots.py`.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`
**Pre-registration:** `reports/pre_registration.md` (already committed)

---

### Task 1: Notebook 02 — funnel

**Files:**
- Create: `notebooks/build_02_funnel.py`, `notebooks/02_funnel.ipynb` (generated, committed)
- Create: `assets/funnel_chart.png` (generated, committed)

**Interfaces:**
- Consumes: `src/funnel.funnel_rates`, `src/plots.plot_funnel`, `data/processed/sessions.parquet`; pre-registration funnel definition.
- Produces: executed notebook 02 with:
  1. Markdown: recap of funnel definition + unit (references pre-registration section).
  2. Code+markdown: session table load; a compact session summary (n sessions, per-step boolean counts, net_cart share).
  3. Code: `funnel_rates` table printed as markdown (step, n, rate, CI) — the headline funnel.
  4. Code: `plot_funnel` → `assets/funnel_chart.png` + inline; markdown interpreting the biggest absolute drop (candidate bottleneck, flagged as a question for notebook 05, no claim yet).
  5. Code+markdown: diagnostics — purchases-without-cart, cart-without-view, direct-purchase share; why the funnel is defined by reached-steps (documents the data reality).

- [ ] **Step 1: Write `build_02_funnel.py`** per spec using `build_and_execute`; keep narrative factual.
- [ ] **Step 2: Build & execute** — `.venv/bin/python notebooks/build_02_funnel.py`; confirm `errors=none`, funnel chart written to `assets/`.
- [ ] **Step 3: Inspect & commit**

```bash
git add notebooks/build_02_funnel.py notebooks/02_funnel.ipynb assets/funnel_chart.png
git commit -m "feat: add executed funnel notebook with headline rates"
```

---

### Task 2: Notebook 03 — segmentation (descriptive)

**Files:**
- Create: `notebooks/build_03_segmentation.py`, `notebooks/03_segmentation.ipynb` (generated, committed)
- Create: `assets/seg_price_tier_view_cart.png`, `assets/seg_visit_cart_purchase.png`, `assets/seg_weekend_cart_purchase.png` (generated, committed)

**Interfaces:**
- Consumes: `src/segments.assign_segments`, `segment_funnel`, `src/plots.plot_segment_rates`, `data/processed/sessions.parquet`.
- Produces: executed notebook 03 covering, purely descriptively:
  1. Markdown: what segmentation is approved, what is descriptive vs confirmatory (confirmatory = notebook 04); explicit note "no inference here".
  2. Code+md: segment sizes table (n_sessions per tier / visit_kind / weekend — report NA price-tier group explicitly).
  3. Code: three segment-funnel charts (each uses `assets_path` from `src/plots.py` for the output path):
     - `plot_segment_rates(segment_funnel(sessions, "price_tier"), "view->cart", ...)` → `assets/seg_price_tier_view_cart.png`;
     - `plot_segment_rates(segment_funnel(sessions, "visit_kind"), "cart->purchase", ...)` → `assets/seg_visit_cart_purchase.png`;
     - `plot_segment_rates(segment_funnel(sessions, "is_weekend"), "cart->purchase", ...)` → `assets/seg_weekend_cart_purchase.png`.
  4. Code+markdown: `plot_price_conversion_curve` → `assets/price_conversion_curve.png` (descriptive continuous view) with a factual reading (e.g., monotonic/non-monotonic shape without significance claims).
  5. Markdown: list of patterns observed (candidate hypotheses listed, to be *tested* in notebook 04).

- [ ] **Step 1: Write `build_03_segmentation.py`** per spec.
- [ ] **Step 2: Build & execute**; confirm the three segment charts + price curve written; spot-check segment sizes vs `data/processed/segment_*.csv`.
- [ ] **Step 3: Inspect & commit**

```bash
git add notebooks/build_03_segmentation.py notebooks/03_segmentation.ipynb assets/seg_price_tier_view_cart.png assets/seg_visit_cart_purchase.png assets/seg_weekend_cart_purchase.png assets/price_conversion_curve.png
git commit -m "feat: add executed descriptive segmentation notebook"
```

---

### Task 3: Notebook 04 — CONFIRMATORY hypothesis tests

**Files:**
- Create: `notebooks/build_04_hypothesis_testing.py`, `notebooks/04_hypothesis_testing.ipynb` (generated, committed)
- Create: `assets/confirm_price_tier_forest.png` (pairwise differences with CIs, family A; generated, committed)

**Interfaces:**
- Consumes: `src/inference` (`chi2_independence`, `two_prop_ztest`, `bonferroni`, `fdr_bh`, `wilson_ci`), `src/segments`, `data/processed/sessions.parquet`, `reports/pre_registration.md`.
- Produces: executed notebook 04 running ONLY the pre-registered tests, in this exact structure:
  1. Markdown: restate the pre-registered decision rule (α, two-sided, corrections, sparse rule, practical-significance rule); cite the pre-registration commit SHA.
  2. **Family A — price tier × view→cart**: build 4×2 contingency (tiers as rows, cart/no-cart among `has_view` sessions, NA tier excluded with an explicit note); `chi2_independence` with `valid` check; pairwise `two_prop_ztest` across all 6 tier pairs in a table (p_a, p_b, diff, CI, p-value, sparse flag); Bonferroni + FDR columns; forest plot `confirm_price_tier_forest.png` of differences with CIs.
  3. **Family B — visit_kind × cart→purchase**: single `two_prop_ztest` (new vs returning).
  4. **Family C — weekend × cart→purchase**: single `two_prop_ztest`.
  5. **Practical-significance table**: for every nominal test that passed its corrected threshold, show diff (p.p.), relative uplift, affected volume (n of sessions swayed), and the pre-registered decision-rule verdict (YES/NO/SPARSE). This is the "two kinds of significance" golden standard — a separate verdict column.
  6. Markdown: verdict summary — which comparisons are statistically AND practically significant; explicit statement that the overall funnel is descriptive (no claim).

- [ ] **Step 1: Write `build_04_hypothesis_testing.py`** per spec. Code cells must call only `src/inference` and `src/segments`; results rendered as markdown tables.
- [ ] **Step 2: Build & execute**; confirm `errors=none`; verify the executed tests exactly match the pre-registered list — a sanity audit: family A must contain exactly 1 chi-square (independence over the 4 tiers) plus 6 pairwise z-tests; family B exactly 1 z-test; family C exactly 1 z-test. No test beyond the pre-registered set may be added.

  **Fidelity check before commit:** re-read `reports/pre_registration.md` and assert the executed comparisons match the document exactly. Any deviation must be recorded in the notebook as an explicitly labelled exploration, never silently.

- [ ] **Step 3: Inspect & commit**

```bash
git add notebooks/build_04_hypothesis_testing.py notebooks/04_hypothesis_testing.ipynb assets/confirm_price_tier_forest.png
git commit -m "feat: add executed confirmatory hypothesis testing notebook"
```

---

### Task 4: Notebook 05 — business / bottleneck / A-B proposal + executive summary

**Files:**
- Create: `notebooks/build_05_business.py`, `notebooks/05_business.ipynb` (generated, committed)
- Create: `reports/05_executive_summary.html` (nbconvert of notebook 05; committed)
- Create: `assets/bottleneck_diagnostic.png` (generated, committed)

**Interfaces:**
- Consumes: `src/funnel.funnel_rates`, `src/segments.segment_funnel`, `src/inference.required_n_per_group`, `achieved_power`, `two_prop_ztest`, notebook-04 verdicts.
- Produces: executed notebook 05 covering:
  1. Markdown: the ONE question — where do we lose most customers, and is the loss concentrated in a segment?
  2. Code+md: **bottleneck diagnosis**: absolute session loss per funnel step (view→cart vs cart→purchase) computed from `funnel_rates`; `bottleneck_diagnostic.png` (volume lost per step + worst-segment overlay); identify the bottleneck step and the worst-performing segment at it, citing notebook-04 outcomes (not pressure-tested numbers).
  3. Code+md: **business hypotheses** — 3–5 concrete product/business hypotheses for the bottleneck, each mapped to evidence (which segment/test supports it) and what would falsify it. Marked as hypotheses, not conclusions.
  4. Code+md: **next-step A/B test proposal** — choose ONE hypothesis; fix p₀ = relevant baseline rate from the funnel; pick MDE and power (justified in business terms, referencing the practical-significance rule); compute `required_n_per_group` and `achieved_power` at realistic available sample sizes; present the pre-registered-style design table for the future test (this is a *proposal*, honestly labelled so — the test itself is out of scope per plan).
  5. Markdown: executive verdict — one paragraph in plain business language: what the funnel loses, where, which segment, what I recommend, and the honest limits (product-attribute segments are proxies, not user attributes; no causal claim).
  6. Code: final verdict variables (bottleneck step, worst segment, recommendation) exposed in variables that notebook 06 and README reuse — single source of truth for consistency checks.

- [ ] **Step 1: Write `build_05_business.py`** per spec.
- [ ] **Step 2: Build & execute**; confirm `errors=none`.
- [ ] **Step 3: Export executive summary**

Run `.venv/bin/python -m jupyter nbconvert --to html --output-dir reports --output 05_executive_summary notebooks/05_business.ipynb`
Verify `reports/05_executive_summary.html` opens (file exists, non-trivial size).

- [ ] **Step 4: Inspect & commit**

```bash
git add notebooks/build_05_business.py notebooks/05_business.ipynb reports/05_executive_summary.html assets/bottleneck_diagnostic.png
git commit -m "feat: add business notebook and executive summary report"
```

---

### Task 5: Notebook 06 — robustness + verdict consistency

**Files:**
- Create: `notebooks/build_06_robustness.py`, `notebooks/06_robustness.ipynb` (generated, committed)

**Interfaces:**
- Consumes: `src/funnel`, `src/segments`, `src/inference`; notebook-04/05 verdict variables or equivalent recomputation.
- Produces: executed notebook 06 with a **robustness battery**, every variant explicitly labelled as sensitivity (never reframed as the headline):
  1. **User-level variant**: funnel per `user_id` (ever-view/ever-cart/ever-purchase over the window); recompute rates and compare direction-of-effect for family B (visit-kind is per-session here and collapses — state that).
  2. **Strict-path variant**: progress-only funnel (view THEN cart THEN purchase within a session); report how the headline rates and the bottleneck change when jumpers are excluded (direct purchase = ~19% of purchase sessions — already documented).
  3. **Net-cart variant**: re-run the overall funnel (and family C only) using `net_cart` instead of `has_cart`; report movement of family-C verdict.
  4. **Quantile-tier sensitivity** for family A: recompute the price-tier comparison using quantile-derived tiers instead of business bands; report whether the verdict survives the redefinition.
  5. **Exploratory (non-confirmatory):** month trend and daypart funnel — descriptive tables/charts with a clear "exploratory, not pre-registered" label and no significance claims.
  6. **Verdict consistency block**: markdown comparing the verdict wording in notebook 05 vs 06; if any robustness variant flips a pre-registered verdict, the flip is documented as a genuine finding (honesty over narrative) and the downstream verdicts (README) must be updated to match.

- [ ] **Step 1: Write `build_06_robustness.py`** per spec.
- [ ] **Step 2: Build & execute**; confirm `errors=none`.
- [ ] **Step 3: Verdict consistency check**

Compare the bottleneck/segment/recommendation stated in notebooks 05/06. If they conflict, fix the downstream statement in this stage (document the change) — nothing gets buried.

- [ ] **Step 4: Inspect & commit**

```bash
git add notebooks/build_06_robustness.py notebooks/06_robustness.ipynb
git commit -m "feat: add robustness notebook with variant funnels and honesty check"
```

---

### Task 6: Stage-06 learning notes + gate

**Files:**
- Create: `notes/05_business.md`, `notes/06_robustness.md`

**Interfaces:**
- Consumes: notebook-04/05/06 conclusions.
- Produces: two RU notes:
  - `05_business.md`: practical vs statistical significance, bottleneck logic (largest absolute loss), sample-size/power for the proposal, how a business answer is written without causal claims.
  - `06_robustness.md`: why sensitivity analysis exists, user-vs-session unit differences, strict-vs-reached funnel, honest result flipping.

- [ ] **Step 1: Write both notes** (RU; same structure as earlier notes).
- [ ] **Step 2: Full-suite run + commit**

```bash
.venv/bin/python -m pytest tests/ -q
git add notes/05_business.md notes/06_robustness.md
git commit -m "docs: add stage 06 learning notes (RU)"
```

---

**Stage verification gate:** notebooks 02–06 all `errors=none`; notebook-04 matches pre-registration line-for-line; exec summary HTML exists; `pytest` green; git log shows pre-registration SHA *before* notebooks 03/04 commits. Report to the user: headline funnel + per-family results + verdict + any robustness flips, with my honest read.