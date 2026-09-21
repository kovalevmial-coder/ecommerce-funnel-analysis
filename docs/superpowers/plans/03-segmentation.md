# Stage 03: Segmentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the funnel rates per approved segment (price tier, visit index, weekday/weekend, hour daypart, month) with per-segment Wilson CIs, plus one descriptive continuous price→conversion curve. All purely descriptive — no tests run here (that is Stage 04/05's confirmatory work).

**Architecture:** `src/segments.py` owns segment definitions and the tidy per-segment funnel frame; `src/plots.py` grows two plotting functions. Segmentation results are descriptive inputs for the pre-registered comparisons later.

**Tech Stack:** polars, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`

---

### Task 1: Segment definitions (`src/segments.py`)

**Files:**
- Create: `src/segments.py`
- Test: `tests/test_segments.py`

**Interfaces:**
- Consumes: `session_features` output (`src/funnel.py`) — columns `hour, dayofweek, month, session_number, median_price, has_view, has_cart, has_purchase`, `net_cart`.
- Produces:
  - `PRICE_TIERS: dict[str, tuple[float, float]]` — fixed business bands: `budget <5`, `mid [5,30)`, `premium [30,150)`, `luxury [150, inf)`. EUR; documented in docstring as merchandising tiers for cosmetics retail.
  - `DAYPARTS: dict[str, tuple[int, int]]` — `night [0,6), morning [6,12), afternoon [12,18), evening [18,24)`.
  - `assign_segments(sessions: pl.DataFrame) -> pl.DataFrame` — adds the categorical label columns `price_tier:str` (from `median_price`, `"NA"` when median missing), `visit_kind:str` (`new` if `session_number == 1` else `returning`), `is_weekend:bool` (dayofweek ∈ {5,6}), `daypart:str` (from `hour` via `DAYPARTS`), `month_label:str` (e.g. `2019-10`..`2020-02`). It derives labels only from raw columns already present in `session_features` (`hour, dayofweek, month, session_number, median_price`) and keeps those raw columns untouched. `assign_segments` is the single owner of every categorical label.

- [ ] **Step 1: Write the failing test**

Synthetic sessions frame with prices at band boundaries (4.9, 5.0, 29.99, 30.0, 149.99, 150.0) → each lands in the documented tier; sessions with `session_number` 1 vs 2 → new/returning; days 4 vs 5 → weekday/weekend; hours 2/8/14/20 → night/morning/afternoon/evening; a null median → `price_tier = "NA"`. Assert every label. Run; verify failure.

- [ ] **Step 2: Implement `PRICE_TIERS`, `DAYPARTS`, `assign_segments`**

Implement with polars `when/then/otherwise`, mapping hour/dayofweek/session_number. Ensure `price_tier` NA-handling is explicit (documented why: sessions can lack priced events; they remain in the funnel but outside price-tier comparisons — reported, not silently dropped).

- [ ] **Step 3: Run and verify**

Run `.venv/bin/python -m pytest tests/test_segments.py -v`. Also assert the bin map is exhaustive: every real price maps into exactly one tier (add a property-style loop over boundary values in the same test file).

- [ ] **Step 4: Commit**

```bash
git add src/segments.py tests/test_segments.py
git commit -m "feat: add fixed business price tiers and segment labels"
```

---

### Task 2: Per-segment funnel (`src/segments.py::segment_funnel`)

**Files:**
- Modify: `src/segments.py` (add `segment_funnel`)
- Test: `tests/test_segments.py` (extend)

**Interfaces:**
- Consumes: `assign_segments`, `funnel_rates` semantics from Stage 02.
- Produces:
  - `segment_funnel(sessions: pl.DataFrame, by: str) -> pl.DataFrame` — same output columns as `funnel_rates` plus `segment:str`: rows = (segment × 3 steps). For `by ∈ {price_tier, visit_kind, is_weekend, daypart, month_label}`. Rates defined exactly as in `funnel_rates` (P(cart|view), P(purchase|cart), P(purchase|view)) but conditioned within each segment's sessions. Keep `n_sessions` column too (segment size — needed for practical significance).

- [ ] **Step 1: Write the failing test**

Extend the Task 1 fixture into a structured sessions frame where segment-level counts are hand-computable; assert `segment_funnel(sessions, "visit_kind")` produces the exact expected numerator/denominator/rate rows for both `new` and `returning`.

- [ ] **Step 2: Implement `segment_funnel`**

Implement by constructing the segment key with `assign_segments` (when not already present) then `group_by(segment)` and computing the three step rates with the same boolean numerators as `funnel_rates`. To avoid logic drift from Stage 02, factor the three-step computation into a private `_funnel_from_flags(df) -> pl.DataFrame` used by BOTH `funnel_rates` and `segment_funnel` (single source of truth — update `tests/test_funnel_rates.py` to keep passing).

- [ ] **Step 3: Run full suite and verify**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PASS (funnel tests included — refactor must not break them).

- [ ] **Step 4: Render real segment funnels (descriptive, not tested)**

Run a scratch script (not committed, or a committed `scripts/explore_segments.py` marked exploratory) that reads `sessions.parquet`, assigns segments, and saves each segment funnel CSV to `data/processed/segment_<by>.csv` — used as descriptive inputs in notebook 03 and to sanity-check counts. No statistics here.

- [ ] **Step 5: Commit**

```bash
git add src/segments.py scripts/explore_segments.py tests/test_segments.py
git commit -m "feat: add per-segment funnel computation"
```

---

### Task 3: Segment visualizations (`src/plots.py::plot_segment_rates`, `plot_price_conversion_curve`)

**Files:**
- Modify: `src/plots.py`
- Test: `tests/test_plots.py` (extend)

**Interfaces:**
- Consumes: `segment_funnel` output; sessions frame for the price curve.
- Produces:
  - `plot_segment_rates(seg_rates: pl.DataFrame, step: str, out_path: Path) -> None` — draws the given step's rates for every segment in `seg_rates`, 95% errors from `ci_low/ci_high`, sorted by rate, labeled with `n_sessions`. One plot per (segment-family, step) — each answers one question. (The segment family is already fixed in `seg_rates`; the `segment` column carries the group names.)
  - `plot_price_conversion_curve(sessions: pl.DataFrame, out_path: Path) -> None` — descriptive continuous view: conversion rate (P(cart|view)) over binned median-price x-axis with confidence ribbons; purpose: expose price-conversion shape without hard cuts.

- [ ] **Step 1: Write the failing tests**

For `plot_segment_rates`: feed a tiny `segment_funnel` frame with 2 segments, assert PNG created at `tmp_path` (magic bytes + non-empty). For `plot_price_conversion_curve`: feed a 50-row sessions fixture, assert PNG file exists.

- [ ] **Step 2: Implement both functions**

Implement in the same shared-style module. Both must save-and-close figures, create parent dirs, answer one question. Alpha/value labels kept minimal per the visualization budget rule.

- [ ] **Step 3: Render to assets for human check**

Run a scratch call on real `sessions.parquet` writing `assets/price_conversion_curve.png` and a couple of segment plots. Inspect in stage report; flag any page-layout or scale issue honestly.

- [ ] **Step 4: Commit**

```bash
git add src/plots.py tests/test_plots.py
git commit -m "feat: add segment and price-conversion visualizations"
```

---

### Task 4: Stage 03 report + learning note

**Files:**
- Create: `notes/03_segmentation.md`

**Interfaces:**
- Consumes: segment funnel CSVs + charts from Tasks 2–3.
- Produces: RU note covering: why fixed business tiers beat quantiles for a company decision, descriptive-vs-confirmatory boundary (nothing tested here), NA price tier handling, small-category caveat (sparse segments → unreliable), and the plan to pre-register comparisons.

- [ ] **Step 1: Write the note.**

- [ ] **Step 2: Commit**

```bash
git add notes/03_segmentation.md
git commit -m "docs: add stage 03 learning note (RU)"
```

---

**Stage verification gate:** `pytest` green; `segment_<by>.csv` artifacts produced; price-curve + sample segment charts rendered to `assets/` for human inspection. Report to the user: segment sizes and the descriptive pattern seen (no inference claims).