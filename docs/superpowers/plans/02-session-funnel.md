# Stage 02: Session Features + Funnel Core

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the 20.7 M event rows into one session-level table (`data/processed/sessions.parquet`) and the core funnel table (view→cart, cart→purchase, view→purchase with Wilson 95% CI), all behind tested `src/` functions.

**Architecture:** `session_features()` does one clean groupby over `user_session` (booleans + counts + session number per user + time-derived columns + median price). `funnel_rates()` consumes that table and emits one tidy rows-of-steps frame. Plotting stays in `src/plots.py` (shared by later stages).

**Tech Stack:** polars, scipy for nothing here (Wilson CI implemented closed-form and cross-checked against statsmodels), matplotlib via `src/plots.py`.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`

---

### Task 1: Session features (`src/funnel.py::session_features`)

**Files:**
- Create: `src/funnel.py`
- Test: `tests/test_funnel_session_features.py`

**Interfaces:**
- Consumes: events frame with columns from `src/data_io.EVENT_SCHEMA` (at minimum `event_time`, `event_type`, `user_id`, `user_session`, `price`).
- Produces:
  - `session_features(events: pl.DataFrame) -> pl.DataFrame` with one row per `user_session` and columns:
    `user_session:str, user_id:int64, month:int, dayofweek:int(0=Mon), hour:int, session_number:int, n_view:int, n_cart:int, n_remove:int, n_purchase:int, has_view:bool, has_cart:bool, has_purchase:bool, net_cart:bool(has_cart and n_cart > n_remove? see below), median_price:float64, n_products:int(distinct product_id per session)`.
  - `net_cart` semantics: `True` iff at least one `cart` event remains after cancelling `remove_from_cart` one-for-one (i.e. `n_cart > n_remove`). This is the conservative "cart survived the session" indicator used only in the robustness stage — the primary funnel uses `has_cart`.
  - `session_number`: 1-based sequential index of the session per `user_id`, ordered by first `event_time` — the "new vs returning visit" segmentation input.
  - Categorical labels (`daypart`, `visit_kind`, `is_weekend`, `month_label`, `price_tier`) are NOT computed here — Stage 03's `assign_segments` owns all categorical derivations from the raw `hour`/`dayofweek`/`month`/`session_number` columns. `median_price`: median of `price` across the session's events (price is float32 in events; median cast to float64).

- [ ] **Step 1: Write the failing test**

Craft a synthetic multi-session events frame (≥5 sessions) with known: mixed event types, a user with 3 sessions (session_number 1,2,3), another user with 1, a session with cart>remove (net_cart True), one with cart≤remove (net_cart False), a session mixing dayparts to pin `daypart`, and known median. Assert every produced column value for the crafted rows. Run; verify failure (module missing).

- [ ] **Step 2: Implement `session_features`**

Implement as a single polars expression pipeline: time extraction (month/dayofweek/hour from already-datetime `event_time`), one group-by `user_session` with all aggregations, then `session_number` via sorted `user_id`-level ranking (polars `rank`/`group_by + cumcount` style over ordered frame). Keep `user_id` int64. Document the `net_cart` definition in the docstring (why: remove_from_cart cancels cart; primary funnel intentionally uses `has_cart`).

- [ ] **Step 3: Run and verify**

Run: `.venv/bin/python -m pytest tests/test_funnel_session_features.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/funnel.py tests/test_funnel_session_features.py
git commit -m "feat: add session-level funnel feature builder"
```

---

### Task 2: Session parquet + funnel rates (`src/funnel.py::funnel_rates`, `scripts/etl_pipeline.py`)

**Files:**
- Create: `scripts/etl_pipeline.py`
- Modify: `src/funnel.py` (add `funnel_rates`)
- Test: `tests/test_funnel_rates.py`

**Interfaces:**
- Consumes: `session_features`; `src/data_io` parquet path helper; `scripts/etl_events.py` output (`data/processed/events.parquet`).
- Produces:
  - `funnel_rates(sessions: pl.DataFrame) -> pl.DataFrame` — exactly 3 rows, columns:
    `step:str, numerator:int, denominator:int, rate:float64, ci_low:float64, ci_high:float64` where steps are `view->cart`, `cart->purchase`, `view->purchase` (rates: P(cart|view), P(purchase|cart), P(purchase|view)).
  - `scripts/etl_pipeline.py` — loads `events.parquet`, builds sessions, writes `data/processed/sessions.parquet`, prints the funnel table + per-step totals. Idempotent.

- [ ] **Step 1: Write the failing test**

With a small synthetic sessions frame (e.g. 200 sessions with known view/cart/purchase booleans — 200 view, 100 cart, 40 purchase, 30 of those with both cart+view), assert `funnel_rates` returns exactly the three steps with correct numerators/denominators and rates. Assert Wilson CI is present and symmetric-wrapped in (0,1). Make the fixture also include sessions with purchase-but-no-view and cart-but-no-view so the step maths is honest about jumpers. Run; verify failure.

- [ ] **Step 2: Implement `funnel_rates`**

Implement via boolean aggregates over the sessions frame:
- view→cart: numerator = count(has_cart & has_view), denominator = count(has_view).
- cart→purchase: numerator = count(has_purchase & has_cart), denominator = count(has_cart).
- view→purchase: numerator = count(has_purchase & has_view), denominator = count(has_view) (note: includes direct purchasers who saw a product but never carted — document why in docstring).
CI: Wilson (score) interval, computed by an inline local helper `_wilson_ci` with a `# TODO(stage-04)` marker — Stage 04 Task 1 replaces it with a call to `src/inference.wilson_ci` and removes the marker.

- [ ] **Step 3: Write `scripts/etl_pipeline.py`**

Implement: read `events.parquet` → `session_features` → `sink_parquet` to `data/processed/sessions.parquet` → `funnel_rates` printed as a markdown table, plus counts of direct-purchase and cart-no-view sessions (diagnostics, will be used in notebook 02).

- [ ] **Step 4: Run pipeline on real data**

Run: `.venv/bin/python scripts/etl_pipeline.py`
Expected: finishes in ~1–3 minutes; prints the funnel table (headline numbers to be captured in the stage report) and diagnostics. Sanity-check: `denominator` for view→cart should be the ~4.28 M "any view" sessions; cart→purchase denominator the ~0.99 M cart sessions.

- [ ] **Step 5: Verify and commit**

```bash
git add src/funnel.py scripts/etl_pipeline.py tests/test_funnel_rates.py
git commit -m "feat: add funnel rates with wilson CI and session parquet ETL"
```

---

### Task 3: Funnel chart (`src/plots.py::plot_funnel`)

**Files:**
- Create: `src/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `funnel_rates` output.
- Produces:
  - `plot_funnel(rates: pl.DataFrame, out_path: Path) -> None` — a single business-facing funnel figure: step bars with absolute session counts, conversion % labels between steps, and per-step Wilson CI error bars. Saves PNG (dpi 150) and closes the figure (no global state, no `plt.show`). Shared matplotlib style set once in `src/plots.py` (consistent font/color for the whole project).

- [ ] **Step 1: Write the failing test**

Call `plot_funnel` on the Task 2 synthetic rates into a `tmp_path`; assert the file exists, is non-empty and starts with the PNG magic bytes; and that `out_path`'s parent dirs are created if missing. Run; verify failure (no module).

- [ ] **Step 2: Implement `src/plots.py`**

Implement the module-level style setup, a `plot_funnel` that draws step bars + rates + CI error bars, and the filename-by-convention helper `assets_path(name: str) -> Path` (returns `assets/<name>.png`). Return `None`; always `plt.close()`.

- [ ] **Step 3: Run and verify**

Run `.venv/bin/python -m pytest tests/test_plots.py -v`. Then visually render once to a scratch path (`assets/funnel_check.png`) and open it in the notebook-free Q/A note in the stage report (a human sanity look is the point; no assertion on aesthetics in CI).

- [ ] **Step 4: Commit**

```bash
git add src/plots.py tests/test_plots.py
git commit -m "feat: add funnel visualization module"
```

---

### Task 4: Stage 02 report + learning note

**Files:**
- Create: `notes/02_funnel.md`

**Interfaces:**
- Consumes: headline funnel numbers from Task 2 Step 4.
- Produces: RU learning note (цель/почему/термины+англ/что сделано/выводы/подводные камни) covering: session-vs-user unit choice, why `view→purchase` ≠ product of step rates (jumpers), remove_from_cart semantics (gross vs net), Wilson CI vs simple Wald for small n.

- [ ] **Step 1: Write the note** (RU; structure inherited from `notes/00_scope_and_plan.md`).

- [ ] **Step 2: Commit**

```bash
git add notes/02_funnel.md
git commit -m "docs: add stage 02 learning note (RU)"
```

---

**Stage verification gate:** `pytest` green; `data/processed/sessions.parquet` exists (~4.54 M rows); funnel table printed and matches expectations; funnel chart rendered to `assets/` for human check. Report to the user: headline funnel rates + the two diagnostic shares.