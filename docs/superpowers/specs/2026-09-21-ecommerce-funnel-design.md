# E-commerce Funnel & Product Analytics — Design

Date: 2026-09-21
Status: Approved by the user (2026-09-21)

## 1. Purpose

Find the «bottleneck» step in the cosmetics-shop user path and propose concrete,
testable business improvements. The deliverable is a reproducible, statistically
honest funnel analysis suitable for a data-analyst resume (format mirrors the
reference project `dimassaa/marketing-ab-testing-analysis`).

## 2. Dataset

E-commerce Events History in Cosmetics Shop (Kaggle, mkechinov), 5 monthly CSVs
(Oct 2019 – Feb 2020), 20,692,840 events, 1,639,358 users, 4,535,941 sessions,
columns `event_time, event_type, product_id, category_id, category_code, brand,
price, user_id, user_session`.

Known data characteristics (verified by scan on 2026-09-21):
- event counts: view 9,657,821; cart 5,768,333; remove_from_cart 3,979,679;
  purchase 1,287,007.
- `category_code` missing in 98.3% of rows; `brand` missing in 42.3%.
- No device / channel / geo / user-agent columns — segmentation by user
  attributes is impossible; segmentation must be product- and time-based.
- 19% of purchase sessions have no `cart` event in the same session;
  20.0% of cart sessions have no `view` (197,350 / 985,781). Step order does not strictly hold
  within sessions → funnel defined via reached-step indicators.

## 3. Decisions (user-approved 2026-09-21)

1. Segmentation re-scoped to: price tier, visit index (new vs returning),
   time (weekday/weekend, hour dayparts, month as exploratory trend).
2. Funnel unit of analysis: user_session (primary) + user-level robustness.
3. Full 5 months processed with polars; no pilot month.
4. Deliverables: src/ + tests/ + stage notebooks + HTML reports + README EN/RU;
   no interactive dashboard.
5. Repository name `ecommerce-funnel-analysis` (push later, after completion).
6. Notebook execution fully verified end-to-end (errors=none per AGENTS.md).

## 4. Funnel definition (pre-registration target)

Per-session indicators: `view` = ≥1 view, `cart` = ≥1 cart, `purchase` = ≥1
purchase. Conversion rates (with Wilson 95% CI each):

- `view→cart` = P(cart | view)
- `cart→purchase` = P(purchase | cart)
- `view→purchase` = P(purchase | view)

Explicit diagnostics (never hidden): share of purchases without cart in the
same session, carts without view, and gross-vs-net cart (effect of
`remove_from_cart`).

## 5. Pipeline and stack

- Python 3.12, `uv` venv, pinned `requirements.txt`; minimal `pyproject.toml`
  for tooling config (pytest pythonpath). This mirrors the reference repo,
  which ships both files.
- polars for event→session aggregation; scipy (chi-square), statsmodels
  (z-test cross-validation), matplotlib + seaborn (charts).
- Data flow: raw `data/*.csv` (gitignored) → integrity check →
  `data/processed/sessions.parquet` (gitignored, reproducible) → funnel tables.
- All reusable logic as tested functions in `src/`; notebooks import from
  `src/` only.

## 6. Hypothesis testing

- Price tier: session price = median `price` of the session's events; tier
  bins derived from the global price distribution at the EDA stage and frozen
  in `reports/pre_registration.md` before any confirmatory test runs.

- Pre-registration committed before confirmatory runs
  (`reports/pre_registration.md`): exact segment sets, exact comparison list,
  α = 0.05, two-sided, per-family Bonferroni (+ FDR as sensitivity),
  sparse-cell rule (skip test when n·p < 5).
- Primary (confirmatory) comparisons:
  1. price tier × view→cart,
  2. visit index × cart→purchase,
  3. weekday/weekend × cart→purchase.
- Each primary family: chi-square test of independence, then pairwise
  two-proportion z-tests with correction.
- Separate statistical significance (p, CI) from practical significance
  (p.p. difference, relative uplift, business volume).

## 7. Business output

- Bottleneck = step with the largest absolute session loss; highlight the
  segment with the worst conversion at that step.
- Business hypotheses per bottleneck mapped to evidence.
- Proposed next-step A/B test: one concrete hypothesis, pre-registered
  parameters (α, power, MDE), required sample size computed with tested
  `src/` functions.

## 8. Deliverables / structure

```
├── src/            tested reusable functions (load, integrity, funnel, segments, inference)
├── tests/          pytest, incl. cross-validation vs scipy/statsmodels
├── notebooks/      stages 01–06, built from build_*.py (nbformat+nbclient), outputs embedded
├── reports/        pre_registration.md + executive-summary HTML
├── assets/         PNG figures referenced by README
├── notes/          Russian learning notes per stage
├── data/           raw CSVs (gitignored) + processed/ (gitignored)
├── docs/           internal process docs (spec, plans)
├── README.md       English, per The Ultimate README Guide
├── README_ru.md    Russian
├── requirements.txt pinned deps
├── .gitignore      AGENTS.md, guide, data, .venv, caches
└── plan.md         committed (project assignment)
```

Notebook stages (mirrors reference project):
`01_data_integrity`, `02_funnel`, `03_segmentation`,
`04_hypothesis_testing`, `05_business`, `06_robustness`.

## 9. Definition of done

- `git status` clean; `pytest` 100% green; all notebooks `errors=none`;
  every README figure exists in `assets/`; verdict consistent across
  README / notebook 05 / report; pre-registration commit precedes
  confirmatory results in git history; AGENTS.md/guide/CSVs stay gitignored.