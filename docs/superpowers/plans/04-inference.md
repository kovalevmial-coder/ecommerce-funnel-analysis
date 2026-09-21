# Stage 04: Inference (src)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and cross-validate the confirmatory statistics: Wilson CI, two-proportion z-test with pooled H0 + Wald CI, chi-square independence wrapper, multiple-comparison corrections, sparse-cell guard, and closed-form sample-size/power for the proposed next-step A/B test.

**Architecture:** All statistical functions live in `src/inference.py`; every formula is checked against `statsmodels`/`scipy` with honest relative tolerances. These functions are called by the confirmatory notebook (04) and the business notebook (05).

**Tech Stack:** NumPy, SciPy, statsmodels (reference), pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`

---

### Task 1: Wilson CI (`src/inference.py::wilson_ci`)

**Files:**
- Create: `src/inference.py`
- Modify: `src/funnel.py` (funnel_rates now calls `inference.wilson_ci` instead of the inline formula)
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]` — Wilson (score) interval lower/upper. Guard: if `n <= 0` raise `ValueError`; if not `0 <= success <= n` raise `ValueError`.

- [ ] **Step 1: Write the failing test**

Known values: (success=0, n=100) → CI ≈ (0, 0.0362); (100,100) → ≈ (0.9638, 1); (50,100) → ≈ (0.4038, 0.5962). Use `pytest.approx(rel=1e-6)`. Also negative path: `n=0` raises. Run; verify failure.

- [ ] **Step 2: Implement `wilson_ci`**

Closed-form Wilson score formula. Docstring: purpose, params, return, exceptions, why Wilson over Wald for small n (documented in docstring).

- [ ] **Step 3: Refactor `src/funnel.py`**

Replace the inline Wilson formula in `funnel_rates` with a call to `inference.wilson_ci`, removing the Stage-02 `TODO(stage-04)` marker. Keep all funnel tests green.

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/ -q`. Additionally in a throwaway REPL command: compare `wilson_ci` against `statsmodels.stats.proportion_confint(..., method="wilson")` on 5 random (n, k) pairs, print differences — expected < 1e-6. Record the check in the stage report.

- [ ] **Step 5: Commit**

```bash
git add src/inference.py src/funnel.py tests/test_inference.py
git commit -m "feat: add wilson CI and dedupe funnel confidence intervals"
```

---

### Task 2: Two-proportion z-test (`src/inference.py::two_prop_ztest`)

**Files:**
- Modify: `src/inference.py`
- Test: `tests/test_inference.py` (extend)

**Interfaces:**
- Consumes: `wilson_ci` (not required — CI here is the Wald difference interval, but keep `wilson_ci` export for reuse).
- Produces:
  - `two_prop_ztest(a_events: int, a_n: int, b_events: int, b_n: int) -> dict` with keys:
    `p_a, p_b, diff (p_a − p_b), z, p_value (two-sided), ci_low, ci_high` (Wald CI of the difference, z-scale = 1.96), `pooled` (pooled H0 rate), `sparse: bool`, `n_ok: bool`.
    Sparse rule: `sparse=True` when `min(n_a·p_a, n_a·(1−p_a), n_b·p_b, n_b·(1−p_b)) < 5`; when sparse, `z = p_value = ci_low = ci_high = None` and `diff_rate` still reported (descriptive). Raises `ValueError` if `a_n<=0 or b_n<=0`.

- [ ] **Step 1: Write the failing test**

Fixture: a=50/1000, b=40/1000. Assert: computed p_a/p_b; pooled = (50+40)/2000; z/p_value match `statsmodels.stats.proportion.proportions_ztest([50,40],[1000,1000], alternative="two-sided")` with `rel=1e-6`; Wald CI matches `statsmodels.stats.proportion.confint_proportions_2indep(..., method="wald")` (rel=1e-3, since CI parametrization differs slightly). Sparse path: a_events=2, a_n=100 → `sparse=True` and z/p/CI None. Run; verify failure.

- [ ] **Step 2: Implement `two_prop_ztest`**

Pooled-variance z under H0, Wald difference CI. Docstring documents formula and the parametrization note for statsmodels CI comparisons. Sparse branch returns None-values as specified.

- [ ] **Step 3: Verify**

Run `.venv/bin/python -m pytest tests/ -q`. Confirm the z/p cross-check is tight (1e-6), the Wald CI cross-check within stated rel tolerance. Record both in stage report.

- [ ] **Step 4: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: add pooled two-proportion z-test with wald CI and sparse guard"
```

---

### Task 3: Chi-square + corrections (`src/inference.py::chi2_independence`, `bonferroni`, `fdr_bh`)

**Files:**
- Modify: `src/inference.py`
- Test: `tests/test_inference.py` (extend)

**Interfaces:**
- Consumes: SciPy.
- Produces:
  - `chi2_independence(table: object) -> dict` — accepts a 2·k or k·k contingency (list of lists / ndarray / polars frame). Keys: `chi2, dof, p_value, expected_min`, `valid: bool` (`valid=False` when `expected_min < 5`). Built on `scipy.stats.chi2_contingency`.
  - `bonferroni(p_values: Sequence[float]) -> list[float]` — min(p·k, 1) per value.
  - `fdr_bh(p_values: Sequence[float]) -> list[float]` — Benjamini–Hochberg corrected values.

- [ ] **Step 1: Write the failing test**

Hand-computed 2×2 → known scipy values (assert equality with `scipy.stats.chi2_contingency` on the same table); `bonferroni([0.01,0.02,0.03])` → `[0.03,0.06,0.09]`; `fdr_bh` matches `statsmodels.stats.multitest.multipletests(method="fdr_bh")` (rel=1e-6). Sparse: a 2×2 with expected cell < 5 → `valid=False`. Run; verify failure.

- [ ] **Step 2: Implement all three**

Chi-square wrapper normalizing input to a NumPy array; return the expected-min check; FDR via rank-sorted B–H; Bonferroni NaN-safe.

- [ ] **Step 3: Verify**

Run full suite. Record cross-checks in stage report.

- [ ] **Step 4: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: add chi-square wrapper and multiple-comparison corrections"
```

---

### Task 4: Sample size and power (`src/inference.py::required_n_per_group`, `achieved_power`)

**Files:**
- Modify: `src/inference.py`
- Test: `tests/test_inference.py` (extend)

**Interfaces:**
- Consumes: NumPy (normal quantiles).
- Produces:
  - `required_n_per_group(base_rate: float, mde: float, alpha: float, power: float) -> float` — closed-form two-proportion sample size (one-sided/H1 direction from `mde` sign; two-sided at `alpha`), returns n per group.
  - `achieved_power(base_rate: float, mde: float, n: int, alpha: float) -> float` — power at given n.

- [ ] **Step 1: Write the failing test**

Cross-check `required_n_per_group(p0=0.02, mde=0.005, alpha=0.05, power=0.8)` against `statsmodels.stats.power.NormalIndPower().solve_power(effect_size=proportion_effectsize(0.02, 0.025), alpha=0.05, power=0.8, ratio=1.0)` with `rel=0.01` (a documented tolerance: the closed form and the statsmodels normal-approx differ on small-effect sizing). Sanity: larger `mde` → smaller n; n positive and finite. Run; verify failure.

- [ ] **Step 2: Implement both**

Closed-form per-group size formula; power via the normal CDF difference. Docstrings document the assumptions (normal approximation, ratio 1:1, continuous-correction not applied) and the tolerance rationale.

- [ ] **Step 3: Verify**

Run full suite green. Record the cross-check tolerance in the stage report.

- [ ] **Step 4: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: add closed-form sample size and power for proportion tests"
```

---

### Task 5: Stage 04 report + learning note

**Files:**
- Create: `notes/04_hypothesis_testing.md`

**Interfaces:**
- Consumes: cross-check results from Tasks 1–4.
- Produces: RU note covering: Type I/II errors, alpha/power/MDE, pooled vs unpooled variance, Wald vs Wilson, sparse cells (n·p<5), Bonferroni vs BH, why statsmodels cross-validation + rel tolerance (formula parametrization differences are not bugs).

- [ ] **Step 1: Write the note.**

- [ ] **Step 2: Commit**

```bash
git add notes/04_hypothesis_testing.md
git commit -m "docs: add stage 04 learning note (RU)"
```

---

**Stage verification gate:** `pytest` green; cross-validation numbers recorded for Wilson (1e-6), z-test (1e-6), Wald CI (1e-3), chi-square (identity), BH (1e-6), sample size (rel 0.01). Report to user: the cross-check evidence table + any mismatch found.