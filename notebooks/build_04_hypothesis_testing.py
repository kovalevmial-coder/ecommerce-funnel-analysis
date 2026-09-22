"""Build and execute notebook 04: pre-registered confirmatory hypothesis tests.

Driven by the Stage 06 plan (Task 3): the notebook executes EXACTLY the
comparison list pinned in reports/pre_registration.md (commit 35317bc) —
family A: one 4x2 chi-square (price tier x view->cart on view sessions) plus
the six pairwise two-proportion z-tests with Bonferroni as the rejection rule
(BH-FDR sensitivity only); family B: one z-test (visit kind x cart->purchase);
family C: one z-test (weekend x cart->purchase). No test outside that list is
invoked — the pre-registration boundary (§6) forbids post-hoc additions.

All statistics come from src.inference and segment labels from src.segments;
this script only declares cells and hands them to scripts.build_notebook, so
the executed .ipynb is reproducible and committed with its outputs. The
family-A forest plot is written to assets/confirm_price_tier_forest.png via
src.plots.assets_path. Outcome numbers are rendered dynamically from the test
results — this build script hard-codes no p-values or verdicts, so frozen data
cannot leave stale prose behind a future rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The build script runs as `python notebooks/build_04_hypothesis_testing.py`,
# so the first sys.path entry is notebooks/ and the repo root is NOT importable;
# insert it explicitly (same pattern as build_02/build_03) before any
# scripts.* import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_notebook import build_and_execute  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --- Cell sources -----------------------------------------------------------
# Cells are declared here as plain strings and handed to the runner; the runner
# converts each to nbformat v4 cells and executes them against the pinned
# `ecommerce-funnel` kernel, failing loudly on any cell error.

# 1. Restate the decision rule BEFORE any result exists (pre-registration
# discipline): the reader must know α, corrections, sparse rule and the
# practical gate before seeing a single p-value. Cites the pinning commit.
_C1_RULE = """\
## Pre-registered decision rule (fixed before this notebook ran)

This notebook executes **only** the confirmatory comparison list of
`reports/pre_registration.md` (pinned **2026-09-22**, commit **`35317bc`**,
before any confirmatory output existed). Nothing below may be tuned after the
fact; any departure would be an explicitly labelled post-hoc exploration
(§6 of the pre-registration), and **no deviation is recorded for this run**.

| Rule | Value (pre-reg §4–§5) |
|---|---|
| Significance level | **α = 0.05, two-sided** for every test |
| Family A multiplicity | 6 pairwise z-tests → **Bonferroni** α\\* = 0.05/6 ≈ 0.0083 (rejection rule) |
| Sensitivity | **BH-FDR** reported for the same 6 pairs — sensitivity only, does **not** set rejection |
| Sparse rule | any cell with expected count < 5 → **no test**, `sparse=True`, rates reported descriptively |
| Practical gate (decision-relevant) | **all three**: 95% CI of the difference excludes 0, **and** \\|relative uplift\\| ≥ 10% of the lower rate, **and** affected volume ≥ 1% of sessions (≥ 45 360 of 4 535 941) |
| Affected-volume operand | for a pair: size of the **smaller** of the two compared groups (§5 gap note) |

**Two kinds of significance are kept separate** (golden standard): a test may
be *statistically* significant (p below its corrected threshold) yet *not*
decision-relevant under the practical gate — the verdict column below reports
the practical answer, never substitutes a bare p-value for it.

**By-design consequence (§5):** luxury holds 42 438 sessions = **0.94 %** of
all sessions — below the 1 % affected-volume bar — so in **every** comparison
it participates in it is **never decision-relevant**, even when statistically
confirmed. This notebook states that fact; it is not a surprise discovered
after the fact.

**Complete pre-registered inventory (nothing else runs):**

1. **Family A** — price tier × view→cart, on **view sessions** only
   (`has_view`), "NA" tier excluded with an explicit count:
   1 omnibus χ² (4×2) + **6** pairwise two-proportion z-tests.
2. **Family B** — visit kind × cart→purchase, on **cart sessions**: 1 z-test.
3. **Family C** — weekend/weekday × cart→purchase, on **cart sessions**: 1 z-test.

Overall funnel figures in notebooks 01/02 stay **descriptive** (pre-reg §1):
they carry no significance claim regardless of what this notebook finds.
"""

# 2. Setup: resolve the repo root from the kernel's CWD (path-independent
# re-execution), import analysis from src only, pin the pre-registered
# constants once so every cell reads the same numbers.
_C2_SETUP = """\
import math
import os
import sys
from itertools import combinations
from pathlib import Path


def _find_repo_root() -> Path:
    \"\"\"Resolve the repo root from the kernel's CWD (cwd-independent cell).

    The build runner launches the kernel from the repo root, but a reader who
    re-executes the .ipynb from another directory must resolve the same data.
    Walk up until a directory contains both src/ and the sessions artifact; the
    is_file() guard below fails loudly instead of silently loading wrong data.
    \"\"\"
    probe = Path(os.getcwd()).resolve()
    for _ in range(8):
        if (probe / "src").is_dir() and (
            probe / "data" / "processed" / "sessions.parquet"
        ).is_file():
            return probe
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    raise RuntimeError(
        "repo root not found from CWD="
        + str(Path(os.getcwd()))
        + "; expected a dir with src/ and data/processed/sessions.parquet"
    )


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

SESSIONS_FILE = REPO_ROOT / "data" / "processed" / "sessions.parquet"

# Fail loudly if the processed artifact is missing — an empty session table
# would otherwise sail through and silently change every downstream number.
if not SESSIONS_FILE.is_file():
    raise FileNotFoundError(f"missing processed artifact: {SESSIONS_FILE}")

# Analysis lives in src only — the notebook renders, it does not re-implement
# test maths. src.plots import applies the shared matplotlib style, and the Agg
# backend set there is exactly what the headless forest-plot save needs.
import polars as pl  # noqa: E402
from IPython.display import Image, Markdown, display  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from src.inference import (  # noqa: E402
    bonferroni,
    chi2_independence,
    fdr_bh,
    two_prop_ztest,
    wilson_ci,
)
from src.segments import assign_segments  # noqa: E402
import src.plots  # noqa: E402,F401  (applies project style, sets Agg backend)
from src.plots import assets_path  # noqa: E402

# Pre-registered constants (reports/pre_registration.md §4–§5, commit 35317bc).
# Single assignment: every threshold check below reads these, not a re-typed
# literal that could drift between cells.
ALPHA = 0.05                 # two-sided, every test
ALPHA_STAR = ALPHA / 6       # Bonferroni for the 6 family-A pairwise tests
VOLUME_MIN = 45_360          # 1% of 4,535,941 sessions (§5)
PREREG_SHA = "35317bc"       # pinning commit of reports/pre_registration.md
N_PAIRWISE = 6               # (4 choose 2) — structural inventory of family A

print(f"repo root:    {REPO_ROOT}")
print(f"sessions:     {SESSIONS_FILE}")
print(f"pre-reg SHA:  {PREREG_SHA}  |  alpha={ALPHA}  |  alpha*={ALPHA_STAR:.6f}")
"""

# 3. Load: one row per user_session; assign_segments is the single owner of
# price_tier / visit_kind / is_weekend — the notebook never derives a label.
_C3_LOAD = """\
# One row per user_session, the frozen unit of the whole project.
sessions = pl.read_parquet(SESSIONS_FILE)
seg = assign_segments(sessions)
del sessions  # keep only the labelled frame; the raw frame is not needed again

print(f"loaded {seg.height:,} session rows")
print(f"view sessions:   {int(seg['has_view'].sum()):,}")
print(f"cart sessions:   {int(seg['has_cart'].sum()):,}")
"""

# 4. Family A input plane + omnibus chi-square. Rates with Wilson CIs are the
# descriptive context the reader needs before the omnibus; the chi-square is
# guarded by its pre-registered validity flag (expected_min >= 5).
_C4_FAMILY_A_CHI2 = """\
# Family A plane (pre-reg §3): view sessions only; the "NA" price tier
# (null median_price) is excluded with an explicit count — pre-reg requires
# it never silently disappears from the comparison.
view = seg.filter(pl.col("has_view") & (pl.col("price_tier") != "NA"))
n_na_view = int(
    seg.filter(pl.col("has_view") & (pl.col("price_tier") == "NA")).height
)

# Merchandising tier order, fixed in pre-reg §2 (not sorted by observed rate —
# ordering the table by outcome would be silent post-hoc styling).
TIERS = ["budget", "mid", "premium", "luxury"]

# Per-tier view->cart counts. Inside the has_view filter, has_cart marks a
# view session that also reached cart — the view->cart numerator of §1.
# wilson_ci reports sampling uncertainty of each observed rate (descriptive
# context around the omnibus test, not a second hypothesis).
tier_stats: dict[str, dict] = {}
for _t in TIERS:
    _sub = view.filter(pl.col("price_tier") == _t)
    _cart = int(_sub["has_cart"].sum())
    _n = _sub.height
    if _n == 0:
        # Empty denominator makes the rate undefined; fail fast instead of
        # dividing by zero and printing a plausible-looking nan.
        raise ValueError(f"tier {_t!r} has no view sessions; cannot test")
    _lo, _hi = wilson_ci(_cart, _n)
    tier_stats[_t] = {
        "cart": _cart, "n": _n, "rate": _cart / _n,
        "ci_low": _lo, "ci_high": _hi,
    }

# 4x2 contingency: rows = tiers (pre-reg order), cols = [reached cart, not].
contingency = [
    [tier_stats[t]["cart"], tier_stats[t]["n"] - tier_stats[t]["cart"]]
    for t in TIERS
]
chi2_res = chi2_independence(contingency)

# Structural fidelity check: dof = (rows-1)*(cols-1) = 3 iff the omnibus
# really spans the four pre-registered tiers — a wrong-shaped table cannot
# silently masquerade as the pre-registered test.
assert chi2_res["dof"] == 3, f"expected dof=3 for the 4x2 table, got {chi2_res['dof']}"

_lines = [
    f"**Family A — price tier × view→cart** "
    # f-string quotes around "NA" must survive into the generated cell:
    # the outer build string consumes one backslash level, so the cell
    # receives \"NA\" (escaped) rather than a string-closing quote.
    f"(view sessions only; \\"NA\\" tier excluded: {n_na_view:,} view sessions).",
    "",
    "| tier | view sessions | cart sessions | view→cart rate | 95% Wilson CI |",
    "|---|---:|---:|---:|---:|",
]
for _t in TIERS:
    _s = tier_stats[_t]
    _lines.append(
        f"| {_t} | {_s['n']:,} | {_s['cart']:,} | {_s['rate']:.4%} "
        f"| [{_s['ci_low']:.4%}, {_s['ci_high']:.4%}] |"
    )
_lines += [
    "",
    f"**Omnibus χ² (4×2):** χ² = {chi2_res['chi2']:.4f}, "
    f"dof = {chi2_res['dof']}, p = {chi2_res['p_value']:.4e}, "
    f"expected_min = {chi2_res['expected_min']:.1f}, "
    f"valid (expected_min ≥ 5) = {chi2_res['valid']}.",
    "",
]
if not chi2_res["valid"]:
    # Pre-reg §4 sparse rule applied to the omnibus: no significance claim,
    # rates stay descriptive. (Not the case in this dataset; the branch exists
    # so the notebook cannot over-claim if data ever changes.)
    _lines.append(
        "**SPARSE:** expected_min < 5 → per pre-reg §4 **no omnibus test**; "
        "tier rates above are reported descriptively only."
    )
elif chi2_res["p_value"] < ALPHA:
    _lines.append(
        f"H₀ (equal view→cart across all four tiers) is **rejected** at "
        f"α = {ALPHA}: at least one tier differs. The six pairwise z-tests "
        f"below localize the difference under Bonferroni α* = "
        f"{ALPHA_STAR:.4f}."
    )
else:
    _lines.append(
        f"H₀ is **not rejected** at α = {ALPHA}. The six pairwise z-tests "
        f"still run exactly as pre-registered (§3) — the inventory is fixed "
        f"before outcomes, not gated on the omnibus result."
    )
display(Markdown("\\n".join(_lines)))
"""

# 5. Family A pairwise: exactly 6 z-tests in fixed tier-pair order; Bonferroni
# is the rejection rule, BH-FDR the sensitivity column (pre-reg §4).
_C5_FAMILY_A_PAIRWISE = """\
# Six pairwise two-proportion z-tests (4 choose 2) on the same view->cart
# plane, pairs generated in the fixed tier order of pre-reg §2. An assertion
# pins the inventory count so a future edit cannot silently add a seventh test.
pairs = list(combinations(TIERS, 2))
pair_results = []
for _a, _b in pairs:
    _res = two_prop_ztest(
        tier_stats[_a]["cart"], tier_stats[_a]["n"],
        tier_stats[_b]["cart"], tier_stats[_b]["n"],
    )
    pair_results.append({"pair": f"{_a} - {_b}", "a": _a, "b": _b, **_res})

assert len(pair_results) == N_PAIRWISE, (
    f"pre-reg requires exactly {N_PAIRWISE} pairwise tests, got {len(pair_results)}"
)

# NaN keeps a sparse slot inside the correction family (bonferroni/fdr_bh
# document NaN passthrough: the slot consumed a test but has no p to adjust).
raw_ps = [
    r["p_value"] if r["p_value"] is not None else math.nan for r in pair_results
]
p_bonf = bonferroni(raw_ps)
p_fdr = fdr_bh(raw_ps)
for _r, _pb, _pf in zip(pair_results, p_bonf, p_fdr):
    _r["p_bonf"] = _pb
    _r["p_fdr"] = _pf
    # Rejection rule = Bonferroni-adjusted p < α (equivalently raw p < α*).
    _r["passes_bonf"] = (not _r["sparse"]) and _pb < ALPHA


def _fmt_p(p: float | None) -> str:
    \"\"\"Format a p-value for markdown; None/NaN become an em-dash cell.\"\"\"
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.3e}"


_lines = [
    "**Family A — pairwise two-proportion z-tests** "
    "(H₀: p_i = p_j, two-sided; diff = p_i − p_j).",
    "",
    "| pair | p_a | p_b | diff (p.p.) | 95% CI (p.p.) | raw p | "
    "Bonferroni p | BH-FDR q | passes α* | sparse |",
    "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
]
for _r in pair_results:
    if _r["sparse"]:
        _lines.append(
            f"| {_r['pair']} | {_r['p_a']:.4%} | {_r['p_b']:.4%} | "
            f"{_r['diff'] * 100:+.2f} | — | — | — | — | no | True |"
        )
    else:
        _lines.append(
            f"| {_r['pair']} | {_r['p_a']:.4%} | {_r['p_b']:.4%} | "
            f"{_r['diff'] * 100:+.2f} | "
            f"[{_r['ci_low'] * 100:+.2f}, {_r['ci_high'] * 100:+.2f}] | "
            f"{_fmt_p(_r['p_value'])} | {_fmt_p(_r['p_bonf'])} | "
            f"{_fmt_p(_r['p_fdr'])} | "
            f"{'yes' if _r['passes_bonf'] else 'no'} | {_r['sparse']} |"
        )
_n_pass = sum(1 for _r in pair_results if _r["passes_bonf"])
_lines += [
    "",
    f"**{ _n_pass } of { N_PAIRWISE }** pairwise tests pass the Bonferroni "
    f"threshold α* = {ALPHA_STAR:.4f}. BH-FDR is shown as sensitivity only "
    f"and does not change any rejection (pre-reg §4).",
]
display(Markdown("\\n".join(_lines)))
"""

# 6. Forest plot: the one chart this notebook owns — each pairwise difference
# with its 95% Wald CI against the zero line, colour = Bonferroni status.
# Answers exactly one question: which tier gaps are established, and how big.
_C6_FOREST = """\
# Forest plot of the six family-A differences (percentage points). Sparse pairs
# would have no CI to draw and are skipped with a count note — honest omission
# beats a fabricated whisker. Colours encode the corrected rejection rule so
# the chart cannot visually over-claim a non-significant gap.
plot_rows = [r for r in pair_results if not r["sparse"]]
n_sparse_skipped = len(pair_results) - len(plot_rows)

fig, ax = plt.subplots(figsize=(8.0, 4.8))
for _yi, _r in enumerate(plot_rows):
    _d = _r["diff"] * 100
    _lo = _r["ci_low"] * 100
    _hi = _r["ci_high"] * 100
    _color = "#55A868" if _r["passes_bonf"] else "#4C72B0"
    ax.errorbar(
        _d, _yi,
        xerr=[[_d - _lo], [_hi - _d]],
        fmt="o", color=_color, ecolor="#333333",
        capsize=3, elinewidth=1, markersize=6, zorder=4,
    )

# Zero line: a CI crossing it is not statistically established at the
# uncorrected level either — the reference the whole chart is read against.
ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=2)

ax.set_yticks(range(len(plot_rows)))
ax.set_yticklabels([r["pair"] for r in plot_rows])
ax.invert_yaxis()  # first pre-registered pair reads at the top
ax.set_xlabel("Difference in view→cart rate (p.p., first − second tier)")
ax.set_title("Family A: pairwise price-tier differences (95% Wald CI)")
ax.set_ylim(len(plot_rows) - 0.5, -0.5)  # headroom after inversion

# Proxy artists so both colours are explainable in the legend.
_proxy_sig = plt.Line2D(
    [0], [0], marker="o", linestyle="none", color="#55A868",
    markersize=6, label="passes Bonferroni α*",
)
_proxy_ns = plt.Line2D(
    [0], [0], marker="o", linestyle="none", color="#4C72B0",
    markersize=6, label="not significant after Bonferroni",
)
ax.legend(handles=[_proxy_sig, _proxy_ns], loc="lower right")

_forest_out = REPO_ROOT / assets_path("confirm_price_tier_forest")
out_path = Path(_forest_out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
plt.close(fig)
display(Image(filename=str(out_path)))
print(f"saved {out_path}")
if n_sparse_skipped:
    print(f"note: {n_sparse_skipped} sparse pair(s) skipped (no CI to draw)")
"""

# 7. Family B: exactly one z-test, direction diff = p_new − p_returning.
_C7_FAMILY_B = """\
# Family B (pre-reg §3): visit kind × cart->purchase on cart sessions only.
# Direction convention is pre-registered: diff = p_new − p_returning, so a
# positive diff means new visitors convert cart→purchase better than returning.
cart = seg.filter(pl.col("has_cart"))
new_cart = cart.filter(pl.col("visit_kind") == "new")
ret_cart = cart.filter(pl.col("visit_kind") == "returning")

fam_b = two_prop_ztest(
    int(new_cart["has_purchase"].sum()), new_cart.height,
    int(ret_cart["has_purchase"].sum()), ret_cart.height,
)
# Single test: α = 0.05, no multiplicity correction applies (pre-reg §4).
fam_b_passes = (
    (not fam_b["sparse"])
    and fam_b["p_value"] is not None
    and fam_b["p_value"] < ALPHA
)
# Affected volume = smaller compared group (§5 gap note): the cart sessions a
# hypothetical action would lift.
fam_b_affected = min(new_cart.height, ret_cart.height)

_lines = [
    "**Family B — visit kind × cart→purchase** (cart sessions; "
    "diff = p_new − p_returning).",
    "",
    "| group | cart sessions | purchase sessions | cart→purchase | 95% Wilson CI |",
    "|---|---:|---:|---:|---:|",
]
for _label, _n, _ev in (
    ("new", new_cart.height, int(new_cart["has_purchase"].sum())),
    ("returning", ret_cart.height, int(ret_cart["has_purchase"].sum())),
):
    _lo, _hi = wilson_ci(_ev, _n)
    _lines.append(
        f"| {_label} | {_n:,} | {_ev:,} | {_ev / _n:.4%} | [{_lo:.4%}, {_hi:.4%}] |"
    )
_lines += ["", f"**z-test result:**"]
if fam_b["sparse"]:
    _lines.append(
        "**SPARSE** (min expected count < 5) → no test per pre-reg §4; "
        "rates above are descriptive only."
    )
else:
    _lines += [
        f"z = {fam_b['z']:.4f}, p = {fam_b['p_value']:.4e}, "
        f"diff = {fam_b['diff'] * 100:+.4f} p.p., "
        f"95% CI [{fam_b['ci_low'] * 100:+.4f}, {fam_b['ci_high'] * 100:+.4f}] p.p. "
        f"— passes α = {ALPHA}: **{'yes' if fam_b_passes else 'no'}**."
    ]
display(Markdown("\\n".join(_lines)))
"""

# 8. Family C: exactly one z-test, direction diff = p_weekend − p_weekday.
_C8_FAMILY_C = """\
# Family C (pre-reg §3): weekend × cart->purchase on cart sessions only.
# Direction: diff = p_weekend − p_weekday; positive = a positive "weekend effect".
we_cart = cart.filter(pl.col("is_weekend"))
wk_cart = cart.filter(pl.col("is_weekend").not_())

fam_c = two_prop_ztest(
    int(we_cart["has_purchase"].sum()), we_cart.height,
    int(wk_cart["has_purchase"].sum()), wk_cart.height,
)
fam_c_passes = (
    (not fam_c["sparse"])
    and fam_c["p_value"] is not None
    and fam_c["p_value"] < ALPHA
)
fam_c_affected = min(we_cart.height, wk_cart.height)

_lines = [
    "**Family C — weekend/weekday × cart→purchase** (cart sessions; "
    "diff = p_weekend − p_weekday).",
    "",
    "| group | cart sessions | purchase sessions | cart→purchase | 95% Wilson CI |",
    "|---|---:|---:|---:|---:|",
]
for _label, _n, _ev in (
    ("weekend", we_cart.height, int(we_cart["has_purchase"].sum())),
    ("weekday", wk_cart.height, int(wk_cart["has_purchase"].sum())),
):
    _lo, _hi = wilson_ci(_ev, _n)
    _lines.append(
        f"| {_label} | {_n:,} | {_ev:,} | {_ev / _n:.4%} | [{_lo:.4%}, {_hi:.4%}] |"
    )
_lines += ["", f"**z-test result:**"]
if fam_c["sparse"]:
    _lines.append(
        "**SPARSE** (min expected count < 5) → no test per pre-reg §4; "
        "rates above are descriptive only."
    )
else:
    _lines += [
        f"z = {fam_c['z']:.4f}, p = {fam_c['p_value']:.4e}, "
        f"diff = {fam_c['diff'] * 100:+.4f} p.p., "
        f"95% CI [{fam_c['ci_low'] * 100:+.4f}, {fam_c['ci_high'] * 100:+.4f}] p.p. "
        f"— passes α = {ALPHA}: **{'yes' if fam_c_passes else 'no'}**."
    ]
display(Markdown("\\n".join(_lines)))
"""

# 9. Practical-significance table: every pre-registered z-test (all eight —
# non-rejections stay visible; silence would read as omission in a confirmatory
# notebook) with the three-part §5 gate and a YES/NO/SPARSE verdict column.
_C9_PRACTICAL = """\
# Practical-significance gate (pre-reg §5). Decision-relevant (YES) requires
# ALL of: corrected threshold passed AND 95% CI excludes 0 AND |relative
# uplift| >= 10% of the lower rate AND affected volume >= 45,360 sessions.
# SPARSE short-circuits everything: a sparse test was never run (§4).


def _rel_uplift(p_a: float, p_b: float, diff: float) -> float:
    \"\"\"|diff| relative to the lower (baseline) rate — §5 condition 2.\"\"\"
    lower = min(p_a, p_b)
    return abs(diff) / lower if lower > 0 else float("inf")


def _gate_verdict(res: dict, passes_threshold: bool, affected_n: int) -> str:
    \"\"\"Apply the pre-registered decision gate to one z-test result.

    Args:
        res: a two_prop_ztest result dict.
        passes_threshold: True iff the test cleared its corrected threshold
            (Bonferroni for family A; plain α for families B and C).
        affected_n: smaller compared group size (§5 gap-note operand).

    Returns:
        "SPARSE" | "YES" | "NO" — never a bare p-value dressed as a verdict.
    \"\"\"
    if res["sparse"]:
        return "SPARSE"
    if not passes_threshold:
        return "NO"
    ci_excludes_zero = res["ci_low"] > 0 or res["ci_high"] < 0
    uplift_ok = _rel_uplift(res["p_a"], res["p_b"], res["diff"]) >= 0.10
    volume_ok = affected_n >= VOLUME_MIN
    return "YES" if (ci_excludes_zero and uplift_ok and volume_ok) else "NO"


# One row per pre-registered z-test: 6 family-A pairs + B + C = 8 (asserted).
practical_rows: list[dict] = []
for _r in pair_results:
    _affected = min(tier_stats[_r["a"]]["n"], tier_stats[_r["b"]]["n"])
    practical_rows.append({
        "test": f"A: {_r['pair']} (view→cart)",
        "res": _r,
        "corr_label": f"Bonf α*={ALPHA_STAR:.4f}",
        "passes": _r["passes_bonf"],
        "p_corr": _r["p_bonf"],
        "affected": _affected,
        "verdict": _gate_verdict(_r, _r["passes_bonf"], _affected),
    })

practical_rows.append({
    "test": "B: new − returning (cart→purchase)",
    "res": fam_b,
    "corr_label": f"α={ALPHA} (single test)",
    "passes": fam_b_passes,
    "p_corr": fam_b["p_value"],
    "affected": fam_b_affected,
    "verdict": _gate_verdict(fam_b, fam_b_passes, fam_b_affected),
})
practical_rows.append({
    "test": "C: weekend − weekday (cart→purchase)",
    "res": fam_c,
    "corr_label": f"α={ALPHA} (single test)",
    "passes": fam_c_passes,
    "p_corr": fam_c["p_value"],
    "affected": fam_c_affected,
    "verdict": _gate_verdict(fam_c, fam_c_passes, fam_c_affected),
})

assert len(practical_rows) == 8, (
    f"pre-reg inventory is 8 z-tests (6+1+1), got {len(practical_rows)}"
)


def _fmt_p_cell(p: float | None) -> str:
    \"\"\"P-value cell for the practical table; None/NaN become an em dash.\"\"\"
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.3e}"


_lines = [
    "**Practical-significance table** — every pre-registered z-test "
    "(non-rejections included so nothing is silently dropped). "
    f"Omnibus χ² is the family-A gate reported above; the §5 practical rule "
    f"applies to tests that carry a CI of a difference.",
    "",
    "| test | raw p | decision p (rule) | passes threshold | diff (p.p.) | "
    "|rel. uplift| | affected n | **verdict** |",
    "|---|---:|---:|:---:|---:|---:|---:|:---:|",
]
for _row in practical_rows:
    _res = _row["res"]
    if _res["sparse"]:
        _lines.append(
            f"| {_row['test']} | — | — | no | — | — | "
            f"{_row['affected']:,} | **SPARSE** |"
        )
        continue
    _rel = _rel_uplift(_res["p_a"], _res["p_b"], _res["diff"])
    _lines.append(
        f"| {_row['test']} | {_fmt_p_cell(_res['p_value'])} | "
        f"{_fmt_p_cell(_row['p_corr'])} ({_row['corr_label']}) | "
        f"{'yes' if _row['passes'] else 'no'} | "
        f"{_res['diff'] * 100:+.4f} | {_rel:.2%} | "
        f"{_row['affected']:,} | **{_row['verdict']}** |"
    )

_n_yes = sum(1 for _row in practical_rows if _row["verdict"] == "YES")
_n_stat = sum(1 for _row in practical_rows if _row["passes"])
_lines += [
    "",
    f"**{_n_stat} of 8** tests pass their corrected threshold; "
    f"**{_n_yes} of 8** are decision-relevant (**YES**) under the full "
    f"three-part gate. Statistically significant ≠ actionable — that gap is "
    f"the point of this table (golden standard: two kinds of significance).",
]
display(Markdown("\\n".join(_lines)))
"""

# 10. Verdict summary: dynamic (built from the result objects — no hard-coded
# outcome prose) covering statistical AND practical outcomes, the by-design
# luxury consequence, and the descriptive-funnel boundary.
_C10_VERDICT = """\
# Verdict summary assembled from the result objects above, so the prose can
# never disagree with the executed tests (single source of truth = this kernel
# state). Static contractual statements (boundaries, luxury-by-design, SHA)
# are included here so the reader gets one complete closing block.
_yes_rows = [row for row in practical_rows if row["verdict"] == "YES"]
_stat_rows = [row for row in practical_rows if row["passes"]]

_yes_md = (
    "\\n".join(f"- **{row['test']}** — diff {row['res']['diff'] * 100:+.4f} p.p., "
              f"rel. uplift {_rel_uplift(row['res']['p_a'], row['res']['p_b'], row['res']['diff']):.1%}, "
              f"affected {row['affected']:,}"
              for row in _yes_rows)
    if _yes_rows
    else "- **none** — no comparison clears all three practical conditions."
)

_md = f\"\"\"\\
## Verdict summary (pre-registered decision rule, commit `{PREREG_SHA}`)

**Family A (price tier × view→cart).**
Omnibus χ²(4×2): p = {chi2_res['p_value']:.4e} — H₀
{"rejected" if chi2_res["valid"] and chi2_res["p_value"] < ALPHA else "not rejected"}
at α = {ALPHA}{" (test valid)" if chi2_res["valid"] else " (SPARSE — descriptive only)"}.
Pairwise (Bonferroni α* = {ALPHA_STAR:.4f}): **{sum(1 for r in pair_results if r['passes_bonf'])} of {N_PAIRWISE}**
pass. BH-FDR (sensitivity) agrees or is reported alongside without setting
rejection.

**Family B (visit kind × cart→purchase).**
p = {fam_b['p_value'] if fam_b['p_value'] is not None else float('nan'):.4e} —
{"statistically significant" if fam_b_passes else "not significant"} at α = {ALPHA};
practical verdict: **{next(row['verdict'] for row in practical_rows if row['test'].startswith('B:'))}**.

**Family C (weekend × cart→purchase).**
p = {fam_c['p_value'] if fam_c['p_value'] is not None else float('nan'):.4e} —
{"statistically significant" if fam_c_passes else "not significant"} at α = {ALPHA};
practical verdict: **{next(row['verdict'] for row in practical_rows if row['test'].startswith('C:'))}**.

**Decision-relevant (statistically AND practically significant):**
{_yes_md}

**By design (pre-reg §5).** Luxury (42 438 sessions = 0.94 %) sits below the
1 % affected-volume bar, so **every** pairwise comparison involving luxury is
**not decision-relevant** regardless of its p-value — the gate says “too small
to act on”. That consequence was pinned before any test ran; it is not a
post-hoc rationalisation.

**Boundaries (unchanged).** The overall funnel in notebooks 01/02 remains
**descriptive** — no significance claim attaches to those headline rates
(pre-reg §1). This notebook ran **exactly** the pre-registered inventory
(1 χ² + 6 + 1 + 1 z-tests); **no additional test, no tuning, no deviation**
from `reports/pre_registration.md` @ `{PREREG_SHA}`. Statistics ≠ causation:
every statement above is an association on session-level data, not a causal
effect of tier, visit kind or weekday.
\"\"\"
display(Markdown(_md))
"""

_C11_CLOSING = """\
## What this notebook did — and did not — do

**Did:** executed the three pre-registered families (§3) against the fixed
parameters (§4–§5), reported both statistical and practical significance in
separate columns, and wrote the family-A forest plot to
`assets/confirm_price_tier_forest.png`.

**Did not:** run any test outside the pre-registered list; adjust α, tiers or
thresholds after seeing outcomes; re-label any exploratory pattern from
notebook 03 as confirmatory; or attach a significance claim to the overall
funnel (descriptive by §1).

Any future re-analysis that departs from this document becomes an explicitly
labelled **exploratory post-hoc** study in notebook 06 territory — never a
silent edit of this notebook’s conclusions (§6).
"""

_CELLS = [
    {"kind": "markdown", "source": _C1_RULE},
    {"kind": "code", "source": _C2_SETUP},
    {"kind": "code", "source": _C3_LOAD},
    {"kind": "code", "source": _C4_FAMILY_A_CHI2},
    {"kind": "code", "source": _C5_FAMILY_A_PAIRWISE},
    {"kind": "code", "source": _C6_FOREST},
    {"kind": "code", "source": _C7_FAMILY_B},
    {"kind": "code", "source": _C8_FAMILY_C},
    {"kind": "code", "source": _C9_PRACTICAL},
    {"kind": "code", "source": _C10_VERDICT},
    {"kind": "markdown", "source": _C11_CLOSING},
]

if __name__ == "__main__":
    # Execute against the pinned kernel; the runner raises on any cell error.
    build_and_execute(
        "04_hypothesis_testing",
        _CELLS,
        OUT_DIR,
        title="Notebook 04 — confirmatory hypothesis tests (pre-registered)",
    )
    print("built notebooks/04_hypothesis_testing.ipynb")
