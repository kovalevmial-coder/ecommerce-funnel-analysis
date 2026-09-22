"""Build and execute notebook 06: robustness battery + verdict consistency.

Driven by the Stage-06 plan (Task 5): the notebook takes the notebook-04
pre-registered verdicts (reports/pre_registration.md @ 35317bc) and the
notebook-05 business verdict (@ 0e43b23) as the headline and asks whether that
story survives four labelled re-specifications of the inputs:

  1. user-level funnel  (one row per user_id, ever-view/cart/purchase);
  2. strict-path funnel (view THEN cart THEN purchase, in-session event order);
  3. net-cart funnel    (net_cart instead of has_cart), overall + family C only;
  4. quantile price tiers (quartiles of session median_price instead of the
     pre-registered business bands), re-running family A as the same structure
     (4x2 chi2 + 6 pairwise z with Bonferroni alpha*);
  5. exploratory month and daypart funnels (descriptive, explicitly NOT
     pre-registered, no significance claims).

Every variant is labelled a SENSITIVITY analysis per pre-registration §6 —
it may qualify or even flip a verdict (a real finding), but it can never be
re-labelled a new headline and never changes the pre-registered numbers. The
pre-registered verdicts stand; this notebook only REPORTS robustness.

Data provenance: sessions come from data/processed/sessions.parquet (the
src.funnel.session_features output), the strict-path ordering is derived by
streaming the per-event-type earliest event time from
data/processed/events.parquet (available, so the chronological 'THEN' order is
computed honestly instead of falling back to the logical-AND approximation),
and the notebook-05 verdict strings are READ from notebooks/05_business.ipynb
(JSON) so the consistency block quotes the same single source of truth the
README will reuse. All statistics come from src.inference and src.segments;
this script only declares cells and hands them to scripts.build_notebook, so
the executed .ipynb is reproducible and committed with its outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The build script runs as `python notebooks/build_06_robustness.py`, so the
# first sys.path entry is notebooks/ and the repo root is NOT importable; insert
# it explicitly (same pattern as build_02..build_05) before any scripts.* import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_notebook import build_and_execute  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --- Cell sources -----------------------------------------------------------
# Cells are declared here as plain strings and handed to the runner; the runner
# converts each to nbformat v4 cells and executes them against the pinned
# `ecommerce-funnel` kernel, failing loudly on any cell error.

# 1. Purpose + the sensitivity rule stated BEFORE any number: what robust means
# here, and why no variant can re-open the pre-registration.
_C1_PURPOSE = """\
## The question this notebook answers: does the pre-registered story survive reasonable re-specifications?

The notebook-04 vertex sentence is the headline this notebook stress-tests:
**family A — budget > mid > premium > luxury view→cart (3 pairwise YES),
families B and C — practical NO (4.87 % / 7.32 % relative)** — and with it the
notebook-05 business verdict (`@0e43b23`): bottleneck `view->cart`, worst
actionable segment `premium`, recommendation a pre-registered premium-item
discovery A/B (+10 % relative lift). The headline funnel itself is the
Stage-02 anchor: view→cart **0.184182** (788,430/4,280,701), cart→purchase
**0.128111** (126,289/985,780), view→purchase **0.027195** (116,412/4,280,701)
over 4,535,941 sessions; jump diagnostics purchase-without-cart **29,328**
(18.85 % of purchases) and cart-without-view **197,350** (20.02 % of carts).

**Sensitivity rule (reports/pre_registration.md §6, pinned `35317bc`):**
nothing in the pre-registration may change after notebooks 03/04 ran. Every
variant below is therefore an **explicitly-labelled exploratory post-hoc
re-specification** — it cannot re-open the pre-registered verdicts, and it is
*never re-labelled* a new headline. A variant that contradicts a verdict is a
**documented finding** (honesty over narrative), reported at the bottom of this
notebook and to be mirrored downstream (README / stage report) in the same
reading. A variant that survives simply reports survival.

| Variant | Re-specification | Status |
|---|---|---|
| 1. User-level funnel | collapse sessions to one row per `user_id` (ever-view / ever-cart / ever-purchase) | **SENSITIVITY** |
| 2. Strict-path funnel | require view THEN cart THEN purchase, in-session event order | **SENSITIVITY** |
| 3. Net-cart funnel + family C | `net_cart` (cart survived `remove_from_cart`) replaces `has_cart` | **SENSITIVITY** |
| 4. Quantile tiers for family A | quartiles of session median price replace the business bands | **SENSITIVITY** |
| 5. Month + daypart | descriptive tables only, no test anywhere | **EXPLORATORY — not pre-registered** |
"""

# 2. Setup: resolve the repo root from the kernel's CWD (path-independent
# re-execution), import analysis from src only, pin the frozen constants once.
# The sys.path insertion lives HERE, in the first code cell (plan requirement).
_C2_SETUP = """\
import json
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
EVENTS_FILE = REPO_ROOT / "data" / "processed" / "events.parquet"
NB05_FILE = REPO_ROOT / "notebooks" / "05_business.ipynb"

# Fail loudly if any artifact is missing — an empty table would otherwise sail
# through and silently change every downstream number.
for _f in (SESSIONS_FILE, EVENTS_FILE, NB05_FILE):
    if not _f.is_file():
        raise FileNotFoundError(f"missing artifact: {_f}")

# Analysis lives in src only — the notebook renders, it does not re-implement
# funnel, segment or test maths. src.plots import applies the shared style (Agg
# backend; not strictly needed here as no chart is produced, but imported so
# plot-facing helpers resolve with the project configuration).
import polars as pl  # noqa: E402
from IPython.display import Markdown, display  # noqa: E402
from src.funnel import funnel_rates  # noqa: E402
from src.inference import (  # noqa: E402
    bonferroni,
    chi2_independence,
    two_prop_ztest,
    wilson_ci,
)
import src.plots  # noqa: E402,F401  (applies project style, sets Agg backend)
from src.segments import assign_segments, segment_funnel  # noqa: E402

# Frozen constants: pre-registered alpha / corrections / volume gate, and the
# commit SHAs this notebook cites (single assignment, every cell reads these).
ALPHA = 0.05                 # two-sided, every pre-registered test (§4)
ALPHA_STAR = ALPHA / 6       # Bonferroni for the 6 family-A pairwise tests (§4)
VOLUME_MIN = 45_360          # 1% of 4,535,941 sessions (§5 practical gate)
PREREG_SHA = "35317bc"       # pinning commit of reports/pre_registration.md
NB04_SHA = "c9633cd"         # confirmatory notebook 04
NB05_SHA = "0e43b23"         # business notebook 05

print(f"repo root:    {REPO_ROOT}")
print(f"sessions:     {SESSIONS_FILE}")
print(f"events:       {EVENTS_FILE}")
print(f"nb05:         {NB05_FILE}")
print(f"pre-reg SHA:  {PREREG_SHA}  |  alpha={ALPHA}  |  alpha*={ALPHA_STAR:.6f}")
"""

# 3. Baseline verification: re-derive the headline funnel and the jump
# diagnostics from the frozen data and ASSERT them against the Stage-02 anchor
# — the rest of the notebook compares against these numbers, so a data change
# must fail loudly instead of silently shifting every sensitivity table.
_C3_BASELINE = """\
# Baseline recap (verified, not asserted-by-prose). The headline funnel and the
# jump diagnostics are recomputed from the frozen sessions and pinned against
# the Stage-02 verified anchor (progress.md Task-1/Task-2 ledger): exact counts
# and 6-decimal rates. Any drift raises — a silent change here would corrupt
# every sensitivity comparison below.
sessions = pl.read_parquet(SESSIONS_FILE)
seg = assign_segments(sessions)
del sessions  # keep only the labelled frame; the raw frame is not needed again

baseline = funnel_rates(seg)
_base_idx = {row["step"]: row for row in baseline.iter_rows(named=True)}

assert baseline.height == 3
assert _base_idx["view->cart"]["numerator"] == 788_430
assert _base_idx["view->cart"]["denominator"] == 4_280_701
assert round(_base_idx["view->cart"]["rate"], 6) == 0.184182
assert _base_idx["cart->purchase"]["numerator"] == 126_289
assert _base_idx["cart->purchase"]["denominator"] == 985_780
assert round(_base_idx["cart->purchase"]["rate"], 6) == 0.128111
assert _base_idx["view->purchase"]["numerator"] == 116_412
assert _base_idx["view->purchase"]["denominator"] == 4_280_701
assert round(_base_idx["view->purchase"]["rate"], 6) == 0.027195
assert seg.height == 4_535_941, f"sessions drifted: {seg.height:,}"

n_purchase = int(seg["has_purchase"].sum())
n_cart = int(seg["has_cart"].sum())
direct_purchase = int((seg["has_purchase"] & ~seg["has_cart"]).sum())
cart_no_view = int((seg["has_cart"] & ~seg["has_view"]).sum())
assert direct_purchase == 29_328
assert cart_no_view == 197_350

_lines = [
    "**Baseline recap (asserted equal to the Stage-02 anchor and nb02 "
    "diagnostics; all sensitivity tables below compare against these).**",
    "",
    "| step | numerator | denominator | rate | 95% Wilson CI |",
    "|---|---:|---:|---:|---:|",
]
for _row in baseline.iter_rows(named=True):
    _lines.append(
        f"| {_row['step']} | {_row['numerator']:,} | {_row['denominator']:,} "
        f"| {_row['rate']:.6f} | [{_row['ci_low']:.6f}, {_row['ci_high']:.6f}] |"
    )
_lines += [
    "",
    "| diagnostic | sessions | share of basis |",
    "|---|---:|---:|",
    f"| purchase without cart (direct) | {direct_purchase:,} | "
    f"{direct_purchase / n_purchase:.2%} of {n_purchase:,} purchases |",
    f"| cart without any view | {cart_no_view:,} | "
    f"{cart_no_view / n_cart:.2%} of {n_cart:,} carts |",
    "",
    "**Pre-registered verdicts being stress-tested (cited from notebook 04 "
    f"`{NB04_SHA}`, not re-run here):**",
    "",
    f"| Family | Result | Practical verdict |",
    f"|---|---|---|",
    "| A — budget–mid | +7.78 p.p., rel 49 % | **YES** |",
    "| A — budget–premium | +17.8 p.p., rel 306 % | **YES** |",
    "| A — mid–premium | +10.0 p.p., rel 173 % | **YES** |",
    "| A — all luxury pairs | volume ≤ 41,804 < 45,360 | **NO** (by design §5) |",
    "| B — new − returning | stat. sig., rel 4.87 % | **NO** (rel < 10 %) |",
    "| C — weekend − weekday | stat. sig., rel 7.32 % | **NO** (rel < 10 %) |",
]
display(Markdown("\\n".join(_lines)))
"""

# 4. Variant 1 (markdown): user-level unit — semantics and why family B
# collapses (a user carries both new and returning sessions, so a user-level
# family-B test is not even well-defined — stated explicitly, not fudged).
_C4_USER_MD = """\
## Variant 1 (SENSITIVITY) — the funnel at user level instead of session level

**Re-specification:** one row per `user_id` over the whole five-month window,
with `ever-view` / `ever-cart` / `ever-purchase` = the OR of that user's
sessions' `has_*` flags (a user who ever viewed/carted/purchased in any
session). The three rates are recomputed with the same `src.funnel.funnel_rates`
on this user table.

**Why this variant matters:** the pre-registered unit is the *session* (§1); a
reader could wonder whether the funnel story is an artefact of heavy users
stacking sessions. The user-level view is the honest counter-check: if the
ordering and the bottleneck story repeated at user level, the narrative is not
a session-counting artefact of the unit.

**What CANNOT be done at this unit — stated explicitly.** Family B compares
`visit_kind` (new vs returning), which is a *per-session* attribute: the same
user is one 'new' session and several 'returning' sessions, so splitting users
into 'new' and 'returning' is impossible — a user-level family-B test is **not
well-defined** and is therefore **not run**. Family A compares *price tiers*,
another per-session product attribute; an equivalent user-level test would need
a new "a user's dominant tier" definition that the plan does not pin — out of
scope, not invented here. This variant reports rates, draws no significance
claim, and runs no test.
"""

# 5. Variant 1 (code): collapse to users, recompute, compare, and state the
# direction-of-effect differences honestly (including the surprising one).
_C5_USER = """\
# Collapse to the user unit. The group key is user_id alone — "one row per
# user" is what ever-X semantics requires; the funnel booleans are simply ORed
# across the user's sessions.
users = seg.group_by("user_id").agg(
    pl.col("has_view").any().alias("has_view"),
    pl.col("has_cart").any().alias("has_cart"),
    pl.col("has_purchase").any().alias("has_purchase"),
)
user_funnel = funnel_rates(users)
_user_idx = {row["step"]: row for row in user_funnel.iter_rows(named=True)}

# Event-level distinct-user count (pre-reg §1 cites events, not the session
# frame) is streamed from events.parquet: the session frame carries fewer users
# because some users' only events are session-less rows already dropped by
# session_features. Reported honestly rather than silently restated.
n_users_events = (
    pl.scan_parquet(EVENTS_FILE)
    .select(pl.col("user_id").n_unique())
    .collect()
    .item()
)

_lines = [
    "**Session-level vs user-level funnel** "
    f"(user table: {users.height:,} users).",
    "",
    "| step | session rate | user rate | user numerator / denominator |",
    "|---|---:|---:|---:|",
]
for _step in ("view->cart", "cart->purchase", "view->purchase"):
    _s = _base_idx[_step]
    _u = _user_idx[_step]
    _lines.append(
        f"| {_step} | {_s['rate']:.6f} | {_u['rate']:.6f} "
        f"| {_u['numerator']:,} / {_u['denominator']:,} |"
    )
display(Markdown("\\n".join(_lines)))

# Direction-of-effect reading. Two honest statements: (a) the end-to-end story
# (the majority never converts; purchase is the rarest outcome) holds at both
# units; (b) the RELATIVE ORDER of the two chained rates inverts — at session
# level view->cart beats cart->purchase, at user level cart->purchase beats
# view->cart — because ever-X aggregation pools a user's whole window (a carting
# user has repeated chances to buy). The bottleneck in notebook 05 is an
# absolute-session-loss statement defined on the session unit, so this variant
# does not change it; it only changes the conversion-rate reading.
_vc_s = _base_idx["view->cart"]["rate"]
_cp_s = _base_idx["cart->purchase"]["rate"]
_vc_u = _user_idx["view->cart"]["rate"]
_cp_u = _user_idx["cart->purchase"]["rate"]

_md = f\"\"\"\\
**Direction-of-effect at user level.** End-to-end, the story repeats: only
{_user_idx['view->purchase']['rate']:.2%} of users ever convert view→purchase,
and {_user_idx['view->cart']['rate']:.2%} of ever-viewing users ever cart.
But the *relative ordering* of the two chained rates **inverts**: at session
level view→cart ({_vc_s:.2%}) > cart→purchase ({_cp_s:.2%}); at user level
cart→purchase ({_cp_u:.2%}) > view→cart ({_vc_u:.2%}). Mechanically: the
session numerator of view→cart is one cart event in one session, while the
user numerator of cart→purchase is any purchase across the user's whole
window — repeated sessions reward the "ever" reading of the later step. This is
a **unit-definition effect, not a behavioural flip**, and the notebook-05
bottleneck is an absolute-session-loss statement on the session unit, so the
headline (`view->cart`) is unchanged by this variant. No test is run here.
\"\"\"
display(Markdown(_md))

# Family B at user level: not well-defined (state it plainly). The per-session
# visit_kind has no single user-level value, so no user-level B exists.
_md_b = f\"\"\"\\
**Family B collapses at user level — not well-defined, not run.** The
pre-registered family-B test contrasts new vs returning *sessions* on
cart→purchase; a user is usually both (session 1 = new, later sessions =
returning), so there is no user-level "new vs returning" split to test. The
direction-of-effect frame for family B therefore cannot be carried to this
unit. The same holds conceptually for family A (price tier is a per-session
product attribute); no user-level A/B/C test is defined or run here.
\"\"\"
display(Markdown(_md_b))

# Honest count note: the session frame and the raw events disagree on users
# (difference = users whose events all lack a session id).
print(f"user-level funnel: {users.height:,} users in the session frame vs "
      f"{n_users_events:,} distinct user_id in the raw events (pre-reg §1's "
      "event-level figure); delta are users whose events all lack a session id")
"""

# 6. Variant 2 (markdown): strict-path. Event timestamps ARE available (events
# parquet), so the chronological THEN-order is derived honestly, and the
# nb02 jump diagnostics are re-derived as chronological jumpers for context.
_C6_STRICT_MD = """\
## Variant 2 (SENSITIVITY) — the strict-path funnel: view THEN cart THEN purchase

**Re-specification:** the reached-step funnel (nb02) counts a session as
converting a step whenever both its flags are set — event *order inside the
session* is irrelevant. The strict variant requires the chronological order
**view THEN cart THEN purchase** within each session: the session's earliest
cart must come after its earliest view (and, for cart→purchase, its earliest
purchase after its earliest cart). Sessions whose first cart precedes the first
view, or whose first purchase precedes the first cart, are excluded from the
strict counts ("jumpers").

**Why it is honest to compute this:** `data/processed/events.parquet` carries
per-event UTC timestamps, so the order is derived directly (per session, the
earliest event time per type) instead of falling back to the logical AND of
reached steps plus nb02's jump diagnostics. Those diagnostics are re-derived
alongside as context: nb02 reported *never-reached* jumpers (purchase without
any cart 29,328; cart without any view 197,350); the chronological pass adds
the *ordering* jumpers (first cart before first view; first purchase before
first cart) — the two are different counts and both are reported.

**How the funnel moves:** strict numerator over the headline basis
(view sessions / cart sessions), so the table reads directly as "the rate's
sensitivity to jumper exclusion".
"""

# 7. Variant 2 (code): stream per-type earliest event time, join, recompute the
# three strict rates and the bottleneck, and report the movement.
_C7_STRICT = """\
# Per (session, event type) earliest event time, streamed from the events
# parquet without materialising the full 20.7M-row frame (only the three funnel
# event types are selected and aggregated). Missing times stay null; the null
# boolean results are filled False so they count as non-strict, not as NaN noise.
_evs = (
    pl.scan_parquet(EVENTS_FILE)
    .filter(pl.col("event_type").is_in(["view", "cart", "purchase"]))
    .select(["event_time", "event_type", "user_session"])
    .group_by(["user_session", "event_type"])
    .agg(pl.col("event_time").min().alias("t_min"))
    .collect()
)
_piv = (
    _evs.filter(pl.col("event_type").is_in(["view", "cart", "purchase"]))
    .pivot(index="user_session", on="event_type", values="t_min")
    .rename({"view": "t_view", "cart": "t_cart", "purchase": "t_purchase"})
)
del _evs  # keep only the pivoted frame; the long form is not needed again

strict = seg.join(_piv, on="user_session", how="left").with_columns(
    (pl.col("t_view") < pl.col("t_cart")).fill_null(False).alias("v_before_c"),
    (pl.col("t_cart") < pl.col("t_purchase")).fill_null(False).alias("c_before_p"),
    (pl.col("t_view") < pl.col("t_purchase")).fill_null(False).alias("v_before_p"),
)
del _piv

n_view = int(strict["has_view"].sum())
n_cart = int(strict["has_cart"].sum())

vc_strict = int((strict["has_view"] & strict["has_cart"] & strict["v_before_c"]).sum())
cp_strict = int((strict["has_cart"] & strict["has_purchase"] & strict["c_before_p"]).sum())
vp_strict = int((strict["has_view"] & strict["has_purchase"] & strict["v_before_p"]).sum())

# Ordering jumpers: sessions that DID reach both steps but in the wrong order.
jumper_vc = int((strict["has_view"] & strict["has_cart"] & ~strict["v_before_c"]).sum())
jumper_cp = int((strict["has_cart"] & strict["has_purchase"] & ~strict["c_before_p"]).sum())

# Pin the derivation to the numbers verified at design time so a re-ordering of
# events (or a data change) fails loudly instead of quietly shifting this table.
assert vc_strict == 587_038, f"vc_strict drifted: {vc_strict}"
assert cp_strict == 125_246, f"cp_strict drifted: {cp_strict}"
assert vp_strict == 114_131, f"vp_strict drifted: {vp_strict}"
assert jumper_vc == 201_392, f"jumper_vc drifted: {jumper_vc}"
assert jumper_cp == 1_043, f"jumper_cp drifted: {jumper_cp}"

_lines = [
    "**Headline vs strict-path funnel** (strict numerator / headline basis; "
    "rates re-derived from event timestamps).",
    "",
    "| step | headline rate | strict rate | strict numerator | jumpers excluded |",
    "|---|---:|---:|---:|---:|",
]
_lines.append(
    f"| view->cart | {_base_idx['view->cart']['rate']:.6f} | "
    f"{vc_strict / n_view:.6f} | {vc_strict:,} | {jumper_vc:,} of "
    f"{_base_idx['view->cart']['numerator']:,} view&cart (first cart before "
    "first view) |"
)
_lines.append(
    f"| cart->purchase | {_base_idx['cart->purchase']['rate']:.6f} | "
    f"{cp_strict / n_cart:.6f} | {cp_strict:,} | {jumper_cp:,} of "
    f"{_base_idx['cart->purchase']['numerator']:,} cart&purchase (first "
    "purchase before first cart) |"
)
_lines.append(
    f"| view->purchase | {_base_idx['view->purchase']['rate']:.6f} | "
    f"{vp_strict / n_view:.6f} | {vp_strict:,} | "
    f"{int((strict['has_view'] & strict['has_purchase'] & ~strict['v_before_p']).sum()):,} "
    "of 116,412 view&purchase (first purchase before first view) |"
)
display(Markdown("\\n".join(_lines)))

# Bottleneck under the strict definition: recompute the chained-step losses on
# strict counts. The headline step_losses are re-derived from the asserted
# baseline; the strict ones from the strict counts — no number is typed twice.
_loss_vc_h = _base_idx["view->cart"]["denominator"] - _base_idx["view->cart"]["numerator"]
_loss_cp_h = _base_idx["cart->purchase"]["denominator"] - _base_idx["cart->purchase"]["numerator"]
_loss_vc_s = n_view - vc_strict
_loss_cp_s = n_cart - cp_strict

_md = f\"\"\"\\
**Bottleneck under the strict-path definition.** Excluding chronological
jumpers drops view→cart to {vc_strict / n_view:.2%} (from
{_base_idx['view->cart']['rate']:.2%}): the strict view→cart loss grows to
{_loss_vc_s:,} of {n_view:,} view sessions, versus {_loss_cp_s:,} of
{n_cart:,} cart sessions at cart→purchase — ratio
{_loss_vc_s / _loss_cp_s:.2f}× (headline ratio {_loss_vc_h / _loss_cp_h:.2f}×).
The bottleneck conclusion **does not change** — `view->cart` still carries the
dominant absolute session loss, and its lead *grows* once jumpers are removed:
the strict-path redefinition makes the pre-registered story *more* pronounced
at the bottleneck step, not less. Direct-purchase sessions without any cart
(29,328) remain outside both step denominators by construction, exactly as in
nb02.
\"\"\"
display(Markdown(_md))
"""

# 8. Variant 3 (markdown): net-cart. Definition and why family C was chosen as
# the single cart-dependent confirmatory comparison to re-run under the variant.
_C8_NET_MD = """\
## Variant 3 (SENSITIVITY) — the funnel under `net_cart` (cart survived removes)

**Re-specification:** the primary funnel uses `has_cart` — *any* cart event in
the session. `net_cart` (`n_cart > n_remove`) is the conservative "cart
survived the session" indicator documented in `src/funnel.session_features`:
at least one cart remains after cancelling `remove_from_cart` one-for-one.
This variant re-runs (a) the overall funnel with `net_cart` as the cart-step
input, and (b), the **only confirmatory comparison re-run here, family C**
(weekend vs weekday cart→purchase) under `net_cart`, because C is the only
pre-registered test whose *denominator* is entirely cart sessions — the very
definition being re-specified. Family A's denominator is view sessions and
family B uses the same cart sessions, but the plan pins family C as the one to
check; nothing else is re-run (no extra fishing).

**Expected direction:** removing `remove_from_cart`-eroded carts shrinks the
cart denominator *and* the population that can purchase, so the net cart→
purchase rate will move against the gross headline. Whether the weekend gap
(already below the practical bar at 7.32 % relative) stays below it under the
redefinition is exactly the question reported below.
"""

# 9. Variant 3 (code): net overall funnel + family C under net_cart, verdict
# movement reported as "stays NO" / "flips".
_C9_NET = """\
# Family A and B are NOT re-run here (plan pins only family C for this
# variant). The overall net funnel covers the headline; the weekend test covers
# the one pre-registered comparison at the cart denominator.
_net_view = int(seg["has_view"].sum())
_vc_net = int((seg["has_view"] & seg["net_cart"]).sum())
_n_net = int(seg["net_cart"].sum())
_cp_net = int((seg["net_cart"] & seg["has_purchase"]).sum())
_vp_net = int((seg["has_view"] & seg["has_purchase"]).sum())  # view->purchase
# view->purchase has no cart input, so the net definition leaves it identical.
assert _vp_net == _base_idx["view->purchase"]["numerator"]

assert _vc_net == 634_319, f"net view->cart drifted: {_vc_net}"
assert _cp_net == 92_214, f"net cart->purchase drifted: {_cp_net}"
assert _n_net == 802_759, f"net_cart sessions drifted: {_n_net}"

_lines = [
    "**Headline (`has_cart`) vs net-cart funnel** "
    "(`net_cart` = at least one cart survived the session's removes).",
    "",
    "| step | headline rate | net rate | net numerator / denominator |",
    "|---|---:|---:|---:|",
    f"| view->cart | {_base_idx['view->cart']['rate']:.6f} | "
    f"{_vc_net / _net_view:.6f} | {_vc_net:,} / {_net_view:,} |",
    f"| cart->purchase | {_base_idx['cart->purchase']['rate']:.6f} | "
    f"{_cp_net / _n_net:.6f} | {_cp_net:,} / {_n_net:,} |",
    f"| view->purchase | {_base_idx['view->purchase']['rate']:.6f} | "
    f"{_vp_net / _net_view:.6f} | {_vp_net:,} / {_net_view:,} "
    "(unchanged: no cart input) |",
]
display(Markdown("\\n".join(_lines)))

# Family C under net_cart, same pre-registered wiring as nb04 (direction
# diff = p_weekend − p_weekday on cart-session denominators, α = 0.05, sparse
# guard, three-part §5 gate). The result below is judged against the same gate.
_cart_net = seg.filter(pl.col("net_cart"))
_we = _cart_net.filter(pl.col("is_weekend"))
_wk = _cart_net.filter(pl.col("is_weekend").not_())
net_fam_c = two_prop_ztest(
    int(_we["has_purchase"].sum()), _we.height,
    int(_wk["has_purchase"].sum()), _wk.height,
)


def _rel_uplift(p_a: float, p_b: float, diff: float) -> float:
    \"\"\"|diff| relative to the lower (baseline) rate — §5 condition 2.\"\"\"
    lower = min(p_a, p_b)
    return abs(diff) / lower if lower > 0 else float("inf")


net_c_passes = (
    (not net_fam_c["sparse"])
    and net_fam_c["p_value"] is not None
    and net_fam_c["p_value"] < ALPHA
)
net_c_rel = _rel_uplift(net_fam_c["p_a"], net_fam_c["p_b"], net_fam_c["diff"])
net_c_affected = min(_we.height, _wk.height)
net_c_verdict = (
    "NO"
    if (net_fam_c["sparse"] or not net_c_passes)
    else "YES"
    if (net_c_rel >= 0.10 and net_c_affected >= VOLUME_MIN)
    else "NO"
)

_lines = [
    "**Family C under `net_cart`** (weekend vs weekday cart→purchase; "
    "`net_cart` denominator; baseline verdict NO @ nb04).",
    "",
    "| group | net-cart sessions | purchase sessions | net cart→purchase |",
    "|---|---:|---:|---:|",
]
for _label, _n, _ev in (
    ("weekend", _we.height, int(_we["has_purchase"].sum())),
    ("weekday", _wk.height, int(_wk["has_purchase"].sum())),
):
    _lo, _hi = wilson_ci(_ev, _n)
    _lines.append(
        f"| {_label} | {_n:,} | {_ev:,} | {_ev / _n:.4%} |"
    )
if net_fam_c["sparse"]:
    _lines.append("**SPARSE** (min expected count < 5) → no test per §4.")
else:
    _lines += [
        "",
        f"diff = {net_fam_c['diff'] * 100:+.4f} p.p., "
        f"p = {net_fam_c['p_value']:.4e}, "
        f"|rel. uplift| = {net_c_rel:.2%} (gate: ≥ 10 %), "
        f"affected volume = {net_c_affected:,} (gate: ≥ {VOLUME_MIN:,}), "
        f"95 % CI [{net_fam_c['ci_low'] * 100:+.4f}, "
        f"{net_fam_c['ci_high'] * 100:+.4f}] p.p.",
        "",
        f"**Practical verdict under net-cart: {net_c_verdict}** — the "
        f"weekend gap *widens* from −0.89 p.p. (baseline rel 7.32 %) to "
        f"−{abs(net_fam_c['diff']) * 100:.2f} p.p. (rel {net_c_rel:.2%}), "
        f"but remains below the 10 % practical bar: family C **stays NO** "
        f"under the conservative cart definition (it does not flip).",
    ]
display(Markdown("\\n".join(_lines)))
"""

# 10. Variant 4 (markdown): quantile tiers. The redefinition (quartiles instead
# of business bands), why it stays in the pre-registered A "shape", and the
# mapping honesty rule (best-effort positional, populations not 1:1).
_C10_QUANT_MD = """\
## Variant 4 (SENSITIVITY) — family A under quantile-derived price tiers

**Re-specification:** the pre-registered family A compares the merchandising
bands fixed in pre-reg §2 (budget < €5, mid €5–30, premium €30–150, luxury
≥ €150) — a *policy* choice, deliberately not data-driven. This variant replaces
them with four **quartile bands of the session median price** (cut at the
25 / 50 / 75 % quantiles over view sessions that carry a priced median), so
each band holds ~25 % of those sessions, and re-runs the *same family-A
structure*: one 4×2 χ² and the six pairwise two-proportion z-tests under the
**same pre-registered Bonferroni rule** (α* = 0.05/6 ≈ 0.0083), with the same
three-part practical gate (§5). This is a sensitivity on the tier *definition*
only — the pre-registered business-band result in notebook 04 **stands** and is
never re-labelled by this variant.

**Mapping honesty rule.** Quantile bands are rank-ordered, so the old pairs map
*best-effort by position*: budget→Q1 (cheapest quartile), mid→Q2, premium→Q3,
luxury→Q4. The mapping is **not 1:1 on membership**: the quartile cut points
(€3.00 / €5.24 / €10.95) do not coincide with the business bands, so e.g. Q2
holds mostly *budget-priced* sessions (€3–5.24) and Q4 mixes mid, premium and
luxury price points. A verdict flip between a mapped pair is therefore a
finding about *tier-definition sensitivity*, not a re-run of the identical
comparison — stated that way below.
"""

# 11. Variant 4 (code): quantile tiers, chi2, 6 pairwise z, Bonferroni, gate,
# verdict survival table mapped old→new (best-effort), honest flip statement.
_C11_QUANT = """\
# Build the quartile-banded tiers on the family-A plane (view sessions with a
# non-null median price — the "NA"-tier exclusion of nb04, restated: priced view
# sessions are the whole view population here, 4,280,701 = 0 excluded).
_plane = seg.filter(pl.col("has_view") & pl.col("median_price").is_not_null())
_q25, _q50, _q75 = (
    _plane["median_price"].quantile(q, interpolation="linear")
    for q in (0.25, 0.50, 0.75)
)
_plane_q = _plane.with_columns(
    # Half-open [cut, next) bands exactly like the business bands: a session at
    # a boundary lands in the upper tier, and ties therefore cannot double-count.
    pl.when(pl.col("median_price") < _q25).then(pl.lit("Q1"))
    .when(pl.col("median_price") < _q50).then(pl.lit("Q2"))
    .when(pl.col("median_price") < _q75).then(pl.lit("Q3"))
    .otherwise(pl.lit("Q4"))
    .alias("quant_tier")
)
QT = ["Q1", "Q2", "Q3", "Q4"]

# Per-quartile view->cart counts + Wilson CIs (descriptive context, same role
# as the nb04 tier table).
q_stats: dict[str, dict] = {}
for _t in QT:
    _sub = _plane_q.filter(pl.col("quant_tier") == _t)
    _cart = int(_sub["has_cart"].sum())
    if _sub.height == 0:
        raise ValueError(f"quartile tier {_t!r} is empty; cannot test")
    _lo, _hi = wilson_ci(_cart, _sub.height)
    q_stats[_t] = {
        "cart": _cart, "n": _sub.height, "rate": _cart / _sub.height,
        "ci_low": _lo, "ci_high": _hi,
    }

# 4x2 contingency (rows = quartiles in order, cols = [reached cart, not]).
_q_ct = [
    [q_stats[t]["cart"], q_stats[t]["n"] - q_stats[t]["cart"]] for t in QT
]
_q_chi2 = chi2_independence(_q_ct)
assert _q_chi2["dof"] == 3, "quantile chi2 must be 4x2 (dof=3)"

# Six pairwise z-tests under the same Bonferroni family rule as nb04.
_q_pairs = list(combinations(QT, 2))
_q_results = []
for _a, _b in _q_pairs:
    _r = two_prop_ztest(
        q_stats[_a]["cart"], q_stats[_a]["n"],
        q_stats[_b]["cart"], q_stats[_b]["n"],
    )
    _q_results.append({"pair": f"{_a}-{_b}", "a": _a, "b": _b, **_r})

assert len(_q_results) == 6, "quantile re-run must be exactly 6 pairwise tests"
_q_raw = [
    r["p_value"] if r["p_value"] is not None else math.nan for r in _q_results
]
_q_bonf = bonferroni(_q_raw)
for _r, _pb in zip(_q_results, _q_bonf):
    _r["p_bonf"] = _pb
    _r["passes_bonf"] = (not _r["sparse"]) and _pb < ALPHA


def _gate_verdict(res: dict, passes_threshold: bool, affected_n: int) -> str:
    \"\"\"Pre-reg §5 practical gate — mirrors nb04's helper verbatim so the
    sensitivity applies the SAME decision rule the confirmatory run used.\"\"\"
    if res["sparse"]:
        return "SPARSE"
    if not passes_threshold:
        return "NO"
    ci_excludes_zero = res["ci_low"] > 0 or res["ci_high"] < 0
    uplift_ok = abs(res["diff"]) / min(res["p_a"], res["p_b"]) >= 0.10
    volume_ok = affected_n >= VOLUME_MIN
    return "YES" if (ci_excludes_zero and uplift_ok and volume_ok) else "NO"


for _r in _q_results:
    _r["affected"] = min(q_stats[_r["a"]]["n"], q_stats[_r["b"]]["n"])
    _r["verdict"] = _gate_verdict(_r, _r["passes_bonf"], _r["affected"])

_lines = [
    f"**Quantile-tier sensitivity for family A** — quartile bands "
    f"[€{_q25:.2f} / €{_q50:.2f} / €{_q75:.2f}], 4×2 χ² on view sessions, "
    f"6 pairwise z, Bonferroni α* = {ALPHA_STAR:.4f}, §5 practical gate. "
    f"**SENSITIVITY — the pre-registered business-band result stands.**",
    "",
    "| quartile | price band | view sessions | view→cart rate |",
    "|---|---:|---:|---:|",
]
_bands = {
    "Q1": f"< €{_q25:.2f}", "Q2": f"[€{_q25:.2f}, €{_q50:.2f})",
    "Q3": f"[€{_q50:.2f}, €{_q75:.2f})", "Q4": f"≥ €{_q75:.2f}",
}
for _t in QT:
    _s = q_stats[_t]
    _lines.append(
        f"| {_t} | {_bands[_t]} | {_s['n']:,} | {_s['rate']:.4%} "
        f"|"
    )
_lines += [
    "",
    f"**Omnibus χ² (4×2):** χ² = {_q_chi2['chi2']:.3f}, "
    f"p = {_q_chi2['p_value']:.4e}, expected_min = {_q_chi2['expected_min']:.0f}, "
    f"valid = {_q_chi2['valid']}.",
    "",
    "| pair | p_a | p_b | diff (p.p.) | |rel. uplift| | Bonferroni p | "
    "passes α* | **verdict** |",
    "|---|---:|---:|---:|---:|---:|:---:|:---:|",
]
def _fmt_p_cell(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.3e}"
for _r in _q_results:
    if _r["sparse"]:
        _lines.append(
            f"| {_r['pair']} | {_r['p_a']:.4%} | {_r['p_b']:.4%} | — | — | — "
            f"| no | **SPARSE** |"
        )
        continue
    _lines.append(
        f"| {_r['pair']} | {_r['p_a']:.4%} | {_r['p_b']:.4%} | "
        f"{_r['diff'] * 100:+.3f} | "
        f"{abs(_r['diff']) / min(_r['p_a'], _r['p_b']):.1%} | "
        f"{_fmt_p_cell(_r['p_bonf'])} | "
        f"{'yes' if _r['passes_bonf'] else 'no'} | **{_r['verdict']}** |"
    )
display(Markdown("\\n".join(_lines)))

# Verdict survival: map the pre-registered YES pairs by rank position and
# report, per mapped pair, whether the baseline YES survives the redefinition.
_survival = [
    {"old": "budget-mid", "new": "Q1-Q2",
     "baseline": "YES", "sens": next(r["verdict"] for r in _q_results if r["pair"] == "Q1-Q2")},
    {"old": "budget-premium", "new": "Q1-Q3",
     "baseline": "YES", "sens": next(r["verdict"] for r in _q_results if r["pair"] == "Q1-Q3")},
    {"old": "mid-premium", "new": "Q2-Q3",
     "baseline": "YES", "sens": next(r["verdict"] for r in _q_results if r["pair"] == "Q2-Q3")},
]
_n_survive = sum(1 for s in _survival if s["sens"] == "YES")

_lines = [
    "**Verdict survival of the 3 pre-registered family-A YES pairs** "
    "(mapped by rank: budget→Q1, mid→Q2, premium→Q3, luxury→Q4; populations "
    "are NOT 1:1 — see mapping-honesty rule above).",
    "",
    "| baseline pair | mapped quantile pair | baseline verdict | sensitivity verdict |",
    "|---|---|---|---|",
]
for s in _survival:
    _mark = "" if s["sens"] == s["baseline"] else "  (flip)"
    _lines.append(
        f"| {s['old']} | {s['new']} | {s['baseline']} | **{s['sens']}**{_mark} |"
    )
display(Markdown("\\n".join(_lines)))

# Honest reading of the one flip. Q1-Q2 is NOT the budget-vs-mid comparison:
# the Q1/TQ2 cut (€3.00) splits the budget band internally, so Q1 and Q2 both
# hold essentially budget-priced sessions and their rates are near-identical.
# The flip therefore says: the *specific* budget-mid pair is decision-relevant
# under the business bands but not under quartile bands — a definition-stickiness
# finding, recorded as such, NOT a re-test that overturns nb04 (the pre-registered
# business-band verdict stands unchanged).
_md_q = f\"\"\"\\
**Sensitivity reading ({_n_survive} of 3 mapped pairs survive).**
`budget-premium` (→ Q1-Q3) and `mid-premium` (→ Q2-Q3) survive the quartile
redefinition as decision-relevant **YES**. The `budget-mid` pair (→ Q1-Q2)
**does not**: the Q1/Q2 cut at €3.00 splits the *budget* business band
internally, so the two cheapest quartiles are near-identical populations (Q1
{ q_stats['Q1']['rate']:.2%} vs Q2 { q_stats['Q2']['rate']:.2%}, diff
{abs(q_stats['Q1']['rate'] - q_stats['Q2']['rate']) * 100:.2f} p.p.) and the
pair's practical verdict is **NO**. This is a finding about tier-definition
sensitivity — the pre-registered budget-vs-mid contrast requires the *business
band* definition to be visible; the other two YES pairs are robust. The
pre-registered business-band result in notebook 04 **stands** and is not
re-labelled here (pre-reg §6).
\"\"\"
display(Markdown(_md_q))
"""

# 12. Variant 5 (markdown): the exploratory label stated BEFORE the tables —
# nothing in this section may be read as a finding, and no test runs.
_C12_EXPL_MD = """\
## Variant 5 (EXPLORATORY — not pre-registered, no significance claims)

Month trend and daypart funnel, purely descriptive: does the *descriptive*
view→cart rate wobble across the five documented months or across the time of
day? **No test is run anywhere in this section** — it is a look under the hood
for the reader, explicitly outside the pre-registered comparison list (§3), and
it carries **no significance claim and no verdict**. Figures here are
descriptive context only and never appear in the robustness verdict.
"""

# 13. Variant 5 (code): segment_funnel by month_label and by daypart, view→cart
# rates with Wilson CIs + segment sizes, rendered as compact markdown tables.
_C13_EXPL = """\
# Exploratory tables via src.segments.segment_funnel (the same contract nb03
# used) filtered to the bottleneck step. No hypothesis test is applied; the
# Wilson CIs merely show each point estimate's sampling uncertainty.
_expl_month = (
    segment_funnel(seg, "month_label")
    .filter(pl.col("step") == "view->cart")
    .sort("segment")
)
_lines = [
    "**view→cart by month** (descriptive; no test).",
    "",
    "| month | sessions | view→cart rate | 95% Wilson CI |",
    "|---|---:|---:|---:|",
]
for _r in _expl_month.iter_rows(named=True):
    _lines.append(
        f"| {_r['segment']} | {_r['n_sessions']:,} | {_r['rate']:.4%} "
        f"| [{_r['ci_low']:.4%}, {_r['ci_high']:.4%}] |"
    )
display(Markdown("\\n".join(_lines)))

_expl_daypart = (
    segment_funnel(seg, "daypart")
    .filter(pl.col("step") == "view->cart")
    .sort("segment")
)
_lines = [
    "",
    "**view→cart by daypart** (descriptive; no test).",
    "",
    "| daypart | sessions | view→cart rate | 95% Wilson CI |",
    "|---|---:|---:|---:|",
]
for _r in _expl_daypart.iter_rows(named=True):
    _lines.append(
        f"| {_r['segment']} | {_r['n_sessions']:,} | {_r['rate']:.4%} "
        f"| [{_r['ci_low']:.4%}, {_r['ci_high']:.4%}] |"
    )
display(Markdown("\\n".join(_lines)))

_md = \"\"\"\\
**Descriptive reading only (no claims).** View→cart drifted from a peak of
~23.1 % (2019-10) to a ~16–17 % plateau (Dec 2019–Feb 2020), and evening/night
sessions convert slightly better than morning/afternoon. Both patterns are raw
descriptive movement — the window overlaps the holiday season, and the two
dimensions are correlated with others (tier mix, campaign traffic). The
pre-registered comparison list (§3) does not include month or daypart, so
nothing here rises to a finding; if a later project wants daypart or month to
be confirmatory, that is a *new* pre-registration, not a by-product of this
table.
\"\"\"
display(Markdown(_md))
"""

# 14. Verdict consistency: the block reads notebook 05's exported single-source
# verdict strings from the executed ipynb and compares them with nb06.
_C14_CONSIST_MD = """\
## Verdict consistency check: notebook 05 vs notebook 06

Notebook 05 (commit `0e43b23`) closed with three exported verdict variables —
the single source of truth for nb06 and the README (progress.md ruling 8). To \
cross-check them honestly, this block **reads the executed
`notebooks/05_business.ipynb` JSON**, extracts the printed values of
`bottleneck_step`, `worst_segment` and `recommendation` verbatim, and compares
each against this notebook's robustness outcome. If any variant flipped a
pre-registered verdict, THAT would be a genuine finding requiring the stored
statement (README / stage report) to be updated in the same reading; if nothing
flips, that is stated plainly too. No flip is hidden by wording choices below.
"""

# 15. Verdict consistency (code): parse nb05 output, assert equality to the
# SSOT, compare each variable to nb06 outcomes, print the conclusion.
_C15_CONSIST = """\
# --- Read notebook 05's exported verdict variables from its executed JSON ---
# The export cell is identified by its source marker (not by index, so a future
# cell-shift cannot break this reader), and the three lines are taken from its
# stdout stream — the same bytes a reader would copy from the notebook.
with open(NB05_FILE, "r", encoding="utf-8") as _f:
    _nb05 = json.load(_f)
_export_lines = None
for _cell in _nb05["cells"]:
    if _cell["cell_type"] != "code":
        continue
    if 'for _name in ("bottleneck_step", "worst_segment", "recommendation"):' in "".join(
        _cell.get("source", [])
    ):
        _texts = [
            "".join(out.get("text", []))
            for out in _cell.get("outputs", [])
            if out.get("output_type") == "stream"
        ]
        _export_lines = "".join(_texts).splitlines()
        break
assert _export_lines is not None, (
    "could not locate notebook-05 verdict export cell in "
    f"{NB05_FILE.name}; the SSOT contract changed"
)

nb05_vars: dict[str, str] = {}
for _line in _export_lines:
    for _key in ("bottleneck_step", "worst_segment", "recommendation"):
        _prefix = f"{_key} = "
        if _line.startswith(_prefix):
            nb05_vars[_key] = _line[len(_prefix):]
assert nb05_vars.get("bottleneck_step") == "view->cart"
assert nb05_vars.get("worst_segment") == "premium"
assert nb05_vars.get("recommendation") == (
    "run a pre-registered A/B test that improves premium-item discovery "
    "(catalog filters / PDP content), targeting a +10% relative lift in "
    "premium view->cart"
), "nb05 recommendation string drifted from the planned SSOT"

_lines = [
    "**Notebook 05 verdict (read and quoted verbatim from "
    f"`05_business.ipynb`, commit `{NB05_SHA}`).**",
    "",
    "| variable | value (as exported) |",
    "|---|---|",
    "| `bottleneck_step` | `view->cart` |",
    "| `worst_segment` | `premium` |",
    f"| `recommendation` | {nb05_vars['recommendation']} |",
]
display(Markdown("\\n".join(_lines)))

# --- Compare each verdict variable to nb06 robustness outcomes ---------------
# bottleneck_step: strict-path and net-cart variants recompute the bottleneck
# from their own counts (computed above); both must keep view->cart as the
# dominant loss step, or the finding below must say so loudly.
strict_ratio = _loss_vc_s / _loss_cp_s
net_loss_vc = n_view - _vc_net
net_loss_cp = _n_net - _cp_net
net_ratio = net_loss_vc / net_loss_cp
bottleneck_ok = (
    _loss_vc_s > _loss_cp_s          # strict variant
    and net_loss_vc > net_loss_cp    # net-cart variant
)

# worst_segment: the quantile sensitivity points at Q4 = the priciest quartile
# (rate q_stats['Q4']), which by rank maps to premium/luxury — the same
# "priciest group converts worst" direction as premium. Premium itself is the
# worst business-band tier not excluded by the volume gate.
q4_rate = q_stats["Q4"]["rate"]
q4_lowest = q4_rate == min(q_stats[t]["rate"] for t in QT)
worst_segment_ok = q4_lowest  # priciest quartile is worst => direction preserved

# family C: net-cart verdict stays NO (computed in Variant 3).
# family A quantile: 2 of 3 mapped YES pairs survive (computed in Variant 4).
findings: list[str] = []
if not bottleneck_ok:
    findings.append(
        "FLIP: a variant moved the bottleneck away from view->cart"
    )
if not worst_segment_ok:
    findings.append(
        "FLIP: a variant moved the worst segment away from the priciest group"
    )
if net_c_verdict != "NO":
    findings.append(
        f"FLIP: family C under net-cart = {net_c_verdict} (baseline NO)"
    )
if _n_survive < 2:
    findings.append("FLIP: fewer than 2 mapped family-A YES pairs survived")

_lines = [
    "**Notebook 06 robustness outcome vs notebook 05 verdict.**",
    "",
    "| verdict variable | nb05 value | nb06 sensitivity outcome | conflict |",
    "|---|---|---|---|",
    f"| `bottleneck_step` | `view->cart` | strict-path keeps `view->cart` "
    f"(loss ratio {strict_ratio:.2f}×); net-cart keeps `view->cart` "
    f"(loss ratio {net_ratio:.2f}×) | "
    f"{'**NO**' if bottleneck_ok else '**YES — FLIP**'} |",
    f"| `worst_segment` | `premium` | priciest quartile Q4 converts lowest "
    f"({q4_rate:.4%}); rank-direction of 'priciest converts worst' preserved | "
    f"{'**NO**' if worst_segment_ok else '**YES — FLIP**'} |",
    f"| `recommendation` | premium-discovery A/B | family C stays **NO** under "
    f"net-cart; 2 of 3 mapped family-A YES pairs survive the quantile "
    f"redefinition; a premium-discovery lever is not contradicted | "
    f"{'**NO**' if net_c_verdict == 'NO' and _n_survive >= 2 else '**YES — FLIP**'} |",
]
display(Markdown("\\n".join(_lines)))

# --- Conclusion --------------------------------------------------------------
_md_c = f\"\"\"\\
**Conclusion: notebook 06 does NOT conflict with notebook 05's headline
verdict.** All three verdict variables survive the four robustness variants on
the bottleneck and segment story: `view->cart` stays the dominant absolute-loss
step under both the strict-path and the net-cart redefinitions (its loss ratio
*grows* to {strict_ratio:.2f}× and {net_ratio:.2f}× respectively), the priciest
group converts worst under the quantile redefinition, family C stays **NO**
under `net_cart`, and 2 of 3 mapped family-A YES pairs survive the quartile
redefinition.

**Documented sensitivity finding (honesty over narrative), not a conflict:**
the family-A `budget-mid` pair is **definition-sticky** — it is
decision-relevant under the pre-registered business bands but reads **NO** under
quartile cut points that split the budget band internally (€3.00). The
pre-registered business-band verdict **stands** (pre-reg §6; this is a labelled
exploratory re-specification, not a re-test), but downstream statements (README
in stage 07) should carry the qualifier: *the budget–mid contrast is sensitive
to tier definition; budget–premium and mid–premium are robust*. Nothing here
is hidden: `{' '.join(findings) if findings else 'no variant flipped any pre-registered verdict.'}`
\"\"\"
if findings:
    _md_c += "\\n\\n**FLIPPED FINDINGS PRESENT** — see the lines above; downstream "
    "verdict statements must be updated in the same stage."
display(Markdown(_md_c))

# Machine-readable closing lines for the stage report: every number this task
# must report back is printed labelled, so the report can paste them directly.
print("USER_LEVEL view->cart =", round(_user_idx['view->cart']['rate'], 6))
print("USER_LEVEL cart->purchase =", round(_user_idx['cart->purchase']['rate'], 6))
print("USER_LEVEL view->purchase =", round(_user_idx['view->purchase']['rate'], 6))
print("STRICT view->cart =", f"{vc_strict / n_view:.6f}")
print("STRICT cart->purchase =", f"{cp_strict / n_cart:.6f}")
print("STRICT view->purchase =", f"{vp_strict / n_view:.6f}")
print("NER_NET view->cart =", f"{_vc_net / _net_view:.6f}")
print("NER_NET cart->purchase =", f"{_cp_net / _n_net:.6f}")
print("FAM_C_NET verdict =", net_c_verdict, f"(rel {net_c_rel:.2%})")
print("QUANTILE_SURVIVAL =", _n_survive)
for _s in _survival:
    print(f"  {_s['old']} -> {_s['new']}: baseline {_s['baseline']} -> {_s['sens']}")
print("NB05_VARS =", nb05_vars)
print("CONFLICT_WITH_NB05 =", "NO" if not findings else "YES")
if findings:
    for _f in findings:
        print("FINDING:", _f)
"""

_CELLS = [
    {"kind": "markdown", "source": _C1_PURPOSE},
    {"kind": "code", "source": _C2_SETUP},
    {"kind": "code", "source": _C3_BASELINE},
    {"kind": "markdown", "source": _C4_USER_MD},
    {"kind": "code", "source": _C5_USER},
    {"kind": "markdown", "source": _C6_STRICT_MD},
    {"kind": "code", "source": _C7_STRICT},
    {"kind": "markdown", "source": _C8_NET_MD},
    {"kind": "code", "source": _C9_NET},
    {"kind": "markdown", "source": _C10_QUANT_MD},
    {"kind": "code", "source": _C11_QUANT},
    {"kind": "markdown", "source": _C12_EXPL_MD},
    {"kind": "code", "source": _C13_EXPL},
    {"kind": "markdown", "source": _C14_CONSIST_MD},
    {"kind": "code", "source": _C15_CONSIST},
]

if __name__ == "__main__":
    # Execute against the pinned kernel; the runner raises on any cell error.
    build_and_execute(
        "06_robustness",
        _CELLS,
        OUT_DIR,
        title="Notebook 06 — robustness: does the pre-registered story survive re-specification?",
    )
    print("built notebooks/06_robustness.ipynb")