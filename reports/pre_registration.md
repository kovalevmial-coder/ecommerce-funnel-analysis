# Pre-registration — Confirmatory Segment Comparisons (Planning Gate)

> Planning-gate document (Stage 05, Task 3). Pinned **before** any confirmatory
> output exists: notebooks 03/04 (segment-conversion confirmatory tests) run in
> Stage 06, **after** this commit. Nothing below may be changed after they run
> (see [Boundary statement](#boundaries-what-may-not-change)).

## 1. Scope

| Item | Value |
|---|---|
| Dataset | E-commerce Events History (Kaggle, mkechinov) — online cosmetics shop |
| Months | 2019-10 … 2020-02, all 5 full months, no pilot month |
| Events / users / sessions | 20 692 840 / 1 639 358 / 4 535 941 |
| Unit of analysis | **session** — one row per `user_session` |
| Reuse caveat | 272 session ids are reused by different users (≈0.006 % of sessions); session-level tables are keyed on `user_session` (Stage-01 measured), contamination is negligible and documented |

**Funnel steps** (reached-step indicators, view → cart → purchase). The three
rates compared in this gate:

| Rate | Definition |
|---|---|
| view→cart | P(cart \| view) — cart sessions among view sessions |
| cart→purchase | P(purchase \| cart) — purchase sessions among cart sessions |
| view→purchase | P(purchase \| view) — purchase sessions among view sessions |

Point estimates are reported with 95 % Wilson CIs (Stage-02 method, `src/funnel`).

> **The overall funnel figures in notebooks 01/02 are DESCRIPTIVE, not
> confirmatory.** They describe what is observed; they do not support any
> significance claim. Confirmatory testing begins only with the comparisons in
> §3 below, on the thresholds fixed in §4–§5.

## 2. Price tiers (explanatory variable → family A)

Fixed business bands (not quantiles), pinned from the pricing *feature* distribution
before any tier × outcome was inspected:

| Tier | Band (EUR) | Sessions (Stage-03 committed) |
|---|---|---|
| budget | `< 5` (incl. 0 and negative prices) | 2 148 161 |
| mid | `[5, 30)` | 1 942 774 |
| premium | `[30, 150)` | 402 568 |
| luxury | `≥ 150` | 42 438 |
| **Total** | | **4 535 941** |

**Tier-validation clause (transparency rule).** If any band holds fewer than
**500 sessions**, it is merged upward with the band below it and the change is
recorded in this document **before any test runs**. Adjustment at planning time
is documented; post-hoc adjustment is forbidden (§6). Check on 2026-09-22: all
four bands ≥ 500 sessions → **merge clause does not trigger**.

## 3. Confirmatory comparisons (complete list)

Everything else in the project (daypart, month, price-conversion curve, gross-vs-net
cart) is **exploratory** and will be labelled as such in the notebooks. Only this
list is confirmatory.

### Family A — price tier × view→cart

- **Omnibus**: χ² test of independence, 4×2 table (4 tiers × {reached cart, did
  not}), on **view sessions** only (denominator = has_view). H₀: the view→cart
  rate is equal across all four tiers. H₁: at least one tier differs. Math:
  H₀: `p_budget = p_mid = p_premium = p_luxury`, H₁: not all equal — where
  `p_tier = P(cart | view, tier)`.
- **Pairwise**: two-proportion z-tests among the 6 tier pairs (4 choose 2),
  on the same view→cart rate. Guard: a pair is tested only if both tiers have a
  non-empty funnel denominator (view sessions > 0 for view→cart; synonymous
  plane: the tier has at least one cart session). All four tiers satisfy this,
  so all **6 pairwise comparisons** run. For tier pair (i, j): H₀: `p_i = p_j`,
  H₁: `p_i ≠ p_j` (two-sided). `diff = p_i − p_j`; the 95 % Wald CI of the
  difference is reported (Stage-04 `src/inference.two_prop_ztest`).

### Family B — visit_kind × cart→purchase

Single two-proportion z-test on **cart sessions** (denominator = has_cart).
Let `p_new` = P(purchase | cart, new session), `p_ret` = P(purchase | cart,
returning session). H₀: `p_new = p_ret`; H₁: `p_new ≠ p_ret`. Direction:
`diff = p_new − p_ret`; `diff > 0` means **new visitors convert cart→purchase
at a higher rate than returning visitors**.

### Family C — weekday/weekend × cart→purchase

Single two-proportion z-test on **cart sessions** (denominator = has_cart).
Let `p_we` = P(purchase | cart, weekend session), `p_wk` = P(purchase | cart,
weekday session). H₀: `p_we = p_wk`; H₁: `p_we ≠ p_wk`. Direction:
`diff = p_we − p_wk`; `diff > 0` is a **positive "weekend effect"** (weekend
sessions convert cart→purchase better than weekday sessions).

## 4. Parameters (fixed before any output)

| Parameter | Value |
|---|---|
| Significance level α | **0.05, two-sided** for every test (z = 1.96) |
| Family A pairwise multiplicity | **per-family Bonferroni**: the 6 pairwise z-tests form one family → α* = 0.05/6 ≈ 0.0083 per test |
| Multiple-comparison sensitivity | **BH-FDR** (Benjamini–Hochberg) reported as *sensitivity* for the same 6 pairwise tests (does not set the rejection rule; Bonferroni does) |
| Sparse-cell rule | in any cell, if expected count `n·p < 5` (equivalently `n(1−p) < 5`); for χ², if `expected_min < 5` → **no test for that comparison**, flag `sparse=True`, report rates descriptively (Stage-04 `chi2_independence.valid`) |

Families B and C are single tests (no multiplicity correction applies).

## 5. Practical-significance rule (decision gate)

A difference is **decision-relevant only if all three hold**:

1. **95 % CI of the difference excludes 0** (direction is statistically established), AND
2. **|relative uplift| ≥ 10 %** of the lower (baseline) rate, AND
3. **affected volume ≥ 1 % of all sessions** (≥ 45 360 of 4 535 941).

**Why these thresholds (business terms).** A cosmetics shop acts on the funnel:

- *+10 % relative uplift.* At the observed view→cart rate 18.42 %, +10 % relative
  = +1.84 pp ≈ **79 000 more view sessions reaching the cart**; at cart→purchase
  12.81 %, +10 % = +1.28 pp ≈ **12 600 more purchases** on ~986 000 cart sessions.
  Below a 10 % relative effect the funnel moves by < 2 pp — a change this shop
  cannot act on (catalog/pricing work is expensive; the metric drifts more than
  that week to week). The threshold is deliberately conservative on **relative**
  effect, not absolute pp, so small bases (premium/luxury) are not rewarded for
  large relative swings on tiny volumes.
- *Affected volume ≥ 1 %.* An action must touch ≥ 1 % of the 4.5 M sessions
  (~45k sessions) to be worth the operational cost of a change. **Consequence by
  design:** luxury (42 438 sessions = 0.94 %) can be statistically confirmed but
  is **never decision-relevant** in any comparison it participates in — the gate
  says "too small to act on". This is recorded here as the accepted reading of
  the rule.
- *CI excludes 0.* Knocks out directions that are not even statistically
  established. Multiplicity is controlled separately (§4).

> **Gap noted (documented, least-surprising default).** The upstream plan fixed
> the three thresholds but not the exact operand of "affected volume". Default
> reading used here: for a pairwise comparison, affected volume = size of the
> **smaller of the two compared groups** (the sessions a hypothetical action
> would lift); for χ² it is the tier-level volume, tier by tier. If a stricter
> operand is intended, that is an exploratory-adjustment decision and must be
> recorded in the follow-up stage, not silently applied post hoc.

## 6. Boundaries (what may not change)

1. **Nothing in this document may be changed after notebooks 03/04 run.**
2. Any later adjustment — to tiers, α, thresholds, the comparison list — becomes an
   explicitly-labelled **exploratory post-hoc analysis**, documented in the
   follow-up stage (notebook 06), and is never re-labelled confirmatory.
3. Tiers and thresholds are pinned here **before** any tier × outcome was
   inspected; "validating a tier post hoc" (anchoring bands to a flattering
   result) is explicitly forbidden.
4. **Commit ordering is the verifiable evidence of the gate.** This document is
   committed **before** notebooks 03/04 exist — they run in Stage 06, after this
   commit, and none of their analysis (nor any `segment_*` outcome study) is
   present in git history at or before this line's commit date. The exact log is
   verified in the stage report.

## 7. Sample-size / power note (capability context, not a tuning knob)

Stage 04 built `required_n_per_group` and `achieved_power` in `src/inference.py`.
They are **not** used to tune any threshold above. In Stage 06 they will be used
to **qualify the capability** of the pre-registered tests: given the actual
observed `n` per segment, what MDE is detectable at 80 % power (α = 0.05
two-sided). The result is reported as context for interpreting null results
(absence of evidence vs evidence of absence), never to re-set α or MDE here.

---

*Pinned 2026-09-22 (Stage 05, Task 3), before any confirmatory segment outcome.*