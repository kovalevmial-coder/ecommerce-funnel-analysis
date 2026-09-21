# Stage 07: README, LICENSE, Final Quality Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the two READMEs (EN + RU) per The Ultimate README Guide, add a LICENSE, and run the definition-of-done audit: clean `git status`, green `pytest`, `errors=none` notebooks, and README/reports/nb05 verdict consistency.

**Architecture:** README is the single source of truth for readers, backed by the executed artifacts. Numbers in README come from computed values, never hand-typed fantasy — each badge/table/figure must trace to a real artifact.

**Tech Stack:** Markdown (GitHub-flavored), HTML report link, shields.io badges.

**Spec:** `docs/superpowers/specs/2026-09-21-ecommerce-funnel-design.md`
**Pre-registration:** `reports/pre_registration.md`

---

### Task 1: README.md (EN) + README_ru.md

**Files:**
- Create: `README.md`
- Create: `README_ru.md`

**Interfaces:**
- Consumes: all notebook/report/asset outputs; the exec-summary verdict variables; ground-truth numbers from Stage 01/06 stage reports.
- Produces: two READMEs following The Ultimate README Guide structure (title+tagline, badges, banner, TOC, introduction, tech stack, project structure, quick start, installation & usage, data, EDA, methodology, results, business impact, testing, limitations, recommendations, support, contributing, license, acknowledgements).

Content decisions (fixed here, no placeholders):
- **Banner**: reuse `assets/funnel_chart.png` (executed headline chart) as the top visual.
- **Badges**: Python 3.12, Tests (n passing computed from `pytest` run — must match reality), License MIT, Pre-registration (link). Use real `shields.io` URLs; count of tests must equal actual `pytest` summary.
- **Numbers**: every figure (events, users, sessions, funnel rates, CIs, verdict) transcribed from the executed notebooks' outputs — verify each against notebook 04/05 before writing.
- **Verdict**: rewritten from the notebook-05/06 consistency check; if robustness flipped anything, README must carry the honest final wording.
- **Quick Start**: Option A `uv` (mirror reference repo); Option B venv+pip; Option C open the report `reports/05_executive_summary.html`.
- **Dataset section**: Kaggle link, row/event/session stats table, quirks table (missing brand/category_code, no device/channel/geo → segmentation re-scope note), cleaning steps.
- **Methodology**: funnel definition + the three CIs; the pre-registration link; the price-tier table; the decision rule (α, corrections, sparse, practical rule).
- **Recommendations**: numbers tied to notebook-05; the proposed A/B test with its own required-n from notebook 05.
- **Testing section**: `.venv/bin/python -m pytest tests/ -q` and expected pass count.

- [ ] **Step 1: Extract the true numbers.** From executed notebooks, compile a single source-of-truth table (events/users/sessions, per-step counts, three rates + CIs, family A/B/C verdicts, bottleneck numbers) into the stage report; transcribe ONLY from there into README.
- [ ] **Step 2: Write `README.md`** (EN) per the structure above; no emojis; formulas via LaTeX math blocks where they add value (decision rule, sample-size relation); every image path verified to exist in `assets/`.
- [ ] **Step 3: Write `README_ru.md`** — faithful Russian translation of the EN version (numbers identical; wording localized). Link each README to the other.
- [ ] **Step 4: Cross-check README claims** against artifacts: for each `![...png]` in README, assert the file exists (a quick check loop). For each headline number, re-run the one-line `src` call that produces it and diffless-compare.
- [ ] **Step 5: Commit**

```bash
git add README.md README_ru.md
git commit -m "docs: add english and russian READMEs with verified figures"
```

---

### Task 2: LICENSE + final audit

**Files:**
- Create: `LICENSE` (MIT text, copyright holder: Dmitrii (dimassaa); year 2026)

**Interfaces:**
- Consumes: repository state.
- Produces: MIT LICENSE, and the definition-of-done audit.

- [ ] **Step 1: Write `LICENSE`** (standard MIT text with the holder/year above).
- [ ] **Step 2: Run the full audit**

```bash
git status --short          # clean except generated .ipynb_checkpoints etc. → all ignored
.venv/bin/python -m pytest tests/ -q   # 100% pass
# per-notebook errors=none check (re-read each executed ipynb for error outputs)
# README/notebook-05/report verdict wording diff-check (must agree)
# every assets/*.png referenced in README exists
git log --oneline --all | wc -l   # one commit per logical change (review for squashed/mixed commits)
git check-ignore AGENTS.md "The Ultimate README Guide.md" data/*.csv data/processed/*.parquet .venv
```

- [ ] **Step 3: Fix anything the audit surfaces** (typos, stale figure, mismatch badge, orphan asset); do it in this task, then re-run the audit to green.
- [ ] **Step 4: Commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license and pass definition-of-done audit"
```

- [ ] **Step 5: Final report + closing note**

RU `notes/07_project_done.md` (what the project delivers, verdict, how to reuse the pipeline for another market/e-commerce dataset). Commit. Provide the user with: final commit hash, the `git push` command ready when they choose (`git remote add origin <url>` + `git branch -M main` + `git push -u origin main`).

---

**Stage verification gate:** `git status` clean (except ignored files); `pytest` 100% pass; all notebooks `errors=none`; README figures ↔ `assets/` bijection verified; verdict identical across README/notebook 05/executive summary; pre-registration commit precedes confirmatory commits in `git log`. Report to the user: the final verdict paragraph, all gate results, and the ready-to-push commands.