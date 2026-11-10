# Input bookkeeping PR preparation

Original fix branch: `work/sim10-input-bookkeeping-20261110`, immediate base
`1805e1d9522a4e45aa611a59d83a4cf051678104` on the Perlmutter trace branch. Its
history includes the synthetic-data and profiling campaigns. The earlier
integration merge `90c05965c` joined `c157b490c` and pinned PR187 `4b068a867`.
Both are merge bases with the current target; there is no single simple shared
base. Full research-vs-target tree comparison touches 1,301 files and should not
be submitted as this fix.

Focused branch: `monarin:fix/gpu-input-group-bookkeeping`.
Target: `slac-lcls/lcls2:features/psana2-gpu` at
`e032e0d562f39ecfee735ebeae8b395f28d4b08e`.
Prepared commit: `6f61a8d760c269b61f0f3a93c4f424d10f5932a2`.
Scope: three runtime files, two unit-test files, concise performance note and
minimal root changelog, 7 files / 310 insertions / 19 deletions.

The target already contains the run-scoped-constants transition test correction;
the research version of that integration test is intentionally not carried over.
No target IPC, field-access or memory-accounting changes are reverted. Exact
source checks are in provenance.json. Full current-target CPU suite: 1,023 passed.
GPU acceptance and performance results are from the original research runtime;
the exact target port has not been run on GPU or rebenchmarked.

PR template read from master `d04c3a1148812b48fa78116c4aa8d3710efcd8e2`:
`.github/pull_request_template.md`, with Motivation, How to test the behavior?,
and Checklist. `docs/CHANGELOG_GUIDE.md` specifies Unreleased/Fixed and a PR link.
The target has no root CHANGELOG.md; a minimal entry is added and its PR-number
link must be added after publication. Draft body keeps Actions and issue-closing
checkboxes unchecked; no closing issue was specified. Independent Claude review passed on 6f61a8d76 with no blocking findings.

Title/body: pr-title.txt and pr-body.md. No PR has been opened in this preparation
step. Opening a draft is possible from the saved fork branch.

Review responses: PREP-R1-01 fixed the stale review status in the body;
PREP-R1-02 verified both GitHub commit/file links through the GitHub connector;
PREP-R1-03 added the optional partial-plan pool.close ownership explanation to
the body. No code changed after review. Review limitations remain in review.json.

The exact reviewer context is preserved as text in review-context.json with its
SHA256; JSON encoding preserves whitespace inside the embedded source diff.

## Publication

Opened [PR #298](https://github.com/slac-lcls/lcls2/pull/298) against
`slac-lcls/lcls2:features/psana2-gpu`. Head `67c327a14` adds only the changelog
PR link after reviewed code commit `6f61a8d76`. GitHub reports mergeable/clean.
No Actions runs reported; target psana/DAQ workflows use schedule/manual
triggers. Exact-base GPU validation remains unrun. See publication.json.
