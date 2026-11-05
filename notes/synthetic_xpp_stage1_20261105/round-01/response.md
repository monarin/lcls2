# Response to review round 1

Reviewed commit: 343f26bc3d7f8b0760744a427115731668e50453.

| Finding | Disposition | Evidence / change |
| --- | --- | --- |
| XPP-S1-01 | Fixed | jungfrau_audit.json/log report all 151152 events, zero issues/damage, both segments in every frame. completion_checks.json verifies that the audit script still matches the reviewed commit, and derives a pass from these fields. |
| XPP-S1-02 | Fixed in result interpretation | completion_checks.py requires 151152 events, zero issues, damage histogram {"0":151152}, and segment presence {"(0, 1)":151152}. README records the explicit issue-disposition policy. The running audit script was not changed. |
| XPP-S1-03 | Fixed | transition_checks.json/log: 35 full-byte comparisons for services 4/6/7/8/9 and seven header-only SMD EndRun checks pass. Stage 2 must repeat the assertions before rewriting. |
| XPP-S1-04 | Fixed in Stage 2 contract | Proposal and README require fresh destination documents from payload/encoding metadata, validity 3–4, fresh destination IDs, and source IDs/hashes only in explicit provenance. No DB writes. |
| XPP-S1-05 | Fixed | access.txt now uses exclusive creation. Original administrative evidence and its original script hash are preserved; no queries or writes replayed. |
| XPP-S1-06 | Fixed | Boundary indices deduplicated. Revised layout report was regenerated while checking and preserving the existing mapping bytes. Both map hashes and exact byte totals match the retained layout_before_review.json. |

Validation after fixes: 11 focused tests pass (unit_checks_round02.log),
transition comparisons pass, and both exact mapping schedules were recomputed.
The added archive helper copies only checksum-verified snapshots into a new
owner-only directory, with exclusive writes and readback verification. It does
not change existing permissions, publish synthetic data or grant collaborator
access. Its output and completion-check results must exist before Stage 1 passes.
