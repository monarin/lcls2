# Responses to review 1

Reviewed implementation: `9452d8d123e8513a1c2d91efd8cfb81d720efd37`.

| ID | Disposition | Evidence / change |
|---|---|---|
| MRC-001 | Addressed | Full job 43382526 completed 0:0; pilot report status pass, 1,000/1,000 events, 5,000 blobs. Full disk readback passed; `readback-result.json` gives each file hash, report hash, sizes, max error 2.0, and source SMD timestamp comparison. Repeat edge job 43384088 result is `edges-corrected.json`. The known codec failure remains outside general acceptance. |
| MRC-002 | Addressed using suggested retained-witness option | `first-event-values.json` records zero nonfinite and zero abs(input)>=2^25 values in event 0. README explicitly says no full-pilot branch counts were recorded, and passing MFX data does not establish coverage of the defective branch. The separate edge probe does exercise it. |
| MRC-003 | Deferred, nonblocking | This bounded trial is complete; harness exceptions leave a non-pass report and the job log retains the reason. A structured exception journal would improve a reusable producer, but is not required for this fixture and does not affect acceptance. Add before making this a general producer. No false-pass path is accepted. |
| MRC-004 | Addressed | `provenance.json` now names frozen and reviewed driver hashes, their artifact associations, and the initial gate driver snapshot limitation. The compiled DRP sources are byte-identical to the repository copies the reviewer could access; both SHA256 values are recorded. |
| MRC-005 | Addressed | README now names committed `edges-original.json` and explains that scratch `edges01.json` is identical. |

Additional validation improvement: `verify_saved.py` now compares the complete
timestamp list with the first N L1Accept timestamps from independently
SHA256-verified source SMD, rather than checking monotonicity alone. The full
1,000-event readback passed this stronger check. A negative test removing the
first source timestamp was rejected. No codec or numerical trial code changed
after the reviewed commit. Exact pinned decoder excerpts with original line
numbers are included so the review can inspect the diagnosis within its
permitted checkout; full sources remain in scratch.
