Extend the 13 existing cdb_prjdatpsana calibration documents from validity 1–1
to 1–2, explicitly authorized by Mona. Change run_end only, retaining document
IDs, payload IDs and all other metadata. Preserve original metadata and a
per-write intent/response journal. No shared/source DB, XTC, permission, or
psana core changes. No run-2 data generation in this task.

Acceptance: exact before/after metadata diff, identical IDs and frozen payload
hashes through normal psana lookup for both runs (23 selections each), and no
experiment-record eligibility for run 3. Run 3 may still use shared fallback.
The old comment saying pilot/run 1 is preserved as historical deployment
provenance; the run/run_end fields control selection.

Review with Claude in a detached checkout at an exact committed SHA, maximum
three rounds. Review the update script and executed evidence; never replay the
writes for review. Historical Stage 1–4 artifacts remain immutable. Stage 4's
old serial checker expects run_end=1 and is not a current-validity test; use
extend.py without --apply in a fresh output directory for read-only checks.
