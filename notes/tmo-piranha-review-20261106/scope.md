# Review contract

Plan only for a standalone reproduction of lcls2 issue 205, per the 2026-11-06 transfer and Mona's request to use the review loop. Codex authors; Claude independently reviews; maximum three rounds, 900 seconds per invocation. No model override.

Deliverable: `notes/tmo-piranha-standalone-test-findings-2026-11-06.md`, a setup plan with verdict, exact call chain, four-level comparison, process lifecycle, safe logging, operational prerequisites, and acceptance criteria. Claims must distinguish source evidence, historical observations, and unvalidated hypotheses. Check proposed allocation JSON and paired-process barrier behavior against source. No runtime tests are required for this document-only task.

Allowed: read source, issue, configuration and existing logs; local plan/evidence/review files and isolated Git commits. Excluded: fix/harness implementation, reproduction execution, hardware opens/UART, production edits, process/job control, credential renewal, publication or messaging. Hardware execution requires a later explicit authorization; this plan grants none.

Base: f3936ee86b35dfc2a93a579c6fb08ac83731eea7. Task branch: notes/tmo-piranha-standalone-plan-20261106. Original checkout and its unrelated changes remain untouched apart from requested notes copies. Review uses a separate clean detached worktree.
