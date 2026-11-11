# Local hold-on-error implementation and review

Implement stages 1–5 of the November 10 staged plan in the existing
`features/piranha-routine-check` branch. Include the explicitly selected
successful-trial rehearsal mechanism, but do not execute it against hardware.
Preserve the default pilot lifecycle, pinned production source and safety gates.

Acceptance: coordinated retention of both roots; exact allocation GCP timeout
classification; immutable original evidence and bounded late evidence; cached
software diagnostics through existing servers; run-scoped release; finite hold
and cleanup deadlines including parent loss; offline regression, race, signal,
failure and software-only client tests. Failures remain failures; no retry.

No hardware opens, DAQ changes, production edits, remote deployments or pushes.
Historical hardware reservations do not authorize new hardware activity.
Use local implementation commits for detached independent review snapshots,
up to three rounds, with a 15-minute limit per invocation. Codex implements;
Claude reviews using the constrained review-loop helper. Missing reviewer or
runtime evidence is a limitation, not an approval. Do not override safety
restrictions if a prerequisite is unavailable.

Base: `6abb2627a` (resolve full SHA in each review context).
Design: `notes/tmo-piranha-hold-plan-20261110.md`.
