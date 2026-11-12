# Scope

Prepare and independently review a staged implementation plan for retaining the
standalone Piranha pilot's live Rogue roots after the issue-205 allocation GCP
exception, so an operator can inspect them through their existing ZMQ servers.
This task changes documentation only: no pilot implementation, hardware calls,
GUI attachment, deployment, or new campaign. Default cleanup remains unchanged
in the future implementation unless the inspection option is selected.

Deliverable: `notes/tmo-piranha-hold-plan-20261110.md`, with explicit defaults,
paired-process coordination, immutable failure evidence, bounded inspection and
cleanup, client semantics, offline tests, and later hardware acceptance stages.
Review at most three rounds. Use an isolated branch and detached review snapshots.
As in the previous Piranha loop, use an in-session independent reviewer instead
of an external Claude process: the previous external-review attempt was rejected
by automatic approval review for exporting source/log content. No new external
review attempt or external disclosure is authorized by this plan task.

Acceptance: claims trace to the unchanged reviewed pilot and deployed source;
the design handles the successful-peer cleanup race, parent/child watchdogs,
late UART traffic, diagnostic side effects and failed inspection; hardware
rehearsal is distinguished from reproducing a real timeout. A passing plan
review is not implementation verification or hardware approval.
