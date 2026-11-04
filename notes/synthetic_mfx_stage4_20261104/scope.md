# Stage 4 pilot acceptance

Authorized: proceed to the next stage; use the implementation/review loop for
all implementations and fixes throughout this session. Base ac5609a27 on
work/synthetic-mfx-scan. Codex coordinates; Claude independently reviews exact
commits in a detached worktree, maximum three review rounds.

Validate the published prjdatpsana run-1 60-event fixture against the first
60 mfx101555026 run-10 source events. Check normal calibration selection IDs
and payload hashes against the frozen/deployed snapshot; detector identity,
segments, raw/supporting-detector and EPICS values; sampled calibrated JF
arrays, masks/status and representative assembled images. Run serial step/event
checks and CPU MPI aggregation with SMD/event batch sizes straddling 20,
including final tails and multiple EB/BD layouts. Record exact runtime and
PS_FAKESTEP_FLAG=0. Keep evidence in this directory and experiment results.

No XTC/calibration/permission changes, logbook registration, full-fixture
production, GPU/performance jobs, or psana core changes are planned. Failures
are investigated; changing psana core requires its broader validation and review.
Mona's independent timestamp output is supplementary evidence, not calibrated
or collaborator acceptance. Actual collaborator-login listing/opening and a
small calibrated DataSource read are required to fully close Stage 4; prepare
an executable handoff if that session is unavailable. Do not impersonate the
collaborator or send messages on the user's behalf.
