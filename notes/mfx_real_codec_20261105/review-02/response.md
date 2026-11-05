# Review-loop closure

Claude passed implementation `ebd9c2d78e134bcc0648a552c4f1c7d51e703b92`.
MRC-001, MRC-002 and MRC-005 are closed. MRC-003 remains an accepted low-priority
deferral: structured failure journaling before reuse as a general producer.

MRC-004's remaining provenance-label ambiguity is corrected in this evidence-only
commit. The top-level `sha256` map was recomputed for every relative source path
from its Git blob at the reviewed implementation commit, and each blob was
compared with the working file. `trial.py` now carries the corrected counter's
hash, matching `trial_versions.reviewed_trial`. The original capture is retained
under `frozen_snapshot_sha256`, matching `trial_versions.frozen_full_trial`.
The latter produced `pilot01/report.json`; the initial one-event gate driver
was not separately frozen. No codec, bridge, trial or verifier code changed.

`size-distribution.json` adds report-hash-pinned arithmetic evidence for the
README's median and percentile claims. `source-archive.json` pins the archived
implementation snapshot. These evidence-only additions did not require another
implementation review round.

This pass covers the bounded trial and its representation of the results.
The upstream GPU decoder's high-magnitude/nonfinite failure remains unresolved;
there is no production codec acceptance, public XTC fixture, or new calibration
deployment. The complete fixture stays in private scratch.
