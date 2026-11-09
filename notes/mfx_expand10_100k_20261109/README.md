# 100,000-event sim10 fixture

Preparation in progress. User-authorized enlargement of the accepted 10k sim10
fixture, retaining the **512 KiB bulk target** for subsequent performance tests.
See the [deferred small-datagram scheduling note](../../psana/psana/gpu/docs/proposals/small_dgram_scheduling_followup.md).

Repeat the accepted 10,000-event sequence ten times into new physical files,
retaining all ten streams (five JF plus five ancillary) and all 32 panel identities.
There are 100,000 unique outer event timestamps but only 10,000 distinct source
payload events. Shift outer datagram timestamps with nanosecond arithmetic;
rewrite SMD offsets/sizes; preserve Configure, detector fields and payload bytes.
Internal detector timestamps/frame counters repeat intentionally. Keep one native
step with no inserted step boundaries. Retain source experiment/run identity
(mfx101555026/run 10) with explicit dir=; no public-run registration or calibration
DB writes. Full-shape frozen calibration remains applicable by panel identity.

Intended root: /sdf/data/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_100k
Evidence root: same parent, sample10_100k_evidence_20261109.
JF event datagram bytes: 335,820,000,000 bytes (312.757 GiB). Full 20-file footprint roughly
351 GiB including ancillary data and SMD, with exact preflight size to follow.
This is a real on-disk repetition, not a loop over the old file or a sparse alias.
The run duration/file working set grows 10x; batch 100 still holds about 320 MiB of
JF input. No new GPU performance result is claimed by generating this fixture.

Acceptance will include a tiny repeated pilot with a partial final cycle,
unit tests across timestamp second boundaries, complete 20-file readback/hashes,
unchanged payloads against the accepted base, all SMD references and aligned
strictly ordered timelines, and serial DataSource traversal of all 100k events.
The fixture is accepted only when its evidence status is complete/accepted and
independent review passes. Existing fixtures will not be overwritten.
