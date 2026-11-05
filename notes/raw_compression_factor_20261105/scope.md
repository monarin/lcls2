# Small raw compression-factor measurement

Measure CPU LZ4 and bitshuffle+LZ4 on the retained original MFX run-10 event 0,
all 32 Jungfrau panels partitioned into their original five DRP groups. Fixed
8 KiB blocks (4096 uint16 elements). Verify input against the previously
validated raw hashes and require lossless round trips for every pixel.
Report compressed bytes and raw/compressed ratio. No throughput, nvCOMP
bitstream equivalence, multi-event representativeness, calibration changes,
new public data, or DRP/psana implementation changes are claimed.

Use isolated scratch dependencies and JSON output, keeping only the small
measurement script/results in this branch. Follow the session's Codex/Claude
review loop, maximum three completed review rounds. No new unit tests are
needed for this bounded measurement; full raw round trips are the validation.
