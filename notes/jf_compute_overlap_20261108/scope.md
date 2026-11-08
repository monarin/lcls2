# Scope
Keep the prior 64 original MFX events, actual 1 BD/1 GPU MPI topology, depth 2,
16-event EB batches, 4-event GPU subbatches, 512 KiB read target and fixed runtime.
Repeat the same calibration twelve times into one batch-owned output, then
accumulate/count once. No framework changes, new data or large benchmark.
Retain the separately labeled single-pass resident control from the baseline.
Acceptance: exact timestamps, 16 callbacks, 4+4+4+4 splits, 32/32 bank counts,
zero admission drains, full first repeated calibrated image equals CPU-v3 hash.
Native Nsight report must show at least one steady adjacent cross-stream gather
CPU launch completes before preceding cube sum ends. Report actual kernel overlap
separately, without representative throughput or universal concurrency claims.
Use restricted independent Claude review, maximum three rounds. Branch is
work/synthetic-mfx-scan; existing user authorization covers commits and push.
