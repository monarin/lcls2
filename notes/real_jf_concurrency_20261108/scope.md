# Real-JF 1,000-event concurrency comparison

User requested the analogous missing-concurrency test on real JF, limited to
1,000 events. Read the first 1,000 events of public mfx101555026 run 10 in place;
do not copy/rewrite the full source run or deploy constants. Use the existing
frozen MFX constants, runtime, 100-bin cube, batch 7, 4 GiB framework budget and
100-event warmup. Dense jungfrau.raw gather replaces sim10 expansion; reuse the
existing calibration and cube kernels. At 512 KiB all 5–7 MiB JF datagrams are
large; the 8 MiB control intentionally makes them small (also small at that
threshold in sim10). Framework default 1 MiB also classifies real JF as large.

Prepare CPU timestamp identity and full raw/calibrated SHA256 references at
indices 0,500,999 in one 1,000-event CPU pass. Run a separate 1,000-event GPU
acceptance with those sample comparisons and all event/bin/presence checks;
no timed per-event D2H in clean runs. Three fresh-process clean repeats with
one/two streams at 512 KiB, one two-stream 8 MiB control, and separate 1,000-event
native traces at both targets. Every DataSource is capped at 1,000 events.
Check kernel device overlap and correlated launch gaps, not just pressure counts.

No core/runtime/kernel changes, 10k campaign or concurrency fix. Preserve results
on project scratch, compact evidence and code in this branch. Use independent
restricted Claude review, max three rounds, response/re-review as needed. Commit
and push only scoped changes to the already authorized fork branch.
