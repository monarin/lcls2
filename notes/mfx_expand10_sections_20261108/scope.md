# Timing attribution for one and two execution streams

User requests which sections dominate both baseline configurations. The clean
10k cases intentionally have no section instrumentation. Run two diagnostic
captures using the unchanged reviewed runner's existing profile mode, with
the same 10k input, batch 7, 100 bins, 4 GiB framework budget, 512 KiB read target,
four KvikIO fallback workers and one GPU. One process at a time, one then two
execution streams. No staging, new data, calibration deployment, kernel changes,
batch tuning or MPI scaling. Diagnostic rates do not replace clean baseline rates.

Acceptance: both native reports pass ordered event/count checks with 10k events,
1000 warmup/9000 measured, expected stream ownership and no admission drains.
Explain inclusive CPU spans, actual GPU kernel durations, nested/overlapping
intervals and uninstrumented time. Never sum inclusive CPU and GPU durations
into a wall-time breakdown. Retain native traces and reproducible analysis on
scratch, concise evidence in Git. Independent restricted Claude review; at most
three rounds. Existing branch work/synthetic-mfx-scan, base 2913e479c.
