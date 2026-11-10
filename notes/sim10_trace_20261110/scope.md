# Scope
Trace the reproducible 10k Sim10 one-BD/one-GPU 1-versus-2-stream staging slowdown
on Perlmutter. Base: accepted cc98b8440e00123bbe86a1d82e6be7986f28a447.
Reuse the frozen runtime, existing five JF files, 512KiB threshold, batch100,
6GiB budget and KvikIO ON/8workers/1MiBtask/16MiBbounce/directOFF.
No framework behavior changes, new dataset, calibration or cube work.

Run clean, host-timed and Nsight 10k pairs on one node, with local-page-cache
conditioning inherited unchanged. Preserve exact event/read/callback/retirement
checks. Reuse prior independent full-input hash validation with before/after
size/owner/inode/mtime comparisons; do not re-read the full 100k for this trace.
Capture one BD only with CUDA/NVTX/OSRT; no profiling helpers or added CUDA waits.
Reuse existing bulk_phase_timing instrumentation, whose exclusive host phases
partition the instrumented main thread (not worker/GPU time). Measure overhead
against clean runs. Inspect steady intervals and startup separately. Remote
traffic remains unverified. Save viewable reports and concise evidence.

Independent Claude implementation/results review, at most three rounds.
Jobs use lcls_g premium, bounded single-node allocation; check Iris balance.
