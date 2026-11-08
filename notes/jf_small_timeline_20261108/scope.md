# Small real-JF timeline

One actual MPI BD and one GPU (three ranks: SMD0, EB, BD), pool depth 2 via
n_gpu_streams=2. First 64 original MFX run-10 events, communication batch 16,
target GPU sub-batch 4 through the existing byte admission policy (1.25 GiB
framework budget; no new sub-batch knob). Two cube bins per stream keep state
small. 512 KiB read target keeps all five JF datagrams large. No core fixes.

Deliver a native .nsys-rep on project scratch, exported SQLite, compact settings,
observed splits, event/bin/pixel checks and timeline viewing guidance. Label
read issue/wait, submit, calibration and cube calls with scoped NVTX wrappers.
A separate two-batch resident control reuses eight original raw images and the
same calibration/cube kernels, bypassing I/O. Measure overlap instead of claiming
that batch sizing guarantees it. No rate sweep or sustained performance claim.
Keep existing frozen runtime and constants. Review implementation/evidence with
restricted Claude, up to three rounds; commit/push scoped work to existing fork.
