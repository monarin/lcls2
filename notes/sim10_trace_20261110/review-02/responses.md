Implementation review PASS on ec0941a31222045e31e5a8ea8fb49c02c4a63a28;
no blocking findings. No code changes after this review.
T1: address operationally before submission: verify the exact nsys executable
and save its --version. Same executable was already checked on Perlmutter.
T2: run import-only install('Integrated-on') against frozen install-B before
submission and verify InputGroupPool methods plus CuPy profiler/NVTX APIs.
T3: deferred nonblocking timeout/cleanup reporting hardening; the job time limit
bounds work, per-case command/log/progress receipts persist, and actual Slurm
exits must be checked. Absence of complete.json never implies acceptance.
T4: deferred cosmetic missing-counter error handling; KeyError fails closed.
No GPU results are approved by this implementation review.

T1 completed: exact executable version 2026.2.1.210 saved remotely.
T2 login preflight attempted twice; missing libfabric.so.1 blocks importing psana on the login node even in serial mode. Deferred to the native timing case, which fails closed if timer installation fails. No runtime settings changed.
Job 61072191 submitted with reviewed implementation ec0941a31.

T2 closed by native evidence: both timing cases and both Nsight cases successfully installed the timer against install-B; all six cases passed.
T3 actual job and six Slurm steps completed 0:0 within 5:43.
