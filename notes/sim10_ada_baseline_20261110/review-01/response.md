# Response to review 01

- ADA-01 fixed: ntasks3/cpus-per-task16; bash -n passes. Also select one UUID
  from the exclusive allocation's multi-GPU mask (historical Slurm AllocTRES
  confirms --exclusive reserves all GPUs despite a request for one).
- ADA-02 documented: retain bounded20min, disclose expected budget and that a
  timeout means incomplete scheduling evidence, never an accepted measurement.
- ADA-03 fixed: audit now verifies each runtime module path/hash against the
  frozen native/Python receipt; added negative evidence mutation tests.
- ADA-04 fixed: native source comparison includes psalg; build config hashes
  separately recorded. README distinguishes existing binary reuse from rebuild.
- ADA-05 documented/enforced: optional psexp.parallel_pread intentionally absent;
  activate() checks find_spec is None. Descriptor-staging does not use debugtools.
- BDS-02 partially hardened: owned GPU-test/MPI process groups receive TERM and
  are reaped (KILL after15s if needed) before input deletion. Two cleanup tests
  pass. Hard-kill during staging can still orphan inputs; README has recovery.
- ADA-06 documented: shared scratch tmp/compiler/pytest caches intentionally
  retained until review for diagnosis.
- ADA-07 documented: only JF bigdata cold-gated; SMD expected warm, backend caches
  uncontrolled. No cache-fraction inference from NIC ratios.

Reviewer limitation clarification: CPU unit run explicitly called
runtime.activate() then identity() for psana, dgram, eventbuilder, smdreader,
gpu_input_group, gpu_input_window, gpu_events, followed by pytest. All1023 passed.
Actual Slurm --test-only accepted the account/GRES request. GPU and performance
remain pending; no measured pass is claimed. Automatic audit added after campaign.
