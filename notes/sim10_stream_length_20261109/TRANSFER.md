# Continue on Perlmutter: controlled Sim10 cold staging comparison

The user asked to prepare a clone and transfer this task, then let their
Perlmutter agent implement and run the rest. Do not resume the SDF allocation.
SDF job44043617 was cancelled while PENDING on 2026-11-09 at16:05:24 Pacific;
it never obtained a node and produced no performance data.

## Objective and repository

First investigate one-BD stream count and input duration, before further BD,
batch-size or worker tuning. Run one BD/one GPU with 1 versus2 CUDA execution
streams and 10k versus100k events, two repeats each in reversed order, plus
small pilots. Use the SAME accepted100k Sim10 files for both event counts.

Prepared clone location:
`/global/homes/m/monarin/lcls2_worktree/benchmarks/sim10-stream-length-20261109`

Receiving branch: `work/perlmutter-sim10-stream-length`.
Source branch: `monarin/work/synthetic-mfx-scan`.
SDF benchmark implementation commit: `21f56de8884dc386d2d0ae65d159a795ed448614`.
Prior accepted scaling report: `notes/sim10_100k_bd_sweep_20261109/README.md`.
Task scripts and this transfer are in `notes/sim10_stream_length_20261109/`.
Use the clone HEAD as the transfer commit; review metadata distinguishes code
review from documentation-only closure commits. Existing Perlmutter checkouts
and their dirty files must remain untouched. Unrelated SDF untracked work was
not transferred.

Scratch task root:
`/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109`

Put builds, installs, generated data, compiler caches, logs and reviewer worktrees
under this root with unique attempt directories. No Perlmutter job is submitted
by this transfer. The clone contains source and provenance, not a new native build.

## Measured context and interpretation limits

SDF cold Weka,100k Sim10,2 streams/BD:1/2/3/4BD medians1.189/2.356/3.479/4.502GiB/s.
Earlier SDF1BD/1stream/10k was2.1884/2.1197GiB/s on another node and separate10k
files. None of these measurements isolates stream count, event count or node.
4.50GiB/s is the best tested SDF staging rate, not a measured hardware ceiling.

Perlmutter `/pscratch` is Lustre, verified by findmnt. This is a new within-site
controlled test; it cannot directly attribute the earlier SDF discrepancy.
Changing source run, CPU/GPU/platform, filesystem, CUDA/runtime versions also
precludes calling it a direct Weka reproduction. Do not label Lustre as Weka.
The same-file10k prefix tests duration, not physical10k-versus100k file layouts.

## Available local Jungfrau source (verified November9)

Preferred full source for locally creating Sim10:
`/pscratch/sd/p/psdatmgr/psdm/mfx/mfx101232725/xtc`
Experiment `mfx101232725`, run442. Five JF streams `s004` through `s008`,
32panels distributed5+6+7+7+7. Original full-run file sizes were listed/readable
under the current account; read-only source, do not alter/drop other users' cache.

An owned, previously validated raw10k prefix also exists:
`/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01/inputs`
The corresponding `inputs-ready.json` and `reference.json` are in its parent.
Five bigdata files total335606159580bytes (about312.56GiB), plus small SMD files.

See `perlmutter-dataset.json` for exact paths, sizes, panel order and hashes.
Current checks: all five bigdata file sizes match prior receipts; Configure
headers readable; all five SMD hashes match the prior reference and complete
(timestamp,service) sequences align. SMD contains16263events, but the owned
bigdata copies are only validated for the first10000! Never request100k raw
events from this prefix or treat its SMD as a complete owned-run index. Current
bigdata full hashes and Configure shapes were not independently revalidated.
The full original source is preferable when fetching closing transition templates.

No Sim10 fixture was found in the inspected task/staging areas; the exact SDF
mfx101555026 and prjdatpsana paths were absent from the checked psdatmgr tree.
This is not an exhaustive global filesystem search. Do not claim a ready100k
Sim10 dataset or silently benchmark raw32MiB events in its place.

## First concrete implementation steps

1. Recheck source Configure/Names, JF schema/raw uint16 shape, physical panel
   order, timestamps/transitions and SMD offsets. Derive mappings from source;
   do not reuse SDF's hardcoded s005..s009 mapping or experiment/run IDs.
2. Adapt the existing sample-every-tenth-uint16 writer to this source. Start
   with a tiny complete fixture and DataSource acceptance, then produce10k.
   Retain five physical JF streams and32panel descriptors. Per-panel seed is
   raw.reshape(-1)[::10],52429uint16 words for512x1024; metadata preserves the
   expansion factor and original shape. Sim10 is not a real compressor.
3. Adapt the reviewed repetition writer to make100k physical events from10k
   seed events, unique outer timestamps, synchronized transition schedule and
   correct SMD offsets. No artificial multi-step scan is needed. Repeated
   payloads are intentional. Keep source bytes immutable and verify receipts.
   Recompute actual event/dgram sizes; do not assume SDF's3358200B/event exactly.
   Expect roughly31GiB10k plus313GiB100k seed fixtures, excluding source/builds.
4. Port the controlled runner and independent audit; only then run GPU pilots
   and the2x2 matrix. Continue implement-review-loop for implementations/fixes.
   Review code and native results separately; no throughput assertions in tests.

Writer starting points:
- `notes/mfx_expand10_10k_20261106/{fixture.py,verify.py,campaign.py}`
- `notes/mfx_expand10_stage2_20261105/{fixture.py,adapter.py}`
- `notes/mfx_expand10_100k_20261109/{repeat.py,validate_repeat.py,test_repeat.py}`
- `notes/synthetic_mfx_stage3_20261104/xtc_fixture.py`

These contain SDF paths, source manifests, calibration receipts and stream
assumptions; they are references, not Perlmutter-ready commands. This staging
benchmark uses `skip_calib_load='all'` and no calibration-consuming inputs, so
it requires no calibration DB deployment. Decouple generation provenance from
unavailable SDF calibration snapshots for this scope; do not substitute foreign
calibration as though accepted. Expansion/calibration/cube correctness is later.

## Runtime and scheduler adaptation

An existing Perlmutter frozen B install is available:
`/pscratch/sd/m/monarin/psana2-gpu/pr187-pr208-perlmutter-20261026/attempt01/install-B`
Environment candidate:
`/pscratch/sd/m/monarin/psana2-gpu/perlmutter-kvikio2610-20261015/attempt01/env`
Eight Python module hashes in `perlmutter-runtime.json` match the SDF baseline.
This is source identity evidence, not native/GPU/MPI ABI or complete-runtime
validation. Recheck imports, CuPy/KvikIO versions and CUDA/Cray MPICH on compute.
Existing bootstrap/build scripts in the older attempt are references only:
do not write its tmp, cache or install paths. Use new task-owned paths.

Use skills psana-perlmutter, psana2-gpu, gpu-cuda-python and implement-review-loop.
Build on login if needed with the checkout's build_psana.sh, explicit scratch
build/install prefixes; inspect --help for supported options. Source
~/goodstuffs/bashrc and ~/psana-nersc/activate_psana_build_env.sh as documented.
Never overwrite another checkout's native install. No build was needed for
preparing this source-only transfer.

Replace OpenMPI OMPI_COMM_WORLD_RANK in runner.py with a verified Perlmutter
rank mechanism (e.g. SLURM_PROCID under srun), preserving CUDA mask setup BEFORE
MPI/psana/CuPy imports. Use srun with verified Cray MPI support, explicit tasks
(SMD0+EB+1BD), GPU allocation/binding, one physical GPU UUID for the BD and empty
CUDA masks for helpers. Maintain constant CPU affinity/NUMA policy across cases;
do not copy SDF CPU IDs or topology assumptions blindly.

Fixed intended controls: batch100, SMD1000, bulk524288B,6GiB GPU budget,
KvikIO compatON,8workers,1MiB tasks,16MiB bounce, automatic direct I/O OFF,
PS_FAKESTEP_FLAG=0, PS_PARALLEL=mpi,1EB,0SRV. Record actual GPU memory size/model;
recompute admission/subbatch counts on the new input. Task receives field
metadata only: framework read/walk/descriptor preparation, no dense gather,
expansion, calibration, cube or payload D2H.

Use account lcls_g and premium QoS per standing preference; check Iris personal
and project charged balances before submission and alert if either <20%.
This transfer does not launch jobs; the receiving agent owns subsequent work.

## Cold methodology and acceptance

Use owner-controlled generated files on Lustre. Port the SDF-specific path/mount
allowlists explicitly. fsync/FADV_DONTNEED then mincore after DataSource setup
must establish zero resident input pages for local-cold labels. Never perform
global cache drops or fall back silently to NVMe. Lustre client/server caches
remain distinct; if eviction does not work, report it rather than label cold.
Inspect available Lustre llite/lctl statistics and Slingshot counters on compute;
Linux Ethernet sysfs RX counters alone may miss KFI/Lustre data traffic. Establish
counter scope/units and timestamp brackets before claiming network-served bytes.
If appropriate counters are inaccessible, retain local-cache evidence and state
that remote-traffic attribution is unverified.

Require exact ordered event identities, sizes/read counts, callbacks and
subbatch histogram, requested stream count exercised, proper GPU/CPU routing,
correct peer count, no pressure drains, complete resource retirement and cleanup.
Audit measured rates from actual bytes and loop time, not assumed seed pixels.
Record first/last wall-clock times and one job identity across cases if possible.

Carry code-review findings from review-01: fixed startup/fill/drain remains timed;
short cases can be more affected; reversed order does not equalize backend cache;
later10k cases follow full-file reads. Report individual repeats and available
traffic/payload ratios, not only medians. Preserve exact audit/test commands and
artifact roots. Original SDF24native-receipt tests have NOT run because the job
was cancelled. Port/review them for new receipts before claiming acceptance.
