# UED diagnostic capture before disk writing

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

This opt-in hook copies the first 51 sampled ePixQuad events at two points:
the lane-0 DMA payload before BEBDetector parses it, and the formatted Dgram
after TEB damage propagation immediately before file-writer submission. It
does not start another DMA reader or change detector programming/timing.

Both copies own their bytes. Event buffers are preallocated and touched
before detector initialization: 204 MiB for 51 pairs, plus up to 16 MiB for
Configure. The hard limit is 64 pairs of at most 2 MiB per stage. Event-time
work is a bounded memory copy and matching lookup; output starts in a separate
thread at **EndRun**. This adds some memory traffic, so it is an instrumented
measurement, not a claim of zero impact on the DAQ.

The capture is **one-shot per DRP process**. Restart the diagnostic DRP/DAQ
for another sample; Pause/Resume or another run does not rearm it. Samples
are selected in worker arrival order and paired by pulse ID and timestamp;
the comparison sorts by timestamp and excludes the earliest sampled image
from gain assessment while retaining its counts and byte comparisons.

## Operator launch

Use a fresh Bash shell on ued-daq as uedopr, with the full DAQ stopped and
the detector available. This uses a separate install from the previous test.
It keeps the existing TEST alias and normal XTC2 destination/routing.

```bash
source /sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/notes/ued-full-daq-20260930/diagnostic-hook/setup.sh
mkdir -p "$UED_GAINMAP_DIAG_DIR"
command -v drp daqmgr
echo "$TESTRELDIR"
echo "$SUBMODULEDIR"
echo "$UED_GAINMAP_CNF"
QT_QPA_PLATFORM=offscreen python "$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/verify-install.py"
python "$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/diagnostic-hook/verify-launch.py"
daqmgr start "$UED_GAINMAP_CNF"
```

Expected install:
`/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq/install-ued-diag-20260930`.
Expected bundle: `lcls2_submodules_09282026_ued_epixquad_test1`.
The launch passes `-k epixDiagDir=...,epixDiagFrames=51` only to epixquad1kfps_0.
Other processes use the same isolated diagnostic install. Original setup-test.sh
and install-ued-gainmap-20260930 remain available for rollback.

In the DAQ GUI, select **TEST**, verify the intended expand2 MapFHL
configuration and **360 Hz**, enable XTC2 recording, then start a normal run.
This uses normal DAQ controls, not the standalone READY/groupca handshake.
Record for about 2–3 seconds, then **end the run normally**. Do not terminate
the DRP before its diagnostic output is saved.

The epix DRP log should show:

```text
EPIX DIAGNOSTIC prepared: .../epix-diag-PID-STAMP frames=51 ...
EPIX DIAGNOSTIC saved: .../epix-diag-PID-STAMP status="complete" paired=51
```

Files appear under `$UED_GAINMAP_DIAG_DIR/epix-diag-PID-STAMP/`:

- `configure.xtc2`: the complete Configure datagram for interpreting samples.
- `input-N.bin`: complete incoming batcher payload, not a Rogue file wrapper.
- `prewrite-N.xtc2`: exactly one Dgram as submitted to the XTC2 writer.
- `manifest.json`: event identity, lengths, writer selection, sample counts,
  finish reason and completion status. Published last, after closing the files.

Output uses a unique directory and never replaces a previous sample. An early
disconnect/destruction attempts a dump marked incomplete. Forced termination
can lose the in-memory sample. An I/O failure publishes no final manifest and
logs `EPIX DIAGNOSTIC save failed`; acquisition is not deliberately aborted.
The hook supports epixquad1kfps lane 0 without cube workers, and is off unless
epixDiagDir is provided. A bad path/count fails startup before camera init.

## Compare with recorded data

After EndRun and file finalization, identify the detector stream's XTC2 file
from the epix DRP log, as for run 3. Diagnostic files go to the diagnostic
directory; normal XTC2 files keep their usual experiment/run location.

On a compatible Rocky 9 host with the diagnostic setup sourced:

```bash
PS_PARALLEL=none python "$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/diagnostic-hook/compare.py" \
  "$UED_GAINMAP_DIAG_DIR/epix-diag-PID-STAMP" \
  "$UED_GAINMAP_DIAG_DIR/comparison-runN" \
  /path/from/drp/log/experiment-rNNNN-s001-c000.xtc2
```

Supply all detector-stream chunks for that run if chunked. Use a new output
directory. The comparator requires finalized files through EndRun and checks
that they do not change while read. It verifies the captured and disk Configure
bytes, detector identity, expand2 pixel map and trbits before assessing gain.

`comparison.json` records full Dgram byte equality, all raw pixel-word
differences, per-frame FP/FN, file hashes, and missing/damaged/unmatched events.
`gain-mismatches.csv` and `after-first-counts.npz` preserve coordinates/counts
at each stage. Incoming pulse ID/timestamp must match the hook manifest;
XTC2 Dgram does not store pulse ID, so disk pairing uses unique timestamps.
The comparison fails incomplete rather than calling absent events clean.

Interpretation:

- FPs already in incoming data: investigate acquisition/configuration and
  possible recording-load effects, rather than disk byte corruption.
- Incoming raw words differ from pre-write raw words: investigate conversion
  or intervening DRP buffer handling, and check the independent XTC field
  interpretation used by the comparator.
- Pre-write and disk raw words differ: investigate writer/buffer lifetime or
  file decoding. Exact Dgram equality additionally checks headers/aux data.

## Build and offline validation

`build.sh` builds in builddir-ued-diag-20260930 and installs to
install-ued-diag-20260930 on drp-ued-cmp003/Rocky 9 using daq_20250402_r9.
No DAQ start, detector access, PV write or ConfigDB mutation is part of the
build or tests.

The C++ buffer test in psdaq/drp/EpixQuadDiagnosticTest.cc passes with
`-fsanitize=address,undefined -Wall -Wextra -Werror -pthread -std=c++17`.
It covers concurrency, post-copy DMA/pebble mutation, bounds, no files before
EndRun, one-shot behavior, mismatched timestamps and interrupted capture.

`test_compare.py` replays three existing run-3 events with synthetic matching
incoming packets. Ten tests pass: byte/pixel equality while retaining the
known 43 FPs, injected input/conversion difference, injected disk corruption,
missing disk event, truncated file, wrong incoming identity, unselected event,
missing EndRun, duplicate timestamp and incomplete capture. This is offline validation, not a
new detector result. See compare-tests.log and the [build record](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/build-record.json), [build log](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/build.log.gz), and verification
logs for [drp-ued-cmp003](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/verify-detector-host.log) and [uedopr](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/verify-operator.log).
Run `bash notes/ued-full-daq-20260930/diagnostic-hook/test-native.sh`
on Rocky 9 to reproduce the sanitized native test.

The same ten replay tests also pass in the final installed DAQ environment
on drp-ued-cmp003; see [installed-runtime tests](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/compare-tests-installed.log).
That host does not mount the SDF XTC2 path, so the test used a byte-exact
three-event prefix on shared storage; [fixture provenance](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/replay-fixture.json)
records its source/hash. Set UED_DIAG_TEST_XTC to that fixture to replay there.
The final formatting rebuild is recorded in [build-final.log.gz](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/build-final.log.gz).
The output directory was prepared and checked writable as uedopr.

The first operator launch exposed a wrapper error: daqmgr executes configuration
code without `__file__`, unlike the `runpy` loader used in the initial check.
The wrapper now resolves its root from `UED_GAINMAP_SRC`, exported by setup.sh.
`verify-launch.py` constructs the installed daqmgr `Runner` as uedopr, checks
the TEST alias, process set, diagnostic arguments and DRP selection, and never
submits jobs. The corrected wrapper passed this check on ued-daq; see
[parser verification](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/verify-launch.log). No binary rebuild was needed.
The original build record preserves the pre-fix helper hashes; the current
wrapper hash is recorded separately in [launch fix](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-full-daq-20260930/diagnostic-hook/launch-fix.json).

The 21:59 launch also exposed missing `UED_GAINMAP_SRC` in daqstat's filtered
Slurm environment. Acquisition participants reached Ready for transitions;
the popup concerned only process-status monitoring. The daqstat row now
explicitly exports the source root and diagnostic destination, including custom
destinations. Parser verification also reloads the configuration with custom
shell variables removed and only the row's explicit environment restored.
This fix applies on the next daqstat launch; an already failed status thread
does not recover automatically. Acquisition need not restart for this popup.
