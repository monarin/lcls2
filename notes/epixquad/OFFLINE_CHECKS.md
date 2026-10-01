# Offline checks

Run from the investigation checkout in a compatible psana environment (NumPy,
pytest and the compiled `psana.dgram` are required for XTC tests). No command
below opens hardware or changes DAQ, timing, power or ConfigDB.

```bash
source /sdf/home/m/monarin/lcls2/setup_env.sh
export PYTHONPATH="$PWD/psdaq:$PYTHONPATH"
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export PS_PARALLEL=none
export PYTHONDONTWRITEBYTECODE=1
export UED_DIAG_TEST_XTC=/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/xtc/uedc00107-r0003-s001-c000.xtc2
python -m pytest -q -p no:cacheprovider \
  psdaq/psdaq/tests/test_epixquad_standalone.py \
  psdaq/psdaq/tests/test_epixquad_capture_writers.py \
  psdaq/psdaq/tests/test_epixquad_external_payload.py \
  psdaq/psdaq/tests/test_epixquad_external_timing.py \
  psdaq/psdaq/tests/test_epixquad_standalone_daq_writer.py \
  notes/ued-full-daq-20260930/test_read_fp_pixels.py \
  notes/ued-full-daq-20260930/diagnostic-hook/test_compare.py \
  notes/ued-full-daq-20260930/isen-ordering-test/test_ordering.py
```

The tests use a committed gzip snapshot of the tested TEST configuration and
the committed expand2 mask. They do not require access to uedopr's home or a
live ConfigDB. Diagnostic replay needs the shared run-3 data; an unavailable
fixture is a skip, not a validated capture comparison.

The full-source Configure replay must check **IsEn at every matrix/pixel write**,
unchanged write-command order, final IsEn=False, unchanged standalone-port
sequence and the no-map-update path. Testing the inner helper alone misses the
expert-settings ordering defect.

To replay a separate minimal PR checkout:

```bash
UED_PR_SOURCE=/path/to/minimal-fix-checkout \
  python notes/ued-full-daq-20260930/closeout/test_pr_source.py
```

For the native diagnostic buffer, use a Rocky 9 host/toolchain with the required
headers and run `notes/ued-full-daq-20260930/diagnostic-hook/test-native.sh`.
It uses ASan/UBSan and checks copy ownership, bounds, concurrency, one-shot
behavior and interrupted capture. Its setup selects the existing diagnostic
environment; it does not start DAQ. Keep native validation distinct from Python
replay and live detector validation.

For a software-only profile preview, run the command in the repeat-test guide
with `--dry-run` for each of the three profiles. Use fresh temporary output
directories. Hardware acceptance cannot be inferred from a successful preview.

## Cleanup validation, October 1, 2026

[Validation record](validation.json): 69 tests passed, 3 skipped because this
psana environment lacks Rogue/PyRogue. All three profile previews and both
full-source Configure replays passed. The retained run-8 reviewer again found
1,084 full frames, no damage/incomplete images, and zero FP/FN outside known-bad
through EndRun. Native C++ and live detector tests were not rerun for this cleanup.
