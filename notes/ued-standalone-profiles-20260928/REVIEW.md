# Standalone profile/capture review — 2026-09-28

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

Scope: shared host profiles and startup for the gain writer, optional GUI, and
new headless recorder. Reviewed CLI override precedence, host/card guards,
startup failure cleanup, stream callback/file-close synchronization, timeout
and interrupt cleanup, raw record integrity, expected gain maps, and provenance.

Validation used Python 3.9.21 and Rogue/PyRogue 6.1.3 from
`/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9` with the private
`lcls2_submodules_09282026_ued_epixquad_test1` bundle. Hardware packages were
imported from the candidate; no PCIe/detector roots were constructed.

- All 11 offline unittest cases passed, including the native Rogue writer and
  reader with synthetic short/error/full-image records, both full-image header
  lengths, and completed/failed/interrupted attempt metadata plus data checksums.
- Both profiles passed all three entry-point dry runs (six CLI invocations).
  Each headless preview recorded 106 repository entries, copied the available
  inputs, and produced the expected map without creating a capture file.
- Wrong-host, unvalidated-profile and mismatched-card tests prevented hardware
  startup. Startup failure and capture timeout tests verified cleanup.
- Review fixes: synchronized stream disarm before file close, rejected truncated
  trailing records, retained the primary acquisition error when cleanup also
  fails, and registered root cleanup before startup.

No blocking software finding remains after these corrections. Hardware use is
not validated: production UED remains gated by its unvalidated profile; bench
settings are carried over but the new capture lifecycle needs a hardware test.
The existing matrix-fill write algorithm is unchanged. This work neither
promotes nor publishes the candidate submodule bundle.

Reproduce the tests with the command in
[STANDALONE_PROFILES.md](../../psdaq/psdaq/debugtools/epixquad1kfps/STANDALONE_PROFILES.md).
