# Direct ASIC readback after UED run 4

All 44 FP locations requested as FH (pixel code 12) read back as code 8
(FL with trbit 1) on two passes. All eight affected ASICs had trbit 1 before
and after the read. This supports an actual retained pixel-configuration
discrepancy, consistent with the low-gain bits already present in incoming DMA.
It does not yet distinguish an incorrect initial write from a later overwrite.

The 44 targets came directly from run4-gainbits/after-first-mismatches.csv,
filtered to FP outside the known-bad baseline. Each was mapped through the
shared raw-to-ASIC layout and inverse-checked, including the bank-coded SACI
column address. All expected values were independently checked against expand2.
Only these addresses were read: no unaffected control pixels or full matrices.
The existing GetPixelBitmap addressing/read sequence was used with explicit
`WritePixelData.get(read=True)` hardware reads, rather than cached values.

The measurement took 1.731 seconds, including preflight and cleanup, outside
the Configure transition. DAQ state was unallocated before and after. The
existing camera server at localhost:9107 and PCIe server at localhost:9103 on
drp-ued-cmp003 were used through VirtualClient, with no new DMA reader,
standalone initialization, map load, pixel-data write, or ConfigDB mutation.

DAQ and Run timing MasterEnable, camera TrigEn/AutoTrigEn, and RdoutEn were
saved and temporarily disabled. For each affected ASIC the local enable and
IsEn were saved, configuration access enabled, and CmdPrepForRead /
PrepareMultiConfig plus row/column selectors used for the read. Cleanup issued
CmdPrepForRead and restored IsEn and local enable. The row/column selectors
are write-only, so their original hardware values cannot be claimed restored.
All trigger/readout gates were restored and read back equal to their originals;
ASIC cleanup and control restoration reported no errors.

## Evidence

- [Per-pixel CSV](measurement/pixels.csv): raw and ASIC coordinates, requested
  code, both hardware reads, trbit and comparison results.
- [Full measurement](measurement/readback.json): identity, states, timestamps,
  script hash, targets, cleanup and timing.
- [Summary and checksums](summary.json).
- [Execution log](execution.log), [preflight](preflight/readback.json), and
  [three passing offline tests](offline-tests.log).
- [Reader](../read-fp-pixels.py) and [offline tests](../test_read_fp_pixels.py).

The tests check the run-4 target selection and a specific rotated coordinate,
two hardware-style read passes with pixel writes forbidden, and cleanup after
an injected read error. All 44 rows in the measurement CSV were checked against
the saved JSON and requested map; both reads are 8 and every trbit is 1.

The next discriminating checks are known-correct FH/FL control addresses to
validate readback specificity, and the same 44 addresses after a clean standalone
write/capture. Neither additional hardware test was performed here. Preserve
this evidence before any new Configure or map rewrite.
