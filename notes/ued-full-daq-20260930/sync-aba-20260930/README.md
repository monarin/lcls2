# Standalone SYNC False / True / False comparison

September 30, 2026, 19:39–19:43 PDT on `drp-ued-cmp003`. The user explicitly
authorized this test with the detector on and DAQ stopped. Preflight found only
the monitor IOC and CA repeater; no DRP, devGui, or standalone capture process.

**Enabling `AcqCore.AsicSyncInjEn` alone did not reproduce the 43 extra FP
pixels seen in full DAQ run 3. All three captures are clean outside the existing
known-bad list after the first frame.**

| Attempt | AsicSyncInjEn | Full / assessed frames | FP outside known-bad | FN outside known-bad | Known-bad FN per assessed frame |
|---|---|---|---|---|---|
| 14, first control | False | 87 / 86 | 0 | 0 | 68 |
| 15, DAQ-matching SYNC | True | 87 / 86 | 0 | 0 | 68 |
| 16, return control | False | 87 / 86 | 0 | 0 | 68 |

None of the 43 DAQ FP coordinates appears in any assessed frame. Each capture
requested 51 frames; in-flight/drain frames bring the saved total to 87.
The same 68 FN pixels occur in every assessed frame (5848 occurrences per
capture), all within the established 352-pixel known-bad list.

## Controlled conditions and verification

Each attempt starts a new hardware session, loads the same baseline, applies
full readout clock period 3, explicitly writes the requested SYNC value, then
programs the same expand2 MapFHL map with the actual DAQ raw-map writer.
The setting is read back before and after map programming; both match in every
attempt. All use the intended September test bundle and source checkout.

Measured trigger source is 3, internal auto period 434028 ticks
(359.99981568 Hz). Full/half readout periods are 3/1; ADC pipeline delay is 68.
This changes the SYNC selection, not `AsicSyncInjDly` or other pulser settings.
The map source is `../../ued-commissioning-20260928/map-inputs/expand2.npy`;
its recorded hash is the same in all captures.

Full saved camera/PCIe YAML comparisons establish:

- Attempt 14 versus 15: exactly one camera field differs,
  `AcqCore.AsicSyncInjEn=False → True`; no PCIe fields differ.
- Attempt 14 versus 16: all camera and PCIe fields match.

Snapshots use the same prepared/read-first and armed/cached stages as
[attempt 13](../spy-comparison-1842/README.md). Disabled-device fields can
remain cached; SaveConfig does not read pixel SRAM. The gain-bit analysis is
independent evidence of the observed image response.

## Integrity, startup frames, and limits

Each native file passes length/hash checks. Image sequences are consecutive:
3669476–3669562, 3669563–3669649, and 3669650–3669736. All record error bytes
are zero; additional 48-byte records are timing-only. ADC status is done/pass,
camera line-buffer errors are zero, and capture cleanup errors are empty.
Recorded payload sizes, sequence checks, and counter snapshots are preserved
in [results.json](results.json).

The first frame is retained but excluded by the existing assessment policy.
Its FP/FN counts were 237/22158, 0/68, and 124/22044 respectively. Thus the
True attempt also happened to have a clean first frame outside known-bad;
this single observation is not enough to establish a startup effect.

These are short captures at the verified internal trigger period. Host image
arrival estimates are 227–232 Hz including transport effects, so they do not
establish sustained 360-Hz throughput. The test shows that the SYNC setting
alone is insufficient to reproduce the persistent DAQ FP pattern under these
standalone conditions. It does not rule out interaction with external timing
or full DAQ Configure ordering. No additional register, ConfigDB, external
sequence, power, IOC, or Slurm experiment was performed.

## Final state and reproduction

All three wrapper capture and restore exit codes are zero. Each restoration
verifies readout disabled, trigger source 3, auto period 15625000 (10 Hz),
auto trigger and trigger enabled, with ADC done/pass. The last capture returns
SYNC to False. Hardware access ended after attempt 16 restoration at 19:43:47
PDT. PCIe remains in standalone configuration; normal DAQ start/Configure is
required before DAQ use.

The new `--asic-sync-inj-en 0|1` option is opt-in and the commissioning wrapper
passes it through. Its regression test checks write-before-map ordering,
readback failure abort, and cleanup. [28 offline tests passed](script-tests.log)
in the configured DAQ environment. These diagnostic source edits do not
require or change the already installed DAQ build.

[launch-record.txt](launch-record.txt) records the executed command sequence.
[archive_compare.py](archive_compare.py) regenerates this archival comparison
without hardware access. It verifies native data and YAML hashes, exact SYNC
readbacks, restore results, frame counts, and overlap with the DAQ run-3 CSV.
Per-attempt files include full YAMLs, software provenance, compressed capture
metadata, review JSON/CSV/NPZ, execution records, and console/restore logs.
The native capture files remain in the ignored `attempt14-mapfhl`,
`attempt15-mapfhl`, and `attempt16-mapfhl` commissioning directories, with
absolute paths and SHA256 hashes recorded in the archived metadata.
