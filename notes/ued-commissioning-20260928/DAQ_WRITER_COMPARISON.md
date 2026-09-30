# DAQ-writer and trigger-rate comparison — prepared 2026-09-29

Status: software prepared and validated offline. **No hardware test started.**
Mona powered off the detector after the September 28 captures and requested
implementation only. The historical production timing settings still need
to be recovered before choosing a production-rate comparison.

## Implementation

`capture_standalone.py --gain-writer daq` imports the actual
`psdaq.configdb.epixquad1kfps_config._apply_raw_pixel_map_program` function,
with a source-path/hash check to prevent silently using another installed
checkout. `--gain-writer standalone` preserves the original writer and remains
the default. Both use the same baseline and capture settings. The adapter sets
ASIC enable/IsEn and mode trbits, calls the helper, and attempts ASIC cleanup
even if enabling or writing fails. The DAQ configuration source is unchanged.
This exercises its map helper, not its complete Configure sequence.

The map passed to DAQ contains register codes 0xc/0x8, not the boolean expected
gain-bit mask. DAQ prepares all ASICs, then fills and updates one ASIC at a time,
using each ASIC's most common value as background. Standalone fills all
backgrounds before writing selected pixels. DAQ rejects selected ghost rows
that cannot be represented in the raw `(4,352,384)` map.

Each record and mismatch export identifies the writer source/function/hash,
requested rate and camera trigger readback. Source/input/software records and
the existing detector identity checks are retained.

`--rate-sequence-hz` loads the baseline and writes the map **once** in one
hardware session, then captures each rate into its own `phaseNN/` directory.
Each phase quiesces/drains/stops acquisition; the next phase restarts the PCIe
event builder and triggering without another baseline load or map write.
Failure stops the sequence and preserves partial evidence. Camera auto-trigger
rate is the varied setting; PCIe timing, delays and event-builder timeout stay
at their recorded profile values. The sequence is not a continuous acquisition
through a rate change and does not reproduce production dual-trigger routing.

## Pending hardware procedure

1. After the detector is available again, perform the usual identity/ADC and
   ownership checks. Use the pinned test bundle and current UED BEAM baseline.
2. At 10 Hz, compare standalone and DAQ writers with the exact expand2 map in
   MapFML and MapFHL. Use three independent programming attempts per combination
   (12 captures), interleaving the two writers. Assign new attempt numbers;
   the last completed attempt is 06. Record 51 images per attempt.
3. Recover the historical run and readout trigger settings. For each writer,
   test 10 Hz → verified comparison rate → 10 Hz using one map write and 51
   images per phase. Do not assume 360 Hz or a run-clock rate is the historical
   recorded-image rate. Match mode/map across comparisons and repeat as needed.
4. If both comparisons remain clean, prepare a separate external-trigger/full
   Configure comparison. The internal-rate selector does not implement it.

Run `review_gain_map.py ATTEMPT NEW_REVIEW_DIR` for single captures or supply
`ATTEMPT/phaseNN` for each rate phase. Keep all frames in the source files and
full mismatch exports. Exclude the first full image of each capture/phase only
in the separate assessment. Use the production module's 352-pixel persistent
list and report overlap with the 43 historical FP coordinates explicitly.
Do not expand the persistent list to hide newly observed mismatches.

The following command is an **offline preview**, using the recorded environment
(`SUBMODULEDIR`, `PYTHONPATH`, headless Qt, and DAQ Python):

```bash
python notes/ued-commissioning-20260928/run_once.py \
  --attempt 9001 --mode MapFML --gain-writer daq \
  --selected-map notes/ued-commissioning-20260928/map-inputs/expand2.npy \
  --frames 51 --rate-hz 10 --watchdog-seconds 300 --dry-run
```

Choose an unused attempt number; previews also preserve artifacts and refuse
overwrites. Both the capture and wrapper dry-run paths avoid hardware access,
including restoration. For a future rate sequence, replace `--rate-hz 10`
with `--rate-sequence-hz 10 VERIFIED_RATE 10` after establishing that rate.
The frame target, timeout and byte limit apply separately to each phase;
the wrapper watchdog covers the whole process, including provenance/analysis.

## Offline validation

24 unit/integration checks passed with DAQ Python 3.9 / Rogue 6.1.3 and the
pinned test bundle. They cover the existing native Rogue file writer/reader,
all five gain modes, ASIC rotations, all banks and boundary rows, DAQ majority
background selection, source-path verification, ASIC failure cleanup,
single-map-write rate sequencing and failure abort, partial-file accounting,
invalid rates and dry-run hardware isolation. See [offline log](daq-writer-offline-tests.log).

DAQ register-model checks compile the unchanged helper from the checked-in
source with hardware imports excluded; production uses a normal module import.
Separately, the actual DAQ module imported and resolved to this checkout with
`pyrogue.Root.start` blocked. No detector or PCIe root was started. These checks
validate software behavior, not hardware equivalence or absence of FPs.
