# ePixQuad ConfigDB and Configure compatibility / rollback

> Historical September 2026 record. For current repeat-test entry points and archived evidence, see [the repeat-test guide](../epixquad/README.md). Process IDs, running-state statements and launch paths below describe that session.

Reviewed September 30, 2026. No ConfigDB, hardware or DAQ writes were made.
Current BEAM still matches the September 28/30 snapshot byte-for-byte under
canonical JSON serialization: SHA256
`c0d91aa6e7485ab1a2a59eea7c8a823698aeca55c5cc32b9f876b7c66f95d73a`.
It has `user.pixel_map_raw`, no `user.pixel_map`, and gain_mode 1. Its config
algorithm is 3.0.0. See [fresh snapshot](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-config-rollback-20260930/beam-current.json.gz) and
[metadata](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-config-rollback-20260930/current-summary.json), read at 22:49 UTC.

## What changed

| Layer | Legacy representation | Current representation |
| --- | --- | --- |
| ConfigDB | `user.pixel_map`, ASIC layout `(16,178,192)` | `user.pixel_map_raw`, raw panel layout `(4,352,384)` |
| Configure panel config | `asicPixelConfig`, `(4,176,192)` per panel, four ASIC tiles | `asicPixelConfig` still exists, now `(352,384)` per panel |
| Configure panel config gain decoding | Older reader combines ASIC config and trbits | Added `cbitsConfig`, `(352,384)`, low nibble plus trbit in bit 4; trbit metadata uses panel order |
| Schema versions | Config 2.0.0 / panel raw 2.0.1 | Config 3.0.0 / panel raw 3.0.1 |

Key source changes are `8e6f55eb2` (April 21, raw DB map added), `df623c333`
(April 22, panel-oriented Configure metadata), `c4d361ba0` (April 24, legacy
DB field removed), `ba8670daa` (April 24, cbitsConfig added), and `b6780ee6d`
(May 1, schema versions bumped). `pixelArray` was not the field name here.
The map is stored in ConfigDB, and the Python configure code derives the
XTC Configure fields from it; these are related but distinct schemas.

The older code directly accesses `cfg['user']['pixel_map']` or
`ocfg['user']['pixel_map']`. [check_schema.py](check_schema.py) extracts only
the relevant pure functions/expressions from actual Git source, with hardware
imports excluded. At legacy commit `a91a52fdcfbdcc34d859e6e47564ee813b9121cd`,
today's snapshot reproduces:

- Map mode: `KeyError: 'user.pixel_map is required when user.gain_mode == 5'`.
- Fixed gain: `KeyError: 'pixel_map'`.

This establishes a concrete Configure-failure path, not a live recreation of
the historical DAQ crash. Choosing fixed gain alone does not avoid it.
Renaming or reshaping the raw map is not a correct migration: ASIC rotations,
tile ordering, type descriptors and the legacy extra rows must be handled.
Blindly keeping both fields would also let old/new writers consume different,
stale maps unless every edit maintained both representations atomically.

## Which rollback is compatible?

The deployed UED release
`/sdf/group/lcls/ds/ana/sw/conda2/rel/ued/lcls2_091026` **already uses the new
schema**, including pixel_map_raw, panel-shaped asicPixelConfig, cbitsConfig
and raw 3.0.1. Pure schema functions from its installed Python module and the
tested checkout produce identical arrays from today's ConfigDB snapshot.
See [offline-check.json](/sdf/scratch/users/m/monarin/ued-epixquad-gainmap-20260930/notes/ued-config-rollback-20260930/offline-check.json) for exact source hashes.
Thus no map-schema restoration is required merely to return to that release.
This statement does not validate its older ADC/startup behavior or make all
code/submodule combinations equivalent.

For a pre-migration release, restore that release's **complete known-good
detector configuration**, or use a preserved legacy alias, before Configure.
The exact older target release has not been supplied; no historical revision
key is certified for it here. Today's BEAM backup already lacks pixel_map and
is therefore not a legacy-schema backup.

Keep C++ DRP, installed Python configuration modules, detector submodule bundle
and psana reader versions paired. An old DRP binary plus new Python may label
the new panel metadata with the old raw algorithm, defeating reader version
dispatch. The new psana class `epix10ka_raw_3_0_1` uses cbitsConfig; the old
2.0.1 class expects ASIC-shaped configuration. Restoring ConfigDB affects
future Configure transitions, not existing XTC files. Retain a compatible
reader for files already recorded with the new schema.

## Prevention for the pending full-DAQ test

The [full-DAQ runbook](../ued-full-daq-20260930/README.md) now reuses **ued/TEST**.
It first saves TEST's complete current contents and the BEAM source entries,
then copies only `epixquad1kfps_0`, `timing_0`, and `trigger_0` from BEAM to TEST
and uploads maps only to TEST. Its prepared launch cnf selects TEST for control;
the GUI must also select TEST. The original TEST contents are restored after
the experiment without deleting an alias. TEST's unrelated detector entry is
preserved. BEAM remains untouched. The backup and three-entry copy completed
September 30 at 17:20 PDT: TEST key 1125 became 1405; BEAM remained 1402.
The original contents are saved in the operator's `alias-backups.json`, with
hashes in the [execution record](../ued-full-daq-20260930/test-prepared.json).
The operator's MapFHL upload was verified at TEST key 1406 on September 30,
17:27 PDT; [readback details](../ued-full-daq-20260930/mapfhl-readback.json)
confirm BEAM and the other TEST entries remain unchanged. Restoration remains
pending. Schema isolation does not
restore hardware settings: returning to a release still requires fresh
detector initialization and Configure.

Before starting an older release, check its actual installed configuration
module and C++/reader schema version against the selected alias. Perform that
check before hardware initialization rather than waiting for a Configure
exception. A longer-term code change could add a clear schema preflight error
or a tested migration layer; neither is implemented here.

## Restoring when actually needed

First End/Unconfigure and stop the test DAQ. Save the currently selected
detector configuration before restoring anything. Prefer a known-good ConfigDB
history key tied to the target release, or an exact exported JSON backup with
its `:types:`, algorithm version, detector identity, trbits and expert settings.
Restore only `epixquad1kfps_0`; do not roll unrelated detector entries backward.

The existing CLI supports a read-only preview followed by an explicit write:

```bash
export UED_CDB_URL=https://psdmint.sdf.slac.stanford.edu/ws-auth/configdb/ws
# OLD_KEY must be a reviewed, compatible revision, not an arbitrary old key.
configdb --url "$UED_CDB_URL" rollback ued/BEAM/epixquad1kfps_0 \
  --key "$OLD_KEY" --user uedopr
# After checking the complete proposed configuration:
configdb --url "$UED_CDB_URL" rollback ued/BEAM/epixquad1kfps_0 \
  --key "$OLD_KEY" --user uedopr --write
```

Without `--write`, rollback only reads/displays. With it, the old content is
written as the latest configuration for this device. Read it back and compare
to the chosen old configuration before restarting with the matched software.
The history request in this investigation timed out after retries; the saved
`history.json` value `[]` records that failure, **not an empty revision history**.
No legacy restore key was guessed.

If the history service remains unavailable, an exact JSON backup can be sent
directly to `configdb.modify_device(alias, backup_dict)` and verified by
`get_configuration`. Do not rebuild a legacy backup through today's
`epixquad_cdict`/gain-map uploader: those deliberately drop pixel_map and
produce the new schema. The fresh gzip backup here can restore today's raw
configuration, but cannot reconstruct the removed legacy representation.

No restore was performed because the live BEAM configuration is unchanged
and is compatible with the currently deployed UED schema.
