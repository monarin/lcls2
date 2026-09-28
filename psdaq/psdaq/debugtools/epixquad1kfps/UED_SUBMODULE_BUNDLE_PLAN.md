# UED standalone software bundle investigation

Investigated 2026-09-28. The private candidate has now been prepared; see
[preparation results](UED_SUBMODULE_BUNDLE_PREPARED.md). This construction plan
does not establish production deployment or hardware validation.

## What the bundle is

The existing superproject is https://github.com/slac-lcls/lcls2_submodules,
default branch `main`. Its Git submodule entries pin detector and PCIe
repositories to exact commits; each repository also pins nested dependencies.
`SUBMODULEDIR` selects a checkout at runtime through the psdaq `enable_*.py`
helpers. It is separate from the lcls2 source repository and the Conda environment.

Current upstream main is `f42780c93414182c652c620eaeb861bcf1b99333`
(2026-07-17). The GitHub API returned no tags and no releases on the investigation
date. There is no README or automated bundle-deployment recipe at this revision;
the root contains `.gitmodules`, the submodule entries, and `patch.sh`.

## Observed naming convention

The SDF deployment directory convention is `lcls2_submodules_MMDDYYYY`:

| Directory suffix | Directory date | Superproject commit |
| --- | --- | --- |
| `03122026` | March 12, 2026 | `f3268923ff37728f4ed9029f541c59206637cc4a` |
| `07142026` | July 14, 2026 | same `f3268923...` |
| `07202026` | July 20, 2026 | `f42780c93414182c652c620eaeb861bcf1b99333` |
| `09092026` | September 9, 2026 | same `f42780c9...` |

These dates identify deployment directories, not source commits or Git tags.
The directories are under `/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/`, owned by
`psrel:xs`. An older `/cds` setup reference uses `lcls2_submodules_170726`, so
legacy names are not globally consistent. Use the observed eight-digit SDF
convention for a new SDF bundle.

Proposed private candidate name: `lcls2_submodules_09282026_ued_epixquad_test1`.
The descriptive suffix is our proposal, not an established site convention.
Use the actual creation date if creating later. A published site bundle would
normally use `lcls2_submodules_MMDDYYYY`; its manifest must identify its exact
source and patches, regardless of directory name.

## Proposed pins

Base the candidate on superproject `f42780c93414182c652c620eaeb861bcf1b99333`.
Change only the `epix-quad-1kfps` gitlink initially.

| Component | Proposed commit | Reason |
| --- | --- | --- |
| `epix-quad-1kfps` | `44a7c1cb9ed1869e75b16e24bb610e7e3ee53698` | Contains merged PRs 19, 20, and 21, including `time.monotonic()` |
| `lcls2-pgp-pcie-apps` | `ccbe3de52fa7f6c09dac0216372121bcfbbc49e5` | Retain existing v4.2.0 software baseline |
| apps' `lcls2-pgp-fw-lib` | `d000e96efd5f3381b4614f7ae183c69011f5010c` | Existing nested pin, retained |
| All other top-level repositories | Existing superproject pins | Avoid unrelated upgrades |

All four inspected installed bundles still pin ePix to
`239cbd54083503efe7f41f452032534d9c989977`, so selecting September's directory
alone does not acquire Larry's fixes.

Compared with that old ePix pin, the proposed revision changes `Top.py`,
`SystemRegs.py`, `.gitignore`, `LICENSE.txt`, and the `ruckus` gitlink. It does
not change the pinned ePix SURF or axi-pcie-core dependencies:

- ePix SURF: `bd00ad1489a3e3d48ef150febb09716a02dba770`.
- ePix axi-pcie-core: `04d00e29f3365fb9a1287c30231882de443e9fba`.
- ePix RUCKUS advances to `937bc47fdd67c3fbd6f7eaa8254c002e42669b15`.

The PCIe apps repository has different SURF/axi-pcie-core pins. Preserve both
trees and check actual Python import origins: Python loads one module per
name, so helper import order matters. Do not flatten or independently update
these nested dependencies.

## Construction procedure

Use a new private directory. Prerequisites are Git, access to the submodule
repositories, and Git LFS if configured LFS filters require it. Git LFS was
not available in this session's default shell. Large firmware LFS objects are
not required for this Python-only use; where Git LFS is installed, set
`GIT_LFS_SKIP_SMUDGE=1` for the clone/update operations to avoid downloading
them. Do not build or load firmware as part of bundle preparation.

Example commands below are a recipe, not commands already executed:

```bash
ued_bundle="$HOME/lcls2_submodules_09282026_ued_epixquad_test1"
export GIT_LFS_SKIP_SMUDGE=1
git clone git@github.com:slac-lcls/lcls2_submodules.git "$ued_bundle"
git -C "$ued_bundle" switch -c ued-epixquad-20260928 \
    f42780c93414182c652c620eaeb861bcf1b99333
git -C "$ued_bundle" submodule update --init --recursive

git -C "$ued_bundle/epix-quad-1kfps" fetch origin main
git -C "$ued_bundle/epix-quad-1kfps" checkout --detach \
    44a7c1cb9ed1869e75b16e24bb610e7e3ee53698
git -C "$ued_bundle/epix-quad-1kfps" submodule sync --recursive
git -C "$ued_bundle/epix-quad-1kfps" submodule update --init --recursive
git -C "$ued_bundle" add epix-quad-1kfps
git -C "$ued_bundle" diff --cached --submodule=log
git -C "$ued_bundle" commit -m "Update ePix Quad software for UED standalone tests"
```

`update --init --recursive` reproduces recorded commits. Do not use
`update --remote`: that follows remote branches rather than the tested pins.
The complete bundle initializes all 14 top-level repositories. A smaller
standalone checkout can initialize only `epix-quad-1kfps` and
`lcls2-pgp-pcie-apps` recursively, but then the full `patch.sh` cannot be run
unchanged because it targets other repositories too.

## Compatibility patches are part of the deployment

The superproject's `patch.sh` modifies checked-out sources. Relevant here, it
comments out `rogue.Version.minVersion('6.4.0')` in the PCIe apps root. This
exact modification is present in the September installed bundle.

The current `daq_20250402_r9` interpreter was checked without hardware access:
Python 3.9.21, Rogue/PyRogue 6.1.3. Therefore an unpatched recursive clone is
not equivalent to the installed bundle. Bypassing a version guard does not
prove API or hardware compatibility.

For the full candidate, inspect the pinned `patch.sh`, apply it from the bundle
root, then inspect every resulting diff. Some commands rely on line numbers,
so verify their intended changes rather than relying on the exit code alone.
For a reduced two-repository checkout, create an explicit scoped patch instead.
Do not copy the installed directory's dirty state wholesale: it also includes
firmware binary differences and an unrelated changed submodule checkout.

Record, alongside the candidate:

- Superproject commit and `git submodule status --recursive`.
- `patch.sh` checksum plus recursive source diffs after patching.
- Python, Rogue/PyRogue versions, environment prefix, and imported module paths.
- lcls2 test-branch commit and later hardware identity/firmware observations.

## Validation and selection for UED

1. Import the modules in the same order as the actual standalone helpers;
   report `ePixQuad`, `lcls2_pgp_pcie_apps`, `surf`, `axipcie`, and timing-library
   `__file__` paths. Imports alone should not instantiate or start hardware roots.
2. Verify the new AsicMask model, ADC timeout, and failed-training PROM guard
   with software tests. Check standalone writer `--dry-run` and offline decoding.
3. Set `SUBMODULEDIR` after environment setup, since current setup scripts
   overwrite it. Ensure psdaq itself resolves to the updated checkout/install.
4. During the hardware session, verify actual board identity, firmware, lane,
   PGP version, ADC status, and full images before running the FP/FN sequence.

The existing axi-pcie-core code supports both `XilinxKcu1500` and
`XilinxVariumC1100`. The bundle is not named for a particular board or OS;
the environment/driver and runtime device/timing choices supply those details.
Actual production firmware and timing compatibility remain to be verified.
The ePix startup check requires PCIe firmware >= v4.2.0; the update corrects
its error-message text but does not introduce that threshold.

This bundle update does not by itself adapt the helpers' hard-coded XpmMini,
PGP4, or PROM-bypass choices. No ADC training, PROM writing, device opening,
DAQ changes, or deployment was performed for this investigation.

## References

- Superproject: https://github.com/slac-lcls/lcls2_submodules/tree/f42780c93414182c652c620eaeb861bcf1b99333
- Patch recipe: https://github.com/slac-lcls/lcls2_submodules/blob/f42780c93414182c652c620eaeb861bcf1b99333/patch.sh
- ePix changes: https://github.com/slaclab/epix-quad-1kfps/compare/239cbd54083503efe7f41f452032534d9c989977...44a7c1cb9ed1869e75b16e24bb610e7e3ee53698
- Git submodule semantics: https://git-scm.com/docs/git-submodule
