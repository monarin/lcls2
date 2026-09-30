# Source in a fresh Bash shell; selects software only, without starting DAQ.
export UED_GAINMAP_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! test -f "$UED_GAINMAP_SRC/install-ued-gainmap-20260930/.ued-build-complete"; then
    echo 'The isolated UED build is not installed yet.' >&2
    return 1
fi
source "$UED_GAINMAP_SRC/setup_env_daq.sh" || return
export TESTRELDIR="$UED_GAINMAP_SRC/install-ued-gainmap-20260930"
export DAQ_TESTRELDIR="$TESTRELDIR"
export DAQ_CONDA_PREFIX="$CONDA_PREFIX"
export DAQ_CONDA_DEFAULT_ENV="$CONDA_DEFAULT_ENV"
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
# Remove the old worktree install entry added by the common setup.
PATH="${PATH#"$UED_GAINMAP_SRC/install/bin:"}"
export PATH="$TESTRELDIR/bin:$PATH"
export PYTHONPATH="$TESTRELDIR/lib/python3.9/site-packages"
export PYTHONDONTWRITEBYTECODE=1
export UED_GAINMAP_CNF="$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/epix_gainmap_test.py"
export UED_GAINMAP_MASK="$UED_GAINMAP_SRC/notes/ued-commissioning-20260928/map-inputs/expand2.npy"
hash -r
