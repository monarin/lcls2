# Source to select the isolated clean PR build; no DAQ state changes.
export UED_GAINMAP_SRC=/sdf/home/m/monarin/lcls2_worktree/ued-epixquad-isen-pr
export UED_GAINMAP_EVIDENCE_SRC=/sdf/home/m/monarin/lcls2_worktree/ued-epix10ka-rdsrv421-daq
source "$UED_GAINMAP_SRC/setup_env_daq.sh" || return
export TESTRELDIR="$UED_GAINMAP_SRC/install-ued-isen-pr"
test -f "$TESTRELDIR/.ued-isen-pr-build-complete" || { echo 'PR build is not ready.' >&2; return 1; }
export DAQ_TESTRELDIR="$TESTRELDIR"
export DAQ_CONDA_PREFIX="$CONDA_PREFIX"
export DAQ_CONDA_DEFAULT_ENV="$CONDA_DEFAULT_ENV"
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export PATH="$TESTRELDIR/bin:$PATH"
export PYTHONPATH="$TESTRELDIR/lib/python3.9/site-packages"
export PYTHONDONTWRITEBYTECODE=1
export UED_GAINMAP_CNF="$UED_GAINMAP_EVIDENCE_SRC/notes/ued-full-daq-20260930/closeout/epix_pr_test.py"
export UED_GAINMAP_MASK="$UED_GAINMAP_EVIDENCE_SRC/notes/ued-commissioning-20260928/map-inputs/expand2.npy"
hash -r
