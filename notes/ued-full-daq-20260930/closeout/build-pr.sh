#!/bin/bash
# Build the clean PR worktree; no hardware or DAQ-control access.
set -eo pipefail
umask 022
cd /sdf/home/m/monarin/lcls2_worktree/ued-epixquad-isen-pr
source setup_env_daq.sh
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export BUILDDIR="$PWD/builddir-ued-isen-pr"
export TESTRELDIR="$PWD/install-ued-isen-pr"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$BUILDDIR/uv-cache"
./build_all.sh -d -c -j 4
uv pip install ./psdaq --no-compile --no-deps --no-build-isolation --no-index --prefix "$TESTRELDIR"
test -x "$TESTRELDIR/bin/drp"
test -x "$TESTRELDIR/bin/daqmgr"
cmp psdaq/psdaq/configdb/epixquad1kfps_config.py "$TESTRELDIR/lib/python3.9/site-packages/psdaq/configdb/epixquad1kfps_config.py"
git rev-parse HEAD > "$TESTRELDIR/.ued-isen-pr-build-complete"
