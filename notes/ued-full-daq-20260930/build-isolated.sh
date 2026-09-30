#!/bin/bash
# Compile/install only; does not connect to hardware or ConfigDB.
set -eo pipefail
umask 022
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.."
source setup_env_daq.sh
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export BUILDDIR="$PWD/builddir-ued-gainmap-20260930"
export TESTRELDIR="$PWD/install-ued-gainmap-20260930"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$BUILDDIR/uv-cache"
# A failed rebuild must not leave the previous completion marker in place.
rm -f "$TESTRELDIR/.ued-build-complete"
# build_all's root wheel hard-codes install/ and can package stale libraries.
# Meson installs the current native libraries and Python modules; install only
# the psdaq entry-point wheel afterwards. No dependency/environment updates.
./build_all.sh -d -c -j "${UED_BUILD_JOBS:-2}"
uv pip install ./psdaq --no-compile --no-deps --no-build-isolation \
    --no-index --prefix "$TESTRELDIR"
test -x "$TESTRELDIR/bin/drp"
test -x "$TESTRELDIR/bin/daqmgr"
cmp psdaq/psdaq/configdb/epixquad1kfps_config.py \
    "$TESTRELDIR/lib/python3.9/site-packages/psdaq/configdb/epixquad1kfps_config.py"
git rev-parse HEAD > "$TESTRELDIR/.ued-build-complete"
