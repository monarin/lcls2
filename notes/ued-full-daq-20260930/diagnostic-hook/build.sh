#!/bin/bash
# Isolated software build only; never starts DAQ or opens a device.
set -eo pipefail
umask 022
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.."
source setup_env_daq.sh
export SUBMODULEDIR=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
export BUILDDIR="$PWD/builddir-ued-diag-20260930"
export TESTRELDIR="$PWD/install-ued-diag-20260930"
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$BUILDDIR/uv-cache"
rm -f "$TESTRELDIR/.ued-diag-build-complete"
./build_all.sh -d -c -j "${UED_BUILD_JOBS:-4}"
uv pip install ./psdaq --no-compile --no-deps --no-build-isolation --no-index --prefix "$TESTRELDIR"
test -x "$TESTRELDIR/bin/drp"
git rev-parse HEAD > "$TESTRELDIR/.ued-diag-build-complete"
