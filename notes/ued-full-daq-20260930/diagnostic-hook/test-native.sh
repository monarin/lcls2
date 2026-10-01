#!/bin/bash
# No hardware/runtime initialization; only the standalone diagnostic buffer class.
set -eo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.."
source notes/ued-full-daq-20260930/setup-test.sh
diag_test_dir=$(mktemp -d /tmp/ued-diag-tests.XXXXXX)
${CXX:-g++} -std=c++17 -O1 -g -Wall -Wextra -Werror -pthread \
    -fsanitize=address,undefined -I"$CONDA_PREFIX/include" \
    psdaq/drp/EpixQuadDiagnostic.cc psdaq/drp/EpixQuadDiagnosticTest.cc \
    -o "$diag_test_dir/test"
"$diag_test_dir/test" "$diag_test_dir"
