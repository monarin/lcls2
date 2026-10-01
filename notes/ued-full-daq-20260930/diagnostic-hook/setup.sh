# Source in a fresh Bash shell on ued-daq; software selection only.
source "$(dirname "${BASH_SOURCE[0]}")/../setup-test.sh" || return
export TESTRELDIR="$UED_GAINMAP_SRC/install-ued-diag-20260930"
if ! test -f "$TESTRELDIR/.ued-diag-build-complete"; then
    echo 'Diagnostic build is not ready.' >&2
    return 1
fi
export DAQ_TESTRELDIR="$TESTRELDIR"
export PATH="$TESTRELDIR/bin:$PATH"
export PYTHONPATH="$TESTRELDIR/lib/python3.9/site-packages"
export UED_GAINMAP_CNF="$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/diagnostic-hook/epix_diagnostic.py"
export UED_GAINMAP_DIAG_DIR="${UED_GAINMAP_DIAG_DIR:-/sdf/home/u/uedopr/daq/gainmap-diagnostic-20260930}"
hash -r
