# Original DAQ gain writer with the IsEn ordering correction.
source "$(dirname "${BASH_SOURCE[0]}")/../diagnostic-hook/setup.sh" || return
export UED_GAINMAP_CNF="$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/isen-ordering-test/epix_isen_test.py"
