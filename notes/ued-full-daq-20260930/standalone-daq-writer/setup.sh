# Source for the opt-in standalone gain writer inside DAQ Configure.
source "$(dirname "${BASH_SOURCE[0]}")/../diagnostic-hook/setup.sh" || return
export UED_GAINMAP_CNF="$UED_GAINMAP_SRC/notes/ued-full-daq-20260930/standalone-daq-writer/epix_standalone_writer.py"
