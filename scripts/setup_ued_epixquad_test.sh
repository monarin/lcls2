# Source this file to select the private UED test bundle and this psdaq checkout.
# It configures the shell only; it does not open or configure detector hardware.
ued_epixquad_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ued_epixquad_bundle=/sdf/home/m/monarin/lcls2_submodules_09282026_ued_epixquad_test1
if [[ ! -f "$ued_epixquad_bundle/epix-quad-1kfps/software/python/ePixQuad/Top.py" ]]; then
    echo "UED test bundle is missing: $ued_epixquad_bundle" >&2
    return 1
fi
source "$ued_epixquad_checkout/setup_env.sh" || return
export SUBMODULEDIR="$ued_epixquad_bundle"
export PYTHONPATH="$ued_epixquad_checkout/psdaq${PYTHONPATH:+:$PYTHONPATH}"
echo "UED test SUBMODULEDIR=$SUBMODULEDIR"
echo "psdaq source=$ued_epixquad_checkout/psdaq"
unset ued_epixquad_bundle ued_epixquad_checkout
