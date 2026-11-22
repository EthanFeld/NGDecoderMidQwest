#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

interface=${1:-gpio}
profile=${2:-default}
case "$interface:$profile" in
    gpio:default|uart:default|uart:low-defer|uart:fast) ;;
    *) echo "Usage: ./build.sh [gpio|uart] [default|low-defer|fast]" >&2; exit 2 ;;
esac
top=nano20k_$interface
relative=".build/fpga/$interface"
if [[ $profile != default ]]; then
    relative=".build/fpga/$profile/$interface"
fi
dir="$PWD/$relative"
mkdir -p "$dir"
export TMPDIR="$dir"
frequency=27
if [[ $interface == gpio ]]; then
    frequency=${QEC_FREQ_MHZ:-50}
fi
seed=${QEC_ROUTE_SEED:-1}

core=rtl/qec_core.v
policy_include=
defines=
parameters=
if [[ $profile == low-defer ]]; then
    python3 gen_policy.py --policy data/policy-low-defer.json --rtl-output "$dir/policy.vh"
    core="$relative/qec_core.v"
    cp rtl/qec_core.v "$core"
    policy_include="-I $relative"
else
    python3 gen_policy.py
fi
if [[ $profile == fast ]]; then
    defines="-D QEC_FAST_CORE"
    parameters="chparam -set CLKS_PER_BIT 27 $top;"
fi
sources=(rtl/qec_priority.v rtl/qec_proximity.v rtl/qec_prune.v rtl/qec_prune_gates.v
         rtl/qec_packet.v rtl/reset_sync.v)
if [[ $interface == uart ]]; then
    sources+=(rtl/uart.v)
fi
sources+=("rtl/$top.v" "$core")
yosys -Q -ql "$dir/synthesis.log" \
    -p "read_verilog -sv $defines $policy_include -I rtl ${sources[*]}; $parameters synth_gowin -family gw2a -top $top -json $relative/synth.json"
nextpnr-himbaechel -q -l "$dir/route.log" \
    --device GW2AR-LV18QN88C8/I7 --vopt family=GW2A-18C --vopt "cst=board/$interface.cst" \
    --freq "$frequency" --seed "$seed" --json "$dir/synth.json" \
    --write "$dir/routed.json" --report "$dir/timing.json"
gowin_pack -d GW2A-18C -o "$dir/$top.fs" "$dir/routed.json"
echo "$dir/$top.fs"
