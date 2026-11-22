#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

target=${1:-all}
if [[ $target == all ]]; then
    for path in core gpio uart; do
        bash "$PWD/verify_rtl.sh" "$path"
    done
    exit
fi
sources=(rtl/qec_core.v rtl/qec_priority.v rtl/qec_proximity.v rtl/qec_prune.v rtl/qec_prune_gates.v)
parameters=()
case "$target" in
    core) top=qec_core ;;
    gpio|uart)
        top=nano20k_$target
        sources+=(rtl/qec_packet.v rtl/reset_sync.v "rtl/$top.v")
        ;;
    *) echo "Usage: ./verify_rtl.sh [all|core|gpio|uart]" >&2; exit 2 ;;
esac
if [[ $target == uart ]]; then
    sources+=(rtl/uart.v)
    parameters+=(-GTIMEOUT_CYCLES=16384)
fi
dir="$PWD/.build/sim_$target"
mkdir -p "$dir"
export TMPDIR="$dir"
python3 gen_policy.py
verilator --cc --exe --build --top-module "$top" --Mdir "$dir" \
    -j "${QEC_SIM_JOBS:-2}" -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Irtl "${parameters[@]}" \
    "${sources[@]}" "$PWD/sim/$target.cpp" > "$dir/compile.log" 2>&1 || {
    cat "$dir/compile.log" >&2
    exit 1
}
vectors=${QEC_VECTOR_DIR:-.build/vectors}
if [[ $target == uart ]]; then
    "$dir/V$top" "$vectors/p0.005.bin"
else
    "$dir/V$top" "$vectors"/*.bin
fi
