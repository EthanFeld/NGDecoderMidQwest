#include "Vqec_core.h"
#include "support.h"

#include <iostream>

static void load(Vqec_core& dut, const Vector& shot, size_t plane, uint16_t id) {
    for (int chunk = 0; chunk < 4; ++chunk) dut.in_plane[chunk] = 0;
    for (size_t j = 0; j < 14; ++j)
        dut.in_plane[j / 4] |= uint32_t(shot.syndrome[plane * 14 + j]) << ((j % 4) * 8);
    dut.in_valid = 1;
    dut.in_first = plane == 0;
    dut.in_last = plane == 15;
    dut.in_id = id;
}

static void shot(Vqec_core& dut, const Vector& data, uint16_t id, bool stall) {
    for (size_t i = 0; i < 16; ++i) {
        check(dut.in_ready, "Core dropped ready during input");
        if (stall && (i + id) % 5 == 0) {
            dut.in_valid = 0;
            tick(dut);
            check(!dut.out_valid, "Result during input stall");
        }
        load(dut, data, i, id);
        tick(dut);
        check(!dut.out_valid, "Premature result during input");
    }
    dut.in_valid = 0;
    for (int i = 0; !dut.out_valid && i < kMaxTail; ++i) {
        check(!dut.in_ready, "Ready asserted during decode");
        tick(dut);
    }
    check(dut.out_valid && dut.in_ready, "Core did not finish");
    const int code = dut.out_accept ? 1 + dut.out_flip : 0;
    check(!dut.out_bad_frame && dut.out_id == id && code == data.code &&
          (!dut.out_flip || dut.out_accept), "Reference disagreement for shot " + std::to_string(id));
}

int main(int argc, char** argv) {
    Vqec_core dut;
    const Vector zero{1, {}};
    dut.reset = 1;
    dut.in_valid = 0;
    tick(dut);
    dut.reset = 0;
    tick(dut);
    for (int broken = 0; broken < 3; ++broken) {
        load(dut, zero, 0, 0x8bad);
        tick(dut);
        load(dut, zero, 1, 0x8bad);
        if (broken == 0) dut.in_last = 1;
        if (broken == 1) dut.in_first = 1;
        if (broken == 2) {
            for (size_t i = 1; i < 15; ++i) {
                load(dut, zero, i, 0x8bad);
                tick(dut);
            }
            load(dut, zero, 15, 0x8bad);
            dut.in_last = 0;
        }
        tick(dut);
        check(dut.out_valid && dut.out_bad_frame && !dut.out_accept && dut.out_id == 0x8bad,
              "Malformed direct frame did not fail closed");
        shot(dut, zero, 0xbeef, true);
    }
    load(dut, zero, 0, 42);
    tick(dut);
    dut.reset = 1;
    dut.in_valid = 0;
    tick(dut);
    check(!dut.out_valid && !dut.out_accept, "Reset produced a correction");
    dut.reset = 0;
    tick(dut);
    shot(dut, zero, 43, false);
    Vector dense{0, {}};
    dense.syndrome.fill(255);
    shot(dut, dense, 44, false);
    shot(dut, zero, 45, false);
    size_t total = 0;
    for (int i = 1; i < argc; ++i) {
        for (const auto& v : vectors(argv[i])) {
            shot(dut, v, static_cast<uint16_t>(total), total % 7 == 0);
            ++total;
        }
    }
    check(total > 0, "No test vectors were supplied");
    std::cout << total << " RTL reference vectors matched. Direct framing, tags, stalls, and reset passed.\n";
}
