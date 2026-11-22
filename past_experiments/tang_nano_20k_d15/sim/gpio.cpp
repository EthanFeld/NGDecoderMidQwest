#include "Vnano20k_gpio.h"
#include "support.h"

#include <algorithm>
#include <iostream>

struct Link {
    Vnano20k_gpio dut;
    int elapsed = 0;
    std::vector<std::array<int, 3>> replies;

    void advance() {
        tick(dut);
        ++elapsed;
        if (!dut.result_valid) {
            check(dut.result_n == 15, "GPIO exported a result symbol while idle");
            return;
        }
        int code = 0;
        while (code < 4 && dut.result_n != (15 ^ (1 << code))) ++code;
        check(code < 4, "Invalid GPIO result symbol");
        replies.push_back({elapsed, dut.result_id, code});
    }
    void idle(int clocks) {
        dut.in_valid = 0;
        for (int i = 0; i < clocks; ++i) advance();
    }
    void send(const std::array<uint8_t, 232>& packet, size_t words = 116, size_t offset = 0) {
        replies.clear();
        for (size_t i = offset; i < words; ++i) {
            check(dut.in_ready, "GPIO unexpectedly stalled during a packet");
            dut.in_word = packet[2 * i] | (uint16_t(packet[2 * i + 1]) << 8);
            dut.in_valid = 1;
            advance();
        }
        dut.in_valid = 0;
    }
    void expect(uint8_t tag, int code) {
        for (int i = 0; i < kMaxTail && replies.empty(); ++i) idle(1);
        check(replies.size() == 1 && replies[0][1] == (tag & 15) && replies[0][2] == code,
              "Missing or wrong GPIO status/tag");
        idle(1);
        check(replies.size() == 1, "GPIO emitted a duplicate response");
    }
};

int main(int argc, char** argv) {
    Link link;
    link.dut.reset_n = 0;
    tick(link.dut);
    link.dut.reset_n = 1;
    link.idle(8);
    const Vector zero{1, {}};
    constexpr uint8_t tag = 0x7e;
    const auto packet = request(zero, tag);
    const int start = link.elapsed + 1;
    link.send(packet);
    const int end = link.elapsed;
    check(link.replies.empty(), "GPIO result before packet integrity check");
    link.expect(tag, 1);
    std::cout << "Empty-frame result took " << link.replies[0][0] - start << " clocks from first word and "
              << link.replies[0][0] - end << " from final CRC word.\n";
    link.send(packet, 114);
    link.idle(50);
    check(link.replies.empty(), "Core result escaped before delayed CRC");
    link.send(packet, 116, 114);
    link.expect(tag, 1);
    for (size_t bit = 0; bit < packet.size() * 8; ++bit) {
        auto damaged = packet;
        damaged[bit / 8] ^= 1 << (bit % 8);
        link.send(damaged);
        link.idle(6);
        for (const auto& reply : link.replies)
            check(reply[2] == 3, "Single-bit corruption escaped integrity checking");
        if (bit >= 16) link.expect(damaged[2], 3);
    }
    link.send(packet, 31);
    link.idle(1040);
    link.expect(tag, 3);
    link.send(packet, 60);
    link.dut.reset_n = 0;
    link.idle(3);
    link.dut.reset_n = 1;
    link.idle(8);
    check(link.replies.empty(), "Partial packet survived reset");
    link.send(packet);
    link.expect(tag, 1);
    size_t total = 0;
    for (int i=1; i<argc; ++i) {
        const bool target = std::string(argv[i]).find("/p0.005.bin") != std::string::npos;
        std::vector<int> accepted;
        for (const auto& shot : vectors(argv[i], target ? 4096 : 512)) {
            const uint8_t id = total & 255;
            const int first = link.elapsed + 1;
            link.send(request(shot, id));
            link.expect(id, shot.code);
            if (shot.code != 0) accepted.push_back(link.replies[0][0] - first);
            ++total;
        }
        if (target && !accepted.empty()) {
            std::sort(accepted.begin(), accepted.end());
            std::cout << "p=.005 accepted clocks from first word "
                      << accepted.size() << " samples, median " << accepted[(accepted.size()-1)/2]
                      << ", p99 " << accepted[(accepted.size()-1)*99/100] << ", max " << accepted.back() << ".\n";
        }
    }
    std::cout << total << " GPIO vectors matched. 1856 single-bit corruptions rejected. "
                 "Timeout and reset recovery passed.\n";
}
