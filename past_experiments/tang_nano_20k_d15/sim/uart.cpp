#include "Vnano20k_uart.h"
#include "support.h"

#include <algorithm>
#include <iostream>

struct Link {
    static constexpr int bit_clocks = 234;
    Vnano20k_uart dut;
    std::vector<uint8_t> received;
    int elapsed = 0, sample_at = 0, bit = 0;
    uint8_t byte = 0;
    bool last_tx = true;

    void advance(int clocks, bool line) {
        dut.uart_rx = line;
        for (int i = 0; i < clocks; ++i) {
            tick(dut);
            ++elapsed;
            const bool tx = dut.uart_tx;
            if (!sample_at && last_tx && !tx) {
                sample_at = elapsed + bit_clocks + bit_clocks / 2;
                byte = 0;
                bit = 0;
            }
            if (elapsed == sample_at) {
                if (bit < 8) {
                    byte |= uint8_t(tx) << bit;
                    sample_at += bit_clocks;
                    ++bit;
                } else {
                    check(tx, "UART reply stop-bit failure");
                    received.push_back(byte);
                    sample_at = 0;
                }
            }
            last_tx = tx;
        }
    }
    void send_byte(uint8_t data, bool valid_stop = true, int clocks = bit_clocks) {
        advance(clocks, false);
        for (int i = 0; i < 8; ++i) advance(clocks, (data & (1 << i)) != 0);
        advance(clocks, valid_stop);
    }
    void send(const std::array<uint8_t, 232>& data, size_t n = 232, int clocks = bit_clocks) {
        received.clear();
        for (size_t i = 0; i < n; ++i) send_byte(data[i], true, clocks);
    }
    void expect(uint8_t tag, uint8_t code) {
        advance(bit_clocks * 82, true);
        check(received.size() == 8 && received[0] == 0x5b && received[1] == 0xd1 &&
              received[2] == tag && received[3] == code, "Wrong UART reply or tag");
        const uint32_t crc = crc32c(received.data(), 4);
        for (int i = 0; i < 4; ++i)
            check(received[i + 4] == uint8_t(crc >> (i * 8)), "Wrong UART reply CRC");
    }
};

int main(int argc, char** argv) {
    check(argc == 2, "Expected one RTL vector file");
    const auto cases = vectors(argv[1]);
    Link link;
    link.dut.reset_button = 0;
    link.advance(8, true);
    link.send_byte(0xb5);
    link.send_byte(0x5a);
    for (int code = 0; code <= 2; ++code) {
        const auto found = std::find_if(cases.begin(), cases.end(), [=](const auto& v) { return v.code == code; });
        check(found != cases.end(), "A UART result class is missing from the vector file");
        const uint8_t tag = 0x40 + code;
        link.send(request(*found, tag), 232, Link::bit_clocks + 4);
        link.expect(tag, code);
    }
    auto damaged = request(Vector{1, {}}, 0x7d);
    damaged[179] ^= 0x08;
    link.send(damaged);
    link.expect(0x7d, 3);
    const auto clean = request(Vector{1, {}}, 0x91);
    link.send(clean, 5);
    link.send_byte(clean[5], false);
    link.expect(0x91, 3);
    link.send(clean, 11);
    link.advance(16384 + Link::bit_clocks, true);
    link.expect(0x91, 3);
    link.send(clean);
    link.expect(0x91, 1);
    link.send(clean, 11);
    link.dut.reset_button = 1;
    link.advance(4, true);
    link.dut.reset_button = 0;
    link.advance(8, true);
    link.send(clean);
    link.expect(0x91, 1);
    std::cout << "UART RTL wire test passed all result classes, CRC, bad stop bits, timeout, and recovery.\n";
}
