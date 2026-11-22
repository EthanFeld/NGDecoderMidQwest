#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int kMaxTail = 8192;

inline void check(bool ok, const std::string& label) {
    if (!ok) throw std::runtime_error(label);
}

template <typename Model>
void tick(Model& dut) {
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
}

struct Vector {
    uint8_t code;
    std::array<uint8_t, 224> syndrome;
};

inline std::vector<Vector> vectors(const char* file, size_t limit = SIZE_MAX) {
    std::ifstream input(file, std::ios::binary);
    check(static_cast<bool>(input), std::string("Cannot read ") + file);
    std::vector<Vector> out;
    while (out.size() < limit) {
        Vector shot{};
        if (!input.read(reinterpret_cast<char*>(&shot.code), 1)) break;
        check(static_cast<bool>(input.read(reinterpret_cast<char*>(shot.syndrome.data()), 224)),
              "Truncated syndrome in test file");
        out.push_back(shot);
    }
    return out;
}

inline uint32_t crc32c(const uint8_t* data, size_t n) {
    uint32_t crc = 0xffffffff;
    for (size_t i = 0; i < n; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = crc & 1 ? 0x82f63b78 : 0;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

inline std::array<uint8_t, 232> request(const Vector& shot, uint8_t id) {
    std::array<uint8_t, 232> packet{0x5a, 0xd1, id, 0x01};
    for (size_t i = 0; i < shot.syndrome.size(); ++i) packet[i + 4] = shot.syndrome[i];
    const uint32_t crc = crc32c(packet.data(), 228);
    for (size_t i = 0; i < 4; ++i) packet[i + 228] = (crc >> (i * 8)) & 0xff;
    return packet;
}
