#include "Vnano20k_gpio.h"
#include "support.h"

#include <array>
#include <iostream>

int main(int argc, char** argv) {
    try {
        check(argc == 2, "Usage: tang_no_wire PACKETS.bin");
        std::ifstream input(argv[1], std::ios::binary);
        check(static_cast<bool>(input), "Cannot open UART request file");

        Vnano20k_gpio model;
        model.in_valid = 0;
        model.in_word = 0;
        model.reset_n = 0;
        tick(model);
        model.reset_n = 1;
        for (int i = 0; i < 8; ++i) tick(model);

        std::array<uint8_t, 232> packet{};
        while (input.read(reinterpret_cast<char*>(packet.data()), packet.size())) {
            check(packet[0] == 0x5a && packet[1] == 0xd1 && packet[3] == 1,
                  "Invalid request header in the vector file");
            constexpr int request_words = packet.size() / 2;
            model.in_valid = 1;
            for (int i = 0; i < request_words; ++i) {
                check(model.in_ready, "Packet engine stalled during streaming input");
                model.in_word = uint16_t(packet[2 * i]) | (uint16_t(packet[2 * i + 1]) << 8);
                tick(model);
                check(!model.result_valid, "Result became valid before all CRC words arrived");
            }

            model.in_valid = 0;
            int cycles_after_crc = 0;
            while (!model.result_valid && cycles_after_crc < kMaxTail) {
                tick(model);
                ++cycles_after_crc;
            }
            check(model.result_valid && model.result_id == (packet[2] & 15),
                  "Missing result or wrong sequence tag");
            int code = 0;
            while (code < 4 && model.result_n != (15 ^ (1 << code))) ++code;
            check(code < 3, "Invalid or bad-packet result for an intact request");
            std::cout << code << ',' << request_words - 1 + cycles_after_crc
                      << ',' << cycles_after_crc << '\n';
            tick(model);
            check(!model.result_valid, "Repeated result");
        }
        check(input.eof() && input.gcount() == 0, "Partial UART request in vector file");
        model.final();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
