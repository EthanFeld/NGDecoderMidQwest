// One request contains a magic word, a tagged header, 16 planes, and CRC-32C.
module qec_packet #(
    parameter TIMEOUT_CYCLES = 1024,
    parameter REPORT_LATENCY = 0
)(
    input wire clk, reset,
    input wire word_valid,
    input wire [15:0] word,
    input wire link_error,
    output wire word_ready, hunting,
    output reg result_valid,
    output reg [1:0] result_code,
    output reg [7:0] result_tag,
    output wire [15:0] core_latency, crc_latency
);
    localparam MAGIC = 16'hd15a;
    localparam HUNT = 3'd0, HEADER = 3'd1, PAYLOAD = 3'd2;
    localparam CRC_LOW = 3'd3, CRC_HIGH = 3'd4, WAIT_CORE = 3'd5;
    localparam DEFER = 2'd0, ACCEPT_0 = 2'd1, ACCEPT_1 = 2'd2, BAD_PACKET = 2'd3;
    reg [2:0] state;
    reg [7:0] tag;
    reg [2:0] word_index;
    reg [3:0] plane_index;
    reg [95:0] partial;
    reg [31:0] crc;
    reg [15:0] crc_low;
    localparam TIMER_BITS = $clog2(TIMEOUT_CYCLES + 1);
    localparam [TIMER_BITS-1:0] TIMEOUT_LIMIT = TIMEOUT_CYCLES - 1;
    reg [TIMER_BITS-1:0] idle_cycles;

    function [31:0] crc_word;
        input [31:0] previous;
        input [15:0] data;
        reg [31:0] value;
        integer k;
        begin
            value = previous;
            for (k = 0; k < 16; k = k + 1)
                value = (value >> 1) ^ ((value[0] ^ data[k]) ? 32'h82f63b78 : 32'b0);
            crc_word = value;
        end
    endfunction
    localparam CRC_AFTER_MAGIC = crc_word(32'hffffffff, MAGIC);

    reg abort_core, plane_valid, plane_first, plane_last;
    reg [111:0] plane;
    wire core_ready, core_valid, core_accept, core_flip, core_bad;
    wire [15:0] core_id;
    reg have_core_result;
    reg [1:0] saved_code;
    qec_core core(
        .clk(clk), .reset(reset | abort_core),
        .in_valid(plane_valid), .in_first(plane_first), .in_last(plane_last),
        .in_plane(plane), .in_id({8'b0, tag}), .in_ready(core_ready),
        .out_valid(core_valid), .out_accept(core_accept), .out_flip(core_flip),
        .out_bad_frame(core_bad), .out_id(core_id)
    );

    assign hunting = state == HUNT;
    assign word_ready = !reset && state != WAIT_CORE;
    wire take_word = word_ready && word_valid;
    wire timeout = !hunting && state != WAIT_CORE && !take_word && idle_cycles == TIMEOUT_LIMIT;
    wire illegal_header = state == HEADER && take_word && word[15:8] != 8'h01;
    wire bad_crc = state == CRC_HIGH && take_word && {word, crc_low} != ~crc;
    wire broken = (!hunting && link_error) || timeout || illegal_header || bad_crc ||
                  (plane_valid && !core_ready) || (core_valid && core_bad);
    wire [1:0] decoded = core_id != {8'b0, tag} ? BAD_PACKET :
                         !core_accept ? DEFER : core_flip ? ACCEPT_1 : ACCEPT_0;
    wire result_ready = state == WAIT_CORE && (core_valid || have_core_result);

    generate
        if (REPORT_LATENCY) begin : latency
            reg [15:0] core_cycles, crc_cycles, saved_core_latency, saved_crc_latency;
            assign core_latency = result_code == BAD_PACKET ? 16'hffff : saved_core_latency;
            assign crc_latency = result_code == BAD_PACKET ? 16'hffff : saved_crc_latency;

            always @(posedge clk) begin
                if (reset || (hunting && take_word)) begin
                    core_cycles <= 0;
                    crc_cycles <= 0;
                    saved_core_latency <= 16'hffff;
                    saved_crc_latency <= 16'hffff;
                end else begin
                    // A count of one at the handshake represents the next edge.
                    if (plane_valid && plane_last && core_ready)
                        core_cycles <= 1;
                    else if (core_valid && core_cycles != 0) begin
                        saved_core_latency <= core_cycles;
                        core_cycles <= 0;
                    end else if (core_cycles != 0 && core_cycles != 16'hffff)
                        core_cycles <= core_cycles + 1'b1;

                    if (state == CRC_HIGH && take_word)
                        crc_cycles <= 1;
                    else if (result_ready && !broken)
                        saved_crc_latency <= crc_cycles;
                    else if (state == WAIT_CORE && crc_cycles != 16'hffff)
                        crc_cycles <= crc_cycles + 1'b1;
                end
            end
        end else begin : no_latency
            assign core_latency = 0;
            assign crc_latency = 0;
        end
    endgenerate

    always @(posedge clk) begin
        if (reset) begin
            state <= HUNT;
            tag <= 0;
            word_index <= 0;
            plane_index <= 0;
            crc <= 0;
            crc_low <= 0;
            partial <= 0;
            idle_cycles <= 0;
            abort_core <= 0;
            plane_valid <= 0;
            plane_first <= 0;
            plane_last <= 0;
            plane <= 0;
            result_valid <= 0;
            result_code <= DEFER;
            result_tag <= 0;
            have_core_result <= 0;
            saved_code <= DEFER;
        end else begin
            abort_core <= 0;
            plane_valid <= 0;
            result_valid <= 0;
            result_code <= DEFER;
            if (core_valid) begin
                have_core_result <= 1;
                saved_code <= decoded;
            end
            if (hunting || take_word || state == WAIT_CORE)
                idle_cycles <= 0;
            else
                idle_cycles <= idle_cycles + 1'b1;

            if (broken) begin
                state <= HUNT;
                abort_core <= 1;
                have_core_result <= 0;
                result_valid <= 1;
                result_code <= BAD_PACKET;
                result_tag <= illegal_header ? word[7:0] : tag;
            end else if (take_word) begin
                case (state)
                    HUNT: if (word == MAGIC && !link_error) begin
                        crc <= CRC_AFTER_MAGIC;
                        tag <= 0;
                        have_core_result <= 0;
                        state <= HEADER;
                    end
                    HEADER: begin
                        tag <= word[7:0];
                        word_index <= 0;
                        plane_index <= 0;
                        crc <= crc_word(crc, word);
                        state <= PAYLOAD;
                    end
                    PAYLOAD: begin
                        crc <= crc_word(crc, word);
                        partial <= {word, partial[95:16]};
                        if (word_index == 6) begin
                            plane <= {word, partial};
                            plane_valid <= 1;
                            plane_first <= plane_index == 0;
                            plane_last <= plane_index == 15;
                            word_index <= 0;
                            plane_index <= plane_index + 1'b1;
                            if (plane_index == 15)
                                state <= CRC_LOW;
                        end else
                            word_index <= word_index + 1'b1;
                    end
                    CRC_LOW: begin
                        crc_low <= word;
                        state <= CRC_HIGH;
                    end
                    CRC_HIGH: state <= WAIT_CORE;
                    default: state <= HUNT;
                endcase
            end else if (result_ready) begin
                // UART may finish sending CRC after the core has produced its result.
                state <= HUNT;
                have_core_result <= 0;
                result_valid <= 1;
                result_tag <= tag;
                result_code <= core_valid ? decoded : saved_code;
            end
        end
    end
endmodule
