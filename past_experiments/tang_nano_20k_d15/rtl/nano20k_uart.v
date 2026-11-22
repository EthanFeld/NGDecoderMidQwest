module nano20k_uart #(
    parameter CLKS_PER_BIT = 234,
    parameter TIMEOUT_CYCLES = 1048576,
    parameter REPORT_LATENCY = 0
)(
    input wire clk, reset_button, uart_rx,
    output wire uart_tx
);
    wire reset;
    reset_sync rst(clk, !reset_button, reset);
    wire rx_valid, rx_bad;
    wire [7:0] rx_data;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) receiver(clk, reset, uart_rx, rx_valid, rx_bad, rx_data);
    wire tx_ready;
    localparam REPLY_BYTES = REPORT_LATENCY ? 16 : 8;
    localparam REPLY_BITS = REPLY_BYTES * 8;
    reg [$clog2(REPLY_BYTES + 1)-1:0] tx_bytes;
    reg [REPLY_BITS-1:0] tx_frame;
    wire tx_valid = tx_bytes != 0;
    wire replying = tx_valid || !tx_ready;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) transmitter(clk, reset, tx_valid, tx_frame[7:0], tx_ready, uart_tx);

    wire hunting, word_ready, result_valid;
    wire [1:0] result_code;
    wire [7:0] result_tag;
    wire [15:0] core_latency, crc_latency;
    reg word_valid, have_low;
    reg [7:0] low_byte;
    reg [15:0] word;
    wire link_error = rx_bad || (rx_valid && (!word_ready || replying));
    qec_packet #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES), .REPORT_LATENCY(REPORT_LATENCY)) packet(
        .clk(clk), .reset(reset), .word_valid(word_valid), .word(word),
        .link_error(link_error), .word_ready(word_ready), .hunting(hunting),
        .result_valid(result_valid), .result_code(result_code), .result_tag(result_tag),
        .core_latency(core_latency), .crc_latency(crc_latency)
    );

    always @(posedge clk) begin
        if (reset) begin
            word_valid <= 0;
            have_low <= 0;
            low_byte <= 0;
            word <= 0;
        end else begin
            word_valid <= 0;
            if (link_error || result_valid)
                have_low <= 0;
            else if (rx_valid) begin
                if (hunting) begin
                    have_low <= rx_data == 8'h5a;
                    if (have_low && rx_data == 8'hd1) begin
                        word <= 16'hd15a;
                        word_valid <= 1;
                        have_low <= 0;
                    end
                end else if (have_low) begin
                    word <= {rx_data, low_byte};
                    word_valid <= 1;
                    have_low <= 0;
                end else begin
                    low_byte <= rx_data;
                    have_low <= 1;
                end
            end
        end
    end

    function [31:0] crc_reply;
        input [31:0] header;
        reg [31:0] value;
        integer k;
        begin
            value = 32'hffffffff;
            for (k = 0; k < 32; k = k + 1)
                value = (value >> 1) ^ ((value[0] ^ header[k]) ? 32'h82f63b78 : 32'b0);
            crc_reply = ~value;
        end
    endfunction
    wire [31:0] response = {6'b0, result_code, result_tag, 16'hd15b};
    wire [31:0] telemetry = {crc_latency, core_latency};
    wire [127:0] response_frame = {crc_reply(telemetry), telemetry, crc_reply(response), response};
    always @(posedge clk) begin
        if (reset) begin
            tx_bytes <= 0;
            tx_frame <= 0;
        end else if (result_valid) begin
            tx_bytes <= REPLY_BYTES;
            tx_frame <= response_frame[REPLY_BITS-1:0];
        end else if (tx_valid && tx_ready) begin
            tx_bytes <= tx_bytes - 1'b1;
            tx_frame <= tx_frame >> 8;
        end
    end
endmodule
