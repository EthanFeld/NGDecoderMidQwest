module nano20k_gpio(
    input wire clk, reset_n,
    input wire in_valid,
    input wire [15:0] in_word,
    output wire in_ready,
    output wire result_valid,
    output wire [3:0] result_n,
    output wire [3:0] result_id
);
    wire reset;
    reset_sync rst(clk, reset_n, reset);
    wire [7:0] tag;
    wire [1:0] code;
    qec_packet packet(
        .clk(clk), .reset(reset), .word_valid(in_valid), .word(in_word),
        .link_error(1'b0), .word_ready(in_ready), .hunting(),
        .result_valid(result_valid), .result_code(code), .result_tag(tag),
        .core_latency(), .crc_latency()
    );
    assign result_n = result_valid ? ~(4'b0001 << code) : 4'b1111;
    assign result_id = tag[3:0];
endmodule
