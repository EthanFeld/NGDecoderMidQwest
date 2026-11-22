`timescale 1ns/1ps
`include "sc_image.vh"
`ifdef SC_LOOKUP

module tb_map;
    localparam integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1);
    localparam integer WORDS = 1 << (W*ROWS);
    logic clk = 0;
    always #5 clk = !clk;
    logic reset = 1, start = 0, tilt_enable = 1, target_write = 0, remainder_read = 0;
    logic [IX-1:0] target_index, remainder_index;
    logic [W-1:0] target_row, remainder_row;
    logic busy, done, deferred, logical_bit;
    logic [7:0] clocks;
    logic expected [0:WORDS-1];
    sc_lookup device(.*);
    string path;

    initial begin
        if (!$value$plusargs("answers=%s", path)) $fatal(1, "expected answer file");
        $readmemh(path, expected);
        repeat (2) @(negedge clk);
        reset = 0;
        for (integer word = 0; word < WORDS; word = word + 1) begin
            for (integer t = 0; t < ROWS; t = t + 1) begin
                target_write = 1;
                target_index = t;
                target_row = W'(word >> (W*t));
                @(negedge clk);
            end
            target_write = 0;
            start = 1;
            @(negedge clk);
            start = 0;
            wait(done);
            @(negedge clk);
            if (logical_bit !== expected[word] || deferred || clocks != 2)
                $fatal(1, "incorrect lookup for syndrome %h", word);
        end
        $display("Verified all %0d MAP syndromes in RTL", WORDS);
        $finish;
    end
    initial begin
        #20000000;
        $fatal(1, "lookup test timed out");
    end
endmodule

`endif
