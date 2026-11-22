`timescale 1ns/1ps
`include "sc_image.vh"

module tb_core;
    localparam integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1);
    logic clk = 0;
    always #5 clk = !clk;
    logic reset = 1, start = 0, tilt_enable = 0, target_write = 0, remainder_read = 0;
    logic [IX-1:0] target_index, remainder_index;
    logic [W-1:0] target_row, remainder_row;
    logic busy, done, deferred, logical_bit;
`ifdef SC_GROWTH
    logic [15:0] clocks;
    sc_cluster_telescope device(.*);
`elsif SC_LOOKUP
    logic [7:0] clocks;
    sc_lookup device(.*);
`else
    logic [7:0] clocks;
    sc_telescope device(.handoff(), .*);
`endif
    string path;
    integer fd, count, mode, expected_cycles, expected_defer, expected_logical;
    time start_time;
    logic [W-1:0] expected_row;
    initial begin
        if (!$value$plusargs("vectors=%s", path)) $fatal(1, "missing vectors");
        fd = $fopen(path, "r");
        if (fd == 0 || $fscanf(fd, "%d", count) != 1) $fatal(1, "invalid vector file");
        repeat (3) @(negedge clk);
        reset = 0;
        for (integer shot = 0; shot < count; shot = shot + 1) begin
            if ($fscanf(fd, "%d%d%d%d", mode, expected_cycles, expected_defer, expected_logical) != 4)
                $fatal(1, "invalid shot header");
            tilt_enable = mode;
            for (integer t = 0; t < ROWS; t = t + 1) begin
                if ($fscanf(fd, "%x", target_row) != 1) $fatal(1, "missing input slice");
                target_write = 1;
                target_index = t;
                @(negedge clk);
            end
            target_write = 0;
            start = 1;
            @(posedge clk);
            start_time = $time;
            @(negedge clk);
            start = 0;
            wait(done);
            if (clocks != expected_cycles || $time-start_time != 10*expected_cycles)
                $fatal(1, "shot %0d got %0d clocks and %0d ns, expected %0d clocks",
                       shot, clocks, $time-start_time, expected_cycles);
            if (deferred != expected_defer || logical_bit != expected_logical)
                $fatal(1, "shot %0d got defer/label %0d/%0d, expected %0d/%0d",
                       shot, deferred, logical_bit, expected_defer, expected_logical);
            @(negedge clk);
            for (integer t = 0; t < ROWS; t = t + 1) begin
                if ($fscanf(fd, "%x", expected_row) != 1) $fatal(1, "missing output slice");
                remainder_read = 1;
                remainder_index = t;
                @(negedge clk);
                if (remainder_row !== expected_row)
                    $fatal(1, "shot %0d row %0d got %x expected %x", shot, t, remainder_row, expected_row);
            end
            remainder_read = 0;
        end
        $display("Surface core matched %0d reference shots and cycle counts", count);
        $finish;
    end
    initial begin
        #20000000;
        $fatal(1, "native simulation timed out");
    end
endmodule
