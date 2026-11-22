`timescale 1ns/1ps
`include "sc_image.vh"
module tb_growth;
    localparam integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1);
    logic clk = 0;
    always #5 clk = !clk;
    logic reset = 1, start = 0, load = 0;
    logic [IX-1:0] load_index;
    logic [W-1:0] load_row;
    logic busy, done, success, logical_bit;
    logic [15:0] clocks;
    sc_growth device(.*);

    string path;
    integer fd, count, wanted_ok, wanted_label, wanted_cycles, rescued, timeouts;
    time start_time;
    initial begin
        if (!$value$plusargs("vectors=%s", path)) $fatal(1, "missing growth vectors");
        fd = $fopen(path, "r");
        if (!$fscanf(fd, "%d", count)) $fatal(1, "invalid growth fixture");
        repeat (3) @(negedge clk);
        reset = 0;
        rescued=0;
        timeouts=0;
        for (integer shot=0; shot<count; shot=shot+1) begin
            if ($fscanf(fd, "%d%d%d", wanted_ok, wanted_label, wanted_cycles) != 3)
                $fatal(1, "invalid growth fixture header");
            for (integer row=0; row<ROWS; row=row+1) begin
                if (!$fscanf(fd, "%x", load_row)) $fatal(1, "missing row");
                load_index = row;
                load = 1;
                @(negedge clk);
            end
            load = 0;
            start = 1;
            @(posedge clk);
            start_time = $time;
            @(negedge clk);
            start = 0;
            wait(done);
            if (clocks !== wanted_cycles || $time-start_time != 10*wanted_cycles)
                $fatal(1, "growth shot %0d clocks %0d expected %0d", shot, clocks, wanted_cycles);
            if (success !== wanted_ok || (success && logical_bit !== wanted_label))
                $fatal(1, "growth shot %0d success/label %b/%b expected %d/%d", shot,
                       success, logical_bit, wanted_ok, wanted_label);
            if(success) rescued=rescued+1;
            else timeouts=timeouts+1;
            @(negedge clk);
        end
        $display("Growth matched %0d shots and exact cycle counts (%0d accepted, %0d declined)",count,rescued,timeouts);
        $finish;
    end
    initial begin
        #30000000;
        $fatal(1, "growth simulation timed out");
    end
endmodule
