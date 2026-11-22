`timescale 1ns/1ps
module tb_link #(
    parameter integer BIT_CYCLES = 8
);
    logic clk = 0;
    always #5 clk = ~clk;
    logic reset = 1, req = 1, res;
    logic got;
    logic [7:0] octet;
    sc_link #(.UART_DIV(BIT_CYCLES)) board(clk, reset, req, res);
    uart_receive #(.DIVISOR(BIT_CYCLES)) sniffer(clk, reset, res, got, octet);

    logic [7:0] pending [0:16383];
    integer head = 0;
    integer tail = 0;
    always @(posedge clk) begin
        if (got) begin
            pending[head] <= octet;
            head <= head + 1;
        end
    end

    task automatic transmit(input logic [7:0] value);
        begin
            @(negedge clk); req = 0;
            repeat (BIT_CYCLES) @(negedge clk);
            for (integer b = 0; b < 8; b = b + 1) begin
                req = value[b];
                repeat (BIT_CYCLES) @(negedge clk);
            end
            req = 1;
            repeat (BIT_CYCLES) @(negedge clk);
        end
    endtask

    integer handle, rounds, tx_count, rx_count, code;
    string path;
    logic [7:0] next_byte;
    initial begin
        if (!$value$plusargs("vectors=%s", path)) $fatal(1,"missing vector file");
        handle = $fopen(path, "r");
        if (!handle || $fscanf(handle,"%d",rounds) != 1) $fatal(1,"invalid vector file");
        repeat (9) @(negedge clk);
        reset = 0;
        for (integer i = 0; i < rounds; i = i + 1) begin
            code = $fscanf(handle,"%d%d",tx_count,rx_count);
            if (code != 2) $fatal(1,"missing frame lengths");
            for (integer j = 0; j < tx_count; j = j + 1) begin
                if ($fscanf(handle,"%x",next_byte) != 1) $fatal(1,"truncated request");
                transmit(next_byte);
            end
            for (integer j = 0; j < rx_count; j = j + 1) begin
                if ($fscanf(handle,"%x",next_byte) != 1) $fatal(1,"truncated expected frame");
                wait (head > tail);
                @(negedge clk);
                if (pending[tail] !== next_byte)
                    $fatal(1,"case %0d response byte %0d got %02x expected %02x",i,j,pending[tail],next_byte);
                tail = tail + 1;
                if (tail == 16384) $fatal(1,"test buffer exhausted");
            end
            repeat (BIT_CYCLES*2) @(negedge clk);
        end
        $display("UART waveform matched %0d reference frames",rounds);
        $finish;
    end
    initial begin
        #400000000;
        $fatal(1,"UART timeout");
    end
endmodule
