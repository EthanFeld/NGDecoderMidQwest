`include "sc_image.vh"
`ifdef SC_LOOKUP

module sc_lookup #(
    parameter integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1)
) (
    input logic clk, reset, start, tilt_enable,
    input logic target_write,
    input logic [IX-1:0] target_index,
    input logic [W-1:0] target_row,
    input logic remainder_read,
    input logic [IX-1:0] remainder_index,
    output wire [W-1:0] remainder_row,
    output logic busy, done,
    output wire deferred,
    output logic logical_bit,
    output logic [7:0] clocks
);
    localparam integer SIZE = W*ROWS, AW = SIZE < 14 ? SIZE : 14;
    localparam integer BANKS = 1 << (SIZE-AW), BW = BANKS > 1 ? $clog2(BANKS) : 1;
    (* ram_style = "block" *) logic [BANKS-1:0] logical_rom [0:(1<<AW)-1];
    logic [W-1:0] rows [0:ROWS-1];
    logic [SIZE-1:0] syndrome;
    logic [BANKS-1:0] rom_data;
    logic [BW-1:0] bank;
    logic ready_result;

    initial $readmemh(`SC_MAP_FILE, logical_rom);
    assign deferred = 0;
    assign remainder_row = 0;

    for (genvar t = 0; t < ROWS; t = t + 1) begin : preload
        assign syndrome[t*W+:W] = rows[t];
        always_ff @(posedge clk)
            if (!busy && target_write && target_index == t) rows[t] <= target_row;
    end

    always_ff @(posedge clk) begin
        if (busy && !ready_result) begin
            rom_data <= logical_rom[syndrome[AW-1:0]];
            bank <= BW'(syndrome >> AW);
        end
        if (reset) begin
            busy <= 0;
            done <= 0;
            ready_result <= 0;
            clocks <= 0;
        end else begin
            done <= 0;
            if (start && !busy) begin
                busy <= 1;
                ready_result <= 0;
                clocks <= 0;
            end else if (busy) begin
                clocks <= clocks + 1'b1;
                if (ready_result) begin
                    busy <= 0;
                    done <= 1;
                    logical_bit <= rom_data[bank];
                end else ready_result <= 1;
            end
        end
    end
endmodule
`endif
