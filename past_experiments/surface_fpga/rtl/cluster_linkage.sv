`include "sc_image.vh"
`ifdef SC_GROWTH

module sc_cluster_telescope #(
    parameter integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1)
) (
    input logic clk, reset, start, tilt_enable,
    input logic target_write,
    input logic [IX-1:0] target_index,
    input logic [W-1:0] target_row,
    input logic remainder_read,
    input logic [IX-1:0] remainder_index,
    output wire [W-1:0] remainder_row,
    output wire busy, done, deferred, logical_bit,
    output wire [15:0] clocks
);
    logic base_busy, base_done, base_defer, base_label;
    logic [7:0] base_cycles;
    logic [W-1:0] base_residual;
    logic activate, used_growth;
    logic grow_busy, grow_done, grow_success, grow_label;
    logic [15:0] grow_cycles;

    assign busy = base_busy || activate || grow_busy;
    assign done = base_done || grow_done;
    assign deferred = used_growth ? !grow_success : base_defer;
    assign logical_bit = used_growth ? (grow_success && grow_label) : base_label;
    assign clocks = used_growth ? {8'd0, base_cycles} + grow_cycles + 16'd1 : {8'd0, base_cycles};
    assign remainder_row = deferred ? base_residual : {W{1'b0}};

    sc_telescope #(.GROW_HANDOFF(1)) base(
        .clk(clk), .reset(reset), .start(start), .tilt_enable(tilt_enable),
        .target_write(target_write), .target_index(target_index), .target_row(target_row),
        .remainder_read(remainder_read), .remainder_index(remainder_index), .remainder_row(base_residual),
        .busy(base_busy), .done(base_done), .deferred(base_defer), .logical_bit(base_label), .clocks(base_cycles),
        .handoff(activate)
    );
    sc_growth deep(
        .clk(clk), .reset(reset), .start(activate), .load(target_write),
        .load_index(target_index), .load_row(target_row), .busy(grow_busy), .done(grow_done),
        .success(grow_success), .logical_bit(grow_label), .clocks(grow_cycles)
    );

    always_ff @(posedge clk) begin
        if (reset || start) used_growth <= 0;
        else if (activate) used_growth <= 1;
    end
endmodule
`endif
