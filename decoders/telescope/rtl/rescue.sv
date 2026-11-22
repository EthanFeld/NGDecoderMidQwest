`include "sc_image.vh"
`ifdef SC_RESCUE

module sc_residual_check #(
    parameter integer W = `SC_WIDTH
) (
    input logic clk, reset, clear,
    input logic raw_valid,
    input logic [W-1:0] raw_row,
    input logic residual_valid, marker,
    input logic [W-1:0] residual_row,
    input logic label,
    output logic local_fit, cut_ok, raw_ok, guard_ok,
    output logic local_fit_gap3, cut_ok_gap3, raw_ok_gap3
);
    `include "sc_patch.vh"
    localparam integer TILES = `SC_PATCH_TILES, VIEWS = `SC_PATCH_VIEWS;
    localparam integer GAP = `SC_PATCH_GAP, CUTS = `SC_PATCH_CUTS, CW = $clog2(CUTS+2);
`ifdef SC_PATCH_GAP3_RAW_MAX
    localparam integer MAX_GAP = 3;
    localparam bit USE_RAW = 1'b1;
`else
    localparam integer MAX_GAP = GAP;
    localparam bit USE_RAW = `SC_PATCH_USE_RAW;
`endif
    localparam logic [W-1:0] EXCLUDED = `SC_PATCH_EXCLUDE;
    logic [TILES-1:0] open, toggles;
    logic [TILES-1:0] history [1:MAX_GAP-1];
    wire [TILES-1:0] carry_next = open ^ toggles;
    wire [TILES-1:0] ongoing [0:MAX_GAP-1];

    for (genvar t = 0; t < TILES; t = t+1) begin : tile_dot
        assign toggles[t] = ^(residual_row & tile_mask(t));
    end
    assign ongoing[0] = carry_next & open;
    for (genvar i = 1; i < MAX_GAP; i = i+1) begin : carry_history
        assign ongoing[i] = ongoing[i-1] & history[i];
        if (i == 1) begin
            always_ff @(posedge clk)
                if (reset || clear) history[i] <= 0;
                else if (residual_valid && !marker) history[i] <= open;
        end else begin
            always_ff @(posedge clk)
                if (reset || clear) history[i] <= 0;
                else if (residual_valid && !marker) history[i] <= history[i-1];
        end
    end

    logic [VIEWS-1:0] fit;
    logic excluded_live;
`ifdef SC_PATCH_GAP3_RAW_MAX
    logic [VIEWS-1:0] fit_gap3;
    assign local_fit_gap3 = (|fit_gap3) && !excluded_live;
`endif
    for (genvar v = 0; v < VIEWS; v = v+1) begin : patch_view
        localparam integer LO = tile_start(v), HI = tile_start(v+1)-1;
        logic failed;
        always_ff @(posedge clk)
            if (reset || clear) failed <= 0;
            else if (residual_valid && !marker) failed <= failed || (|ongoing[GAP-1][HI:LO]);
        assign fit[v] = !failed && !(|open[HI:LO]);
`ifdef SC_PATCH_GAP3_RAW_MAX
        logic failed_gap3;
        always_ff @(posedge clk)
            if (reset || clear) failed_gap3 <= 0;
            else if (residual_valid && !marker) failed_gap3 <= failed_gap3 || (|ongoing[2][HI:LO]);
        assign fit_gap3[v] = !failed_gap3 && !(|open[HI:LO]);
`endif
    end
    assign local_fit = (|fit) && !excluded_live;

    always_ff @(posedge clk) begin
        if (reset || clear) begin
            open <= 0;
            excluded_live <= 0;
        end else if (residual_valid && !marker) begin
            open <= carry_next;
            excluded_live <= excluded_live || (|(residual_row & EXCLUDED));
        end
    end

    logic [CUTS-1:0] raw_sheets, res_sheets, raw_cross, res_cross;
    wire [CUTS-1:0] raw_toggles, res_toggles, raw_prefix, res_prefix;
    for (genvar i = 0; i < CUTS; i = i+1) begin : strip_parity
        assign raw_toggles[i] = ^(raw_row & strip_mask(i));
        assign res_toggles[i] = ^(residual_row & strip_mask(i));
        assign raw_prefix[i] = ^raw_sheets[i:0];
        assign res_prefix[i] = ^res_sheets[i:0];
    end
    localparam integer GROUPS = (CUTS+3)/4, PAD = 4*GROUPS;
    wire [PAD-1:0] raw_pad = PAD'(raw_cross), res_pad = PAD'(res_cross);
    wire [2:0] raw_counts [0:GROUPS-1], res_counts [0:GROUPS-1];

    function automatic [2:0] sum_four(input logic [3:0] bits);
        sum_four = (3'(bits[0])+3'(bits[1])) + (3'(bits[2])+3'(bits[3]));
    endfunction

    for (genvar j = 0; j < GROUPS; j = j+1) begin : counts
        assign raw_counts[j] = sum_four(raw_pad[j*4+:4]);
        assign res_counts[j] = sum_four(res_pad[j*4+:4]);
    end
    logic [CW-1:0] res_count, raw_count, rx_sum, raw_sum;
    logic late;
    always_comb begin
        rx_sum = 0;
        raw_sum = 0;
        for (integer j = 0; j < GROUPS; j = j+1) begin
            rx_sum = rx_sum + CW'(res_counts[j]);
            raw_sum = raw_sum + CW'(raw_counts[j]);
        end
    end

    always_ff @(posedge clk) begin
        if (reset || clear) begin
            raw_sheets <= 0;
            res_sheets <= 0;
            late <= 0;
        end else begin
            if (USE_RAW && raw_valid) raw_sheets <= raw_sheets ^ raw_toggles;
            if (residual_valid && !marker) res_sheets <= res_sheets ^ res_toggles;
            late <= residual_valid && marker;
            if (residual_valid && marker) begin
                raw_cross <= raw_prefix;
                res_cross <= res_prefix;
            end
            if (late) begin
                raw_count <= raw_sum;
                res_count <= rx_sum;
            end
        end
    end

    // The proposed class fixes the crossing parity at the first cut. Its
    // complement toggles every cut, including the first one.
    assign cut_ok = label ? res_count >= CW'(CUTS+1-`SC_PATCH_RES_MAX)
                          : res_count <= CW'(`SC_PATCH_RES_MAX);
    assign raw_ok = !`SC_PATCH_USE_RAW || (label ? raw_count >= CW'(CUTS+1-`SC_PATCH_RAW_MAX)
                                                 : raw_count <= CW'(`SC_PATCH_RAW_MAX));
`ifdef SC_PATCH_GAP3_RAW_MAX
    assign cut_ok_gap3 = label ? res_count >= CW'(CUTS+1-`SC_PATCH_GAP3_RES_MAX)
                               : res_count <= CW'(`SC_PATCH_GAP3_RES_MAX);
    assign raw_ok_gap3 = label ? raw_count >= CW'(CUTS+1-`SC_PATCH_GAP3_RAW_MAX)
                               : raw_count <= CW'(`SC_PATCH_GAP3_RAW_MAX);
`else
    assign local_fit_gap3 = 1'b0;
    assign cut_ok_gap3 = 1'b0;
    assign raw_ok_gap3 = 1'b0;
`endif
`ifdef SC_GROW_CUT_LIMIT
    assign guard_ok = label ? raw_count >= CW'(CUTS+1-`SC_GROW_CUT_LIMIT)
                            : raw_count <= CW'(`SC_GROW_CUT_LIMIT);
`else
    assign guard_ok = raw_ok;
`endif
endmodule
`endif
