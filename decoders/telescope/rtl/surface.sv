`include "sc_image.vh"

// A stencil owns one injective spatial displacement. A spatial pass gives
// even edges priority over both neighbours, and a time-spanning pass moves
// oldest-first through the window. Back-to-back slices are supported.
module sc_pair_stage #(
    parameter integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1),
    parameter integer FAMILY = 0, SKIP = 0, ALT = -1, ALT_SKIP = 0
) (
    input logic clk, reset, clear, swap, in_valid,
    input logic [IX-1:0] in_index,
    input logic [W-1:0] in_row,
    output logic out_valid,
    output logic [IX-1:0] out_index,
    output logic [W-1:0] out_row
);
    `include "sc_stencils.vh"
    logic [1:0] flush;
    logic [W-1:0] previous, earlier;
    wire shifted = ALT >= 0 && swap;
    wire [1:0] gap = shifted ? 2'(ALT_SKIP) : 2'(SKIP);
    wire [W-1:0] old_row = gap == 0 ? in_row : gap == 1 ? previous : earlier;
    wire [W-1:0] new_row = in_row;

    logic [W-1:0] removed_here, removed_there;
    if (ALT < 0) begin : fixed
        wire [W-1:0] candidates = old_row & pull_back(FAMILY, new_row) & enabled_edges(FAMILY, in_index);
        wire [W-1:0] neighbour = pull_back(FAMILY, candidates) | push_forward(FAMILY, candidates);
        assign removed_here = SKIP == 0 ? candidates & (even_edges(FAMILY) | ~neighbour) : candidates;
        assign removed_there = push_forward(FAMILY, removed_here);
    end else begin : tilted
        wire [W-1:0] cand_main = old_row & pull_back(FAMILY, new_row) & enabled_edges(FAMILY, in_index);
        wire [W-1:0] cand_alt = old_row & pull_back(ALT, new_row) & enabled_edges(ALT, in_index);
        wire [W-1:0] suppress_main = pull_back(FAMILY, cand_main) | push_forward(FAMILY, cand_main);
        wire [W-1:0] suppress_alt = pull_back(ALT, cand_alt) | push_forward(ALT, cand_alt);
        wire [W-1:0] kept_main = SKIP == 0 ? cand_main & (even_edges(FAMILY) | ~suppress_main) : cand_main;
        wire [W-1:0] kept_alt = ALT_SKIP == 0 ? cand_alt & (even_edges(ALT) | ~suppress_alt) : cand_alt;
        assign removed_here = shifted ? kept_alt : kept_main;
        assign removed_there = shifted ? push_forward(ALT, kept_alt) : push_forward(FAMILY, kept_main);
    end

    always_ff @(posedge clk) begin
        if (reset || clear) begin
            flush <= 0;
            out_valid <= 0;
        end else begin
            flush <= {flush[0], in_valid && in_index == ROWS && gap != 0};
            out_valid <= 0;
            if (in_valid) begin
                previous <= new_row ^ removed_there;
                earlier <= previous;
                if (in_index >= IX'(gap)) begin
                    out_valid <= 1;
                    out_index <= in_index - IX'(gap);
                    out_row <= old_row ^ removed_here ^ (gap == 0 ? removed_there : {W{1'b0}});
                end
            end else if (flush[0] && gap == 2) begin
                out_valid <= 1;
                out_index <= ROWS-1;
                out_row <= earlier;
            end else if ((flush[0] && gap == 1) || (flush[1] && gap == 2)) begin
                out_valid <= 1;
                out_index <= ROWS;
                out_row <= 0;
            end
        end
    end
endmodule

module sc_telescope #(
    parameter integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1),
    parameter bit GROW_HANDOFF = 0
) (
    input logic clk, reset, start, tilt_enable,
    input logic target_write,
    input logic [IX-1:0] target_index,
    input logic [W-1:0] target_row,
    input logic remainder_read,
    input logic [IX-1:0] remainder_index,
    output logic [W-1:0] remainder_row,
    output logic busy, done, deferred, logical_bit,
    output logic handoff,
    output logic [7:0] clocks
);
    localparam integer STAGES = `SC_SHAPES, A = $clog2(ROWS), DEPTH = 3*(1 << A);
    localparam logic [2*STAGES-1:0] SKIPS = `SC_SKIPS;
    `include "sc_stencils.vh"
    (* ram_style = "block", no_rw_check *) logic [W-1:0] shot_mem [0:DEPTH-1];
    logic [W-1:0] read_data, final_row;
    logic [IX-1:0] read_index, source_index, final_index;
    logic launch, feeding, source_valid, finish, final_valid;
    logic use_swap, allow_tilt, chosen_alternate, chosen_raw, parity, first_parity, interior_live;
    logic [IX-1:0] occupied, first_occupied;
    wire normal_read = feeding && read_index < ROWS;
    wire [1:0] read_bank = busy || chosen_raw ? 2'd0 : chosen_alternate ? 2'd2 : 2'd1;
    wire [A-1:0] read_slice = busy ? read_index[A-1:0] : remainder_index[A-1:0];
    wire [1:0] write_bank = !busy ? 2'd0 : use_swap ? 2'd2 : 2'd1;
    wire [A-1:0] write_slice = busy ? final_index[A-1:0] : target_index[A-1:0];
    wire select_alternate = use_swap && occupied < first_occupied;
    wire [STAGES:0] valid;
    wire [IX-1:0] place [0:STAGES];
    wire [W-1:0] rows [0:STAGES];

`ifdef SC_RESCUE
    `include "sc_patch.vh"
    logic patch_fit, cut_ok, raw_ok, guard_ok, first_fit, first_cut;
    logic patch_fit_gap3, cut_ok_gap3, raw_ok_gap3;
    wire ordinary_fit = first_cut && cut_ok && raw_ok &&
                        (`SC_PATCH_BOTH ? (first_fit && patch_fit) : (first_fit || patch_fit));
`ifdef SC_PATCH_GAP3_RAW_MAX
    logic first_fit_gap3, first_cut_gap3;
    wire extended_fit = first_cut_gap3 && cut_ok_gap3 && raw_ok_gap3 &&
                        first_fit_gap3 && patch_fit_gap3;
`else
    wire extended_fit = 1'b0;
`endif
    wire certified = allow_tilt && use_swap && occupied != 0 && (parity == first_parity) &&
                     (ordinary_fit || extended_fit);
    sc_residual_check #(.W(W)) patch(
        .clk(clk), .reset(reset), .clear(launch),
        .raw_valid(source_valid && source_index != ROWS), .raw_row(read_data),
        .residual_valid(valid[STAGES]), .marker(place[STAGES] == ROWS), .residual_row(rows[STAGES]),
        .label(parity), .local_fit(patch_fit), .cut_ok(cut_ok), .raw_ok(raw_ok), .guard_ok(guard_ok),
        .local_fit_gap3(patch_fit_gap3), .cut_ok_gap3(cut_ok_gap3), .raw_ok_gap3(raw_ok_gap3)
    );
`else
    wire certified = 1'b0;
`endif
    wire will_defer = occupied != 0 && !certified;
`ifdef SC_GROW_CUT_GUARD
    // An ordinary acceptance needs another opinion when its logical bit
    // implies too many odd cuts in the original syndrome.
    wire retry_weak = GROW_HANDOFF && allow_tilt && !will_defer && !guard_ok;
`else
    wire retry_weak = 1'b0;
`endif
    wire take_growth = GROW_HANDOFF && allow_tilt && (will_defer || retry_weak);

    assign valid[0] = source_valid;
    assign place[0] = source_index;
    assign rows[0] = read_data;
    assign remainder_row = deferred ? read_data : {W{1'b0}};

    for (genvar s = 0; s < STAGES; s = s + 1) begin : stencil
        localparam integer OTHER = s == 0 ? 1 : s == 1 ? 0 : -1;
        localparam integer OTHER_SKIP = s == 0 ? SKIPS[2+:2] : SKIPS[0+:2];
        sc_pair_stage #(.W(W), .ROWS(ROWS), .IX(IX), .FAMILY(s),
                        .SKIP(SKIPS[2*s+:2]), .ALT(OTHER), .ALT_SKIP(OTHER_SKIP)) pair_stage(
            .clk(clk), .reset(reset), .clear(launch), .swap(use_swap),
            .in_valid(valid[s]), .in_index(place[s]), .in_row(rows[s]),
            .out_valid(valid[s+1]), .out_index(place[s+1]), .out_row(rows[s+1])
        );
    end

    always_ff @(posedge clk) begin
        if ((!busy && target_write) || (busy && final_valid && final_index != ROWS))
            shot_mem[{write_bank, write_slice}] <= busy ? final_row : target_row;
        if (normal_read || (!busy && remainder_read))
            read_data <= shot_mem[{read_bank, read_slice}];
    end

    // Boundary edges contribute to a complete answer only. Defers retain
    // the frontier before those edges, so the CPU can choose a boundary.
    always_ff @(posedge clk) begin
        if (reset || launch) begin
            final_valid <= 0;
            parity <= 0;
        end else begin
            final_valid <= valid[STAGES];
            if (valid[STAGES]) begin
                final_index <= place[STAGES];
                final_row <= rows[STAGES];
                interior_live <= |(rows[STAGES] & ~boundary_bits(0, place[STAGES]));
                parity <= parity ^ ^(rows[STAGES] & logical_bits(0, place[STAGES]));
            end
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            launch <= 0;
            feeding <= 0;
            source_valid <= 0;
            finish <= 0;
            busy <= 0;
            done <= 0;
            handoff <= 0;
            clocks <= 0;
        end else begin
            launch <= 0;
            done <= 0;
            handoff <= 0;
            source_valid <= feeding;
            if (feeding) begin
                source_index <= read_index;
                read_index <= read_index + 1'b1;
                if (read_index == ROWS) feeding <= 0;
            end
            if (start && !busy) begin
                launch <= 1;
                feeding <= 1;
                busy <= 1;
                finish <= 0;
                read_index <= 0;
                use_swap <= 0;
                allow_tilt <= tilt_enable;
                occupied <= 0;
                clocks <= 0;
            end else if (busy) begin
                clocks <= clocks + 1'b1;
                if (final_valid) begin
                    if (final_index == ROWS) finish <= 1;
                    else if (interior_live) occupied <= occupied + 1'b1;
                end
                if (finish) begin
                    if (allow_tilt && !use_swap && occupied != 0) begin
                        launch <= 1;
                        feeding <= 1;
                        finish <= 0;
                        read_index <= 0;
                        use_swap <= 1;
                        first_occupied <= occupied;
                        first_parity <= parity;
`ifdef SC_RESCUE
                        first_fit <= patch_fit;
                        first_cut <= cut_ok;
`ifdef SC_PATCH_GAP3_RAW_MAX
                        first_fit_gap3 <= patch_fit_gap3;
                        first_cut_gap3 <= cut_ok_gap3;
`endif
`endif
                        occupied <= 0;
                    end else begin
                        done <= !take_growth;
                        handoff <= take_growth;
                        busy <= 0;
                        chosen_alternate <= select_alternate;
                        chosen_raw <= retry_weak;
                        deferred <= will_defer || retry_weak;
                        logical_bit <= !retry_weak && (occupied == 0 || certified) &&
                                       (use_swap && !select_alternate ? first_parity : parity);
                    end
                end
            end
        end
    end
endmodule
