`include "policy.vh"

// Ingest sixteen projected planes and evaluate the bounded cluster rule.
// Each plane handshake carries one 112-bit Z-check plane in circuit order.
module qec_core(
    input wire clk, reset,
    input wire in_valid, in_first, in_last,
    input wire [111:0] in_plane,
    input wire [15:0] in_id,
    output wire in_ready,
    output reg out_valid, out_accept, out_flip, out_bad_frame,
    output reg [15:0] out_id
);
    localparam MAX_EVENTS = `QEC_EVENT_LIMIT;
    localparam SLOT_BITS = $clog2(MAX_EVENTS);
    localparam [SLOT_BITS:0] CAPACITY = MAX_EVENTS;
    localparam [10:0] RAW_CAPACITY = `QEC_RAW_EVENT_LIMIT;
    localparam [SLOT_BITS:0] STANDARD_CAPACITY = `QEC_STANDARD_EVENT_LIMIT;
    localparam [10:0] NEAR_CAPACITY = `QEC_NEAR_EVENT_LIMIT;
    localparam [3:0] FIRST_SPAN = `QEC_FIRST_SPAN, SECOND_SPAN = `QEC_SECOND_SPAN;
    localparam [4:0] BOUNDARY_DEPTH = `QEC_BOUNDARY_DEPTH;
    localparam [3:0] CROSSING_LIMIT = `QEC_CROSSING_LIMIT, EXTENDED_SPAN = `QEC_EXTENDED_SPAN;
    localparam [3:0] DENSE_SPAN = `QEC_DENSE_SPAN, DENSE_CROSSINGS = `QEC_DENSE_CROSSINGS;
    localparam [4:0] DENSE_DEPTH = `QEC_DENSE_DEPTH, REPAIR_DEPTH = `QEC_REPAIR_DEPTH;
    localparam [3:0] NEAR_SPAN = `QEC_NEAR_SPAN, WIDE_SPAN = `QEC_WIDE_SPAN;
    localparam [1:0] NEAR_GROUPS = `QEC_NEAR_GROUPS;
    localparam [5:0] NEAR_BUDGET = `QEC_NEAR_BUDGET;
    localparam [3:0] WIDE_LOW = `QEC_WIDE_LOW, WIDE_HIGH = `QEC_WIDE_HIGH;
    localparam [4:0] WIDE_DEPTH = `QEC_WIDE_DEPTH;
    localparam [3:0] WIDE_CROSSINGS = `QEC_WIDE_CROSSINGS;
    localparam [3:0] REPAIR_LOW = `QEC_REPAIR_LOW, REPAIR_HIGH = `QEC_REPAIR_HIGH;
    localparam [3:0] REPAIR_BANDS = `QEC_REPAIR_BANDS;
    localparam [7:0] VARIANTS = `QEC_VARIANT_BITS;
    localparam [1:0] LAST_VARIANT = `QEC_VARIANTS - 1;

    localparam IDLE = 4'd0, LOAD = 4'd1, SCAN = 4'd2, SELECT = 4'd3, READ_SITE = 4'd4;
    localparam GROW = 4'd6, APPLY = 4'd7, CLOSE = 4'd8, NEXT_GRAPH = 4'd9;
    localparam FILTER_TAIL = 4'd10, REPORT = 4'd11;
    localparam LANES = 32, LANE_BITS = 5;
    localparam BANKS = (MAX_EVENTS + LANES - 1) / LANES;
    localparam BANK_BITS = $clog2(BANKS);

    reg [3:0] state;
    reg [4:0] loaded, scanned, cleaned;
    reg [1:0] tail_count;
    reg [15:0] shot_id;
    reg [111:0] planes [0:15];
    reg [111:0] pending;
    reg [3:0] scan_round;
    reg [7:0] scan_rows;
    reg [13:0] band_nonzero;
    reg [3:0] scan_y;
    reg [2:0] scan_x;
    always @* begin
        for (integer b = 0; b < 14; b = b + 1)
            band_nonzero[b] = |pending[b * 8 +: 8];
        scan_y = 0;
        for (integer b = 13; b >= 0; b = b - 1)
            if (band_nonzero[b])
                scan_y = b[3:0];
        scan_rows = pending[{scan_y, 3'b000} +: 8];
        scan_x = 0;
        for (integer x = 7; x >= 0; x = x - 1)
            if (scan_rows[x])
                scan_x = x[2:0];
    end
    wire [3:0] scan_u = {1'b0, scan_x} + ((scan_y + 4'd2) >> 1);
    wire [3:0] scan_v = {1'b0, scan_x} + 4'd7 - ((scan_y + 4'd1) >> 1);
    wire [6:0] scan_site = {scan_y, scan_x};

    reg [SLOT_BITS:0] count;
    reg [BANK_BITS-1:0] plane_begin [0:15];
    reg [BANK_BITS-1:0] plane_end [0:14];
    reg [MAX_EVENTS-1:0] present;
    reg over_capacity;
    wire packing = state == LOAD || state == FILTER_TAIL || state == SCAN;
    wire more_planes = scanned != cleaned;
    wire extract = packing && (|pending) && !over_capacity;
    wire add_event = extract && count != CAPACITY;
    wire load_scan = packing && !(|pending) && more_planes && !over_capacity;
    wire [SLOT_BITS-1:0] last_slot = count[SLOT_BITS-1:0] - 1'b1;
    assign in_ready = !reset && (state == IDLE || state == LOAD);
    wire receive = in_valid && in_ready;
    wire framing_ok = state == IDLE ? (in_first && !in_last) :
                                      (!in_first && (in_last == (loaded == 15)));
    wire start_frame = receive && framing_ok && state == IDLE;
    wire filtered_valid;
    wire [111:0] filtered;
    wire flush_filter = state == FILTER_TAIL;
    wire step_filter = (receive && framing_ok && state == LOAD) || flush_filter;
    qec_prune prune(clk, start_frame, step_filter, flush_filter ? 112'b0 : in_plane,
                    filtered_valid, filtered);

    // Count density before pair pruning. The buffer itself stores the survivors.
    reg [10:0] raw_count;
    reg raw_pending;
    wire [3:0] pop [0:15];
    reg [3:0] recorded [0:15];
    wire [4:0] paired_pop [0:7];
    wire [5:0] quad_pop [0:3];
    wire [6:0] half_pop [0:1];
    for (genvar b = 0; b < 16; b = b + 1) begin : count_band
        if (b < 14) begin
            wire [7:0] val = in_plane[b*8+:8];
            wire [2:0] x = {2'b0, val[0]} + {2'b0, val[1]} + {2'b0, val[2]} + {2'b0, val[3]};
            wire [2:0] y = {2'b0, val[4]} + {2'b0, val[5]} + {2'b0, val[6]} + {2'b0, val[7]};
            assign pop[b] = {1'b0, x} + {1'b0, y};
        end else
            assign pop[b] = 0;
        always @(posedge clk) if (receive && framing_ok)
            recorded[b] <= pop[b];
    end
    for (genvar b = 0; b < 8; b = b + 1) begin : sum_pairs
        assign paired_pop[b] = {1'b0, recorded[2*b]} + {1'b0, recorded[2*b+1]};
    end
    for (genvar b = 0; b < 4; b = b + 1) begin : sum_quads
        assign quad_pop[b] = {1'b0, paired_pop[2*b]} + {1'b0, paired_pop[2*b+1]};
    end
    for (genvar b = 0; b < 2; b = b + 1) begin : sum_halves
        assign half_pop[b] = {1'b0, quad_pop[2*b]} + {1'b0, quad_pop[2*b+1]};
    end
    wire [7:0] raw_plane = {1'b0, half_pop[0]} + {1'b0, half_pop[1]};

    always @(posedge clk) begin
        raw_pending <= receive && framing_ok;
        if (start_frame) begin
            raw_count <= 0;
            cleaned <= 0;
            pending <= 0;
            count <= 0;
            present <= 0;
            over_capacity <= 0;
            scanned <= 0;
            scan_round <= 0;
        end else begin
            if (raw_pending)
                raw_count <= raw_count + {3'b0, raw_plane};
            if (packing && filtered_valid) begin
                planes[cleaned[3:0]] <= filtered;
                cleaned <= cleaned + 1'b1;
            end
        end
        if (extract) begin
            pending[scan_site] <= 0;
            if (add_event) begin
                present <= {present[MAX_EVENTS-2:0], 1'b1};
                count <= count + 1'b1;
            end else
                over_capacity <= 1;
        end else if (load_scan) begin
            pending <= planes[scanned[3:0]];
            plane_begin[scanned[3:0]] <= count[SLOT_BITS-1:LANE_BITS];
            if (scanned != 0)
                plane_end[scanned[3:0] - 1'b1] <= last_slot[SLOT_BITS-1:LANE_BITS];
            scan_round <= scanned[3:0];
            scanned <= scanned + 1'b1;
        end
    end

    reg [MAX_EVENTS-1:0] unseen, frontier, component;
    reg [MAX_EVENTS-1:0] keep_interior, keep_top, keep_bottom;
    reg [SLOT_BITS-1:0] selected_index;
    wire [MAX_EVENTS-1:0] selected_onehot = {{(MAX_EVENTS-1){1'b0}}, 1'b1} << selected_index;
    reg [11:0] focus;
    reg [3:0] low_y, high_y;
    reg parity, top_parity;
    reg [3:0] early_top_depth, early_bottom_depth;
    reg [3:0] top_depth, bottom_depth;
    reg [12:0] extended_cuts;
    reg two_hop, candidate_flip, frame;
    reg [1:0] variant;
    reg [3:0] verified_masks;
    reg dense, near_allowed, multiple_sites;
    reg [1:0] orphans, near_groups;
    reg [11:0] orphan_site;
    reg used_near, used_wide, used_three_hop;
    reg [4:0] deepest, report_depth;
    reg report_candidate, report_flip, report_repaired;

    wire frontier_found, unseen_found;
    wire [SLOT_BITS-1:0] frontier_index, unseen_index;
    qec_priority #(MAX_EVENTS) frontier_priority(frontier, frontier_found, frontier_index);
    qec_priority #(MAX_EVENTS) unseen_priority(unseen, unseen_found, unseen_index);
    wire select_from_frontier = state == SELECT && frontier_found;
    wire select_new_site = (state == SELECT && (frontier_found || (!(|component) && unseen_found))) ||
                           (state == CLOSE && unseen_found);
    wire [SLOT_BITS-1:0] pick_index = select_from_frontier ? frontier_index : unseen_index;

    // A slot address is a bank index followed by a lane index.
    reg [BANK_BITS-1:0] bank, last_bank;
    wire [BANK_BITS-1:0] read_bank = state == READ_SITE ? selected_index[SLOT_BITS-1:LANE_BITS] : bank;
    wire [11:0] read_positions [0:LANES-1];
    for (genvar lane = 0; lane < LANES; lane = lane + 1) begin : site_ram
        (* ram_style = "distributed" *) reg [11:0] entries [0:BANKS-1];
        always @(posedge clk) begin
            if (add_event && count[LANE_BITS-1:0] == lane)
                entries[count[SLOT_BITS-1:LANE_BITS]] <= {scan_round, scan_u, scan_v};
        end
        assign read_positions[lane] = entries[read_bank];
    end
    wire [11:0] selected_site = read_positions[selected_index[LANE_BITS-1:0]];

    // Every two-hop neighbor lies in the five-round window around its site.
    wire [3:0] read_round = selected_site[11:8];
    wire [3:0] first_round = read_round < 2 ? 4'd0 : read_round - 4'd2;
    wire [3:0] last_round = read_round + 4'd2;
    wire [BANK_BITS-1:0] start_bank = plane_begin[first_round];
    wire [BANK_BITS-1:0] stop_bank = read_round >= 13 ?
                                     last_slot[SLOT_BITS-1:LANE_BITS] : plane_end[last_round];

    wire [4:0] t = {1'b0, focus[11:8]};
    wire [4:0] u = {1'b0, focus[7:4]};
    wire [4:0] v = {1'b0, focus[3:0]};
    wire [4:0] t_m2 = t - 5'd2, t_m1 = t - 5'd1, t_p1 = t + 5'd1, t_p2 = t + 5'd2;
    wire [4:0] u_m2 = u - 5'd2, u_m1 = u - 5'd1, u_p1 = u + 5'd1, u_p2 = u + 5'd2;
    wire [4:0] v_m2 = v - 5'd2, v_m1 = v - 5'd1, v_p1 = v + 5'd1, v_p2 = v + 5'd2;
    wire [LANES-1:0] near_one, near_two;
    for (genvar n = 0; n < LANES; n = n + 1) begin : lanes
        qec_proximity near_site(read_positions[n], t_m2, t_m1, t, t_p1, t_p2,
                                u_m2, u_m1, u, u_p1, u_p2, v_m2, v_m1, v, v_p1, v_p2,
                                near_one[n], near_two[n]);
    end
    wire [BANKS*LANES-1:0] fresh_one, fresh_two;
    reg [MAX_EVENTS-1:0] applied_one;
    for (genvar b = 0; b < BANKS; b = b + 1) begin : accumulated_near
        reg [LANES-1:0] one, two;
        wire active = bank == b;
        assign fresh_one[b*LANES+:LANES] = active ? near_one : one;
        assign fresh_two[b*LANES+:LANES] = active ? near_two : two;
        always @(posedge clk) begin
            if (state == READ_SITE) begin
                one <= 0;
                two <= 0;
            end else if (state == GROW && active) begin
                one <= near_one;
                two <= near_two;
            end
        end
    end
    // Cache rows on the first pass and reuse them in all boundary treatments.
    (* ram_style = "block" *) reg [MAX_EVENTS+11:0] neighbor_cache [0:MAX_EVENTS-1];
    reg [MAX_EVENTS+11:0] cached;
    always @(posedge clk) begin
        if (state == GROW && bank == last_bank) begin
            applied_one <= fresh_one[MAX_EVENTS-1:0];
            neighbor_cache[selected_index] <= {focus, fresh_two[MAX_EVENTS-1:0]};
        end
        if (select_new_site && two_hop)
            cached <= neighbor_cache[pick_index];
    end
    wire [11:0] active_site = two_hop ? cached[MAX_EVENTS+11:MAX_EVENTS] : focus;
    wire [3:0] active_y = active_site[7:4] + 4'd6 - active_site[3:0];
    wire [MAX_EVENTS-1:0] neighbors = two_hop ? cached[MAX_EVENTS-1:0] : applied_one;
    wire [MAX_EVENTS-1:0] discover = unseen & neighbors & ~selected_onehot;

    wire component_top = low_y == 0;
    wire component_bottom = high_y == 13;
    wire [3:0] component_span = high_y - low_y;
    wire near_boundary = near_allowed && !component_top && !component_bottom &&
                         component_span <= NEAR_SPAN && (low_y == 1 || high_y == 12);
    wire anchor_top = component_top || (near_boundary && low_y == 1);
    wire anchor_bottom = component_bottom || (near_boundary && high_y == 12);
    wire wide_even = !dense && !parity && component_span <= WIDE_SPAN &&
                     low_y >= WIDE_LOW && high_y <= WIDE_HIGH;
    wire [3:0] tdepth = high_y + 1'b1, bdepth = 4'd14 - low_y;
    wire [3:0] tmax = anchor_top && tdepth > top_depth ? tdepth : top_depth;
    wire [3:0] bmax = anchor_bottom && bdepth > bottom_depth ? bdepth : bottom_depth;
    wire [4:0] depth_total = {1'b0, tmax} + {1'b0, bmax};
    wire [12:0] span_cuts;
    for (genvar cut = 0; cut < 13; cut = cut + 1) begin : check_cuts
        assign span_cuts[cut] = low_y <= cut && high_y > cut;
    end
    function [3:0] count_cuts(input [12:0] cuts);
        begin
            count_cuts = 0;
            for (integer cut = 0; cut < 13; cut = cut + 1)
                count_cuts = count_cuts + {3'b0, cuts[cut]};
        end
    endfunction

    wire [1:0] next_variant = two_hop ? variant + 1'b1 : 2'd0;
    wire [1:0] freeze = VARIANTS[{next_variant, 1'b0} +: 2];
    wire [1:0] boundary_groups = {|keep_bottom, |keep_top};
    wire [1:0] current_mask = VARIANTS[{variant, 1'b0} +: 2] & boundary_groups;
    wire [1:0] next_mask = freeze & boundary_groups;
    wire repeated_pass = two_hop && (verified_masks[next_mask] || current_mask == next_mask);
    wire [MAX_EVENTS-1:0] variant_events = keep_interior |
               (freeze[0] ? {MAX_EVENTS{1'b0}} : keep_top) |
               (freeze[1] ? {MAX_EVENTS{1'b0}} : keep_bottom);
    wire same_prediction = variant == 0 || candidate_flip == frame;
    wire [3:0] span_one = dense ? DENSE_SPAN : FIRST_SPAN;
    wire [3:0] span_two = dense ? DENSE_SPAN : SECOND_SPAN;
    wire [4:0] depth_limit = dense ? DENSE_DEPTH : BOUNDARY_DEPTH;
    wire [3:0] cut_count = count_cuts(extended_cuts);
    wire [3:0] cut_limit = dense ? DENSE_CROSSINGS : CROSSING_LIMIT;
    wire crossings_ok = cut_count <= cut_limit;
`ifdef QEC_FAST_CORE
    // Bounds only expand as a component grows. An even interior component
    // can use the wider rule only while it stays inside these bounds.
    wire irrecoverable_span = (|component) && component_span > (two_hop ? span_two : span_one) &&
                              (dense || component_span > WIDE_SPAN || low_y < WIDE_LOW || high_y > WIDE_HIGH);
    // Every variant accumulates cuts in the same mask, so later work
    // cannot recover a shot whose crossing threshold is already exceeded.
    wire early_defer = ((state == READ_SITE || state == APPLY) && !crossings_ok) ||
                       (state == SELECT && irrecoverable_span);
`endif
    wire [4:0] early_depth = {1'b0, early_top_depth} + {1'b0, early_bottom_depth};
    wire [4:0] current_depth = {1'b0, top_depth} + {1'b0, bottom_depth};
    wire [4:0] next_depth = (freeze[0] ? {1'b0, early_top_depth} : 5'b0) +
                             (freeze[1] ? {1'b0, early_bottom_depth} : 5'b0);
    wire [4:0] completed_depth = two_hop ? current_depth : early_depth;
    wire [4:0] max_completed_depth = completed_depth > deepest ? completed_depth : deepest;
    wire [5:0] near_cost = {1'b0, report_depth} + {2'b0, cut_count};
    wire extensions_ok = (!used_near || (!report_repaired && near_cost <= NEAR_BUDGET)) &&
                         (!used_wide || (!used_near && !report_repaired && report_depth <= WIDE_DEPTH &&
                                         cut_count <= WIDE_CROSSINGS));
    wire early_closure = !two_hop && !(|keep_interior) &&
                         early_depth <= depth_limit && early_depth <= 12 &&
                         early_top_depth <= span_two + 1'b1 && early_bottom_depth <= span_two + 1'b1;
    wire [3:0] orphan_y = orphan_site[7:4] + 4'd6 - orphan_site[3:0];
    wire [3:0] band_difference = low_y >= orphan_y ? low_y - orphan_y : orphan_y - low_y;
    wire separated_by_three;
    qec_three_hop repair_distance(orphan_site, active_site, separated_by_three);
    wire repair_possible = !dense && !multiple_sites && low_y >= REPAIR_LOW && high_y <= REPAIR_HIGH &&
                           (orphans == 0 || (orphans == 1 && separated_by_three && band_difference <= REPAIR_BANDS));
    wire repair_complete = orphans == 0 || (orphans == 2 && current_depth <= REPAIR_DEPTH);

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            loaded <= 0;
            shot_id <= 0;
            out_id <= 0;
            out_valid <= 0;
            out_accept <= 0;
            out_flip <= 0;
            out_bad_frame <= 0;
        end else begin
            out_valid <= 0;
            out_accept <= 0;
            out_flip <= 0;
            out_bad_frame <= 0;
            if (receive && !framing_ok) begin
                out_valid <= 1;
                out_bad_frame <= 1;
                out_id <= state == IDLE ? in_id : shot_id;
                state <= IDLE;
`ifdef QEC_FAST_CORE
            end else if (early_defer) begin
                state <= IDLE;
                out_valid <= 1;
                out_id <= shot_id;
`endif
            end else begin
                case (state)
                    IDLE: if (start_frame) begin
                        shot_id <= in_id;
                        loaded <= 1;
                        state <= LOAD;
                    end
                    LOAD: if (receive) begin
                        loaded <= loaded + 1'b1;
                        if (in_last) begin
                            state <= FILTER_TAIL;
                            tail_count <= 0;
                        end
                    end
                    FILTER_TAIL: begin
                        tail_count <= tail_count + 1'b1;
                        if (tail_count == 1)
                            state <= SCAN;
                    end
                    SCAN: if (over_capacity || raw_count > RAW_CAPACITY) begin
                        state <= IDLE;
                        out_valid <= 1;
                        out_id <= shot_id;
                    end else if (!(|pending) && !more_planes && cleaned == 16) begin
                        unseen <= present;
                        frontier <= 0;
                        component <= 0;
                        keep_top <= 0;
                        keep_bottom <= 0;
                        keep_interior <= 0;
                        top_parity <= 0;
                        early_top_depth <= 0;
                        early_bottom_depth <= 0;
                        extended_cuts <= 0;
                        two_hop <= 0;
                        variant <= 0;
                        verified_masks <= 0;
                        frame <= 0;
                        candidate_flip <= 0;
                        orphans <= 0;
                        near_groups <= 0;
                        dense <= raw_count > STANDARD_CAPACITY;
                        near_allowed <= raw_count <= NEAR_CAPACITY;
                        used_near <= 0;
                        used_wide <= 0;
                        used_three_hop <= 0;
                        deepest <= 0;
                        state <= SELECT;
                    end
                    SELECT: if ((|component) && !frontier_found) begin
                        state <= CLOSE;
                    end else if (frontier_found || unseen_found) begin
                        selected_index <= pick_index;
                        state <= two_hop ? APPLY : READ_SITE;
                    end else
                        state <= NEXT_GRAPH;
                    READ_SITE: begin
                        focus <= selected_site;
                        bank <= start_bank;
                        last_bank <= stop_bank;
                        state <= GROW;
                    end
                    GROW: if (bank != last_bank)
                        bank <= bank + 1'b1;
                    else
                        state <= APPLY;
                    APPLY: begin
                        unseen <= unseen & ~(discover | selected_onehot);
                        frontier <= (frontier & ~selected_onehot) | discover;
                        component <= component | selected_onehot;
                        if (|component) begin
                            parity <= !parity;
                            multiple_sites <= 1;
                            low_y <= active_y < low_y ? active_y : low_y;
                            high_y <= active_y > high_y ? active_y : high_y;
                        end else begin
                            parity <= 1;
                            multiple_sites <= 0;
                            low_y <= active_y;
                            high_y <= active_y;
                        end
`ifdef QEC_FAST_CORE
                        state <= |((frontier & ~selected_onehot) | discover) ? SELECT : CLOSE;
`else
                        state <= SELECT;
`endif
                    end
                    CLOSE: begin
                        state <= unseen_found ? (two_hop ? APPLY : READ_SITE) : NEXT_GRAPH;
                        if (unseen_found)
                            selected_index <= unseen_index;
                        component <= 0;
                        if (component_span >= EXTENDED_SPAN)
                            extended_cuts <= extended_cuts | span_cuts;
                        if (component_span > (two_hop ? span_two : span_one) && wide_even)
                            used_wide <= 1;
                        if ((component_span > (two_hop ? span_two : span_one) && !wide_even) ||
                            (two_hop && parity &&
                             ((anchor_top && anchor_bottom) ||
                              (!anchor_top && !anchor_bottom && !repair_possible) ||
                              (near_boundary && near_groups == NEAR_GROUPS) ||
                              depth_total > depth_limit))) begin
                            state <= IDLE;
                            out_valid <= 1;
                            out_id <= shot_id;
                        end else if (parity && two_hop) begin
                            if (anchor_top || anchor_bottom) begin
                                frame <= frame ^ anchor_top;
                                top_depth <= tmax;
                                bottom_depth <= bmax;
                                if (near_boundary) begin
                                    near_groups <= near_groups + 1'b1;
                                    used_near <= 1;
                                end
                            end else begin
                                if (orphans == 0)
                                    orphan_site <= active_site;
                                orphans <= orphans + 1'b1;
                            end
                        end else if (parity) begin
                            if (component_top) begin
                                keep_top <= keep_top | component;
                                top_parity <= !top_parity;
                                if (tdepth > early_top_depth) early_top_depth <= tdepth;
                            end else if (component_bottom) begin
                                keep_bottom <= keep_bottom | component;
                                if (bdepth > early_bottom_depth) early_bottom_depth <= bdepth;
                            end else
                                keep_interior <= keep_interior | component;
                        end
                    end
                    NEXT_GRAPH: begin
                        if (!crossings_ok || (two_hop && !repair_complete) || early_closure ||
                            (two_hop && (!same_prediction || variant == LAST_VARIANT)) ||
                            next_depth > depth_limit) begin
                            report_candidate <= crossings_ok && repair_complete && (early_closure ||
                                                (two_hop && same_prediction && variant == LAST_VARIANT));
                            report_flip <= early_closure ? top_parity : frame;
                            report_depth <= max_completed_depth;
                            report_repaired <= used_three_hop || (two_hop && orphans == 2);
                            state <= REPORT;
                        end else begin
                            if (two_hop && variant == 0)
                                candidate_flip <= frame;
                            if (two_hop) begin
                                verified_masks[current_mask] <= 1;
                                deepest <= max_completed_depth;
                                used_three_hop <= used_three_hop || orphans == 2;
                            end
                            two_hop <= 1;
                            variant <= next_variant;
                            orphans <= 0;
                            near_groups <= 0;
                            top_depth <= freeze[0] ? early_top_depth : 4'b0;
                            bottom_depth <= freeze[1] ? early_bottom_depth : 4'b0;
                            if (repeated_pass) begin
                                // An absent boundary component makes these treatments equal.
                                unseen <= 0;
                                state <= NEXT_GRAPH;
                            end else begin
                                unseen <= variant_events;
                                frame <= freeze[0] && top_parity;
                                state <= SELECT;
                            end
                        end
                    end
                    REPORT: begin
                        out_valid <= 1;
                        out_id <= shot_id;
                        out_accept <= report_candidate && extensions_ok;
                        out_flip <= report_candidate && extensions_ok && report_flip;
                        state <= IDLE;
                    end
                    default: state <= IDLE;
                endcase
            end
        end
    end
endmodule
