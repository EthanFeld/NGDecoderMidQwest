`include "sc_image.vh"
`ifdef SC_GROWTH

// Grow odd components on the original detector graph. Boundary edges
// terminate a component without joining all sites along that boundary.
module sc_growth #(
    parameter integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1)
) (
    input logic clk, reset, start,
    input logic load,
    input logic [IX-1:0] load_index,
    input logic [W-1:0] load_row,
    output logic busy, done, success, logical_bit,
    output logic [15:0] clocks
);
    `include "sc_growth.vh"
    localparam integer G = `SC_GROW_GROUPS, AGE = `SC_GROW_AGES;
    localparam integer S = `SC_GROW_SBITS, P = 1 << S, POINTS = P*ROWS;
    localparam integer A = $clog2(POINTS), Q = 4;
    localparam bit DUAL_PORT = `SC_GROW_DUAL_PORT;
    localparam integer RANK = $clog2($clog2(W*ROWS+1));
    localparam integer META = DUAL_PORT ? RANK+1 : A+1;
    localparam integer RW = A+META+4;
    localparam logic [AGE-1:0] MAX_AGE = AGE'(`SC_GROW_MAXAGE);
    localparam logic [AGE-1:0] INITIAL_AGE = AGE'(`SC_GROW_INITIAL_AGE);

    typedef enum logic [2:0] {IDLE, EPOCH, LATCH, SELECT, PICK, FIND, PARENT} phase_t;
    phase_t phase;
    logic [IX-1:0] layer, candidate_layer;
    logic initial_epoch, frame_bit, conflict;
    logic [W-1:0] raw_row [0:ROWS-1], live_row [0:ROWS-1], frozen_row [0:ROWS-1];
    logic [W-1:0] touched_row [0:ROWS-1];
    wire [W-1:0] member_en, last_en, winner_en, lost_en;
    wire [ROWS-1:0] any_active;
    localparam bit PACKED_TOUCH = W > 32;
    localparam bit PREFETCH_ROWS = `SC_GROW_PREFETCH_ROWS;
    wire [IX-1:0] edge_layer = PREFETCH_ROWS ? candidate_layer : layer;

    (* ram_style = "block", no_rw_check *) logic [W*AGE-1:0] age_mem [0:ROWS-1];
    logic [W*AGE-1:0] age_q, older, aug_age;
    logic [W-1:0] past_activity, activity_q;
    wire [W*AGE-1:0] current_age;
    wire [W-1:0] activity = DUAL_PORT && PREFETCH_ROWS ? activity_q : frozen_row[layer];
    logic [W-1:0] pending [0:G-1];
    wire [W-1:0] hit [0:G-1];
    logic [G-1:0] family_hit;
    wire [G-1:0] nonempty, row_hit;
    wire any_hit = |row_hit;
    localparam integer GB = $clog2(G);
    localparam integer SEGMENTS = (W+7)/8, PADDED = 8*SEGMENTS;
    localparam integer SB = SEGMENTS == 1 ? 1 : $clog2(SEGMENTS);

    function automatic [GB-1:0] first_group(input logic [G-1:0] bitmap);
        first_group = 0;
        for (integer i=G-1; i>=0; i=i-1)
            if (bitmap[i]) first_group = GB'(i);
    endfunction
    function automatic [SB-1:0] first_segment(input logic [SEGMENTS-1:0] bitmap);
        first_segment = 0;
        for (integer i=SEGMENTS-1; i>=0; i=i-1)
            if (bitmap[i]) first_segment = SB'(i);
    endfunction
    function automatic [2:0] first_site(input logic [7:0] bitmap);
        first_site = 0;
        for (integer i=7; i>=0; i=i-1)
            if (bitmap[i]) first_site = 3'(i);
    endfunction

    wire [GB-1:0] group_select = first_group(nonempty);
    wire is_boundary = group_select == G-1;
    wire [A-1:0] from_vertex, to_vertex;
    wire [W-1:0] side_bitmap = boundary_type(edge_layer);
    wire choice_label;
    logic [A-1:0] picked_a, picked_b;
    logic [GB-1:0] picked_group;
    logic [S-1:0] picked_site;
    logic picked_label, picked_terminal, picked_valid;
    wire available = PACKED_TOUCH ? (|nonempty) : picked_valid;
    wire [GB-1:0] selected_group = PACKED_TOUCH ? group_select : picked_group;
    wire [A-1:0] selected_a = PACKED_TOUCH ? from_vertex : picked_a;
    wire [A-1:0] selected_b = PACKED_TOUCH ? to_vertex : picked_b;

    // On small patches, the first selection for a slice gets one clock.
    // The remaining selections overlap the root search of the prior edge.
    if (!PACKED_TOUCH) begin : picker_registers
        always_ff @(posedge clk) begin
            if (phase == SELECT || phase == FIND) begin
                picked_a <= from_vertex;
                picked_b <= to_vertex;
                picked_group <= group_select;
                picked_site <= from_vertex[S-1:0];
                picked_label <= choice_label;
                picked_terminal <= is_boundary;
                picked_valid <= |nonempty;
            end
        end
    end

    function automatic [S-1:0] first_lane(input logic [W-1:0] bitmap);
        logic [PADDED-1:0] lanes;
        logic [SEGMENTS-1:0] have_segment;
        logic [SB-1:0] segment;
        lanes = PADDED'(bitmap);
        for (integer i=0; i<SEGMENTS; i=i+1) have_segment[i] = |lanes[i*8+:8];
        segment = first_segment(have_segment);
        first_lane = S'({segment, first_site(lanes[{segment, 3'b0}+:8])});
    endfunction

    wire [S-1:0] next_site [0:G-1];
    if (PACKED_TOUCH) begin : compact_picker
        wire [S-1:0] site = first_lane(pending[group_select]);
        wire [IX-1:0] earlier_layer = edge_layer - IX'(past_edge(group_select));
        assign from_vertex = A'({earlier_layer, site});
        assign to_vertex = A'({edge_layer, partner(group_select, site)});
        assign choice_label = side_bitmap[site];
        for (genvar g=0; g<G; g=g+1) assign next_site[g] = site;
    end else begin : fast_picker
        wire [S-1:0] partners [0:G-1];
        wire [G-1:0] before_layer;
        for (genvar g=0; g<G; g=g+1) begin : neighbors
            assign next_site[g] = first_lane(pending[g]);
            assign partners[g] = partner(g, next_site[g]);
            assign before_layer[g] = past_edge(g);
        end
        wire [IX-1:0] earlier_layer = edge_layer - IX'(before_layer[group_select]);
        assign from_vertex = A'({earlier_layer, next_site[group_select]});
        assign to_vertex = A'({edge_layer, partners[group_select]});
        assign choice_label = side_bitmap[next_site[G-1]];
    end

    for (genvar u=0; u<W; u=u+1) begin : grow_cell
        // Preload the age reached just before the first possible collision.
        assign current_age[u*AGE+:AGE] = initial_epoch ? (activity[u] ? INITIAL_AGE : '0) : age_q[u*AGE+:AGE];
        wire [AGE-1:0] old_age = current_age[u*AGE+:AGE];
        assign aug_age[u*AGE+:AGE] = activity[u] && old_age != MAX_AGE ? old_age+1'b1 : old_age;
    end
    for (genvar g=0; g<G; g=g+1) begin : family
        localparam bit CROSS = past_edge(g);
        wire [W*AGE-1:0] left = CROSS ? older : current_age;
        wire [W*AGE-1:0] right = pull_age(g, current_age);
        wire [W-1:0] left_live = CROSS ? past_activity : activity;
        wire [W-1:0] right_live = pull_activity(g, activity);
        wire [W*AGE-1:0] weight;
        if (DUAL_PORT && PREFETCH_ROWS) begin : cache_weight
            logic [W*AGE-1:0] next_weight;
            always_ff @(posedge clk)
                if (begin_epoch || read_next_row) next_weight <= edge_weight(g, scan_address);
            assign weight = next_weight;
        end else begin : weight_mask
            assign weight = edge_weight(g, layer);
        end
        wire [S-1:0] site = PACKED_TOUCH ? next_site[g] : picked_site;
        wire selected = phase == PICK && available && selected_group == g;
        assign nonempty[g] = PACKED_TOUCH ? family_hit[g] : (|pending[g]);
        assign row_hit[g] = |hit[g];
        if (PACKED_TOUCH || DUAL_PORT) begin : packed_candidates
            wire [W-1:0] mask;
            for (genvar u=0; u<W; u=u+1) begin : candidate
                assign mask[u] = site == u;
                always_ff @(posedge clk) begin
                    if (phase == LATCH) pending[g][u] <= hit[g][u];
                    else if (selected && mask[u]) pending[g][u] <= 0;
                end
            end
            always_ff @(posedge clk) begin
                if (phase == LATCH) family_hit[g] <= row_hit[g];
                else if (selected) family_hit[g] <= |(pending[g] & ~mask);
            end
        end else begin : wide_candidates
            always_ff @(posedge clk) begin
                if (phase == LATCH && any_hit) pending[g] <= hit[g];
                else if (selected) pending[g][site] <= 0;
            end
        end
        for (genvar u=0; u<W; u=u+1) begin : lane
            wire [AGE-1:0] l = left[u*AGE+:AGE], r = right[u*AGE+:AGE];
            wire [AGE-1:0] half_lower = weight[u*AGE+:AGE];
            wire one_active = left_live[u] || right_live[u];
            wire two_active = left_live[u] && right_live[u];
            wire ready = (l[0] ^ r[0]) ? one_active : two_active;
            if ((PACKED_TOUCH || DUAL_PORT) && `SC_GROW_MAXAGE == 6) begin : short_thresholds
                // The only enabled thresholds are four and six. Compare
                // half the old sum using the carry from its bottom bit.
                wire carry = l[0] && r[0];
                wire middle_xor = l[1] ^ r[1];
                wire middle_pair = l[1] && r[1];
                wire middle_one = middle_xor ^ carry;
                wire middle_two = (middle_pair && !carry) || (middle_xor && carry);
                wire high_zero = !(l[2] || r[2]);
                wire middle_zero = !(l[1] || r[1] || carry);
                wire lower_one = high_zero && middle_one && !middle_pair;
                wire lower_two = (high_zero && middle_two) || ((l[2] ^ r[2]) && middle_zero);
                assign hit[g][u] = ready && ((half_lower[0] && lower_one) || (half_lower[1] && lower_two));
            end else begin : general_thresholds
                wire [AGE:0] old_sum = {1'b0, l} + {1'b0, r};
                assign hit[g][u] = ready && half_lower != 0 && old_sum[AGE:1] == half_lower;
            end
        end
    end

    // An untouched record denotes a one-vertex component. The dual-port
    // schedule stores rank and a size-two flag in place of full size.
    (* ram_style = "block", no_rw_check *) logic [A-1:0] next_member [0:POINTS-1];
    logic [RW-1:0] qa, qb;
    wire valid_a, valid_b;
    logic qa_valid, qb_valid;
    logic qa_raw, qb_raw;
    logic [A-1:0] a, b, lost, kept;
    logic terminal, side;
    logic parent_mark;
    localparam logic [META-1:0] SINGLETON = DUAL_PORT ? 0 : 1;
    wire singleton_a = PACKED_TOUCH ? frozen_row[a[A-1:S]][a[S-1:0]] :
                       DUAL_PORT ? qa_raw : raw_row[a[A-1:S]][a[S-1:0]];
    wire singleton_b = PACKED_TOUCH ? frozen_row[b[A-1:S]][b[S-1:0]] :
                       DUAL_PORT ? qb_raw : raw_row[b[A-1:S]][b[S-1:0]];
    wire [RW-1:0] ra = valid_a ? qa : {1'b0, a, SINGLETON, singleton_a, 2'd0};
    wire [RW-1:0] rb = valid_b ? qb : {1'b0, b, SINGLETON, singleton_b, 2'd0};
    wire link_a = ra[RW-1], link_b = rb[RW-1] && !terminal;
    wire [A-1:0] next_a = link_a ? ra[RW-2-:A] : a;
    wire [A-1:0] next_b = link_b ? rb[RW-2-:A] : b;
    wire advance_find = phase == FIND && (link_a || link_b);
    wire [A-1:0] req_a = phase == PICK ? selected_a : next_a;
    wire [A-1:0] req_b = phase == PICK ? selected_b : next_b;
    wire read_records = (phase == PICK && available) || advance_find;
    wire old_a = ra[2], old_b = !terminal && rb[2];
    wire [1:0] bound_a = ra[1:0], bound_b = terminal ? (side ? 2'd2 : 2'd1) : rb[1:0];
    wire [META-1:0] size_a = ra[3+:META], size_b = rb[3+:META];
    wire [RANK-1:0] rank_a = ra[3+:RANK], rank_b = rb[3+:RANK];
    wire pair_a = ra[3+RANK], pair_b = rb[3+RANK];
    wire take_b = !terminal && (DUAL_PORT ? rank_b > rank_a : size_b > size_a);
    wire [A-1:0] winner = take_b ? b : a;
    wire [A-1:0] loser = take_b ? a : b;
    wire [A-1:0] tail_a = ra[RW-2-:A], tail_b = rb[RW-2-:A];
    wire [A-1:0] winner_tail = take_b ? tail_b : tail_a;
    wire [A-1:0] loser_tail = take_b ? tail_a : tail_b;
    wire [A-1:0] new_tail = terminal ? tail_a : loser_tail;
    wire a_single = a == tail_a, b_single = b == tail_b;
    wire new_pair = terminal ? pair_a : a_single && b_single;
    wire [RANK-1:0] rank = (take_b ? rank_b : rank_a) + RANK'(!terminal && rank_a == rank_b);
    wire [META-1:0] total = DUAL_PORT ? META'({new_pair, rank}) : size_a + (terminal ? META'(0) : size_b);
    wire [1:0] all_bound = bound_a | bound_b;
    wire combined = old_a ^ old_b;
    wire act_a = old_a && bound_a == 0, act_b = old_b && bound_b == 0;
    wire new_act = combined && all_bound == 0;
    wire diff_a = act_a != new_act, diff_b = !terminal && (act_b != new_act);
    wire update_needed = diff_a || diff_b;
    wire [A-1:0] job_head = diff_a && diff_b ? winner : diff_a ? a : b;
    wire [A-1:0] job_tail = diff_a && diff_b ? new_tail : diff_a ? tail_a : tail_b;
    wire short_job = diff_a && diff_b ? a_single && b_single : diff_a ? a_single || pair_a : b_single || pair_b;

    logic [A-1:0] heads [0:Q-1], tails [0:Q-1];
    logic [Q-1:0] values, short_jobs;
    logic [$clog2(Q)-1:0] head_ix, tail_ix;
    logic [$clog2(Q+1)-1:0] queued;
    logic writing, value;
    logic [A-1:0] member, last_member, next_q;
    wire enter = busy && !writing && queued != 0;
    wire same_root = !terminal && a == b;
    wire accepted_merge = phase == FIND && !link_a && !link_b && !same_root &&
                          (!update_needed || queued != Q);
    wire short_dequeue = `SC_GROW_SHORT_UPDATES && enter && short_jobs[head_ix];
    wire [A-1:0] write_head = short_dequeue ? heads[head_ix] : member;
    wire [A-1:0] write_tail = tails[head_ix];
    wire write_value = short_dequeue ? values[head_ix] : value;
    wire finished_edge = phase == FIND && !link_a && !link_b && (same_root || accepted_merge);
    wire drained = (phase == PICK && !available) || (DUAL_PORT && finished_edge && !(|nonempty));
    wire begin_epoch = phase == EPOCH && queued == 0 && !writing && |any_active;
    wire advance_row = ((phase == LATCH && !any_hit) || drained) && layer != ROWS-1;
    // No remaining candidate uses the age data after its row is latched.
    // At the faster clock, issue the next SRAM read without waiting on hit.
    wire read_next_row = PREFETCH_ROWS ? (phase == LATCH) : advance_row;
    wire [IX-1:0] scan_address = begin_epoch ? 0 : layer+1'b1;
    wire add_job = accepted_merge && update_needed;
    wire take_parent = DUAL_PORT ? parent_mark : phase == PARENT;
    wire member_read = (enter && !short_dequeue) || (busy && writing && member != last_member);
    wire [A-1:0] member_address = enter ? heads[head_ix] : next_q;

    assign busy = phase != IDLE;

    if (PACKED_TOUCH) begin : touched_columns
        // Every input writes the full detector window before start. Clear
        // each validity row while it arrives, using one write port per site.
        wire loading = load && !busy;
        wire update = loading || accepted_merge || take_parent;
        wire [A-S-1:0] address = loading ? (A-S)'(load_index) :
                                  take_parent ? lost[A-1:S] : winner[A-1:S];
        wire [W-1:0] hit_a, hit_b;
        wire [W-1:0] which = take_parent ? lost_en : winner_en;
        for (genvar u = 0; u < W; u = u+1) begin : column
            (* ram_style = "distributed" *) logic visited [0:ROWS-1];
            always_ff @(posedge clk)
                if (update && (loading || which[u])) visited[address] <= !loading;
            assign hit_a[u] = visited[a[A-1:S]];
            assign hit_b[u] = visited[b[A-1:S]];
        end
        assign valid_a = hit_a[a[S-1:0]];
        assign valid_b = hit_b[b[S-1:0]];
    end else begin : touched_planes
        assign valid_a = qa_valid;
        assign valid_b = qb_valid;
        if (DUAL_PORT) begin : latch_endpoints
            // A link can only point to another initialized record. Keep
            // the flags during traversal and read the bitplanes at pick.
            always_ff @(posedge clk) begin
                if (phase == PICK && available) begin
                    qa_valid <= touched_row[selected_a[A-1:S]][selected_a[S-1:0]] || (parent_mark && selected_a == lost);
                    qb_valid <= touched_row[selected_b[A-1:S]][selected_b[S-1:0]] || (parent_mark && selected_b == lost);
                    qa_raw <= raw_row[selected_a[A-1:S]][selected_a[S-1:0]];
                    qb_raw <= raw_row[selected_b[A-1:S]][selected_b[S-1:0]];
                end
            end
        end else begin : refresh_flags
            always_ff @(posedge clk) begin
                if (read_records) begin
                    qa_valid <= touched_row[req_a[A-1:S]][req_a[S-1:0]];
                    qb_valid <= touched_row[req_b[A-1:S]][req_b[S-1:0]];
                end
            end
        end
    end

    for (genvar u = 0; u < W; u = u+1) begin : column_enable
        assign member_en[u] = write_head[S-1:0] == u;
        assign last_en[u] = write_tail[S-1:0] == u;
        assign winner_en[u] = winner[S-1:0] == u;
        assign lost_en[u] = lost[S-1:0] == u;
    end
    for (genvar t = 0; t < ROWS; t = t+1) begin : bit_planes
        assign any_active[t] = |live_row[t];
        wire active_write = busy && (writing || short_dequeue) && write_head[A-1:S] == t;
        wire last_write = short_dequeue && write_tail[A-1:S] == t;
        wire winner_write = accepted_merge && winner[A-1:S] == t;
        wire lost_write = take_parent && lost[A-1:S] == t;
        wire clear = !busy && start;
        wire loading = !busy && load && load_index == t;
        always_ff @(posedge clk) begin
            if (loading) raw_row[t] <= load_row;
            if (begin_epoch) frozen_row[t] <= live_row[t];
        end
        for (genvar u = 0; u < W; u = u+1) begin : site_enable
            always_ff @(posedge clk) begin
                if (PACKED_TOUCH ? loading : clear) live_row[t][u] <= PACKED_TOUCH ? load_row[u] : raw_row[t][u];
                else if ((active_write && member_en[u]) || (last_write && last_en[u])) live_row[t][u] <= write_value;
                if (!PACKED_TOUCH) begin
                    if (clear) touched_row[t][u] <= 0;
                    else if ((winner_write && winner_en[u]) || (lost_write && lost_en[u]))
                        touched_row[t][u] <= 1;
                end
            end
        end
    end

    if (DUAL_PORT) begin : atomic_records
        // The two ports either chase parents or commit one merge. The
        // roots are distinct when both ports write.
        (* ram_style = "block", no_rw_check *) logic [RW-1:0] records [0:POINTS-1];
        wire [A-1:0] port_a = accepted_merge ? winner : req_a;
        wire [A-1:0] port_b = accepted_merge ? loser : req_b;
        always_ff @(posedge clk) begin
            if (accepted_merge) records[port_a] <= {1'b0, new_tail, total, combined, all_bound};
            else if (read_records) qa <= records[port_a];
        end
        always_ff @(posedge clk) begin
            if (accepted_merge && !terminal) records[port_b] <= {1'b1, winner, (META+3)'(0)};
            else if (read_records) qb <= records[port_b];
        end
        always_ff @(posedge clk) begin
            // The next record read observes this flag through forwarding.
            parent_mark <= !reset && accepted_merge && !terminal;
        end
    end else begin : serial_records
        (* ram_style = "block", no_rw_check *) logic [RW-1:0] records [0:POINTS-1];
        always_ff @(posedge clk) begin
            if (read_records) begin
                qa <= records[req_a];
                qb <= records[req_b];
            end
            if (accepted_merge || take_parent)
                records[take_parent ? lost : winner] <= take_parent ? {1'b1, kept, (META+3)'(0)} :
                                                                      {1'b0, new_tail, total, combined, all_bound};
        end
    end

    always_ff @(posedge clk) begin
        if (begin_epoch || read_next_row) age_q <= age_mem[scan_address];
        if (phase == LATCH) age_mem[layer] <= aug_age;
        if (accepted_merge && !terminal) next_member[winner_tail] <= loser;
        if (member_read) next_q <= next_member[member_address];
    end

    if (DUAL_PORT && PREFETCH_ROWS) begin : snapshot_row
        always_ff @(posedge clk) begin
            if (begin_epoch) activity_q <= live_row[0];
            else if (read_next_row) activity_q <= frozen_row[scan_address];
        end
    end

    always_ff @(posedge clk) begin
        if (reset || (!busy && start)) begin
            queued <= 0;
            head_ix <= 0;
            tail_ix <= 0;
            writing <= 0;
        end else if (busy) begin
            if (add_job) begin
                heads[tail_ix] <= job_head;
                tails[tail_ix] <= job_tail;
                values[tail_ix] <= new_act;
                short_jobs[tail_ix] <= short_job;
                tail_ix <= tail_ix+1'b1;
            end
            if (enter) begin
                writing <= !short_dequeue;
                member <= heads[head_ix];
                last_member <= tails[head_ix];
                value <= values[head_ix];
                head_ix <= head_ix+1'b1;
            end else if (writing) begin
                if (member == last_member) writing <= 0;
                else member <= next_q;
            end
            case ({add_job, enter})
                2'b10: queued <= queued+1'b1;
                2'b01: queued <= queued-1'b1;
                default: ;
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            phase <= IDLE;
            done <= 0;
            clocks <= 0;
        end else begin
            done <= 0;
            if (phase == IDLE) begin
                if (start) begin
                    phase <= EPOCH;
                    clocks <= 0;
                    conflict <= 0;
                    frame_bit <= 0;
                    initial_epoch <= 1;
                end
            end else begin
                clocks <= clocks+1'b1;
                if (accepted_merge) begin
                    frame_bit <= frame_bit ^ (old_a && bound_a==2) ^
                                 (old_b && bound_b==2) ^ (combined && all_bound==2);
                    if (all_bound == 3) conflict <= 1;
                    lost <= loser;
                    kept <= winner;
                end
                if (clocks == `SC_GROW_CYCLES-1 || conflict) begin
                    success <= 0;
                    done <= 1;
                    phase <= IDLE;
                end else if (drained) begin
                    if (edge_layer == ROWS-1) begin
                        initial_epoch <= 0;
                        phase <= EPOCH;
                    end else begin
                        if (!PREFETCH_ROWS) layer <= layer+1'b1;
                        phase <= LATCH;
                    end
                end else case (phase)
                    EPOCH: if (queued == 0 && !writing) begin
                        if (!(|any_active)) begin
                            success <= 1;
                            logical_bit <= frame_bit;
                            done <= 1;
                            phase <= IDLE;
                        end else begin
                            layer <= 0;
                            older <= 0;
                            past_activity <= 0;
                            phase <= LATCH;
                        end
                    end
                    LATCH: begin
                        older <= current_age;
                        past_activity <= activity;
                        if (PREFETCH_ROWS) begin
                            candidate_layer <= layer;
                            layer <= layer+1'b1;
                        end
                        if (any_hit) begin
                            phase <= PACKED_TOUCH ? PICK : SELECT;
                        end else if (layer == ROWS-1) begin
                            initial_epoch <= 0;
                            phase <= EPOCH;
                        end else if (!PREFETCH_ROWS) layer <= layer+1'b1;
                    end
                    SELECT: phase <= PICK;
                    PICK: begin
                        a <= selected_a;
                        b <= selected_b;
                        terminal <= PACKED_TOUCH ? is_boundary : picked_terminal;
                        side <= PACKED_TOUCH ? choice_label : picked_label;
                        phase <= FIND;
                    end
                    FIND: if (link_a || link_b) begin
                        a <= next_a;
                        b <= next_b;
                    end else if (same_root) phase <= PICK;
                    else if (accepted_merge) phase <= DUAL_PORT || terminal ? PICK : PARENT;
                    PARENT: phase <= PICK;
                    default: phase <= IDLE;
                endcase
            end
        end
    end
endmodule
`endif
