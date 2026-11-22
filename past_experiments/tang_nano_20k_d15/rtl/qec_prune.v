// The matching removes a component only when both sites have degree one.
module qec_prune(
    input wire clk, start, step,
    input wire [111:0] plane,
    output reg valid,
    output wire [111:0] filtered
);
    reg [111:0] prev_bits, cur_bits;
    reg [111:0] left_one, mid_one, right_one, to_filter;
    reg [4:0] seen;
    wire [111:0] degrees, drop;
    qec_degree_one degree(prev_bits, cur_bits, plane, degrees);
    qec_drop_pairs drop_pairs(left_one, mid_one, right_one, drop);
    assign filtered = to_filter & ~drop;

    always @(posedge clk) begin
        valid <= 0;
        if (start) begin
            prev_bits <= 0;
            cur_bits <= plane;
            left_one <= 0;
            mid_one <= 0;
            right_one <= 0;
            seen <= 1;
        end else if (step) begin
            left_one <= mid_one;
            mid_one <= right_one;
            right_one <= degrees;
            prev_bits <= cur_bits;
            cur_bits <= plane;
            to_filter <= prev_bits;
            seen <= seen + 1'b1;
            valid <= seen >= 2;
        end
    end
endmodule
