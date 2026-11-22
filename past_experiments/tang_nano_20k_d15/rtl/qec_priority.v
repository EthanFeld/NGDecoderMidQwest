// A balanced first-set-bit tree keeps selection out of a linear chain.
module qec_priority #(
    parameter SLOTS = 176,
    parameter BITS = $clog2(SLOTS)
)(
    input wire [SLOTS-1:0] mask,
    output wire found,
    output wire [BITS-1:0] address
);
    localparam LEAVES = 1 << BITS;
    wire [2*LEAVES-1:1] valid;
    wire [BITS-1:0] index [1:2*LEAVES-1];
    for (genvar i = 0; i < LEAVES; i = i + 1) begin : leaf
        if (i < SLOTS)
            assign valid[LEAVES+i] = mask[i];
        else
            assign valid[LEAVES+i] = 1'b0;
        assign index[LEAVES+i] = i;
    end
    for (genvar i = 1; i < LEAVES; i = i + 1) begin : branch
        assign valid[i] = valid[2*i] || valid[2*i+1];
        assign index[i] = valid[2*i] ? index[2*i] : index[2*i+1];
    end
    assign found = valid[1];
    assign address = index[1];
endmodule
