module reset_sync(input wire clk, reset_n, output wire reset);
    reg [3:0] ready = 4'b0;
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            ready <= 0;
        else
            ready <= {ready[2:0], 1'b1};
    end
    assign reset = !ready[3];
endmodule
