module uart_rx #(
    parameter CLKS_PER_BIT = 234
)(
    input wire clk, reset, rx,
    output reg valid, framing_error,
    output reg [7:0] data
);
    localparam IDLE = 3'd0, START = 3'd1, PAYLOAD = 3'd2, STOP = 3'd3, WAIT_HIGH = 3'd4;
    localparam TICK_BITS = $clog2(CLKS_PER_BIT);
    localparam [TICK_BITS-1:0] BIT_TICKS = CLKS_PER_BIT - 1;
    localparam [TICK_BITS-1:0] HALF_TICKS = CLKS_PER_BIT / 2 - 1;
    reg [2:0] state, bit_index;
    reg [TICK_BITS-1:0] ticks;
    (* async_reg = "true" *) reg rx_meta, rx_sync;
    always @(posedge clk) begin
        if (reset) begin
            rx_meta <= 1;
            rx_sync <= 1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            ticks <= 0;
            bit_index <= 0;
            valid <= 0;
            framing_error <= 0;
            data <= 0;
        end else begin
            valid <= 0;
            framing_error <= 0;
            case (state)
                IDLE: if (!rx_sync) begin
                    ticks <= HALF_TICKS;
                    state <= START;
                end
                START: if (ticks != 0)
                    ticks <= ticks - 1'b1;
                else if (rx_sync)
                    state <= IDLE;
                else begin
                    ticks <= BIT_TICKS;
                    bit_index <= 0;
                    state <= PAYLOAD;
                end
                PAYLOAD: if (ticks != 0)
                    ticks <= ticks - 1'b1;
                else begin
                    ticks <= BIT_TICKS;
                    data <= {rx_sync, data[7:1]};
                    bit_index <= bit_index + 1'b1;
                    if (bit_index == 7)
                        state <= STOP;
                end
                STOP: if (ticks != 0)
                    ticks <= ticks - 1'b1;
                else begin
                    valid <= rx_sync;
                    framing_error <= !rx_sync;
                    state <= rx_sync ? IDLE : WAIT_HIGH;
                end
                WAIT_HIGH: if (rx_sync)
                    state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter CLKS_PER_BIT = 234
)(
    input wire clk, reset, valid,
    input wire [7:0] data,
    output wire ready, tx
);
    localparam TICK_BITS = $clog2(CLKS_PER_BIT);
    localparam [TICK_BITS-1:0] BIT_TICKS = CLKS_PER_BIT - 1;
    reg [TICK_BITS-1:0] ticks;
    reg [3:0] remaining;
    reg [9:0] bits;
    assign ready = remaining == 0;
    assign tx = bits[0];
    always @(posedge clk) begin
        if (reset) begin
            ticks <= 0;
            remaining <= 0;
            bits <= 10'b1111111111;
        end else if (ready && valid) begin
            bits <= {1'b1, data, 1'b0};
            remaining <= 10;
            ticks <= BIT_TICKS;
        end else if (!ready) begin
            if (ticks != 0)
                ticks <= ticks - 1'b1;
            else begin
                bits <= {1'b1, bits[9:1]};
                remaining <= remaining - 1'b1;
                ticks <= BIT_TICKS;
            end
        end
    end
endmodule
