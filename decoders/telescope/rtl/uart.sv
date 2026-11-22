module uart_receive #(
    parameter integer DIVISOR = 27
) (
    input logic clk, reset, serial_in,
    output logic valid,
    output logic [7:0] data
);
    typedef enum logic [1:0] {READY, START_BIT, DATA_BITS, STOP_BIT} phase;
    phase state;
    (* ASYNC_REG = "TRUE" *) logic rx_in, rx;
    logic [$clog2(DIVISOR+1)-1:0] tick;
    logic [2:0] bit_index;
    logic [7:0] staged;

    always_ff @(posedge clk) begin
        if (reset) begin
            rx_in <= 1; rx <= 1;
            state <= READY; valid <= 0; tick <= 0; bit_index <= 0; staged <= 0;
        end else begin
            rx_in <= serial_in;
            rx <= rx_in;
            valid <= 0;
            if (state == READY) begin
                if (!rx) begin tick <= DIVISOR/2-1; state <= START_BIT; end
            end else if (tick != 0) tick <= tick - 1'b1;
            else case (state)
                START_BIT: if (rx) state <= READY;
                           else begin tick <= DIVISOR-1; bit_index <= 0; state <= DATA_BITS; end
                DATA_BITS: begin
                    staged[bit_index] <= rx;
                    tick <= DIVISOR-1;
                    if (bit_index == 7) state <= STOP_BIT;
                    else bit_index <= bit_index + 1'b1;
                end
                STOP_BIT: begin
                    if (rx) begin valid <= 1; data <= staged; end
                    state <= READY;
                end
                default: state <= READY;
            endcase
        end
    end
endmodule

module uart_send #(
    parameter integer DIVISOR = 27
) (
    input logic clk, reset, valid,
    input logic [7:0] data,
    output logic ready, serial_out
);
    logic active;
    logic [9:0] packet;
    logic [3:0] bit_index;
    logic [$clog2(DIVISOR+1)-1:0] tick;
    assign ready = !active;
    assign serial_out = active ? packet[0] : 1'b1;

    always_ff @(posedge clk) begin
        if (reset) begin active <= 0; tick <= 0; bit_index <= 0; end
        else if (!active) begin
            if (valid) begin
                packet <= {1'b1, data, 1'b0};
                bit_index <= 0;
                tick <= DIVISOR-1;
                active <= 1;
            end
        end else if (tick != 0) tick <= tick - 1'b1;
        else if (bit_index == 9) active <= 0;
        else begin
            packet <= {1'b1, packet[9:1]};
            bit_index <= bit_index + 1'b1;
            tick <= DIVISOR-1;
        end
    end
endmodule
