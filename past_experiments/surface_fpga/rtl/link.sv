`include "sc_image.vh"

module sc_link #(
    parameter integer UART_DIV = 135,
    parameter integer W = `SC_WIDTH, ROWS = `SC_ROWS, IX = $clog2(ROWS+1)
) (
    input logic clk, reset, rx,
    output logic tx
);
    localparam integer ROW_BYTES = (W+7)/8, BYTES = ROWS*ROW_BYTES;
    localparam integer LANE_W = ROW_BYTES > 1 ? $clog2(ROW_BYTES) : 1;
    localparam logic [31:0] TAG = `SC_TAG;
    localparam logic [7:0] FLAG = 8'h7e, ESC = 8'h7d;
    localparam logic [7:0] HEADER = 8'd12;

    logic got, sent, tx_ready;
    logic [7:0] incoming, outgoing;
    uart_receive #(.DIVISOR(UART_DIV)) receiver(clk, reset, rx, got, incoming);
    uart_send #(.DIVISOR(UART_DIV)) transmitter(clk, reset, sent, outgoing, tx_ready, tx);

    function automatic [15:0] crc_step(input logic [15:0] old, input logic [7:0] byte_data);
        logic [15:0] polynomial;
        begin
            polynomial = old ^ {byte_data, 8'd0};
            for (integer i = 0; i < 8; i = i + 1)
                polynomial = (polynomial << 1) ^ (polynomial[15] ? 16'h1021 : 16'd0);
            crc_step = polynomial;
        end
    endfunction

    typedef enum logic [2:0] {CLOSED, OPEN_FLAG, ADDR, PICK, ENCODE, SEND, TAIL, CLOSE_FLAG} tx_state;
    tx_state phase;
    logic [7:0] command, rx_position, tx_position, tx_length, body_octet, encoded, tail_byte, raw;
    logic [15:0] rx_crc, tx_crc, token, live_token;
    logic [31:0] tag_buffer;
    logic opened, escape_rx, owns_image, padding_error, waiting, launch, run_tilt, last_octet, send_tail;
    logic busy, done, deferred, label_bit;
    wire [15:0] clocks;
`ifndef SC_GROWTH
    logic [7:0] fast_clocks;
    assign clocks = {8'd0, fast_clocks};
`endif
    logic [IX-1:0] rx_plane, tx_plane;
    logic [LANE_W-1:0] rx_lane, tx_lane;
    logic [8*ROW_BYTES-1:0] shifted_row, assembled;
    wire [W-1:0] reduced;
    wire [8*ROW_BYTES-1:0] padded = {{(8*ROW_BYTES-W){1'b0}}, reduced};
    wire [7:0] body_rx = escape_rx ? incoming ^ 8'h20 : incoming;
    wire idle = !waiting && !busy && phase == CLOSED && tx_ready;
    wire receive_data = got && opened && incoming != FLAG && (escape_rx || incoming != ESC);
    wire payload = receive_data && owns_image && rx_position >= 7 && rx_position < 7+BYTES;
    wire read_payload = deferred && tx_position >= HEADER && tx_position < tx_length;
    wire read_row = read_payload && phase == ADDR;
    wire write_row = payload && rx_lane == ROW_BYTES-1;
    wire advance = tx_ready && (phase == TAIL || (phase == SEND && !send_tail));
    wire escaping = raw == FLAG || raw == ESC;

    assign assembled = (shifted_row >> 8) | (8*ROW_BYTES)'(body_rx) << (8*(ROW_BYTES-1));

`ifdef SC_GROWTH
    sc_cluster_telescope core(
`elsif SC_LOOKUP
    sc_lookup core(
`else
    sc_telescope core(
`endif
        .clk(clk), .reset(reset), .start(launch), .tilt_enable(run_tilt),
        .target_write(write_row), .target_index(rx_plane), .target_row(assembled[W-1:0]),
        .remainder_read(read_row), .remainder_index(tx_plane), .remainder_row(reduced),
        .busy(busy), .done(done), .deferred(deferred), .logical_bit(label_bit),
`ifdef SC_GROWTH
        .clocks(clocks)
`else
        .clocks(fast_clocks)
`endif
    );

    always_ff @(posedge clk) begin
        if (reset) begin
            opened <= 0;
            waiting <= 0;
            launch <= 0;
            escape_rx <= 0;
            rx_position <= 0;
            owns_image <= 0;
        end else begin
            launch <= 0;
            if (done) waiting <= 0;
            if (got && incoming == FLAG) begin
                if (opened && !escape_rx && rx_position == 9+BYTES && rx_crc == 0 && !padding_error &&
                    (command == 8'h45 || command == 8'h46) && tag_buffer == TAG && owns_image && idle) begin
                    waiting <= 1;
                    launch <= 1;
                    live_token <= token;
                    run_tilt <= command == 8'h45;
                end
                opened <= 1;
                escape_rx <= 0;
                rx_position <= 0;
                rx_crc <= 16'hffff;
                owns_image <= 0;
                rx_plane <= 0;
                rx_lane <= 0;
                padding_error <= 0;
            end else if (got && opened && incoming == ESC && !escape_rx) escape_rx <= 1;
            else if (receive_data) begin
                escape_rx <= 0;
                if (rx_position == 9+BYTES) opened <= 0;
                else begin
                    rx_position <= rx_position + 1'b1;
                    rx_crc <= crc_step(rx_crc, body_rx);
                    case (rx_position)
                        0: begin command <= body_rx; owns_image <= idle; end
                        1: token[7:0] <= body_rx;
                        2: token[15:8] <= body_rx;
                        3: tag_buffer[7:0] <= body_rx;
                        4: tag_buffer[15:8] <= body_rx;
                        5: tag_buffer[23:16] <= body_rx;
                        6: tag_buffer[31:24] <= body_rx;
                        default: if (payload) begin
                            shifted_row <= assembled;
                            if (write_row) begin
                                rx_lane <= 0;
                                rx_plane <= rx_plane + 1'b1;
                                if (|((assembled >> W))) padding_error <= 1;
                            end else rx_lane <= rx_lane + 1'b1;
                        end
                    endcase
                end
            end
        end
    end

    always_comb begin
        tx_length = HEADER + (deferred ? 8'(BYTES) : 8'd0);
        case (tx_position)
            0: body_octet = 8'h92;
            1: body_octet = live_token[7:0];
            2: body_octet = live_token[15:8];
            3: body_octet = TAG[7:0];
            4: body_octet = TAG[15:8];
            5: body_octet = TAG[23:16];
            6: body_octet = TAG[31:24];
            7: body_octet = deferred ? 8'd3 : 8'd1;
            8: body_octet = clocks[7:0];
            9: body_octet = clocks[15:8];
            10: body_octet = {7'd0, label_bit};
            11: body_octet = deferred ? 8'(BYTES) : 8'd0;
            default: body_octet = padded[{tx_lane, 3'b000}+:8];
        endcase
        sent = phase == OPEN_FLAG || phase == SEND || phase == TAIL || phase == CLOSE_FLAG;
        case (phase)
            SEND: outgoing = encoded;
            TAIL: outgoing = tail_byte;
            default: outgoing = FLAG;
        endcase
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            phase <= CLOSED;
        end else if (done && waiting) begin
            phase <= OPEN_FLAG;
            tx_position <= 0;
            tx_plane <= 0;
            tx_lane <= 0;
            tx_crc <= 16'hffff;
        end else begin
            if (advance) begin
                tx_position <= tx_position + 1'b1;
                if (read_payload) begin
                    if (tx_lane == ROW_BYTES-1) begin tx_lane <= 0; tx_plane <= tx_plane + 1'b1; end
                    else tx_lane <= tx_lane + 1'b1;
                end
            end
            case (phase)
                OPEN_FLAG: if (tx_ready) phase <= ADDR;
                ADDR: phase <= PICK;
                PICK: begin
                    raw <= tx_position < tx_length ? body_octet :
                           tx_position == tx_length ? tx_crc[15:8] : tx_crc[7:0];
                    last_octet <= tx_position == tx_length+1;
                    phase <= ENCODE;
                end
                ENCODE: begin
                    if (tx_position < tx_length) tx_crc <= crc_step(tx_crc, raw);
                    encoded <= escaping ? ESC : raw;
                    tail_byte <= raw ^ 8'h20;
                    send_tail <= escaping;
                    phase <= SEND;
                end
                SEND: if (tx_ready) phase <= send_tail ? TAIL : last_octet ? CLOSE_FLAG : ADDR;
                TAIL: if (tx_ready) phase <= last_octet ? CLOSE_FLAG : ADDR;
                CLOSE_FLAG: if (tx_ready) phase <= CLOSED;
                default: phase <= CLOSED;
            endcase
        end
    end
endmodule

`ifndef SC_CLOCK_MULT
`define SC_CLOCK_MULT 5
`endif
`ifndef SC_CLOCK_DIV
`define SC_CLOCK_DIV 1
`endif

module tang_nano20k (
    input logic board_clock, usb_receive,
    output logic usb_transmit
);
    localparam integer MULT = `SC_CLOCK_MULT, DIV = `SC_CLOCK_DIV;
    localparam integer FABRIC_MHZ = 27*MULT/DIV;
    logic fabric, locked;
    if (MULT == DIV) begin : direct_clock
        assign fabric = board_clock;
        assign locked = 1'b1;
    end else begin : pll_clock
        rPLL #(
            .FCLKIN("27.0"), .IDIV_SEL(DIV-1), .FBDIV_SEL(MULT-1),
            .ODIV_SEL(FABRIC_MHZ < 50 ? 32 : FABRIC_MHZ < 80 ? 16 : 8)
        ) pll (
            .CLKIN(board_clock), .CLKOUT(fabric), .LOCK(locked),
            .CLKOUTP(), .CLKOUTD(), .CLKOUTD3(), .CLKFB(1'b0),
            .RESET(1'b0), .RESET_P(1'b0), .FBDSEL(6'b0), .IDSEL(6'b0), .ODSEL(6'b0),
            .PSDA(4'b0), .DUTYDA(4'b0), .FDLY(4'b0)
        );
    end
    (* ASYNC_REG = "TRUE" *) logic [2:0] release_reset = 3'b111;
    always_ff @(posedge fabric or negedge locked) begin
        if (!locked) release_reset <= 3'b111;
        else release_reset <= {release_reset[1:0], 1'b0};
    end
    sc_link #(.UART_DIV(FABRIC_MHZ)) endpoint(.clk(fabric), .reset(release_reset[2]),
                                          .rx(usb_receive), .tx(usb_transmit));
endmodule
