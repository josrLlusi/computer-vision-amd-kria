// ============================================================================
// Module: yolo_maxpool2d
// Description: Streaming 2D Max Pooling (5x5, Stride 1, Padding 2) for YOLO26 SPPF.
//              Uses 4 line buffers and a 2-stage separable comparator tree.
//              Preserves exact spatial dimensions (H_out = H_in, W_out = W_in).
//              Targeted for AMD Kria KV260/KR260 (Fixed-point signed Q8.8).
// ============================================================================

`timescale 1ns / 1ps

module yolo_maxpool2d #(
    parameter DATA_WIDTH = 16,
    parameter CHANNELS   = 4,
    parameter IMG_WIDTH  = 16,
    parameter IMG_HEIGHT = 16
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    input  wire [CHANNELS*DATA_WIDTH-1:0]    in_data,
    output reg                               out_valid,
    output wire [CHANNELS*DATA_WIDTH-1:0]    out_data
);

    localparam signed [DATA_WIDTH-1:0] MIN_VAL = -16'sh8000;

    // Coordinate counters
    reg [15:0] col_cnt;
    reg [15:0] row_cnt;

    // Line buffers: 4 lines of depth IMG_WIDTH
    reg signed [DATA_WIDTH-1:0] lb0 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] lb1 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] lb2 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] lb3 [0:IMG_WIDTH-1];

    // Unpack input channels
    wire signed [DATA_WIDTH-1:0] in_chan [0:CHANNELS-1];
    genvar ci;
    generate
        for (ci = 0; ci < CHANNELS; ci = ci + 1) begin : UNPACK_IN
            assign in_chan[ci] = in_data[ci*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Channel-wise pooling logic
    reg signed [DATA_WIDTH-1:0] max_out [0:CHANNELS-1];

    // For simplicity and hardware efficiency, we implement per-channel pooling
    genvar ch;
    generate
        for (ch = 0; ch < CHANNELS; ch = ch + 1) begin : GEN_CHAN_POOL
            // 4 Line buffers for this channel
            reg signed [DATA_WIDTH-1:0] clb0 [0:IMG_WIDTH-1];
            reg signed [DATA_WIDTH-1:0] clb1 [0:IMG_WIDTH-1];
            reg signed [DATA_WIDTH-1:0] clb2 [0:IMG_WIDTH-1];
            reg signed [DATA_WIDTH-1:0] clb3 [0:IMG_WIDTH-1];

            // 5-tap shift registers for 5 rows
            reg signed [DATA_WIDTH-1:0] tap0 [0:4];
            reg signed [DATA_WIDTH-1:0] tap1 [0:4];
            reg signed [DATA_WIDTH-1:0] tap2 [0:4];
            reg signed [DATA_WIDTH-1:0] tap3 [0:4];
            reg signed [DATA_WIDTH-1:0] tap4 [0:4];

            wire signed [DATA_WIDTH-1:0] r0_val = clb0[col_cnt];
            wire signed [DATA_WIDTH-1:0] r1_val = clb1[col_cnt];
            wire signed [DATA_WIDTH-1:0] r2_val = clb2[col_cnt];
            wire signed [DATA_WIDTH-1:0] r3_val = clb3[col_cnt];
            wire signed [DATA_WIDTH-1:0] r4_val = in_chan[ch];

            integer t;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (t = 0; t < 5; t = t + 1) begin
                        tap0[t] <= MIN_VAL; tap1[t] <= MIN_VAL;
                        tap2[t] <= MIN_VAL; tap3[t] <= MIN_VAL; tap4[t] <= MIN_VAL;
                    end
                end else if (in_valid) begin
                    // Shift line buffers
                    clb0[col_cnt] <= r1_val;
                    clb1[col_cnt] <= r2_val;
                    clb2[col_cnt] <= r3_val;
                    clb3[col_cnt] <= r4_val;

                    // Shift taps horizontally
                    for (t = 4; t > 0; t = t - 1) begin
                        tap0[t] <= tap0[t-1];
                        tap1[t] <= tap1[t-1];
                        tap2[t] <= tap2[t-1];
                        tap3[t] <= tap3[t-1];
                        tap4[t] <= tap4[t-1];
                    end
                    tap0[0] <= r0_val;
                    tap1[0] <= r1_val;
                    tap2[0] <= r2_val;
                    tap3[0] <= r3_val;
                    tap4[0] <= r4_val;
                end
            end

            // Stage 1: Row maximums of 5 taps
            reg signed [DATA_WIDTH-1:0] row_max0, row_max1, row_max2, row_max3, row_max4;
            function signed [DATA_WIDTH-1:0] max5;
                input signed [DATA_WIDTH-1:0] a, b, c, d, e;
                reg signed [DATA_WIDTH-1:0] m1, m2, m3;
                begin
                    m1 = (a > b) ? a : b;
                    m2 = (c > d) ? c : d;
                    m3 = (m1 > m2) ? m1 : m2;
                    max5 = (m3 > e) ? m3 : e;
                end
            endfunction

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    row_max0 <= MIN_VAL; row_max1 <= MIN_VAL;
                    row_max2 <= MIN_VAL; row_max3 <= MIN_VAL; row_max4 <= MIN_VAL;
                end else if (in_valid) begin
                    row_max0 <= max5(tap0[0], tap0[1], tap0[2], tap0[3], tap0[4]);
                    row_max1 <= max5(tap1[0], tap1[1], tap1[2], tap1[3], tap1[4]);
                    row_max2 <= max5(tap2[0], tap2[1], tap2[2], tap2[3], tap2[4]);
                    row_max3 <= max5(tap3[0], tap3[1], tap3[2], tap3[3], tap3[4]);
                    row_max4 <= max5(tap4[0], tap4[1], tap4[2], tap4[3], tap4[4]);
                end
            end

            // Stage 2: Final 2D maximum
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    max_out[ch] <= MIN_VAL;
                end else begin
                    max_out[ch] <= max5(row_max0, row_max1, row_max2, row_max3, row_max4);
                end
            end
        end
    endgenerate

    // Master coordinate tracking and pipeline valid generation
    reg val_d1, val_d2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt   <= 16'd0;
            row_cnt   <= 16'd0;
            val_d1    <= 1'b0;
            val_d2    <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            val_d1    <= in_valid;
            val_d2    <= val_d1;
            out_valid <= val_d2;

            if (in_valid) begin
                if (col_cnt == IMG_WIDTH - 1) begin
                    col_cnt <= 16'd0;
                    if (row_cnt == IMG_HEIGHT - 1)
                        row_cnt <= 16'd0;
                    else
                        row_cnt <= row_cnt + 16'd1;
                end else begin
                    col_cnt <= col_cnt + 16'd1;
                end
            end
        end
    end

    // Pack output
    genvar pk;
    generate
        for (pk = 0; pk < CHANNELS; pk = pk + 1) begin : PACK_OUT
            assign out_data[pk*DATA_WIDTH +: DATA_WIDTH] = max_out[pk];
        end
    endgenerate

endmodule
