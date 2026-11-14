// ============================================================================
// Module: yolo_bottleneck
// Description: Standard YOLO26 Residual Bottleneck Block.
//              Architecture:
//                Input -> Conv1 (c_in -> c_hidden, k1) 
//                      -> Conv2 (c_hidden -> c_out, k2) 
//                      -> Add (Input + Conv2) if shortcut & (c_in == c_out)
//              Uses 16-bit Signed Q8.8 Fixed-Point format.
//              Optimized for AMD Kria KV260/KR260.
// ============================================================================

`timescale 1ns / 1ps

module yolo_bottleneck #(
    parameter DATA_WIDTH   = 16, // Q8.8
    parameter FRAC_BITS    = 8,
    parameter IN_CHANNELS  = 4,
    parameter OUT_CHANNELS = 4,
    parameter HIDDEN_CHANS = 4,
    parameter KERNEL_SIZE1 = 3,  // Default 3x3
    parameter KERNEL_SIZE2 = 3,  // Default 3x3
    parameter SHORTCUT     = 1,  // 1: Enable residual shortcut
    parameter IMG_WIDTH    = 16,
    parameter IMG_HEIGHT   = 16
)(
    input  wire                              clk,
    input  wire                              rst_n,
    
    // Feature stream input
    input  wire                              in_valid,
    input  wire [IN_CHANNELS*DATA_WIDTH-1:0] in_data,
    
    // Feature stream output
    output wire                              out_valid,
    output wire [OUT_CHANNELS*DATA_WIDTH-1:0] out_data
);

    // ------------------------------------------------------------------------
    // Conv 1: IN_CHANNELS -> HIDDEN_CHANS
    // ------------------------------------------------------------------------
    wire                              cv1_valid;
    wire [HIDDEN_CHANS*DATA_WIDTH-1:0] cv1_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(HIDDEN_CHANS),
        .KERNEL_SIZE(KERNEL_SIZE1),
        .STRIDE(1),
        .PADDING(KERNEL_SIZE1 == 3 ? 1 : 0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(1)
    ) u_conv1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(cv1_valid),
        .out_data(cv1_data)
    );

    // ------------------------------------------------------------------------
    // Conv 2: HIDDEN_CHANS -> OUT_CHANNELS
    // ------------------------------------------------------------------------
    wire                              cv2_valid;
    wire [OUT_CHANNELS*DATA_WIDTH-1:0] cv2_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(HIDDEN_CHANS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE2),
        .STRIDE(1),
        .PADDING(KERNEL_SIZE2 == 3 ? 1 : 0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(1)
    ) u_conv2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(cv1_valid),
        .in_data(cv1_data),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(cv2_valid),
        .out_data(cv2_data)
    );

    // ------------------------------------------------------------------------
    // Residual Shortcut Alignment FIFO
    // Buffers in_data until cv2_valid is asserted, then performs saturated add
    // ------------------------------------------------------------------------
    localparam FIFO_DEPTH = 256;
    reg [IN_CHANNELS*DATA_WIDTH-1:0] sc_fifo [0:FIFO_DEPTH-1];
    reg [7:0] wr_ptr, rd_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 8'd0;
            rd_ptr <= 8'd0;
        end else begin
            if (in_valid) begin
                sc_fifo[wr_ptr] <= in_data;
                wr_ptr <= wr_ptr + 8'd1;
            end
            if (cv2_valid) begin
                rd_ptr <= rd_ptr + 8'd1;
            end
        end
    end

    wire [IN_CHANNELS*DATA_WIDTH-1:0] sc_data = sc_fifo[rd_ptr];

    // Unpack channels
    wire signed [DATA_WIDTH-1:0] cv2_chan [0:OUT_CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] sc_chan  [0:IN_CHANNELS-1];
    genvar ui;
    generate
        for (ui = 0; ui < OUT_CHANNELS; ui = ui + 1) begin : UNPACK_CV2
            assign cv2_chan[ui] = cv2_data[ui*DATA_WIDTH +: DATA_WIDTH];
        end
        for (ui = 0; ui < IN_CHANNELS; ui = ui + 1) begin : UNPACK_SC
            assign sc_chan[ui]  = sc_data[ui*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Saturated channel-wise addition
    reg signed [DATA_WIDTH-1:0] res_add [0:OUT_CHANNELS-1];
    integer c;
    reg signed [DATA_WIDTH:0] sum_extended;

    always @(*) begin
        for (c = 0; c < OUT_CHANNELS; c = c + 1) begin
            if (SHORTCUT && (IN_CHANNELS == OUT_CHANNELS)) begin
                sum_extended = $signed(cv2_chan[c]) + $signed(sc_chan[c]);
                if (sum_extended > 17'sh07FFF)
                    res_add[c] = 16'sh7FFF;
                else if (sum_extended < -17'sh08000)
                    res_add[c] = 16'sh8000;
                else
                    res_add[c] = sum_extended[DATA_WIDTH-1:0];
            end else begin
                res_add[c] = cv2_chan[c];
            end
        end
    end

    // Pack output
    genvar pk;
    generate
        for (pk = 0; pk < OUT_CHANNELS; pk = pk + 1) begin : PACK_OUT
            assign out_data[pk*DATA_WIDTH +: DATA_WIDTH] = res_add[pk];
        end
    endgenerate

    assign out_valid = cv2_valid;

endmodule
