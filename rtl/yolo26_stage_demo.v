// ============================================================================
// Module: yolo26_stage_demo
// Description: Top-Level Integration of YOLO26 Building Blocks.
//              Demonstrates the exact cascade from Backbone P5 (Figure S1):
//                Input -> Conv (Downsample 3x3/2) 
//                      -> C3k2 (with C3k inner units) 
//                      -> SPPF (Spatial Pyramid Pooling Fast, k=5) 
//                      -> C2PSA (Position-Sensitive Attention CSP) 
//                      -> Output to Neck / Detect Head
//              Targeted for AMD Kria KV260/KR260 FPGA Accelerator.
// ============================================================================

`timescale 1ns / 1ps

module yolo26_stage_demo #(
    parameter DATA_WIDTH  = 16,
    parameter FRAC_BITS   = 8,
    parameter CHANNELS    = 4,  // Scaled channels for simulation/demo
    parameter IMG_WIDTH   = 16,
    parameter IMG_HEIGHT  = 16
)(
    input  wire                              clk,
    input  wire                              rst_n,
    
    // Top-level input stream
    input  wire                              in_valid,
    input  wire [CHANNELS*DATA_WIDTH-1:0]    in_data,
    
    // Top-level output stream
    output wire                              out_valid,
    output wire [CHANNELS*DATA_WIDTH-1:0]    out_data
);

    // ------------------------------------------------------------------------
    // Stage 1: Conv 3x3 / 2 (Downsample by 2, halves width and height)
    // ------------------------------------------------------------------------
    wire                              conv_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]    conv_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .KERNEL_SIZE(3),
        .STRIDE(2), // Stride 2 downsamples
        .PADDING(1),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(1) // SiLU
    ) u_stage_conv (
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
        .out_valid(conv_valid),
        .out_data(conv_data)
    );

    // Output dimension of Stage 1 is halved:
    localparam DS_WIDTH  = IMG_WIDTH / 2;
    localparam DS_HEIGHT = IMG_HEIGHT / 2;

    // ------------------------------------------------------------------------
    // Stage 2: C3k2 Block (with C3k inner units)
    // ------------------------------------------------------------------------
    wire                              c3k2_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]    c3k2_data;

    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .NUM_BLOCKS(2),
        .C3K_MODE(1), // C3k mode enabled
        .ATTN_MODE(0),
        .IMG_WIDTH(DS_WIDTH),
        .IMG_HEIGHT(DS_HEIGHT)
    ) u_stage_c3k2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(conv_valid),
        .in_data(conv_data),
        .out_valid(c3k2_valid),
        .out_data(c3k2_data)
    );

    // ------------------------------------------------------------------------
    // Stage 3: SPPF Block (Spatial Pyramid Pooling Fast, k=5)
    // ------------------------------------------------------------------------
    wire                              sppf_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]    sppf_data;

    yolo_sppf #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .POOL_SIZE(5),
        .IMG_WIDTH(DS_WIDTH),
        .IMG_HEIGHT(DS_HEIGHT)
    ) u_stage_sppf (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(c3k2_valid),
        .in_data(c3k2_data),
        .out_valid(sppf_valid),
        .out_data(sppf_data)
    );

    // ------------------------------------------------------------------------
    // Stage 4: C2PSA Block (Position-Sensitive Attention CSP)
    // ------------------------------------------------------------------------
    yolo_c2psa #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .NUM_PSA(2),
        .NUM_HEADS(2),
        .IMG_WIDTH(DS_WIDTH),
        .IMG_HEIGHT(DS_HEIGHT)
    ) u_stage_c2psa (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(sppf_valid),
        .in_data(sppf_data),
        .out_valid(out_valid),
        .out_data(out_data)
    );

endmodule
