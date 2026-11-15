// ============================================================================
// Module: yolo_c3k
// Description: C3k Sub-block for YOLO26.
//              Contains two stacked Bottleneck units with 3x3 convolution paths.
//              Referenced in YOLO26 paper Fig S2 and Kria RTL Roadmap.
// ============================================================================

`timescale 1ns / 1ps

module yolo_c3k #(
    parameter DATA_WIDTH   = 16,
    parameter FRAC_BITS    = 8,
    parameter CHANNELS     = 4,
    parameter SHORTCUT     = 1,
    parameter IMG_WIDTH    = 16,
    parameter IMG_HEIGHT   = 16
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    input  wire [CHANNELS*DATA_WIDTH-1:0]    in_data,
    output wire                              out_valid,
    output wire [CHANNELS*DATA_WIDTH-1:0]    out_data
);

    wire                            b1_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]  b1_data;

    // Bottleneck 1
    yolo_bottleneck #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .HIDDEN_CHANS(CHANNELS),
        .KERNEL_SIZE1(3),
        .KERNEL_SIZE2(3),
        .SHORTCUT(SHORTCUT),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_bottleneck_1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .out_valid(b1_valid),
        .out_data(b1_data)
    );

    // Bottleneck 2
    yolo_bottleneck #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .HIDDEN_CHANS(CHANNELS),
        .KERNEL_SIZE1(3),
        .KERNEL_SIZE2(3),
        .SHORTCUT(SHORTCUT),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_bottleneck_2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(b1_valid),
        .in_data(b1_data),
        .out_valid(out_valid),
        .out_data(out_data)
    );

endmodule
