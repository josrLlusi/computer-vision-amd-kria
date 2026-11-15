`timescale 1ns / 1ps

module yolo_neck_fusion #(
    parameter DATA_WIDTH = 16,  // Q8.8
    parameter FRAC_BITS  = 8,
    parameter P3_CH      = 128,
    parameter P4_CH      = 128,
    parameter P5_CH      = 256,
    parameter IMG_WIDTH  = 80,  // Lebar feature map P3 (IMG / 8)
    parameter IMG_HEIGHT = 80   // Tinggi feature map P3 (IMG / 8)
)(
    input  wire                           clk,
    input  wire                           rst_n,

    input  wire [(P3_CH*DATA_WIDTH)-1:0]  p3_data_in,
    input  wire                           p3_valid_in,
    output wire                           p3_ready_out,

    input  wire [(P4_CH*DATA_WIDTH)-1:0]  p4_data_in,
    input  wire                           p4_valid_in,
    output wire                           p4_ready_out,

    input  wire [(P5_CH*DATA_WIDTH)-1:0]  p5_data_in,
    input  wire                           p5_valid_in,
    output wire                           p5_ready_out,

    output wire [(P3_CH*DATA_WIDTH)-1:0]  head_small_data,
    output wire                           head_small_valid,
    input  wire                           head_small_ready,

    output wire [(P4_CH*DATA_WIDTH)-1:0]  head_medium_data,
    output wire                           head_medium_valid,
    input  wire                           head_medium_ready,

    output wire [(P5_CH*DATA_WIDTH)-1:0]  head_large_data,
    output wire                           head_large_valid,
    input  wire                           head_large_ready
);

    localparam int P4_W = (IMG_WIDTH  > 1) ? (IMG_WIDTH  / 2) : 1;
    localparam int P4_H = (IMG_HEIGHT > 1) ? (IMG_HEIGHT / 2) : 1;
    localparam int P5_W = (P4_W > 1) ? (P4_W / 2) : 1;
    localparam int P5_H = (P4_H > 1) ? (P4_H / 2) : 1;

    // ==========================================
    // 1. JALUR TOP-DOWN (Atas ke Bawah)
    // ==========================================

    // Split P5 (1x1) ke Upsample dan ke Skip FIFO untuk Bottom-Up Stage 2
    wire                          p5_up_in_ready, p5_skip_in_ready;
    wire                          p5_in_fire;
    wire [(P5_CH*DATA_WIDTH)-1:0] p5_skip_data;
    wire                          p5_skip_valid;
    wire                          p5_skip_ready;

    assign p5_in_fire   = p5_valid_in && p5_up_in_ready && p5_skip_in_ready;
    assign p5_ready_out = p5_up_in_ready && p5_skip_in_ready;

    stream_fifo #(
        .WIDTH(P5_CH * DATA_WIDTH),
        .DEPTH((P5_W * P5_H) + 8)
    ) u_fifo_p5_skip (
        .clk(clk), .rst_n(rst_n),
        .in_data(p5_data_in),    .in_valid(p5_in_fire),     .in_ready(p5_skip_in_ready),
        .out_data(p5_skip_data), .out_valid(p5_skip_valid), .out_ready(p5_skip_ready)
    );

    // 1A. Upsample P5 (P5_W x P5_H -> P4_W x P4_H)
    wire [(P5_CH*DATA_WIDTH)-1:0] p5_up_data;
    wire                          p5_up_valid;
    wire                          p5_up_ready;

    yolo_upsample #(
        .CHANNELS(P5_CH),
        .DATA_WIDTH(DATA_WIDTH),
        .IN_WIDTH(P5_W),
        .IN_HEIGHT(P5_H)
    ) u_upsample_p5 (
        .clk(clk), .rst_n(rst_n),
        .data_in(p5_data_in),   .valid_in(p5_in_fire),    .ready_out(p5_up_in_ready),
        .data_out(p5_up_data),  .valid_out(p5_up_valid),  .ready_in(p5_up_ready)
    );

    // 1B. Concat (P5_up + P4)
    wire [((P5_CH+P4_CH)*DATA_WIDTH)-1:0] concat_p5_p4_data;
    wire                                  concat_p5_p4_valid;

    yolo_concat #(
        .CH_A(P5_CH), .CH_B(P4_CH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_p5_p4 (
        .clk(clk), .rst_n(rst_n),
        .data_a(p5_up_data),          .valid_a(p5_up_valid),          .ready_a(p5_up_ready),
        .data_b(p4_data_in),          .valid_b(p4_valid_in),          .ready_b(p4_ready_out),
        .data_out(concat_p5_p4_data), .valid_out(concat_p5_p4_valid), .ready_in(1'b1)
    );

    // 1C. C3k2 TD1 (valid-only -> buffered by stream_fifo)
    wire [(P4_CH*DATA_WIDTH)-1:0] c3k2_td1_raw_data;
    wire                          c3k2_td1_raw_valid;

    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(P5_CH + P4_CH),
        .OUT_CHANNELS(P4_CH),
        .NUM_BLOCKS(2),
        .C3K_MODE(0),
        .ATTN_MODE(0),
        .IMG_WIDTH(P4_W),
        .IMG_HEIGHT(P4_H)
    ) u_c3k2_td1 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(concat_p5_p4_valid),
        .in_data(concat_p5_p4_data),
        .out_valid(c3k2_td1_raw_valid),
        .out_data(c3k2_td1_raw_data)
    );

    wire [(P4_CH*DATA_WIDTH)-1:0] c3k2_td1_data;
    wire                          c3k2_td1_valid;
    wire                          c3k2_td1_ready;
    wire                          td1_fifo_in_ready;

    stream_fifo #(
        .WIDTH(P4_CH * DATA_WIDTH),
        .DEPTH((P4_W * P4_H) + 8)
    ) u_fifo_td1_buf (
        .clk(clk), .rst_n(rst_n),
        .in_data(c3k2_td1_raw_data), .in_valid(c3k2_td1_raw_valid), .in_ready(td1_fifo_in_ready),
        .out_data(c3k2_td1_data),    .out_valid(c3k2_td1_valid),    .out_ready(c3k2_td1_ready)
    );

    // Split TD1 ke Upsample dan ke Skip FIFO untuk Bottom-Up Stage 1
    wire                          td1_up_in_ready, td1_skip_in_ready;
    wire                          td1_fire;
    wire [(P4_CH*DATA_WIDTH)-1:0] td1_skip_data;
    wire                          td1_skip_valid;
    wire                          td1_skip_ready;

    assign td1_fire       = c3k2_td1_valid && td1_up_in_ready && td1_skip_in_ready;
    assign c3k2_td1_ready = td1_up_in_ready && td1_skip_in_ready;

    stream_fifo #(
        .WIDTH(P4_CH * DATA_WIDTH),
        .DEPTH((P4_W * P4_H) + 8)
    ) u_fifo_td1_skip (
        .clk(clk), .rst_n(rst_n),
        .in_data(c3k2_td1_data),  .in_valid(td1_fire),        .in_ready(td1_skip_in_ready),
        .out_data(td1_skip_data), .out_valid(td1_skip_valid), .out_ready(td1_skip_ready)
    );

    // 1D. Upsample TD1 (P4_W x P4_H -> IMG_WIDTH x IMG_HEIGHT)
    wire [(P4_CH*DATA_WIDTH)-1:0] td1_up_data;
    wire                          td1_up_valid;
    wire                          td1_up_ready;

    yolo_upsample #(
        .CHANNELS(P4_CH),
        .DATA_WIDTH(DATA_WIDTH),
        .IN_WIDTH(P4_W),
        .IN_HEIGHT(P4_H)
    ) u_upsample_td1 (
        .clk(clk), .rst_n(rst_n),
        .data_in(c3k2_td1_data), .valid_in(td1_fire),        .ready_out(td1_up_in_ready),
        .data_out(td1_up_data),  .valid_out(td1_up_valid),   .ready_in(td1_up_ready)
    );

    // 1E. Concat (TD1_up + P3)
    wire [((P4_CH+P3_CH)*DATA_WIDTH)-1:0] concat_td1_p3_data;
    wire                                  concat_td1_p3_valid;

    yolo_concat #(
        .CH_A(P4_CH), .CH_B(P3_CH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_td1_p3 (
        .clk(clk), .rst_n(rst_n),
        .data_a(td1_up_data),          .valid_a(td1_up_valid),          .ready_a(td1_up_ready),
        .data_b(p3_data_in),           .valid_b(p3_valid_in),           .ready_b(p3_ready_out),
        .data_out(concat_td1_p3_data), .valid_out(concat_td1_p3_valid), .ready_in(1'b1)
    );

    // 1F. C3k2 Top (valid-only -> buffered by stream_fifo)
    wire [(P3_CH*DATA_WIDTH)-1:0] c3k2_top_raw_data;
    wire                          c3k2_top_raw_valid;

    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(P4_CH + P3_CH),
        .OUT_CHANNELS(P3_CH),
        .NUM_BLOCKS(2),
        .C3K_MODE(0),
        .ATTN_MODE(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_c3k2_top (
        .clk(clk), .rst_n(rst_n),
        .in_valid(concat_td1_p3_valid),
        .in_data(concat_td1_p3_data),
        .out_valid(c3k2_top_raw_valid),
        .out_data(c3k2_top_raw_data)
    );

    wire [(P3_CH*DATA_WIDTH)-1:0] c3k2_top_data;
    wire                          c3k2_top_valid;
    wire                          c3k2_top_ready;
    wire                          top_fifo_in_ready;

    stream_fifo #(
        .WIDTH(P3_CH * DATA_WIDTH),
        .DEPTH((IMG_WIDTH * IMG_HEIGHT) + 8)
    ) u_fifo_top_buf (
        .clk(clk), .rst_n(rst_n),
        .in_data(c3k2_top_raw_data), .in_valid(c3k2_top_raw_valid), .in_ready(top_fifo_in_ready),
        .out_data(c3k2_top_data),    .out_valid(c3k2_top_valid),    .out_ready(c3k2_top_ready)
    );

    // Split Top ke Head Small dan ke Downsample Top
    wire top_ds_in_ready;
    wire top_fire;
    assign top_fire         = c3k2_top_valid && head_small_ready && top_ds_in_ready;
    assign c3k2_top_ready   = head_small_ready && top_ds_in_ready;
    assign head_small_data  = c3k2_top_data;
    assign head_small_valid = top_fire;

    // ==========================================
    // 2. JALUR BOTTOM-UP (Bawah ke Atas)
    // ==========================================

    // 2A. Downsample Top (IMG_WIDTH x IMG_HEIGHT -> P4_W x P4_H)
    wire [(P3_CH*DATA_WIDTH)-1:0] top_down_data;
    wire                          top_down_valid;
    wire                          top_down_ready;

    yolo_conv_downsample #(
        .IN_CH(P3_CH), .OUT_CH(P3_CH),
        .DATA_WIDTH(DATA_WIDTH),
        .IN_WIDTH(IMG_WIDTH),
        .IN_HEIGHT(IMG_HEIGHT)
    ) u_downsample_top (
        .clk(clk), .rst_n(rst_n),
        .data_in(c3k2_top_data),  .valid_in(top_fire),        .ready_out(top_ds_in_ready),
        .data_out(top_down_data), .valid_out(top_down_valid), .ready_in(top_down_ready)
    );

    // 2B. Concat (Top_down + TD1_skip)
    wire [((P3_CH+P4_CH)*DATA_WIDTH)-1:0] concat_top_td1_data;
    wire                                  concat_top_td1_valid;

    yolo_concat #(
        .CH_A(P3_CH), .CH_B(P4_CH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_top_td1 (
        .clk(clk), .rst_n(rst_n),
        .data_a(top_down_data),         .valid_a(top_down_valid),         .ready_a(top_down_ready),
        .data_b(td1_skip_data),         .valid_b(td1_skip_valid),         .ready_b(td1_skip_ready),
        .data_out(concat_top_td1_data), .valid_out(concat_top_td1_valid), .ready_in(1'b1)
    );

    // 2C. C3k2 Mid (valid-only -> buffered by stream_fifo)
    wire [(P4_CH*DATA_WIDTH)-1:0] c3k2_mid_raw_data;
    wire                          c3k2_mid_raw_valid;

    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(P3_CH + P4_CH),
        .OUT_CHANNELS(P4_CH),
        .NUM_BLOCKS(2),
        .C3K_MODE(0),
        .ATTN_MODE(0),
        .IMG_WIDTH(P4_W),
        .IMG_HEIGHT(P4_H)
    ) u_c3k2_mid (
        .clk(clk), .rst_n(rst_n),
        .in_valid(concat_top_td1_valid),
        .in_data(concat_top_td1_data),
        .out_valid(c3k2_mid_raw_valid),
        .out_data(c3k2_mid_raw_data)
    );

    wire [(P4_CH*DATA_WIDTH)-1:0] c3k2_mid_data;
    wire                          c3k2_mid_valid;
    wire                          c3k2_mid_ready;
    wire                          mid_fifo_in_ready;

    stream_fifo #(
        .WIDTH(P4_CH * DATA_WIDTH),
        .DEPTH((P4_W * P4_H) + 8)
    ) u_fifo_mid_buf (
        .clk(clk), .rst_n(rst_n),
        .in_data(c3k2_mid_raw_data), .in_valid(c3k2_mid_raw_valid), .in_ready(mid_fifo_in_ready),
        .out_data(c3k2_mid_data),    .out_valid(c3k2_mid_valid),    .out_ready(c3k2_mid_ready)
    );

    // Split Mid ke Head Medium dan ke Downsample Mid
    wire mid_ds_in_ready;
    wire mid_fire;
    assign mid_fire          = c3k2_mid_valid && head_medium_ready && mid_ds_in_ready;
    assign c3k2_mid_ready    = head_medium_ready && mid_ds_in_ready;
    assign head_medium_data  = c3k2_mid_data;
    assign head_medium_valid = mid_fire;

    // 2D. Downsample Mid (P4_W x P4_H -> P5_W x P5_H)
    wire [(P4_CH*DATA_WIDTH)-1:0] mid_down_data;
    wire                          mid_down_valid;
    wire                          mid_down_ready;

    yolo_conv_downsample #(
        .IN_CH(P4_CH), .OUT_CH(P4_CH),
        .DATA_WIDTH(DATA_WIDTH),
        .IN_WIDTH(P4_W),
        .IN_HEIGHT(P4_H)
    ) u_downsample_mid (
        .clk(clk), .rst_n(rst_n),
        .data_in(c3k2_mid_data),  .valid_in(mid_fire),        .ready_out(mid_ds_in_ready),
        .data_out(mid_down_data), .valid_out(mid_down_valid), .ready_in(mid_down_ready)
    );

    // 2E. Concat (Mid_down + P5_skip)
    wire [((P4_CH+P5_CH)*DATA_WIDTH)-1:0] concat_mid_p5_data;
    wire                                  concat_mid_p5_valid;

    yolo_concat #(
        .CH_A(P4_CH), .CH_B(P5_CH),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_mid_p5 (
        .clk(clk), .rst_n(rst_n),
        .data_a(mid_down_data),         .valid_a(mid_down_valid),         .ready_a(mid_down_ready),
        .data_b(p5_skip_data),          .valid_b(p5_skip_valid),          .ready_b(p5_skip_ready),
        .data_out(concat_mid_p5_data),  .valid_out(concat_mid_p5_valid),  .ready_in(1'b1)
    );

    // 2F. C3k2 Bot -> Output Head Large (valid-only -> buffered by stream_fifo)
    wire [(P5_CH*DATA_WIDTH)-1:0] c3k2_bot_raw_data;
    wire                          c3k2_bot_raw_valid;

    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(P4_CH + P5_CH),
        .OUT_CHANNELS(P5_CH),
        .NUM_BLOCKS(2),
        .C3K_MODE(1),
        .ATTN_MODE(1),
        .IMG_WIDTH(P5_W),
        .IMG_HEIGHT(P5_H)
    ) u_c3k2_bot (
        .clk(clk), .rst_n(rst_n),
        .in_valid(concat_mid_p5_valid),
        .in_data(concat_mid_p5_data),
        .out_valid(c3k2_bot_raw_valid),
        .out_data(c3k2_bot_raw_data)
    );

    wire bot_fifo_in_ready;
    stream_fifo #(
        .WIDTH(P5_CH * DATA_WIDTH),
        .DEPTH((P5_W * P5_H) + 8)
    ) u_fifo_bot_buf (
        .clk(clk), .rst_n(rst_n),
        .in_data(c3k2_bot_raw_data), .in_valid(c3k2_bot_raw_valid), .in_ready(bot_fifo_in_ready),
        .out_data(head_large_data),  .out_valid(head_large_valid),  .out_ready(head_large_ready)
    );

endmodule
