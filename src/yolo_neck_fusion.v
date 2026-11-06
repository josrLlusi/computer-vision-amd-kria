`timescale 1ns / 1ps

module yolo_neck_fusion #(
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,

    // --- Input dari Backbone (P3, P4, P5) ---
    input wire [(256*DATA_WIDTH)-1:0]  p3_data_in,
    input wire                         p3_valid_in,
    output wire                        p3_ready_out,

    input wire [(512*DATA_WIDTH)-1:0]  p4_data_in,
    input wire                         p4_valid_in,
    output wire                        p4_ready_out,

    input wire [(1024*DATA_WIDTH)-1:0] p5_data_in,
    input wire                         p5_valid_in,
    output wire                        p5_ready_out,

    // --- Output ke Detect Head (Small, Medium, Large) ---
    output wire [(256*DATA_WIDTH)-1:0]  head_small_data,
    output wire                         head_small_valid,
    input wire                          head_small_ready,

    output wire [(512*DATA_WIDTH)-1:0]  head_medium_data,
    output wire                         head_medium_valid,
    input wire                          head_medium_ready,

    output wire [(1024*DATA_WIDTH)-1:0] head_large_data,
    output wire                         head_large_valid,
    input wire                          head_large_ready
);

    // ==========================================
    // Sinyal Interkoneksi Internal
    // ==========================================
    // Top-Down Stage 1 (P5 -> P4)
    wire [(1024*DATA_WIDTH)-1:0] p5_up_data;
    wire                         p5_up_valid;
    wire                         p5_up_ready;

    wire [(1536*DATA_WIDTH)-1:0] concat_p5_p4_data;
    wire                         concat_p5_p4_valid;
    wire                         concat_p5_p4_ready;

    wire [(512*DATA_WIDTH)-1:0]  c3k2_td1_data;
    wire                         c3k2_td1_valid;
    wire                         c3k2_td1_ready;

    // Top-Down Stage 2 (TD1 -> P3)
    wire [(512*DATA_WIDTH)-1:0]  td1_up_data;
    wire                         td1_up_valid;
    wire                         td1_up_ready;

    wire [(768*DATA_WIDTH)-1:0]  concat_td1_p3_data;
    wire                         concat_td1_p3_valid;
    wire                         concat_td1_p3_ready;

    wire [(256*DATA_WIDTH)-1:0]  c3k2_top_data;
    wire                         c3k2_top_valid;
    wire                         c3k2_top_ready;

    // Bottom-Up Stage 1 (Top -> Medium)
    wire [(256*DATA_WIDTH)-1:0]  top_down_data;
    wire                         top_down_valid;
    wire                         top_down_ready;

    wire [(768*DATA_WIDTH)-1:0]  concat_top_td1_data;
    wire                         concat_top_td1_valid;
    wire                         concat_top_td1_ready;

    wire [(512*DATA_WIDTH)-1:0]  c3k2_mid_data;
    wire                         c3k2_mid_valid;
    wire                         c3k2_mid_ready;

    // Bottom-Up Stage 2 (Medium -> Large)
    wire [(512*DATA_WIDTH)-1:0]  mid_down_data;
    wire                         mid_down_valid;
    wire                         mid_down_ready;

    wire [(1536*DATA_WIDTH)-1:0] concat_mid_p5_data;
    wire                         concat_mid_p5_valid;
    wire                         concat_mid_p5_ready;

    wire [(1024*DATA_WIDTH)-1:0] c3k2_bot_data;
    wire                         c3k2_bot_valid;
    wire                         c3k2_bot_ready;

    // ==========================================
    // 1. JALUR TOP-DOWN (Atas ke Bawah)
    // ==========================================

    // 1A. Upsample P5 (1024 ch)
    yolo_upsample #(
        .CHANNELS(1024),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_upsample_p5 (
        .clk(clk), .rst_n(rst_n),
        .data_in(p5_data_in),   .valid_in(p5_valid_in),   .ready_out(p5_ready_out),
        .data_out(p5_up_data), .valid_out(p5_up_valid), .ready_in(p5_up_ready)
    );

    // 1B. Concat (P5_up + P4) -> 1024 + 512 = 1536 ch
    yolo_concat #(
        .CH_A(1024), .CH_B(512),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_p5_p4 (
        .clk(clk), .rst_n(rst_n),
        .data_a(p5_up_data),     .valid_a(p5_up_valid),     .ready_a(p5_up_ready),
        .data_b(p4_data_in),    .valid_b(p4_valid_in),    .ready_b(p4_ready_out),
        .data_out(concat_p5_p4_data), .valid_out(concat_p5_p4_valid), .ready_in(concat_p5_p4_ready)
    );

    // 1C. C3k2 (1536 -> 512 ch)
    yolo_c3k2 #(
        .IN_CH(1536), .OUT_CH(512),
        .DATA_WIDTH(DATA_WIDTH), .C3K_MODE(0)
    ) u_c3k2_td1 (
        .clk(clk), .rst_n(rst_n),
        .data_in(concat_p5_p4_data), .valid_in(concat_p5_p4_valid), .ready_out(concat_p5_p4_ready),
        .data_out(c3k2_td1_data),    .valid_out(c3k2_td1_valid),    .ready_in(c3k2_td1_ready)
    );

    // 1D. Upsample TD1 (512 ch)
    yolo_upsample #(
        .CHANNELS(512),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_upsample_td1 (
        .clk(clk), .rst_n(rst_n),
        .data_in(c3k2_td1_data), .valid_in(c3k2_td1_valid), .ready_out(c3k2_td1_ready),
        .data_out(td1_up_data),  .valid_out(td1_up_valid),  .ready_in(td1_up_ready)
    );

    // 1E. Concat (TD1_up + P3) -> 512 + 256 = 768 ch
    yolo_concat #(
        .CH_A(512), .CH_B(256),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_td1_p3 (
        .clk(clk), .rst_n(rst_n),
        .data_a(td1_up_data),    .valid_a(td1_up_valid),    .ready_a(td1_up_ready),
        .data_b(p3_data_in),    .valid_b(p3_valid_in),    .ready_b(p3_ready_out),
        .data_out(concat_td1_p3_data), .valid_out(concat_td1_p3_valid), .ready_in(concat_td1_p3_ready)
    );

    // 1F. C3k2 Top (768 -> 256 ch) -> Output Head Small
    yolo_c3k2 #(
        .IN_CH(768), .OUT_CH(256),
        .DATA_WIDTH(DATA_WIDTH), .C3K_MODE(0)
    ) u_c3k2_top (
        .clk(clk), .rst_n(rst_n),
        .data_in(concat_td1_p3_data), .valid_in(concat_td1_p3_valid), .ready_out(concat_td1_p3_ready),
        .data_out(c3k2_top_data),     .valid_out(c3k2_top_valid),     .ready_in(c3k2_top_ready)
    );

    // ==========================================
    // 2. JALUR BOTTOM-UP (Bawah ke Atas)
    // ==========================================

    // 2A. Downsample Top (256 ch, Stride 2)
    yolo_conv_downsample #(
        .IN_CH(256), .OUT_CH(256),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_downsample_top (
        .clk(clk), .rst_n(rst_n),
        .data_in(c3k2_top_data),  .valid_in(c3k2_top_valid),  .ready_out(c3k2_top_ready),
        .data_out(top_down_data), .valid_out(top_down_valid), .ready_in(top_down_ready)
    );

    // 2B. Concat (Top_down + TD1) -> 256 + 512 = 768 ch
    yolo_concat #(
        .CH_A(256), .CH_B(512),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_top_td1 (
        .clk(clk), .rst_n(rst_n),
        .data_a(top_down_data),  .valid_a(top_down_valid),  .ready_a(top_down_ready),
        .data_b(c3k2_td1_data),  .valid_b(c3k2_td1_valid),  .ready_b(), // Internal fanout
        .data_out(concat_top_td1_data), .valid_out(concat_top_td1_valid), .ready_in(concat_top_td1_ready)
    );

    // 2C. C3k2 Mid (768 -> 512 ch) -> Output Head Medium
    yolo_c3k2 #(
        .IN_CH(768), .OUT_CH(512),
        .DATA_WIDTH(DATA_WIDTH), .C3K_MODE(0)
    ) u_c3k2_mid (
        .clk(clk), .rst_n(rst_n),
        .data_in(concat_top_td1_data), .valid_in(concat_top_td1_valid), .ready_out(concat_top_td1_ready),
        .data_out(c3k2_mid_data),      .valid_out(c3k2_mid_valid),      .ready_in(c3k2_mid_ready)
    );

    // 2D. Downsample Mid (512 ch, Stride 2)
    yolo_conv_downsample #(
        .IN_CH(512), .OUT_CH(512),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_downsample_mid (
        .clk(clk), .rst_n(rst_n),
        .data_in(c3k2_mid_data),  .valid_in(c3k2_mid_valid),  .ready_out(c3k2_mid_ready),
        .data_out(mid_down_data), .valid_out(mid_down_valid), .ready_in(mid_down_ready)
    );

    // 2E. Concat (Mid_down + P5_up) -> 512 + 1024 = 1536 ch
    yolo_concat #(
        .CH_A(512), .CH_B(1024),
        .DATA_WIDTH(DATA_WIDTH)
    ) u_concat_mid_p5 (
        .clk(clk), .rst_n(rst_n),
        .data_a(mid_down_data), .valid_a(mid_down_valid), .ready_a(mid_down_ready),
        .data_b(p5_up_data),    .valid_b(p5_up_valid),    .ready_b(),
        .data_out(concat_mid_p5_data), .valid_out(concat_mid_p5_valid), .ready_in(concat_mid_p5_ready)
    );

    // 2F. C3k2 Bot (1536 -> 1024 ch, Attention=True) -> Output Head Large[cite: 4]
    yolo_c3k2 #(
        .IN_CH(1536), .OUT_CH(1024),
        .DATA_WIDTH(DATA_WIDTH), .C3K_MODE(1)
    ) u_c3k2_bot (
        .clk(clk), .rst_n(rst_n),
        .data_in(concat_mid_p5_data), .valid_in(concat_mid_p5_valid), .ready_out(concat_mid_p5_ready),
        .data_out(c3k2_bot_data),      .valid_out(c3k2_bot_valid),      .ready_in(c3k2_bot_ready)
    );

    // ==========================================
    // Assignment Output
    // ==========================================
    assign head_small_data  = c3k2_top_data;
    assign head_small_valid = c3k2_top_valid;
    assign c3k2_top_ready   = head_small_ready;

    assign head_medium_data  = c3k2_mid_data;
    assign head_medium_valid = c3k2_mid_valid;
    assign c3k2_mid_ready    = head_medium_ready;

    assign head_large_data  = c3k2_bot_data;
    assign head_large_valid = c3k2_bot_valid;
    assign c3k2_bot_ready   = head_large_ready;

endmodule