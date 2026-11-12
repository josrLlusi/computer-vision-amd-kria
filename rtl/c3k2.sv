`timescale 1ns / 1ps
// ============================================================================
// c3k2.sv - Blok C3k2 YOLO26 (varian c3k=False, n=1 Bottleneck, shortcut=True)
//           dipakai pada layer 2 dan layer 4 yolo26n.
//
//   a        = cv1(x)                      Conv 1x1  C1 -> 2C
//   y0, y1   = chunk(a, 2)                 y0 = saluran 0..C-1, y1 = C..2C-1
//   z        = y1 + m.cv2(m.cv1(y1))       Bottleneck: Conv 3x3 s1 C->CB, CB->C, + residual
//   out      = cv2(cat[y0, y1, z])         Conv 1x1  3C -> C2
//
// Semua Conv = Conv2d + BN-folded bias + SiLU (Q16.16), lihat conv_kxk.sv.
// Penjumlahan residual: wrap 32-bit per saluran (sama dengan golden Python).
//
// Streaming tanpa frame buffer: cabang (y0, y1) ditahan di stream_fifo selama
// Bottleneck menghitung. Latensi Bottleneck ~2*(W+2) piksel, sehingga
// FIFO_DEPTH default = 2*(W+2) + 16 entri.
//
// File bobot: {PREFIX}.cv1.{w,b}.hex, {PREFIX}.m0.cv1.*, {PREFIX}.m0.cv2.*, {PREFIX}.cv2.*
// ============================================================================
module c3k2 #(
    parameter int IMAGE_WIDTH  = 160,
    parameter int IMAGE_HEIGHT = 160,
    parameter int C1           = 32,     // saluran masuk
    parameter int C2           = 64,     // saluran keluar
    parameter int C            = 16,     // saluran per chunk (= C2 * e)
    parameter int CB           = 8,      // saluran tengah Bottleneck (= C * 0.5)
    // PAR (IC, OC) untuk keempat Conv
    parameter int P_CV1_IC = 8,  parameter int P_CV1_OC = 2,
    parameter int P_M1_IC  = 4,  parameter int P_M1_OC  = 1,
    parameter int P_M2_IC  = 4,  parameter int P_M2_OC  = 1,
    parameter int P_CV2_IC = 16, parameter int P_CV2_OC = 4,
    parameter int FIFO_DEPTH   = 2 * (IMAGE_WIDTH + 2) + 16,
    parameter     PREFIX       = "L2",
    parameter     SIGMOID_FILE = "sigmoid_lut.hex"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [(C1*32)-1:0] pixel_in,
    input  logic               valid_in,
    output logic               ready_in,

    output logic [(C2*32)-1:0] pixel_out,
    output logic               valid_out,
    input  logic               ready_out
);

    // ---------------- cv1: Conv 1x1  C1 -> 2C ----------------
    logic [(2*C*32)-1:0] a_data;
    logic                a_valid, a_ready;

    conv_kxk #(
        .IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .IN_CHANNELS(C1), .OUT_CHANNELS(2*C), .KERNEL(1), .STRIDE(1),
        .PAR_IC(P_CV1_IC), .PAR_OC(P_CV1_OC),
        .WEIGHT_FILE({PREFIX, ".cv1.w.hex"}), .BIAS_FILE({PREFIX, ".cv1.b.hex"}),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_cv1 (
        .clk, .rst_n,
        .pixel_in(pixel_in), .valid_in(valid_in), .ready_in(ready_in),
        .pixel_out(a_data), .valid_out(a_valid), .ready_out(a_ready)
    );

    // ---------------- Fork: a -> FIFO (y0,y1) dan -> Bottleneck (y1) ----------------
    logic                fifo_in_ready, b_in_ready;
    logic [(2*C*32)-1:0] q_data;
    logic                q_valid, q_ready;

    assign a_ready = fifo_in_ready && b_in_ready;   // beat diambil bila KEDUA cabang siap

    stream_fifo #(.WIDTH(2*C*32), .DEPTH(FIFO_DEPTH)) u_fifo (
        .clk, .rst_n,
        .in_data(a_data), .in_valid(a_valid && b_in_ready), .in_ready(fifo_in_ready),
        .out_data(q_data), .out_valid(q_valid), .out_ready(q_ready)
    );

    // ---------------- Bottleneck: Conv 3x3 s1 C->CB, CB->C ----------------
    logic [(CB*32)-1:0] t_data;
    logic               t_valid, t_ready;
    logic [(C*32)-1:0]  u_data;
    logic               u_valid, u_ready;

    conv_kxk #(
        .IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .IN_CHANNELS(C), .OUT_CHANNELS(CB), .KERNEL(3), .STRIDE(1),
        .PAR_IC(P_M1_IC), .PAR_OC(P_M1_OC),
        .WEIGHT_FILE({PREFIX, ".m0.cv1.w.hex"}), .BIAS_FILE({PREFIX, ".m0.cv1.b.hex"}),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_m_cv1 (
        .clk, .rst_n,
        .pixel_in(a_data[(2*C*32)-1 : C*32]),           // y1
        .valid_in(a_valid && fifo_in_ready), .ready_in(b_in_ready),
        .pixel_out(t_data), .valid_out(t_valid), .ready_out(t_ready)
    );

    conv_kxk #(
        .IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .IN_CHANNELS(CB), .OUT_CHANNELS(C), .KERNEL(3), .STRIDE(1),
        .PAR_IC(P_M2_IC), .PAR_OC(P_M2_OC),
        .WEIGHT_FILE({PREFIX, ".m0.cv2.w.hex"}), .BIAS_FILE({PREFIX, ".m0.cv2.b.hex"}),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_m_cv2 (
        .clk, .rst_n,
        .pixel_in(t_data), .valid_in(t_valid), .ready_in(t_ready),
        .pixel_out(u_data), .valid_out(u_valid), .ready_out(u_ready)
    );

    // ---------------- Join: residual + concat [y0, y1, z] ----------------
    logic [(3*C*32)-1:0] cat_data;
    logic                cat_valid, cat_ready;

    assign cat_valid = q_valid && u_valid;
    assign q_ready   = cat_ready && u_valid;
    assign u_ready   = cat_ready && q_valid;

    always_comb begin
        cat_data[(2*C*32)-1:0] = q_data;                                  // y0, y1
        for (int c = 0; c < C; c++)                                       // z = y1 + u
            cat_data[(2*C + c)*32 +: 32] = q_data[(C + c)*32 +: 32] + u_data[c*32 +: 32];
    end

    // ---------------- cv2: Conv 1x1  3C -> C2 ----------------
    conv_kxk #(
        .IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .IN_CHANNELS(3*C), .OUT_CHANNELS(C2), .KERNEL(1), .STRIDE(1),
        .PAR_IC(P_CV2_IC), .PAR_OC(P_CV2_OC),
        .WEIGHT_FILE({PREFIX, ".cv2.w.hex"}), .BIAS_FILE({PREFIX, ".cv2.b.hex"}),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_cv2 (
        .clk, .rst_n,
        .pixel_in(cat_data), .valid_in(cat_valid), .ready_in(cat_ready),
        .pixel_out(pixel_out), .valid_out(valid_out), .ready_out(ready_out)
    );

endmodule
