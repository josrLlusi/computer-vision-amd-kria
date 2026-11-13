`timescale 1ns / 1ps
// ============================================================================
// backbone_l0_l5.sv - Backbone YOLO26n layer 0..5 (bagian tim: blok yang ditugaskan)
//
//   input 3ch IMGxIMG
//     L0 Conv 3x3/2   3 ->  16   (IMG   -> IMG/2)    "Conv 64"
//     L1 Conv 3x3/2  16 ->  32   (IMG/2 -> IMG/4)    "Conv 128"
//     L2 C3k2        32 ->  64   (IMG/4)             "C3k2 x2 256, e=0.25"
//     L3 Conv 3x3/2  64 ->  64   (IMG/4 -> IMG/8)    "Conv 256"
//     L4 C3k2        64 -> 128   (IMG/8)             "C3k2 x2 512, e=0.25"  --> p3_* (ke neck)
//     L5 Conv 3x3/2 128 -> 128   (IMG/8 -> IMG/16)   "Conv 512"             --> l5_* (ke layer 6)
//
// Keluaran layer 4 dipakai dua tujuan (P3 ke neck & masukan layer 5), sehingga
// beat L4 hanya diambil bila p3_ready DAN layer 5 siap.
// File bobot: L0.w.hex, L0.b.hex, L1.*, L2.cv1.*, L2.m0.cv1.*, ..., L5.*
// ============================================================================
module backbone_l0_l5 #(
    parameter int IMG          = 640,   // lebar = tinggi citra masuk (kelipatan 16)
    parameter     SIGMOID_FILE = "sigmoid_lut.hex"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [(3*32)-1:0]   pixel_in,
    input  logic                valid_in,
    output logic                ready_in,

    output logic [(128*32)-1:0] p3_data,     // keluaran layer 4 (P3/8), 128 ch, IMG/8
    output logic                p3_valid,
    input  logic                p3_ready,

    output logic [(128*32)-1:0] l5_data,     // keluaran layer 5 (P4/16), 128 ch, IMG/16
    output logic                l5_valid,
    input  logic                l5_ready
);
    logic [(16*32)-1:0]  s0;  logic v0, r0;
    logic [(32*32)-1:0]  s1;  logic v1, r1;
    logic [(64*32)-1:0]  s2;  logic v2, r2;
    logic [(64*32)-1:0]  s3;  logic v3, r3;
    logic [(128*32)-1:0] s4;  logic v4, r4, r5_in;

    // L0
    conv_kxk #(.IMAGE_WIDTH(IMG), .IMAGE_HEIGHT(IMG), .IN_CHANNELS(3), .OUT_CHANNELS(16),
               .KERNEL(3), .STRIDE(2), .PAR_IC(3), .PAR_OC(2),
               .WEIGHT_FILE("L0.w.hex"), .BIAS_FILE("L0.b.hex"), .SIGMOID_FILE(SIGMOID_FILE))
    u_l0 (.clk, .rst_n, .pixel_in(pixel_in), .valid_in(valid_in), .ready_in(ready_in),
          .pixel_out(s0), .valid_out(v0), .ready_out(r0));

    // L1
    conv_kxk #(.IMAGE_WIDTH(IMG/2), .IMAGE_HEIGHT(IMG/2), .IN_CHANNELS(16), .OUT_CHANNELS(32),
               .KERNEL(3), .STRIDE(2), .PAR_IC(4), .PAR_OC(2),
               .WEIGHT_FILE("L1.w.hex"), .BIAS_FILE("L1.b.hex"), .SIGMOID_FILE(SIGMOID_FILE))
    u_l1 (.clk, .rst_n, .pixel_in(s0), .valid_in(v0), .ready_in(r0),
          .pixel_out(s1), .valid_out(v1), .ready_out(r1));

    // L2 C3k2
    c3k2 #(.IMAGE_WIDTH(IMG/4), .IMAGE_HEIGHT(IMG/4), .C1(32), .C2(64), .C(16), .CB(8),
           .P_CV1_IC(8), .P_CV1_OC(2), .P_M1_IC(4), .P_M1_OC(1),
           .P_M2_IC(4),  .P_M2_OC(1),  .P_CV2_IC(16), .P_CV2_OC(4),
           .PREFIX("L2"), .SIGMOID_FILE(SIGMOID_FILE))
    u_l2 (.clk, .rst_n, .pixel_in(s1), .valid_in(v1), .ready_in(r1),
          .pixel_out(s2), .valid_out(v2), .ready_out(r2));

    // L3
    conv_kxk #(.IMAGE_WIDTH(IMG/4), .IMAGE_HEIGHT(IMG/4), .IN_CHANNELS(64), .OUT_CHANNELS(64),
               .KERNEL(3), .STRIDE(2), .PAR_IC(8), .PAR_OC(2),
               .WEIGHT_FILE("L3.w.hex"), .BIAS_FILE("L3.b.hex"), .SIGMOID_FILE(SIGMOID_FILE))
    u_l3 (.clk, .rst_n, .pixel_in(s2), .valid_in(v2), .ready_in(r2),
          .pixel_out(s3), .valid_out(v3), .ready_out(r3));

    // L4 C3k2
    c3k2 #(.IMAGE_WIDTH(IMG/8), .IMAGE_HEIGHT(IMG/8), .C1(64), .C2(128), .C(32), .CB(16),
           .P_CV1_IC(4), .P_CV1_OC(4), .P_M1_IC(2), .P_M1_OC(1),
           .P_M2_IC(2),  .P_M2_OC(1),  .P_CV2_IC(8), .P_CV2_OC(8),
           .PREFIX("L4"), .SIGMOID_FILE(SIGMOID_FILE))
    u_l4 (.clk, .rst_n, .pixel_in(s3), .valid_in(v3), .ready_in(r3),
          .pixel_out(s4), .valid_out(v4), .ready_out(r4));

    // Fork L4 -> P3 (neck) & L5
    assign p3_data  = s4;
    assign p3_valid = v4 && r5_in;
    assign r4       = p3_ready && r5_in;

    // L5
    conv_kxk #(.IMAGE_WIDTH(IMG/8), .IMAGE_HEIGHT(IMG/8), .IN_CHANNELS(128), .OUT_CHANNELS(128),
               .KERNEL(3), .STRIDE(2), .PAR_IC(8), .PAR_OC(2),
               .WEIGHT_FILE("L5.w.hex"), .BIAS_FILE("L5.b.hex"), .SIGMOID_FILE(SIGMOID_FILE))
    u_l5 (.clk, .rst_n, .pixel_in(s4), .valid_in(v4 && p3_ready), .ready_in(r5_in),
          .pixel_out(l5_data), .valid_out(l5_valid), .ready_out(l5_ready));

endmodule
