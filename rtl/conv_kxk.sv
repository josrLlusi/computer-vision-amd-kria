`timescale 1ns / 1ps
// ============================================================================
// conv_kxk.sv - Blok "Conv" YOLO26 GENERIK (Conv2d + BN-folded bias + SiLU)
//
// Mode yang didukung (padding = KERNEL/2, sama dengan Ultralytics autopad):
//   KERNEL=1, STRIDE=1 : Conv 1x1   (cv1 / cv2 di C3k2, SPPF)
//   KERNEL=3, STRIDE=1 : Conv 3x3   (Bottleneck di C3k2)
//   KERNEL=3, STRIDE=2 : Conv 3x3/2 (blok biru backbone; identik dengan conv3x3_s2.sv)
//
// Aritmetika Q16.16, akumulator 64-bit, hasil acc[47:16], SiLU via LUT (lihat
// conv3x3_s2.sv). MAC time-multiplexed: PAR_OC x PAR_IC x KERNEL^2 pengali,
// (OUT/PAR_OC)*(IN/PAR_IC) siklus per piksel keluaran.
//
// Padding kanan/bawah untuk 3x3 stride 1 (streaming, tanpa frame buffer):
//   modul berjalan pada grid (W+1) x (H+1). Kolom x=W dan baris y=H adalah
//   piksel "fantom" bernilai 0 yang disisipkan sendiri oleh modul (tidak
//   mengambil data dari hulu, ready_in = 0 pada posisi tersebut).
//
// Packing: pixel_in[c*32 +: 32] = saluran c, pixel_out[oc*32 +: 32] = saluran oc
// Bobot  : alamat ((oc*IN_CHANNELS + ic)*K*K) + (ky*K + kx)   (= weight.flatten())
// ============================================================================
module conv_kxk #(
    parameter int IMAGE_WIDTH  = 160,
    parameter int IMAGE_HEIGHT = 160,
    parameter int IN_CHANNELS  = 16,
    parameter int OUT_CHANNELS = 16,
    parameter int KERNEL       = 3,
    parameter int STRIDE       = 1,
    parameter int PAR_IC       = 1,
    parameter int PAR_OC       = 1,
    parameter bit USE_SILU     = 1'b1,
    parameter     WEIGHT_FILE  = "w.hex",
    parameter     BIAS_FILE    = "b.hex",
    parameter     SIGMOID_FILE = "sigmoid_lut.hex"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [(IN_CHANNELS*32)-1:0]  pixel_in,
    input  logic                         valid_in,
    output logic                         ready_in,

    output logic [(OUT_CHANNELS*32)-1:0] pixel_out,
    output logic                         valid_out,
    input  logic                         ready_out
);

    // ------------------------------------------------------------------
    // Konstanta
    // ------------------------------------------------------------------
    localparam int EXT   = (KERNEL == 3 && STRIDE == 1) ? 1 : 0;  // kolom/baris fantom
    localparam int GW    = IMAGE_WIDTH  + EXT;
    localparam int GH    = IMAGE_HEIGHT + EXT;
    localparam int XW    = (GW > 1) ? $clog2(GW) : 1;
    localparam int YW    = (GH > 1) ? $clog2(GH) : 1;
    localparam int KK    = KERNEL * KERNEL;
    localparam int N_ICG = IN_CHANNELS  / PAR_IC;
    localparam int N_OCG = OUT_CHANNELS / PAR_OC;
    localparam int ICGW  = (N_ICG > 1) ? $clog2(N_ICG) : 1;
    localparam int OCGW  = (N_OCG > 1) ? $clog2(N_OCG) : 1;
    localparam int NW    = OUT_CHANNELS * IN_CHANNELS * KK;

    initial begin
        if (!((KERNEL == 1 && STRIDE == 1) || (KERNEL == 3 && (STRIDE == 1 || STRIDE == 2))))
            $fatal(1, "conv_kxk: hanya mendukung K1S1, K3S1, K3S2");
        if (STRIDE == 2 && (IMAGE_WIDTH % 2 || IMAGE_HEIGHT % 2))
            $fatal(1, "conv_kxk: stride 2 butuh lebar/tinggi genap");
        if (IN_CHANNELS % PAR_IC || OUT_CHANNELS % PAR_OC)
            $fatal(1, "conv_kxk: PAR_IC/PAR_OC harus membagi habis IN/OUT_CHANNELS");
    end

    // ------------------------------------------------------------------
    // ROM
    // ------------------------------------------------------------------
    logic signed [31:0] weights [0:NW-1];
    logic signed [31:0] biases  [0:OUT_CHANNELS-1];
    logic        [31:0] sig_lut [0:2048];

    initial begin
        $readmemh(WEIGHT_FILE,  weights);
        $readmemh(BIAS_FILE,    biases);
        $readmemh(SIGMOID_FILE, sig_lut);
    end

    // ------------------------------------------------------------------
    // Handshake, grid, trigger
    // ------------------------------------------------------------------
    typedef enum logic {S_STREAM, S_COMPUTE} state_t;
    state_t state;

    logic [XW-1:0] gx;
    logic [YW-1:0] gy;
    logic out_pending, phantom, step, trigger;

    assign out_pending = valid_out && !ready_out;
    assign phantom     = (EXT == 1) && ((gx == XW'(IMAGE_WIDTH)) || (gy == YW'(IMAGE_HEIGHT)));
    assign ready_in    = (state == S_STREAM) && !out_pending && !phantom;
    assign step        = (state == S_STREAM) && !out_pending && (phantom || valid_in);

    always_comb begin
        if (KERNEL == 1)      trigger = step;                              // tiap piksel
        else if (STRIDE == 2) trigger = step && gx[0] && gy[0];            // (ganjil, ganjil)
        else                  trigger = step && (gx != '0) && (gy != '0);  // grid diperluas
    end

    // Piksel efektif yang masuk window (0 untuk fantom)
    logic [(IN_CHANNELS*32)-1:0] pix_eff;
    assign pix_eff = phantom ? '0 : pixel_in;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            gx <= '0;
            gy <= '0;
        end else if (step) begin
            if (gx == XW'(GW - 1)) begin
                gx <= '0;
                gy <= (gy == YW'(GH - 1)) ? '0 : gy + 1'b1;
            end else begin
                gx <= gx + 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------
    // Window KxK per saluran: win[ic][baris][kolom], baris 0 = teratas,
    // kolom 0 = piksel terbaru (kanan). Untuk K=1: win[ic][0][0] = piksel.
    // ------------------------------------------------------------------
    logic signed [31:0] win [0:IN_CHANNELS-1][0:KERNEL-1][0:KERNEL-1];

    generate
        if (KERNEL == 3) begin : g_k3
            logic [(IN_CHANNELS*32)-1:0] line_buf_0 [0:GW-1];   // baris y-1
            logic [(IN_CHANNELS*32)-1:0] line_buf_1 [0:GW-1];   // baris y-2
            logic [(IN_CHANNELS*32)-1:0] lb0_rd, lb1_rd;
            assign lb0_rd = line_buf_0[gx];
            assign lb1_rd = line_buf_1[gx];

            always_ff @(posedge clk) begin
                if (step) begin
                    line_buf_0[gx] <= pix_eff;
                    line_buf_1[gx] <= lb0_rd;
                end
            end

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    for (int c = 0; c < IN_CHANNELS; c++)
                        for (int i = 0; i < 3; i++)
                            for (int j = 0; j < 3; j++)
                                win[c][i][j] <= '0;
                end else if (step) begin
                    for (int c = 0; c < IN_CHANNELS; c++) begin
                        win[c][0][2] <= win[c][0][1];
                        win[c][0][1] <= win[c][0][0];
                        win[c][0][0] <= lb1_rd[c*32 +: 32];
                        win[c][1][2] <= win[c][1][1];
                        win[c][1][1] <= win[c][1][0];
                        win[c][1][0] <= lb0_rd[c*32 +: 32];
                        win[c][2][2] <= win[c][2][1];
                        win[c][2][1] <= win[c][2][0];
                        win[c][2][0] <= pix_eff[c*32 +: 32];
                    end
                end
            end
        end else begin : g_k1
            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    for (int c = 0; c < IN_CHANNELS; c++) win[c][0][0] <= '0;
                end else if (step) begin
                    for (int c = 0; c < IN_CHANNELS; c++) win[c][0][0] <= pix_eff[c*32 +: 32];
                end
            end
        end
    endgenerate

    // ------------------------------------------------------------------
    // Aritmetika
    // ------------------------------------------------------------------
    function automatic logic signed [63:0] mul_q16(input logic signed [31:0] a,
                                                   input logic signed [31:0] b);
        logic signed [63:0] a64, b64;
        a64 = a;
        b64 = b;
        return a64 * b64;
    endfunction

    function automatic logic signed [31:0] silu_q16(input logic signed [31:0] x);
        logic [21:0]        xo;
        logic [10:0]        idx;
        logic [9:0]         frac;
        logic [31:0]        s0, s1, sig;
        logic signed [63:0] prod;
        if (x >= 32'sh0010_0000)          return x;
        else if (x < -32'sh0010_0000)     return '0;
        else begin
            xo   = 22'(x + 32'sh0010_0000);
            idx  = xo[20:10];
            frac = xo[9:0];
            s0   = sig_lut[idx];
            s1   = sig_lut[idx + 1];
            sig  = s0 + (((s1 - s0) * frac) >> 10);
            prod = mul_q16(x, sig);
            return prod[47:16];
        end
    endfunction

    // ------------------------------------------------------------------
    // MAC time-multiplexed
    // ------------------------------------------------------------------
    logic [ICGW-1:0]    g_ic;
    logic [OCGW-1:0]    g_oc;
    logic               mask_top, mask_left;
    logic signed [63:0] acc [0:PAR_OC-1];

    function automatic logic signed [63:0] mac_step(input int p);
        int oc;
        logic signed [31:0] b;
        logic signed [63:0] sum;
        oc  = int'(g_oc) * PAR_OC + p;
        b   = biases[oc];
        sum = (g_ic == '0) ? {{16{b[31]}}, b, 16'h0000} : acc[p];
        for (int q = 0; q < PAR_IC; q++) begin
            int ic;
            ic = int'(g_ic) * PAR_IC + q;
            for (int ky = 0; ky < KERNEL; ky++) begin
                for (int kx = 0; kx < KERNEL; kx++) begin
                    logic signed [31:0] xv;
                    xv = win[ic][ky][KERNEL-1-kx];
                    if (KERNEL == 3 && ((mask_top && ky == 0) || (mask_left && kx == 0))) xv = '0;
                    sum = sum + mul_q16(xv, weights[((oc*IN_CHANNELS + ic)*KK) + (ky*KERNEL + kx)]);
                end
            end
        end
        return sum;
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state     <= S_STREAM;
            g_ic      <= '0;
            g_oc      <= '0;
            mask_top  <= 1'b0;
            mask_left <= 1'b0;
            valid_out <= 1'b0;
            pixel_out <= '0;
            for (int p = 0; p < PAR_OC; p++) acc[p] <= '0;
        end else begin
            if (valid_out && ready_out) valid_out <= 1'b0;

            case (state)
                S_STREAM: begin
                    if (trigger) begin
                        state     <= S_COMPUTE;
                        g_ic      <= '0;
                        g_oc      <= '0;
                        mask_top  <= (gy == YW'(1));
                        mask_left <= (gx == XW'(1));
                    end
                end

                S_COMPUTE: begin
                    for (int p = 0; p < PAR_OC; p++) begin
                        logic signed [63:0] s;
                        logic signed [31:0] r;
                        s = mac_step(p);
                        acc[p] <= s;
                        if (g_ic == ICGW'(N_ICG - 1)) begin
                            r = s[47:16];
                            pixel_out[(int'(g_oc)*PAR_OC + p)*32 +: 32] <= USE_SILU ? silu_q16(r) : r;
                        end
                    end

                    if (g_ic == ICGW'(N_ICG - 1)) begin
                        g_ic <= '0;
                        if (g_oc == OCGW'(N_OCG - 1)) begin
                            g_oc      <= '0;
                            valid_out <= 1'b1;
                            state     <= S_STREAM;
                        end else begin
                            g_oc <= g_oc + 1'b1;
                        end
                    end else begin
                        g_ic <= g_ic + 1'b1;
                    end
                end
            endcase
        end
    end

endmodule
