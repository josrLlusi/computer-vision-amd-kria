`timescale 1ns / 1ps
// ============================================================================
// conv3x3_s2.sv  -  Blok "Conv" YOLO26 (k=3, s=2) untuk FPGA, versi v2
//
//   y = SiLU( BN( Conv2d(x, k=3, stride=2, padding=1, bias=False) ) )
//
// BatchNorm DILEBUR (folded) ke bobot + bias oleh skrip Python, sehingga
// hardware cukup menghitung:  y = SiLU( sum(x*w') + b' )
//
// Format aritmetika : Q16.16 signed 32-bit
// Akumulator        : 64-bit (Q32.32), hasil = acc[47:16] (truncate)
// SiLU              : x * sigmoid(x), sigmoid dari LUT 2049 titik (langkah 1/64,
//                     rentang [-16,16)) + interpolasi linear. x>=16 -> x, x<-16 -> 0
//
// Arsitektur:
//   * Streaming (line buffer + window 3x3), tanpa frame buffer eksternal.
//   * MAC time-multiplexed: tiap siklus menghitung PAR_OC filter x PAR_IC
//     saluran masuk x 9 tap  (= PAR_OC*PAR_IC*9 pengali). Satu piksel keluaran
//     butuh (OUT_CHANNELS/PAR_OC)*(IN_CHANNELS/PAR_IC) siklus.
//   * Handshake valid/ready di kedua sisi (gaya AXI-Stream):
//       masuk : pixel_in, valid_in, ready_in   (beat diterima saat valid_in & ready_in)
//       keluar: pixel_out, valid_out, ready_out (beat diambil saat valid_out & ready_out)
//
// Konvensi packing (WAJIB sama dengan testbench & skrip Python):
//   pixel_in [c*32  +: 32] = saluran masuk c    (c = 0 di LSB)
//   pixel_out[oc*32 +: 32] = saluran keluar oc  (oc = 0 di LSB)
//   bobot  : alamat ((oc*IN_CHANNELS + ic)*9) + (ky*3 + kx)  (= weight.flatten())
//   bias   : alamat oc
//
// Syarat: IMAGE_WIDTH & IMAGE_HEIGHT genap, PAR_IC | IN_CHANNELS, PAR_OC | OUT_CHANNELS
// ============================================================================
module conv3x3_s2 #(
    parameter int IMAGE_WIDTH  = 640,
    parameter int IMAGE_HEIGHT = 640,
    parameter int IN_CHANNELS  = 3,
    parameter int OUT_CHANNELS = 16,
    parameter int PAR_IC       = 3,     // saluran masuk diproses paralel per siklus
    parameter int PAR_OC       = 2,     // filter diproses paralel per siklus
    parameter bit USE_SILU     = 1'b1,  // 0 = tanpa aktivasi (debug konvolusi murni)
    parameter     WEIGHT_FILE  = "tb_weight.hex",
    parameter     BIAS_FILE    = "tb_bias.hex",
    parameter     SIGMOID_FILE = "sigmoid_lut.hex"
)(
    input  logic clk,
    input  logic rst_n,

    // Sisi masuk
    input  logic [(IN_CHANNELS*32)-1:0]  pixel_in,
    input  logic                         valid_in,
    output logic                         ready_in,

    // Sisi keluar
    output logic [(OUT_CHANNELS*32)-1:0] pixel_out,
    output logic                         valid_out,
    input  logic                         ready_out
);

    // ------------------------------------------------------------------
    // Konstanta turunan
    // ------------------------------------------------------------------
    localparam int XW    = (IMAGE_WIDTH  > 1) ? $clog2(IMAGE_WIDTH)  : 1;
    localparam int YW    = (IMAGE_HEIGHT > 1) ? $clog2(IMAGE_HEIGHT) : 1;
    localparam int N_ICG = IN_CHANNELS  / PAR_IC;   // jumlah grup saluran masuk
    localparam int N_OCG = OUT_CHANNELS / PAR_OC;   // jumlah grup filter
    localparam int ICGW  = (N_ICG > 1) ? $clog2(N_ICG) : 1;
    localparam int OCGW  = (N_OCG > 1) ? $clog2(N_OCG) : 1;
    localparam int NW    = OUT_CHANNELS * IN_CHANNELS * 9;

    // Pemeriksaan parameter saat elaborasi
    initial begin
        if (IMAGE_WIDTH % 2 || IMAGE_HEIGHT % 2)
            $fatal(1, "conv3x3_s2: IMAGE_WIDTH/HEIGHT harus genap");
        if (IN_CHANNELS % PAR_IC || OUT_CHANNELS % PAR_OC)
            $fatal(1, "conv3x3_s2: PAR_IC/PAR_OC harus membagi habis IN/OUT_CHANNELS");
    end

    // ------------------------------------------------------------------
    // ROM: bobot, bias, LUT sigmoid
    // ------------------------------------------------------------------
    logic signed [31:0] weights [0:NW-1];
    logic signed [31:0] biases  [0:OUT_CHANNELS-1];
    logic        [31:0] sig_lut [0:2048];     // sigmoid(-16 + i/64) dalam Q0.16

    initial begin
        $readmemh(WEIGHT_FILE,  weights);
        $readmemh(BIAS_FILE,    biases);
        $readmemh(SIGMOID_FILE, sig_lut);
    end

    // ------------------------------------------------------------------
    // Handshake & FSM
    // ------------------------------------------------------------------
    typedef enum logic {S_STREAM, S_COMPUTE} state_t;
    state_t state;

    logic out_pending;
    assign out_pending = valid_out && !ready_out;

    // Terima piksel hanya saat tidak sedang menghitung dan buffer keluaran bebas
    assign ready_in = (state == S_STREAM) && !out_pending;

    logic accept;
    assign accept = valid_in && ready_in;

    // Koordinat piksel yang SEDANG diterima
    logic [XW-1:0] pixel_x;
    logic [YW-1:0] pixel_y;

    // Posisi keluaran: piksel terbaru window berada di (y ganjil, x ganjil)
    // -> window menutupi baris 2oy-1..2oy+1 dan kolom 2ox-1..2ox+1 (padding 1)
    logic trigger;
    assign trigger = accept && pixel_x[0] && pixel_y[0];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pixel_x <= '0;
            pixel_y <= '0;
        end else if (accept) begin
            if (pixel_x == XW'(IMAGE_WIDTH - 1)) begin
                pixel_x <= '0;
                pixel_y <= (pixel_y == YW'(IMAGE_HEIGHT - 1)) ? '0 : pixel_y + 1'b1;
            end else begin
                pixel_x <= pixel_x + 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------
    // TODO 1: Line Buffer (BRAM) - satu alamat = semua saluran satu kolom
    // ------------------------------------------------------------------
    logic [(IN_CHANNELS*32)-1:0] line_buf_0 [0:IMAGE_WIDTH-1];   // baris y-1
    logic [(IN_CHANNELS*32)-1:0] line_buf_1 [0:IMAGE_WIDTH-1];   // baris y-2
    logic [(IN_CHANNELS*32)-1:0] lb0_rd, lb1_rd;

    assign lb0_rd = line_buf_0[pixel_x];
    assign lb1_rd = line_buf_1[pixel_x];

    always_ff @(posedge clk) begin
        if (accept) begin
            line_buf_0[pixel_x] <= pixel_in;
            line_buf_1[pixel_x] <= lb0_rd;
        end
    end

    // ------------------------------------------------------------------
    // TODO 2: Window 3x3 per saluran masuk
    //   window_3x3[ic][baris][kolom], kolom 0 = piksel terbaru (kanan),
    //   kolom 2 = piksel terlama (kiri). Window "beku" selama S_COMPUTE
    //   karena ready_in = 0 (tidak ada piksel baru yang diterima).
    // ------------------------------------------------------------------
    logic signed [31:0] window_3x3 [0:IN_CHANNELS-1][0:2][0:2];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int c = 0; c < IN_CHANNELS; c++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        window_3x3[c][i][j] <= '0;
        end else if (accept) begin
            for (int c = 0; c < IN_CHANNELS; c++) begin
                window_3x3[c][0][2] <= window_3x3[c][0][1];
                window_3x3[c][0][1] <= window_3x3[c][0][0];
                window_3x3[c][0][0] <= lb1_rd[c*32 +: 32];
                window_3x3[c][1][2] <= window_3x3[c][1][1];
                window_3x3[c][1][1] <= window_3x3[c][1][0];
                window_3x3[c][1][0] <= lb0_rd[c*32 +: 32];
                window_3x3[c][2][2] <= window_3x3[c][2][1];
                window_3x3[c][2][1] <= window_3x3[c][2][0];
                window_3x3[c][2][0] <= pixel_in[c*32 +: 32];
            end
        end
    end

    // ------------------------------------------------------------------
    // Fungsi aritmetika Q16.16
    // ------------------------------------------------------------------
    function automatic logic signed [63:0] mul_q16(input logic signed [31:0] a,
                                                   input logic signed [31:0] b);
        logic signed [63:0] a64, b64;
        a64 = a;
        b64 = b;
        return a64 * b64;
    endfunction

    // SiLU(x) = x * sigmoid(x), sigmoid dari LUT + interpolasi linear
    function automatic logic signed [31:0] silu_q16(input logic signed [31:0] x);
        logic [21:0]        xo;      // x + 16.0  (0 .. 2^21-1)
        logic [10:0]        idx;
        logic [9:0]         frac;
        logic [31:0]        s0, s1, sig;
        logic signed [63:0] prod;
        if (x >= 32'sh0010_0000)          return x;      // x >= 16
        else if (x < -32'sh0010_0000)     return '0;     // x < -16
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
    // TODO 3: MAC time-multiplexed
    // ------------------------------------------------------------------
    logic [ICGW-1:0]    g_ic;            // grup saluran masuk yang sedang dihitung
    logic [OCGW-1:0]    g_oc;            // grup filter yang sedang dihitung
    logic               mask_top;        // baris atas window = padding (oy == 0)
    logic               mask_left;       // kolom kiri window = padding (ox == 0)
    logic signed [63:0] acc [0:PAR_OC-1];

    // Satu langkah MAC untuk filter ke-p dari grup g_oc, grup saluran g_ic:
    // PAR_IC x 9 perkalian + akumulasi (dimulai dari bias pada grup pertama).
    // Dipanggil di dalam always_ff -> disintesis sebagai logika kombinasional
    // di depan register acc / pixel_out (PAR_OC x PAR_IC x 9 pengali).
    function automatic logic signed [63:0] mac_step(input int p);
        int oc;
        logic signed [31:0] b;
        logic signed [63:0] sum;
        oc  = int'(g_oc) * PAR_OC + p;
        b   = biases[oc];
        sum = (g_ic == '0) ? {{16{b[31]}}, b, 16'h0000} : acc[p];   // bias Q16.16 -> Q32.32
        for (int q = 0; q < PAR_IC; q++) begin
            int ic;
            ic = int'(g_ic) * PAR_IC + q;
            for (int ky = 0; ky < 3; ky++) begin
                for (int kx = 0; kx < 3; kx++) begin
                    logic signed [31:0] xv;
                    // kx = 0 (kiri) = kolom register 2 (terlama)
                    xv = window_3x3[ic][ky][2-kx];
                    if ((mask_top && ky == 0) || (mask_left && kx == 0)) xv = '0;   // padding
                    sum = sum + mul_q16(xv, weights[((oc*IN_CHANNELS + ic)*9) + (ky*3 + kx)]);
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
            // Beat keluaran diambil oleh hilir
            if (valid_out && ready_out) valid_out <= 1'b0;

            case (state)
                S_STREAM: begin
                    if (trigger) begin
                        state     <= S_COMPUTE;
                        g_ic      <= '0;
                        g_oc      <= '0;
                        mask_top  <= (pixel_y == YW'(1));
                        mask_left <= (pixel_x == XW'(1));
                    end
                end

                S_COMPUTE: begin
                    for (int p = 0; p < PAR_OC; p++) begin
                        logic signed [63:0] s;
                        logic signed [31:0] r;
                        s = mac_step(p);
                        acc[p] <= s;
                        if (g_ic == ICGW'(N_ICG - 1)) begin
                            // Grup filter selesai -> truncate [47:16], aktivasi, tulis ke buffer keluaran
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
