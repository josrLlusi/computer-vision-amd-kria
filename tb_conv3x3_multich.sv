`timescale 1ns / 1ps
// ============================================================================
// tb_conv3x3_multich.sv (v2) - Testbench self-checking untuk conv3x3_s2 v2
//
// Parameter dibaca dari tb_params.svh (dibuat ekstrak_uji_multichannel.py).
// Plusargs opsional:
//   +BUBBLE=<persen>       siklus kosong acak pada valid_in       (default 0)
//   +BACKPRESSURE=<persen> siklus ready_out=0 acak dari hilir     (default 0)
//   +FRAMES=<n>            kirim citra yang sama n kali berturut  (default 1)
//   +SEED=<n>              seed acak
// Keluaran: sim_output.hex (urutan [frame][oy][ox][oc]) + ringkasan LULUS/GAGAL
// ============================================================================
module tb_conv3x3_multich;

    `include "tb_params.svh"

    localparam int N_PIX  = TB_IMAGE_WIDTH * TB_IMAGE_HEIGHT;
    localparam int N_IN   = N_PIX * TB_IN_CHANNELS;
    localparam int N_OPIX = TB_OUT_WIDTH * TB_OUT_HEIGHT;
    localparam int N_OUT  = N_OPIX * TB_OUT_CHANNELS;
    localparam int CYC_PER_OUT = (TB_IN_CHANNELS / TB_PAR_IC) * (TB_OUT_CHANNELS / TB_PAR_OC);

    // ---------------- DUT ----------------
    logic clk, rst_n;
    logic [(TB_IN_CHANNELS*32)-1:0]  pixel_in;
    logic                            valid_in, ready_in;
    logic [(TB_OUT_CHANNELS*32)-1:0] pixel_out;
    logic                            valid_out, ready_out;

    conv3x3_s2 #(
        .IMAGE_WIDTH (TB_IMAGE_WIDTH),
        .IMAGE_HEIGHT(TB_IMAGE_HEIGHT),
        .IN_CHANNELS (TB_IN_CHANNELS),
        .OUT_CHANNELS(TB_OUT_CHANNELS),
        .PAR_IC      (TB_PAR_IC),
        .PAR_OC      (TB_PAR_OC),
        .USE_SILU    (TB_USE_SILU),
        .WEIGHT_FILE ("tb_weight.hex"),
        .BIAS_FILE   ("tb_bias.hex"),
        .SIGMOID_FILE("sigmoid_lut.hex")
    ) dut (.*);

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;      // 100 MHz
    end

    // ---------------- Data uji ----------------
    logic [31:0] mem_in   [0:N_IN-1];
    logic [31:0] mem_gold [0:N_OUT-1];

    int     fout;
    int     bubble_pct = 0, bp_pct = 0, frames = 1, seed = 1;
    longint out_pix_cnt = 0, mismatches = 0, max_diff = 0, cycles = 0;
    longint t_start;

    // ---------------- Hilir: ready_out acak ----------------
    initial ready_out = 1'b1;
    always @(negedge clk) ready_out = ($urandom_range(99) >= bp_pct);

    always @(posedge clk) cycles++;

    // ---------------- Stimulus ----------------
    initial begin
        rst_n    = 1'b0;
        valid_in = 1'b0;
        pixel_in = '0;
        void'($value$plusargs("BUBBLE=%d", bubble_pct));
        void'($value$plusargs("BACKPRESSURE=%d", bp_pct));
        void'($value$plusargs("FRAMES=%d", frames));
        if ($value$plusargs("SEED=%d", seed)) void'($urandom(seed));

        $readmemh("tb_input.hex",  mem_in);
        $readmemh("tb_golden.hex", mem_gold);
        if ($isunknown(mem_in[N_IN-1]) || $isunknown(mem_gold[N_OUT-1])) begin
            $display("ERROR: file hex tidak lengkap. Jalankan ekstrak_uji_multichannel.py dulu.");
            $finish;
        end
        fout = $fopen("sim_output.hex", "w");

        $display("==================================================");
        $display(" Uji conv3x3_s2 v2 (Conv 3x3/2 + pad 1 + bias + %s)", TB_USE_SILU ? "SiLU" : "tanpa aktivasi");
        $display("  Input    : %0d x %0d x %0d ch", TB_IMAGE_WIDTH, TB_IMAGE_HEIGHT, TB_IN_CHANNELS);
        $display("  Keluaran : %0d x %0d x %0d ch", TB_OUT_WIDTH, TB_OUT_HEIGHT, TB_OUT_CHANNELS);
        $display("  PAR      : IC=%0d OC=%0d -> %0d pengali, %0d siklus/piksel keluaran",
                 TB_PAR_IC, TB_PAR_OC, TB_PAR_IC*TB_PAR_OC*9, CYC_PER_OUT);
        $display("  Frame=%0d  Bubble=%0d%%  Backpressure=%0d%%", frames, bubble_pct, bp_pct);
        $display("==================================================");

        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        t_start = cycles;

        for (int f = 0; f < frames; f++) begin
            for (int p = 0; p < N_PIX; p++) begin
                // Gelembung acak pada valid_in
                while (bubble_pct > 0 && $urandom_range(99) < bubble_pct) begin
                    @(negedge clk) valid_in = 1'b0;
                end
                @(negedge clk);
                for (int c = 0; c < TB_IN_CHANNELS; c++)
                    pixel_in[c*32 +: 32] = mem_in[p*TB_IN_CHANNELS + c];
                valid_in = 1'b1;
                // Tahan data sampai diterima (valid_in & ready_in pada posedge berikutnya)
                #1;
                while (!ready_in) begin
                    @(negedge clk);
                    #1;
                end
                @(posedge clk);
                if ((p+1) % (TB_IMAGE_WIDTH*32) == 0)
                    $display("  [frame %0d] baris input %0d / %0d", f, (p+1)/TB_IMAGE_WIDTH, TB_IMAGE_HEIGHT);
            end
        end
        @(negedge clk) valid_in = 1'b0;

        // Tunggu semua keluaran
        wait (out_pix_cnt == longint'(N_OPIX) * frames);
        repeat (5) @(posedge clk);
        finish_test();
    end

    // ---------------- Monitor & pembanding ----------------
    always @(posedge clk) begin
        if (rst_n && valid_out && ready_out) begin
            if (out_pix_cnt >= longint'(N_OPIX) * frames) begin
                $display("ERROR: keluaran BERLEBIH");
                mismatches++;
            end else begin
                int pix;
                pix = int'(out_pix_cnt % N_OPIX);
                for (int oc = 0; oc < TB_OUT_CHANNELS; oc++) begin
                    logic signed [31:0] got, exp;
                    longint d;
                    got = pixel_out[oc*32 +: 32];
                    exp = mem_gold[pix*TB_OUT_CHANNELS + oc];
                    $fwrite(fout, "%08x\n", got);
                    d = longint'(got) - longint'(exp);
                    if (d < 0) d = -d;
                    if (d > max_diff) max_diff = d;
                    if (d != 0) begin
                        if (mismatches < 10)
                            $display("  MISMATCH frame=%0d oy=%0d ox=%0d oc=%0d : RTL=%08x GOLD=%08x (beda %0d LSB)",
                                     out_pix_cnt / N_OPIX, pix / TB_OUT_WIDTH, pix % TB_OUT_WIDTH,
                                     oc, got, exp, d);
                        mismatches++;
                    end
                end
            end
            out_pix_cnt++;
        end
    end

    // Watchdog
    initial begin
        #1;
        wait (rst_n);
        while (cycles - t_start < (longint'(N_PIX) * 8 + longint'(N_OPIX) * CYC_PER_OUT * 8 + 10000) * frames)
            @(posedge clk);
        $display("ERROR: WATCHDOG timeout!");
        finish_test();
    end

    task automatic finish_test();
        $fclose(fout);
        $display("--------------------------------------------------");
        $display("  Piksel keluaran diterima : %0d / %0d", out_pix_cnt, longint'(N_OPIX) * frames);
        $display("  Mismatch (bit-exact)     : %0d", mismatches);
        $display("  Selisih maksimum         : %0d LSB", max_diff);
        $display("  Siklus clock             : %0d (%0.1f siklus/piksel input)", cycles - t_start,
                 real'(cycles - t_start) / (real'(N_PIX) * frames));
        $display("--------------------------------------------------");
        if (mismatches == 0 && out_pix_cnt == longint'(N_OPIX) * frames)
            $display("  STATUS: LULUS VERIFIKASI (bit-exact)");
        else
            $display("  STATUS: GAGAL");
        $display("==================================================");
        $stop;   // $stop (bukan $finish) agar jendela wave ModelSim tetap hidup
    endtask

endmodule