`timescale 1ns / 1ps
// ============================================================================
// tb_backbone.sv - Testbench self-checking untuk conv_kxk / c3k2 / backbone_l0_l5
// Mode & dimensi dibaca dari tb_cfg.svh (dibuat ekstrak_uji_backbone.py).
//
// Plusargs: +BUBBLE=<%>  +BACKPRESSURE=<%>  +FRAMES=<n>  +SEED=<n>
// Keluaran: sim_o1.hex (dan sim_o2.hex untuk mode chain) + ringkasan LULUS/GAGAL
//
// stream_source & stream_sink dapat dipakai ulang untuk menguji blok lain.
// ============================================================================

// ---------------------------------------------------------------------------
// Sumber stream: membaca hex [beat][ch], mengirim dengan valid/ready + gelembung acak
// ---------------------------------------------------------------------------
module stream_source #(
    parameter int CH      = 3,
    parameter int N_BEATS = 16,
    parameter     IN_FILE = "tb_input.hex"
)(
    input  logic               clk,
    input  logic               rst_n,
    output logic [(CH*32)-1:0] data,
    output logic               valid,
    input  logic               ready,
    output logic               done
);
    logic [31:0] mem [0:N_BEATS*CH-1];
    int bubble = 0, frames = 1;

    initial begin
        void'($value$plusargs("BUBBLE=%d", bubble));
        void'($value$plusargs("FRAMES=%d", frames));
        $readmemh(IN_FILE, mem);
        valid = 1'b0;
        data  = '0;
        done  = 1'b0;
        wait (rst_n);
        for (int f = 0; f < frames; f++) begin
            for (int p = 0; p < N_BEATS; p++) begin
                while (bubble > 0 && $urandom_range(99) < bubble) begin
                    @(negedge clk) valid = 1'b0;
                end
                @(negedge clk);
                for (int c = 0; c < CH; c++) data[c*32 +: 32] = mem[p*CH + c];
                valid = 1'b1;
                #1;
                while (!ready) begin
                    @(negedge clk);
                    #1;
                end
                @(posedge clk);
            end
        end
        @(negedge clk) valid = 1'b0;
        done = 1'b1;
    end
endmodule

// ---------------------------------------------------------------------------
// Penerima stream: ready acak, membandingkan tiap beat dengan golden
// ---------------------------------------------------------------------------
module stream_sink #(
    parameter int CH        = 16,
    parameter int N_BEATS   = 16,
    parameter int OUT_W     = 4,
    parameter     GOLD_FILE = "gold_o1.hex",
    parameter     OUT_FILE  = "sim_o1.hex",
    parameter     NAME      = "o1"
)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic [(CH*32)-1:0] data,
    input  logic               valid,
    output logic               ready,
    input  logic               report,     // naik -> cetak ringkasan
    output logic               done,
    output longint             mismatches,
    output longint             received
);
    logic [31:0] gold [0:N_BEATS*CH-1];
    int bp = 0, frames = 1, fout;
    longint max_diff = 0;

    initial begin
        void'($value$plusargs("BACKPRESSURE=%d", bp));
        void'($value$plusargs("FRAMES=%d", frames));
        $readmemh(GOLD_FILE, gold);
        if ($isunknown(gold[N_BEATS*CH-1])) $display("ERROR: %s tidak lengkap/tidak ada", GOLD_FILE);
        fout       = $fopen(OUT_FILE, "w");
        mismatches = 0;
        received   = 0;
        done       = 1'b0;
        ready      = 1'b1;
    end

    always @(negedge clk) ready = ($urandom_range(99) >= bp);

    always @(posedge clk) begin
        if (rst_n && valid && ready) begin
            if (received >= longint'(N_BEATS) * frames) begin
                $display("ERROR [%s]: beat BERLEBIH", NAME);
                mismatches++;
            end else begin
                int p;
                p = int'(received % N_BEATS);
                for (int c = 0; c < CH; c++) begin
                    logic signed [31:0] got, exp;
                    longint d;
                    got = data[c*32 +: 32];
                    exp = gold[p*CH + c];
                    $fwrite(fout, "%08x\n", got);
                    d = longint'(got) - longint'(exp);
                    if (d < 0) d = -d;
                    if (d > max_diff) max_diff = d;
                    if (d != 0) begin
                        if (mismatches < 8)
                            $display("  MISMATCH [%s] frame=%0d y=%0d x=%0d ch=%0d : RTL=%08x GOLD=%08x (beda %0d LSB)",
                                     NAME, received / N_BEATS, p / OUT_W, p % OUT_W, c, got, exp, d);
                        mismatches++;
                    end
                end
            end
            received++;
            if (received == longint'(N_BEATS) * frames) begin
                done = 1'b1;
                $fflush(fout);
            end
        end
    end

    always @(posedge report) begin
        $fclose(fout);
        $display("  [%s] beat diterima %0d / %0d, mismatch %0d, selisih maks %0d LSB",
                 NAME, received, longint'(N_BEATS) * frames, mismatches, max_diff);
    end
endmodule

// ---------------------------------------------------------------------------
// Testbench utama
// ---------------------------------------------------------------------------
module tb_backbone;
    `include "tb_cfg.svh"

    localparam int N_IN = TB_IN_W * TB_IN_H;
    localparam int N_O1 = TB_O1_W * TB_O1_H;
    localparam int N_O2 = TB_O2_W * TB_O2_H;

    logic clk, rst_n;
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    logic [(TB_IN_CH*32)-1:0] in_data;  logic in_valid, in_ready, src_done;
    logic [(TB_O1_CH*32)-1:0] o1_data;  logic o1_valid, o1_ready, o1_done;
    logic [(TB_O2_CH*32)-1:0] o2_data;  logic o2_valid, o2_ready, o2_done;
    longint o1_mis, o1_rcv, o2_mis, o2_rcv;
    longint cycles = 0;
    int     frames = 1, seed;
    logic   report = 1'b0;

    always @(posedge clk) cycles++;

    stream_source #(.CH(TB_IN_CH), .N_BEATS(N_IN)) u_src (
        .clk, .rst_n, .data(in_data), .valid(in_valid), .ready(in_ready), .done(src_done));

    stream_sink #(.CH(TB_O1_CH), .N_BEATS(N_O1), .OUT_W(TB_O1_W),
                  .GOLD_FILE("gold_o1.hex"), .OUT_FILE("sim_o1.hex"), .NAME("out1")) u_snk1 (
        .clk, .rst_n, .data(o1_data), .valid(o1_valid), .ready(o1_ready), .report(report),
        .done(o1_done), .mismatches(o1_mis), .received(o1_rcv));

    // ---------------- DUT sesuai mode ----------------
    generate
        if (TB_MODE == 0) begin : g_conv
            conv_kxk #(.IMAGE_WIDTH(TB_IN_W), .IMAGE_HEIGHT(TB_IN_H),
                       .IN_CHANNELS(TB_IN_CH), .OUT_CHANNELS(TB_O1_CH),
                       .KERNEL(TB_K), .STRIDE(TB_S), .PAR_IC(TB_PIC), .PAR_OC(TB_POC),
                       .WEIGHT_FILE("U.w.hex"), .BIAS_FILE("U.b.hex")) dut (
                .clk, .rst_n, .pixel_in(in_data), .valid_in(in_valid), .ready_in(in_ready),
                .pixel_out(o1_data), .valid_out(o1_valid), .ready_out(o1_ready));
            assign o2_done = 1'b1;
            initial begin o2_mis = 0; o2_rcv = 0; end
        end else if (TB_MODE == 1) begin : g_c3k2
            c3k2 #(.IMAGE_WIDTH(TB_IN_W), .IMAGE_HEIGHT(TB_IN_H),
                   .C1(TB_C1), .C2(TB_C2), .C(TB_C), .CB(TB_CB),
                   .P_CV1_IC(TB_P1I), .P_CV1_OC(TB_P1O), .P_M1_IC(TB_PM1I), .P_M1_OC(TB_PM1O),
                   .P_M2_IC(TB_PM2I), .P_M2_OC(TB_PM2O), .P_CV2_IC(TB_P2I), .P_CV2_OC(TB_P2O),
                   .PREFIX(TB_PREFIX)) dut (
                .clk, .rst_n, .pixel_in(in_data), .valid_in(in_valid), .ready_in(in_ready),
                .pixel_out(o1_data), .valid_out(o1_valid), .ready_out(o1_ready));
            assign o2_done = 1'b1;
            initial begin o2_mis = 0; o2_rcv = 0; end
        end else begin : g_chain
            backbone_l0_l5 #(.IMG(TB_IMG)) dut (
                .clk, .rst_n, .pixel_in(in_data), .valid_in(in_valid), .ready_in(in_ready),
                .p3_data(o1_data), .p3_valid(o1_valid), .p3_ready(o1_ready),
                .l5_data(o2_data), .l5_valid(o2_valid), .l5_ready(o2_ready));
            stream_sink #(.CH(TB_O2_CH), .N_BEATS(N_O2), .OUT_W(TB_O2_W),
                          .GOLD_FILE("gold_o2.hex"), .OUT_FILE("sim_o2.hex"), .NAME("out2")) u_snk2 (
                .clk, .rst_n, .data(o2_data), .valid(o2_valid), .ready(o2_ready), .report(report),
                .done(o2_done), .mismatches(o2_mis), .received(o2_rcv));
        end
    endgenerate

    // ---------------- Kontrol ----------------
    initial begin
        void'($value$plusargs("FRAMES=%d", frames));
        if ($value$plusargs("SEED=%d", seed)) void'($urandom(seed));
        rst_n  = 1'b0;
        $display("==================================================");
        $display(" tb_backbone: mode %0s", TB_MODE == 0 ? "conv_kxk" : TB_MODE == 1 ? "c3k2" : "backbone_l0_l5");
        $display("  Input  : %0d x %0d x %0d ch", TB_IN_W, TB_IN_H, TB_IN_CH);
        $display("  Out1   : %0d x %0d x %0d ch", TB_O1_W, TB_O1_H, TB_O1_CH);
        if (TB_MODE == 2) $display("  Out2   : %0d x %0d x %0d ch", TB_O2_W, TB_O2_H, TB_O2_CH);
        $display("==================================================");
        repeat (3) @(posedge clk);
        @(negedge clk) rst_n = 1'b1;
        wait (o1_done && o2_done);
        repeat (5) @(posedge clk);
        finish_test();
    end

    // Watchdog: berhenti bila tidak ada keluaran baru selama 2 juta siklus
    initial begin
        longint last_cnt, idle;
        last_cnt = -1;
        idle     = 0;
        wait (rst_n);
        forever begin
            @(posedge clk);
            if (o1_rcv + o2_rcv != last_cnt) begin
                last_cnt = o1_rcv + o2_rcv;
                idle = 0;
            end else begin
                idle = idle + 1;
            end
            if (idle > 2_000_000) begin
                $display("ERROR: WATCHDOG - tidak ada keluaran selama 2 juta siklus (deadlock?)");
                finish_test();
            end
        end
    end

    task automatic finish_test();
        bit ok;
        $display("--------------------------------------------------");
        report = 1'b1;
        #1;
        ok = (o1_mis == 0) && (o1_rcv == longint'(N_O1) * frames);
        if (TB_MODE == 2)
            ok = ok && (o2_mis == 0) && (o2_rcv == longint'(N_O2) * frames);
        $display("  Siklus clock : %0d", cycles);
        $display("--------------------------------------------------");
        $display("  STATUS: %s", ok ? "LULUS VERIFIKASI (bit-exact)" : "GAGAL");
        $display("==================================================");
        $stop;
    endtask
endmodule
