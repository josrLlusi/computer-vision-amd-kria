`timescale 1ns / 1ps

module tb_yolo_top;

    // Resolusi 32x32 -> P3=4x4 (16), P4=2x2 (4), P5=1x1 (1)
    localparam int IMG       = 32;
    localparam int P3_EXP    = (IMG / 8)  * (IMG / 8);   // 16
    localparam int P4_EXP    = (IMG / 16) * (IMG / 16);  // 4
    localparam int P5_EXP    = (IMG / 32) * (IMG / 32);  // 1
    localparam int WORDS_EXP = (P3_EXP * 32) + (P4_EXP * 32) + (P5_EXP * 64); // 704 words

    logic              clk;
    logic              rst_n;

    logic [(3*32)-1:0] pixel_in;
    logic              valid_in;
    logic              ready_in;

    logic              host_en;
    logic              host_we;
    logic [2:0]        host_win;
    logic [17:0]       host_waddr;
    logic [31:0]       host_wdata;
    logic [31:0]       host_rdata;

    logic              head_busy;
    logic              head_done;
    logic [7:0]        n_rows;
    logic [31:0]       head_cycles;

    // Clock 100 MHz (periode 10 ns)
    initial clk = 1'b0;
    always #5 clk = ~clk;

    yolo_top #(
        .IMG(IMG),
        .P3_ACT_BASE(18'd0),
        .P4_ACT_BASE(18'd512),  // 16 * (128/4)
        .P5_ACT_BASE(18'd640)   // 512 + 4 * (128/4)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .host_en(host_en),
        .host_we(host_we),
        .host_win(host_win),
        .host_waddr(host_waddr),
        .host_wdata(host_wdata),
        .host_rdata(host_rdata),
        .head_busy(head_busy),
        .head_done(head_done),
        .n_rows(n_rows),
        .head_cycles(head_cycles)
    );

    // Counter handshake tiap stage
    int   cnt_in     = 0;
    int   cnt_p3     = 0;
    int   cnt_p4     = 0;
    int   cnt_l7     = 0;
    int   cnt_l8     = 0;
    int   cnt_p5     = 0;
    int   cnt_hs     = 0;
    int   cnt_hm     = 0;
    int   cnt_hl     = 0;
    int   cnt_ram_wr = 0;
    logic head_triggered = 1'b0;
    logic input_done     = 1'b0;

    always @(posedge clk) begin
        if (!rst_n) begin
            head_triggered <= 1'b0;
        end else begin
            if (valid_in && ready_in)                           cnt_in     <= cnt_in + 1;
            if (dut.p3_bb_valid && dut.p3_bb_ready)             cnt_p3     <= cnt_p3 + 1;
            if (dut.p4_fire)                                    cnt_p4     <= cnt_p4 + 1;
            if (dut.l7_valid && dut.l7_ready)                   cnt_l7     <= cnt_l7 + 1;
            if (dut.l8_valid && dut.l8_ready)                   cnt_l8     <= cnt_l8 + 1;
            if (dut.p5_neck_valid && dut.p5_neck_ready)         cnt_p5     <= cnt_p5 + 1;
            if (dut.head_small_valid && dut.head_small_ready)   cnt_hs     <= cnt_hs + 1;
            if (dut.head_medium_valid && dut.head_medium_ready) cnt_hm     <= cnt_hm + 1;
            if (dut.head_large_valid && dut.head_large_ready)   cnt_hl     <= cnt_hl + 1;
            if (dut.bridge_we)                                  cnt_ram_wr <= cnt_ram_wr + 1;
            if (dut.head_start_pulse) begin
                head_triggered <= 1'b1;
                $display("[%0t] BRIDGE SELESAI -> Pulsa start Detect Head aktif!", $time);
            end
        end
    end

    // Monitor progres berkala setiap 2.000 siklus clock setelah input selesai
    initial begin
        wait (input_done === 1'b1);
        forever begin
            #20_000;
            $display("[%0t] PROGRES -> P3:%0d/%0d | P4:%0d/%0d | L7:%0d | L8:%0d | P5:%0d/%0d | Neck(S/M/L):%0d/%0d/%0d | RAM:%0d",
                     $time, cnt_p3, P3_EXP, cnt_p4, P4_EXP,
                     cnt_l7, cnt_l8, cnt_p5, P5_EXP,
                     cnt_hs, cnt_hm, cnt_hl, cnt_ram_wr);
        end
    end

    int px;
    initial begin
`ifdef DUMP_VCD
        $dumpfile("tb_yolo_top.vcd");
        $dumpvars(1, tb_yolo_top);
`endif

        rst_n      = 1'b0;
        pixel_in   = '0;
        valid_in   = 1'b0;
        host_en    = 1'b0;
        host_we    = 1'b0;
        host_win   = 3'd0;
        host_waddr = '0;
        host_wdata = 32'd0;

        repeat (10) @(posedge clk);
        rst_n = 1'b1;
        repeat (5) @(posedge clk);

        $display("[%0t] Mulai mengirim frame %0dx%0d (%0d piksel)...", $time, IMG, IMG, IMG * IMG);

        for (px = 0; px < IMG * IMG; px = px + 1) begin
            @(posedge clk);
            valid_in <= 1'b1;
            pixel_in[31:0]  <= 32'h0000_4000 + (px & 8'hFF); // R ~0.25 (Q16.16)
            pixel_in[63:32] <= 32'h0000_8000;                // G ~0.50
            pixel_in[95:64] <= 32'h0000_C000;                // B ~0.75
            while (!ready_in) @(posedge clk);

            if ((px % 256) == 255)
                $display("[%0t] Terkirim %0d/%0d piksel...", $time, px + 1, IMG * IMG);
        end

        @(posedge clk);
        valid_in   <= 1'b0;
        input_done <= 1'b1;
        $display("[%0t] Semua %0d piksel selesai dikirim. Memantau L6..L10, Neck & Bridge...", $time, IMG * IMG);

        fork
            begin
                wait (head_done === 1'b1);
                $display("[%0t] HEAD DONE! cycles=%0d, n_rows=%0d", $time, head_cycles, n_rows);
            end
            begin
                wait (head_triggered === 1'b1);
                repeat (2000) @(posedge clk);
                if (!head_done)
                    $display("[%0t] INFO: Head sudah terpicu (head_busy=%0b), menunggu prog ROM.", $time, head_busy);
            end
            begin
                wait (input_done === 1'b1);
                #2_500_000; // Timeout 250.000 siklus clock setelah input selesai
                $display("[%0t] TIMEOUT: Pipeline tertahan setelah 250.000 siklus.", $time);
            end
        join_any

        $display("==========================================================");
        $display("RINGKASAN PIPELINE END-TO-END (IMG=%0d):", IMG);
        $display("  Input Pixels  : %0d / %0d", cnt_in, IMG * IMG);
        $display("  Backbone Out  : P3=%0d/%0d | P4=%0d/%0d | L7=%0d | L8=%0d | P5=%0d/%0d",
                 cnt_p3, P3_EXP, cnt_p4, P4_EXP, cnt_l7, cnt_l8, cnt_p5, P5_EXP);
        $display("  Neck Fusion   : Small=%0d/%0d | Medium=%0d/%0d | Large=%0d/%0d",
                 cnt_hs, P3_EXP, cnt_hm, P4_EXP, cnt_hl, P5_EXP);
        $display("  ACT RAM Write : %0d / %0d words | Head Triggered=%0b",
                 cnt_ram_wr, WORDS_EXP, head_triggered);
        $display("==========================================================");

        if (head_triggered && (cnt_ram_wr == WORDS_EXP))
            $display("STATUS: PASS (End-to-end streaming tanpa deadlock!)");
        else
            $display("STATUS: CEK STAGE DI ATAS (Ada beat yang belum tuntas)");

        $finish;
    end

endmodule