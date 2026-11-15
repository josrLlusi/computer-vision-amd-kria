`timescale 1ns/1ps
// File-driven testbench. Run it from the directory that holds the .hex files:
//   prog.hex wgt.hex par.hex lut.hex sig.hex act_init.hex
// Plusargs: +actwords=N (dump first N activation words)  +maxcyc=N
// Writes:   rows.txt (detections), act_dump.hex, and prints CYCLES / ROWS.
module tb_head;
    parameter AAW = 16, WAW = 16, PAW = 14, LAW = 12, PRW = 10, K = 300, IW = 9, OUTAW = 12;

    reg clk = 0, rst_n = 0, start = 0;
    always #5 clk = ~clk;

    wire done;
    reg  [OUTAW-1:0] host_raddr = 0;
    wire [15:0] host_rdata;
    wire [IW:0] n_rows;

    yolo26_head_top #(.AAW(AAW), .WAW(WAW), .PAW(PAW), .LAW(LAW), .PRW(PRW),
                      .K(K), .IW(IW), .OUTAW(OUTAW)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .done(done),
        .host_raddr(host_raddr), .host_rdata(host_rdata), .n_rows(n_rows));

    integer fd, i, cyc, maxcyc, actwords, nr;
    reg signed [15:0] v0, v1, v2, v3, v5;
    reg        [15:0] v4;
    reg seen_done;

    initial begin
        if (!$value$plusargs("maxcyc=%d", maxcyc))     maxcyc = 400000000;
        if (!$value$plusargs("actwords=%d", actwords)) actwords = 0;
        $readmemh("prog.hex",     dut.u_prog.mem);
        $readmemh("wgt.hex",      dut.u_wgt.mem);
        $readmemh("par.hex",      dut.u_par.mem);
        $readmemh("lut.hex",      dut.u_lut.mem);
        $readmemh("sig.hex",      dut.u_sig.mem);
        $readmemh("act_init.hex", dut.u_act.mem);
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);
        start = 1; @(negedge clk); start = 0;     // stimulus on the falling edge: no race with the DUT
        cyc = 0; seen_done = 0;
        while (!seen_done && cyc < maxcyc) begin
            @(posedge clk); cyc = cyc + 1;
            if (done) seen_done = 1;
        end
        if (!seen_done) begin
            $display("TIMEOUT after %0d cycles", cyc);
            $finish;
        end
        $display("CYCLES=%0d", cyc);
        nr = n_rows;
        $display("ROWS=%0d", nr);
        if (actwords > 0) $writememh("act_dump.hex", dut.u_act.mem, 0, actwords - 1);
        fd = $fopen("rows.txt", "w");
        for (i = 0; i < nr; i = i + 1) begin
            host_raddr = i*6 + 0; @(posedge clk); @(negedge clk); v0 = host_rdata;
            host_raddr = i*6 + 1; @(posedge clk); @(negedge clk); v1 = host_rdata;
            host_raddr = i*6 + 2; @(posedge clk); @(negedge clk); v2 = host_rdata;
            host_raddr = i*6 + 3; @(posedge clk); @(negedge clk); v3 = host_rdata;
            host_raddr = i*6 + 4; @(posedge clk); @(negedge clk); v4 = host_rdata;
            host_raddr = i*6 + 5; @(posedge clk); @(negedge clk); v5 = host_rdata;
            $fdisplay(fd, "%0d %0d %0d %0d %0d %0d", v0, v1, v2, v3, v4, v5);
        end
        $fclose(fd);
        $finish;
    end
endmodule
