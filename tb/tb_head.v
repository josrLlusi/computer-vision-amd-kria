`timescale 1ns/1ps
// Direct-load testbench (fast): preloads the RAMs with $readmemh, runs, dumps results.
// Run from the directory holding prog/wgt/par/lut/sig/act_init .hex.   +actwords=N +boxwords=N +maxcyc=N
module tb_head;
    parameter LANES = 16, LOGL = 4, AAW = 16, WAW = 16, PAW = 14, LAW = 12, PRW = 10, BAW = 14,
              K = 300, IW = 9, OUTAW = 11, YXW = 10, CW = 10;
    reg clk = 0, rst_n = 0, start = 0;
    always #5 clk = ~clk;
    wire busy, done; wire [IW:0] n_rows; wire [31:0] cycles, h_rdata;

    yolo26_head_core #(.LANES(LANES), .LOGL(LOGL), .AAW(AAW), .WAW(WAW), .PAW(PAW), .LAW(LAW), .PRW(PRW),
                       .BAW(BAW), .K(K), .IW(IW), .OUTAW(OUTAW), .YXW(YXW), .CW(CW)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .busy(busy), .done(done), .n_rows(n_rows), .cycles(cycles),
        .h_en(1'b0), .h_we(1'b0), .h_win(3'd0), .h_waddr(18'd0), .h_wdata(32'd0), .h_rdata(h_rdata));

    integer fd, i, cyc, maxcyc, actwords, boxwords, nr;
    reg seen_done;
    initial begin
        if (!$value$plusargs("maxcyc=%d", maxcyc))     maxcyc = 400000000;
        if (!$value$plusargs("actwords=%d", actwords)) actwords = 0;
        if (!$value$plusargs("boxwords=%d", boxwords)) boxwords = 0;
        $readmemh("prog.hex",     dut.u_prog.mem);
        $readmemh("wgt.hex",      dut.u_wgt.mem);
        $readmemh("par.hex",      dut.u_par.mem);
        $readmemh("lut.hex",      dut.u_lut.mem);
        $readmemh("sig.hex",      dut.u_sig.mem);
        $readmemh("act_init.hex", dut.u_act.mem);
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);
        start = 1; @(negedge clk); start = 0;
        cyc = 0; seen_done = 0;
        while (!seen_done && cyc < maxcyc) begin
            @(posedge clk); cyc = cyc + 1;
            if (done) seen_done = 1;
        end
        if (!seen_done) begin $display("TIMEOUT after %0d cycles", cyc); $finish; end
        $display("CYCLES=%0d", cycles);
        nr = n_rows;
        $display("ROWS=%0d", nr);
        if (actwords > 0) $writememh("act_dump.hex", dut.u_act.mem, 0, actwords - 1);
        if (boxwords > 0) $writememh("box_dump.hex", dut.u_box.mem, 0, boxwords - 1);
        fd = $fopen("rows.txt", "w");
        for (i = 0; i < nr; i = i + 1)
            $fdisplay(fd, "%0d %0d %0d %0d %0d %0d",
                $signed(dut.u_out.mem[i*6+0][15:0]), $signed(dut.u_out.mem[i*6+1][15:0]),
                $signed(dut.u_out.mem[i*6+2][15:0]), $signed(dut.u_out.mem[i*6+3][15:0]),
                dut.u_out.mem[i*6+4][15:0], dut.u_out.mem[i*6+5][15:0]);
        $fclose(fd);
        $finish;
    end
endmodule
