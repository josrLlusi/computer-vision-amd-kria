`timescale 1ns/1ps
// PS-style testbench: everything goes through the AXI4-Lite slave (like devmem / PYNQ MMIO would).
//   load prog/par/wgt/lut/sig/act  ->  write CTRL.start  ->  poll STATUS  ->  read N_ROWS, CYCLES, out rows, act RAM
// plusargs: +nprog= +npar= +nwgt= +nlut= +nin= +nact=
module tb_axil;
    parameter LANES = 16, LOGL = 4, AAW = 16, WAW = 16, PAW = 14, LAW = 12, PRW = 10, BAW = 14,
              K = 300, IW = 9, OUTAW = 11;
    localparam AWID = 8*LANES, NCHW = LANES/4, NCHP = 2;
    reg aclk = 0, aresetn = 0;
    always #5 aclk = ~aclk;

    reg  [23:0] awaddr = 0; reg awvalid = 0; wire awready;
    reg  [31:0] wdata = 0;  reg wvalid = 0;  wire wready;
    wire [1:0] bresp; wire bvalid; reg bready = 0;
    reg  [23:0] araddr = 0; reg arvalid = 0; wire arready;
    wire [31:0] rdata; wire [1:0] rresp; wire rvalid; reg rready = 0;
    wire irq;

    yolo26_head_axil #(.LANES(LANES), .LOGL(LOGL), .AAW(AAW), .WAW(WAW), .PAW(PAW), .LAW(LAW), .PRW(PRW),
                       .BAW(BAW), .K(K), .IW(IW), .OUTAW(OUTAW)) dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready), .irq(irq));

    // ---------------------------------------------------------------- AXI master tasks
    reg aw_done, w_done;
    task axi_write(input [23:0] a, input [31:0] d);
        begin
            @(negedge aclk); awaddr = a; awvalid = 1; wdata = d; wvalid = 1; aw_done = 0; w_done = 0;
            while (!(aw_done && w_done)) begin
                @(posedge aclk);
                if (awvalid && awready) aw_done = 1;
                if (wvalid && wready)   w_done = 1;
                @(negedge aclk);
                if (aw_done) awvalid = 0;
                if (w_done)  wvalid = 0;
            end
            bready = 1;
            @(posedge aclk);
            while (!bvalid) @(posedge aclk);
            @(negedge aclk); bready = 0;
        end
    endtask
    task axi_read(input [23:0] a, output [31:0] d);
        reg got;
        begin
            @(negedge aclk); araddr = a; arvalid = 1; got = 0;
            while (!got) begin @(posedge aclk); if (arvalid && arready) got = 1; end
            @(negedge aclk); arvalid = 0; rready = 1;
            @(posedge aclk);
            while (!rvalid) @(posedge aclk);
            d = rdata;
            @(negedge aclk); rready = 0;
        end
    endtask

    // ---------------------------------------------------------------- data
    reg [31:0]     t_prog [0:32767];
    reg [63:0]     t_par  [0:32767];
    reg [AWID-1:0] t_wgt  [0:65535];
    reg [31:0]     t_lut  [0:32767];
    reg [31:0]     t_sig  [0:255];
    reg [AWID-1:0] t_in   [0:65535];
    reg [AWID-1:0] t_act;
    integer nprog, npar, nwgt, nlut, nin, nact, i, c, fd, nr, cycles_r, guard;
    reg [31:0] v, st;
    reg [31:0] r0, r1, r2, r3, r4, r5;
    localparam [23:0] B_PROG = 24'h100000, B_PAR = 24'h200000, B_WGT = 24'h300000, B_LUT = 24'h400000,
                      B_SIG = 24'h500000, B_ACT = 24'h600000, B_OUT = 24'h700000;

    task wr_wide(input [23:0] base, input integer idx, input integer nch, input [AWID-1:0] w);
        integer ch;
        begin
            for (ch = 0; ch < nch; ch = ch + 1)
                axi_write(base + ((idx*nch + ch) * 4), w[32*ch +: 32]);
        end
    endtask

    initial begin
        if (!$value$plusargs("nprog=%d", nprog)) nprog = 0;
        if (!$value$plusargs("npar=%d",  npar))  npar = 0;
        if (!$value$plusargs("nwgt=%d",  nwgt))  nwgt = 0;
        if (!$value$plusargs("nlut=%d",  nlut))  nlut = 0;
        if (!$value$plusargs("nin=%d",   nin))   nin = 0;
        if (!$value$plusargs("nact=%d",  nact))  nact = 0;
        $readmemh("prog.hex", t_prog); $readmemh("par.hex", t_par); $readmemh("wgt.hex", t_wgt);
        $readmemh("lut.hex", t_lut);   $readmemh("sig.hex", t_sig); $readmemh("act_init.hex", t_in);
        repeat (5) @(negedge aclk); aresetn = 1; repeat (3) @(negedge aclk);

        axi_read(24'h000010, v);  $display("ID     = %h (expect 59323648)", v);
        axi_read(24'h000014, v);  $display("CFG    = LANES %0d  K %0d", v[31:16], v[15:0]);
        for (i = 0; i < nprog; i = i + 1) axi_write(B_PROG + i*4, t_prog[i]);
        for (i = 0; i < npar;  i = i + 1) wr_wide(B_PAR, i, 2, {{(AWID-64){1'b0}}, t_par[i]});
        for (i = 0; i < nwgt;  i = i + 1) wr_wide(B_WGT, i, NCHW, t_wgt[i]);
        for (i = 0; i < nlut;  i = i + 1) axi_write(B_LUT + i*4, t_lut[i]);
        for (i = 0; i < 256;   i = i + 1) axi_write(B_SIG + i*4, t_sig[i]);
        for (i = 0; i < nin;   i = i + 1) wr_wide(B_ACT, i, NCHW, t_in[i]);
        $display("loaded over AXI-Lite; starting");

        axi_write(24'h000000, 32'd1);                   // CTRL.start
        guard = 0; st = 0;
        while (!st[1] && guard < 3000000) begin axi_read(24'h000004, st); guard = guard + 1; end
        if (!st[1]) begin $display("TIMEOUT waiting for done"); $finish; end
        $display("irq = %0d (expect 1)", irq);
        axi_read(24'h000008, v); nr = v;              $display("ROWS=%0d", nr);
        axi_read(24'h00000C, v); cycles_r = v;        $display("CYCLES=%0d", cycles_r);

        fd = $fopen("rows.txt", "w");
        for (i = 0; i < nr; i = i + 1) begin
            axi_read(B_OUT + (i*6+0)*4, r0); axi_read(B_OUT + (i*6+1)*4, r1); axi_read(B_OUT + (i*6+2)*4, r2);
            axi_read(B_OUT + (i*6+3)*4, r3); axi_read(B_OUT + (i*6+4)*4, r4); axi_read(B_OUT + (i*6+5)*4, r5);
            $fdisplay(fd, "%0d %0d %0d %0d %0d %0d", $signed(r0[15:0]), $signed(r1[15:0]), $signed(r2[15:0]),
                      $signed(r3[15:0]), r4[15:0], r5[15:0]);
        end
        $fclose(fd);

        fd = $fopen("act_dump.hex", "w");
        for (i = 0; i < nact; i = i + 1) begin
            for (c = 0; c < NCHW; c = c + 1) begin axi_read(B_ACT + (i*NCHW + c)*4, v); t_act[32*c +: 32] = v; end
            $fdisplay(fd, "%0h", t_act);
        end
        $fclose(fd);
        $finish;
    end
endmodule
