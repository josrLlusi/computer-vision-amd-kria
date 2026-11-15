`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// yolo26_head_axil : AXI4-Lite slave around yolo26_head_core, for the Zynq UltraScale+ PS (Kria K26).
// Address map (24-bit, 32-bit accesses, full-word writes only; WSTRB is ignored):
//   0x000000  regs   +0x00 CTRL   W: bit0 = start (self-clearing)
//                    +0x04 STATUS R: bit0 busy, bit1 done (sticky, cleared by start)
//                    +0x08 N_ROWS  R: number of detection rows
//                    +0x0C CYCLES  R: cycles of the last run
//                    +0x10 ID      R: 0x59323648 ("Y26H")
//                    +0x14 CFG     R: {LANES[15:0], K[15:0]}
//   0x100000 prog  0x200000 par(64b)  0x300000 wgt(8*LANES b)  0x400000 lut  0x500000 sig
//   0x600000 act (only while idle)    0x700000 out (6 words per row, 32-bit each)
// Wide RAMs are accessed as consecutive 32-bit chunks, little-endian (chunk 0 = bits [31:0]).
// -----------------------------------------------------------------------------
module yolo26_head_axil #(
    parameter LANES = 16, LOGL = 4,
    parameter AAW = 16, WAW = 16, PAW = 14, LAW = 12, PRW = 10, BAW = 14,
    parameter K = 300, IW = 9, OUTAW = 11, YXW = 10, CW = 10,
    parameter D_PROG = (1<<PRW), D_PAR = (1<<PAW), D_WGT = (1<<WAW), D_LUT = (1<<LAW),
              D_ACT = (1<<AAW), D_BOX = (1<<BAW)
)(
    input  wire        aclk,
    input  wire        aresetn,
    input  wire [23:0] s_axi_awaddr,  input  wire s_axi_awvalid, output wire s_axi_awready,
    input  wire [31:0] s_axi_wdata,   input  wire [3:0] s_axi_wstrb, input wire s_axi_wvalid, output wire s_axi_wready,
    output wire [1:0]  s_axi_bresp,   output wire s_axi_bvalid,  input  wire s_axi_bready,
    input  wire [23:0] s_axi_araddr,  input  wire s_axi_arvalid, output wire s_axi_arready,
    output wire [31:0] s_axi_rdata,   output wire [1:0] s_axi_rresp, output wire s_axi_rvalid, input wire s_axi_rready,
    output wire        irq
);
    reg        start_p;
    reg        h_en, h_we;
    reg [2:0]  h_win;
    reg [17:0] h_waddr;
    reg [31:0] h_wdata;
    wire [31:0] h_rdata, cycles;
    wire        busy, core_done;
    wire [IW:0] n_rows;
    reg         done_sticky;

    yolo26_head_core #(.LANES(LANES), .LOGL(LOGL), .AAW(AAW), .WAW(WAW), .PAW(PAW), .LAW(LAW), .PRW(PRW),
                       .BAW(BAW), .K(K), .IW(IW), .OUTAW(OUTAW), .YXW(YXW), .CW(CW),
                       .D_PROG(D_PROG), .D_PAR(D_PAR), .D_WGT(D_WGT), .D_LUT(D_LUT), .D_ACT(D_ACT), .D_BOX(D_BOX)) u_core (
        .clk(aclk), .rst_n(aresetn), .start(start_p), .busy(busy), .done(core_done), .n_rows(n_rows), .cycles(cycles),
        .h_en(h_en), .h_we(h_we), .h_win(h_win), .h_waddr(h_waddr), .h_wdata(h_wdata), .h_rdata(h_rdata));

    // ---------------- write channel
    reg aw_got, w_got, bvalid;
    reg [23:0] awa; reg [31:0] wd;
    assign s_axi_awready = !aw_got && !bvalid;
    assign s_axi_wready  = !w_got  && !bvalid;
    assign s_axi_bvalid  = bvalid;
    assign s_axi_bresp   = 2'b00;

    // ---------------- read channel
    reg [1:0]  rst_s;
    reg        rvalid;
    reg [31:0] rdat;
    reg [23:0] raa;
    assign s_axi_arready = (rst_s == 2'd0);
    assign s_axi_rvalid  = rvalid;
    assign s_axi_rdata   = rdat;
    assign s_axi_rresp   = 2'b00;

    reg [31:0] reg_rd;
    always @* begin
        case (raa[7:2])
            6'd1:    reg_rd = {30'b0, done_sticky, busy};
            6'd2:    reg_rd = {{(31-IW){1'b0}}, n_rows};
            6'd3:    reg_rd = cycles;
            6'd4:    reg_rd = 32'h59323648;
            6'd5:    reg_rd = {LANES[15:0], K[15:0]};
            default: reg_rd = 32'd0;
        endcase
    end

    assign irq = done_sticky;

    always @(posedge aclk) begin
        if (!aresetn) begin
            aw_got <= 0; w_got <= 0; bvalid <= 0; rst_s <= 0; rvalid <= 0;
            start_p <= 0; h_en <= 0; h_we <= 0; done_sticky <= 0;
            h_win <= 0; h_waddr <= 0; h_wdata <= 0;
        end else begin
            start_p <= 1'b0; h_en <= 1'b0; h_we <= 1'b0;
            if (core_done) done_sticky <= 1'b1;

            if (s_axi_awvalid && s_axi_awready) begin aw_got <= 1'b1; awa <= s_axi_awaddr; end
            if (s_axi_wvalid  && s_axi_wready)  begin w_got  <= 1'b1; wd  <= s_axi_wdata; end
            if (aw_got && w_got && !bvalid) begin
                if (awa[23:20] == 4'd0) begin
                    if (awa[7:2] == 6'd0 && wd[0] && !busy) begin start_p <= 1'b1; done_sticky <= 1'b0; end
                end else begin
                    h_en <= 1'b1; h_we <= 1'b1; h_win <= awa[22:20] - 3'd1; h_waddr <= awa[19:2]; h_wdata <= wd;
                end
                bvalid <= 1'b1; aw_got <= 1'b0; w_got <= 1'b0;
            end
            if (bvalid && s_axi_bready) bvalid <= 1'b0;

            if (s_axi_arvalid && s_axi_arready) begin
                raa <= s_axi_araddr; rst_s <= 2'd1;
                if (s_axi_araddr[23:20] != 4'd0) begin
                    h_en <= 1'b1; h_we <= 1'b0; h_win <= s_axi_araddr[22:20] - 3'd1; h_waddr <= s_axi_araddr[19:2];
                end
            end
            case (rst_s)
                2'd1: rst_s <= 2'd2;
                2'd2: begin rdat <= (raa[23:20] == 4'd0) ? reg_rd : h_rdata; rvalid <= 1'b1; rst_s <= 2'd3; end
                2'd3: if (s_axi_rready) begin rvalid <= 1'b0; rst_s <= 2'd0; end
                default: ;
            endcase
        end
    end
endmodule
