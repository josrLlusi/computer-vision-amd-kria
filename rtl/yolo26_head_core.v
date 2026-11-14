`timescale 1ns/1ps
// Host-side width adapter: 32-bit host accesses into a W-bit-wide RAM port (W multiple of 32).
module hostmux #(
    parameter W = 32, parameter AW = 10, parameter HAW = 18
)(
    input  wire            clk,
    input  wire            sel_we,        // this window selected and writing
    input  wire [HAW-1:0]  h_waddr,       // 32-bit word index inside the window
    input  wire [31:0]     h_wdata,
    output wire [W/8-1:0]  b_we,
    output wire [AW-1:0]   b_addr,
    output wire [W-1:0]    b_din,
    input  wire [W-1:0]    b_dout,
    output wire [31:0]     rdata
);
    localparam NCH = W / 32;
    localparam LN  = $clog2(NCH);
    wire [HAW-1:0] wsh   = h_waddr >> LN;
    wire [HAW-1:0] chunk = h_waddr & (NCH - 1);
    assign b_addr = wsh[AW-1:0];
    wire [W/8-1:0] be = ({(W/8){1'b0}} | 4'hF) << (4 * chunk);
    assign b_we  = sel_we ? be : {(W/8){1'b0}};
    assign b_din = {NCH{h_wdata}};
    reg [HAW-1:0] chunk_d;
    always @(posedge clk) chunk_d <= chunk;
    wire [W-1:0] shd = b_dout >> (32 * chunk_d);
    assign rdata = shd[31:0];
endmodule

// -----------------------------------------------------------------------------
// yolo26_head_core : conv_engine + det_post + all RAMs, with a 32-bit host port.
//  host windows (h_win): 0 prog  1 par  2 wgt  3 lut  4 sig  5 act (idle only)  6 out
//  host read data is valid 1 cycle after the request cycle.
// -----------------------------------------------------------------------------
module yolo26_head_core #(
    parameter LANES = 16, LOGL = 4,
    parameter AAW = 16, WAW = 16, PAW = 14, LAW = 12, PRW = 10, BAW = 14,
    parameter K = 300, IW = 9, OUTAW = 11, HAW = 18,
    parameter YXW = 10,           // bits for a grid coordinate (grid <= 2**YXW)
    parameter CW  = 10,           // bits for a class index (nc <= 2**CW)
    parameter D_PROG = (1<<PRW), D_PAR = (1<<PAW), D_WGT = (1<<WAW), D_LUT = (1<<LAW),
              D_ACT = (1<<AAW), D_BOX = (1<<BAW)
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    output wire              busy,
    output reg               done,
    output wire [IW:0]       n_rows,
    output reg  [31:0]       cycles,

    input  wire              h_en,
    input  wire              h_we,
    input  wire [2:0]        h_win,
    input  wire [HAW-1:0]    h_waddr,
    input  wire [31:0]       h_wdata,
    output wire [31:0]       h_rdata
);
    localparam AW8 = 8 * LANES;

    // ---------------- engine / post signals
    wire [PRW-1:0] e_prog_addr, p_prog_addr, prog_addr;
    wire [31:0]    prog_rdata;
    wire [AAW-1:0] e_act_raddr, p_act_raddr, act_raddr, act_waddr;
    wire [AW8-1:0] act_rdata, act_wdata;
    wire           act_we;
    wire           box_we;  wire [BAW-1:0] box_waddr, box_raddr;  wire [15:0] box_wdata, box_rdata;
    wire [WAW-1:0] w_raddr; wire [AW8-1:0] w_rdata;
    wire [PAW-1:0] p_raddr; wire [63:0] p_rdata;
    wire [LAW-1:0] l_raddr; wire [31:0] l_rdata;
    wire [7:0]     sig_raddr; wire [31:0] sig_rdata;
    wire           out_we;  wire [OUTAW-1:0] out_waddr; wire [31:0] out_wdata;

    reg [1:0] ts; localparam T_IDLE = 0, T_ENG = 1, T_POST = 2;
    assign busy = (ts != T_IDLE);
    wire post_phase = (ts == T_POST);
    assign prog_addr = post_phase ? p_prog_addr : e_prog_addr;
    assign act_raddr = post_phase ? p_act_raddr : e_act_raddr;

    // ---------------- host decode
    wire hw0 = h_en && h_we && (h_win == 3'd0);
    wire hw1 = h_en && h_we && (h_win == 3'd1);
    wire hw2 = h_en && h_we && (h_win == 3'd2);
    wire hw3 = h_en && h_we && (h_win == 3'd3);
    wire hw4 = h_en && h_we && (h_win == 3'd4);
    wire hw5 = h_en && h_we && (h_win == 3'd5) && !busy;

    wire [31:0] rd_prog, rd_par, rd_wgt, rd_lut, rd_sig, rd_act, rd_out;
    reg  [2:0]  win_d;
    reg         busy_d;
    always @(posedge clk) begin win_d <= h_win; busy_d <= busy; end
    assign h_rdata = (win_d == 3'd0) ? rd_prog : (win_d == 3'd1) ? rd_par : (win_d == 3'd2) ? rd_wgt :
                     (win_d == 3'd3) ? rd_lut  : (win_d == 3'd4) ? rd_sig : (win_d == 3'd5) ? (busy_d ? 32'd0 : rd_act) :
                     (win_d == 3'd6) ? rd_out  : 32'd0;

    // ---------------- RAMs (port A = core, port B = host)
    wire [3:0]  pb_we;  wire [PRW-1:0] pb_a;  wire [31:0] pb_d, pb_q;
    hostmux #(32, PRW, HAW) hm0 (clk, hw0, h_waddr, h_wdata, pb_we, pb_a, pb_d, pb_q, rd_prog);
    ram_tdp #(32, PRW, D_PROG) u_prog (.clk(clk), .a_we(4'b0), .a_addr(prog_addr), .a_din(32'd0), .a_dout(prog_rdata),
                               .b_we(pb_we), .b_addr(pb_a), .b_din(pb_d), .b_dout(pb_q));

    wire [7:0]  qb_we;  wire [PAW-1:0] qb_a;  wire [63:0] qb_d, qb_q;
    hostmux #(64, PAW, HAW) hm1 (clk, hw1, h_waddr, h_wdata, qb_we, qb_a, qb_d, qb_q, rd_par);
    ram_tdp #(64, PAW, D_PAR) u_par (.clk(clk), .a_we(8'b0), .a_addr(p_raddr), .a_din(64'd0), .a_dout(p_rdata),
                              .b_we(qb_we), .b_addr(qb_a), .b_din(qb_d), .b_dout(qb_q));

    wire [LANES-1:0] wb_we;  wire [WAW-1:0] wb_a;  wire [AW8-1:0] wb_d, wb_q;
    hostmux #(AW8, WAW, HAW) hm2 (clk, hw2, h_waddr, h_wdata, wb_we, wb_a, wb_d, wb_q, rd_wgt);
    ram_tdp #(AW8, WAW, D_WGT) u_wgt (.clk(clk), .a_we({LANES{1'b0}}), .a_addr(w_raddr), .a_din({AW8{1'b0}}), .a_dout(w_rdata),
                               .b_we(wb_we), .b_addr(wb_a), .b_din(wb_d), .b_dout(wb_q));

    wire [3:0]  lb_we;  wire [LAW-1:0] lb_a;  wire [31:0] lb_d, lb_q;
    hostmux #(32, LAW, HAW) hm3 (clk, hw3, h_waddr, h_wdata, lb_we, lb_a, lb_d, lb_q, rd_lut);
    ram_tdp #(32, LAW, D_LUT) u_lut (.clk(clk), .a_we(4'b0), .a_addr(l_raddr), .a_din(32'd0), .a_dout(l_rdata),
                              .b_we(lb_we), .b_addr(lb_a), .b_din(lb_d), .b_dout(lb_q));

    wire [3:0]  sb_we;  wire [7:0] sb_a;  wire [31:0] sb_d, sb_q;
    hostmux #(32, 8, HAW) hm4 (clk, hw4, h_waddr, h_wdata, sb_we, sb_a, sb_d, sb_q, rd_sig);
    ram_tdp #(32, 8) u_sig (.clk(clk), .a_we(4'b0), .a_addr(sig_raddr), .a_din(32'd0), .a_dout(sig_rdata),
                            .b_we(sb_we), .b_addr(sb_a), .b_din(sb_d), .b_dout(sb_q));

    // activation RAM: port A = core read ; port B = core write while busy, host while idle
    wire [LANES-1:0] ab_hwe;  wire [AAW-1:0] ab_ha;  wire [AW8-1:0] ab_hd, ab_q;
    hostmux #(AW8, AAW, HAW) hm5 (clk, hw5, h_waddr, h_wdata, ab_hwe, ab_ha, ab_hd, ab_q, rd_act);
    wire [LANES-1:0] ab_we = busy ? {LANES{act_we}} : ab_hwe;
    wire [AAW-1:0]   ab_a  = busy ? act_waddr : ab_ha;
    wire [AW8-1:0]   ab_d  = busy ? act_wdata : ab_hd;
    ram_tdp_act #(AW8, AAW, D_ACT) u_act (.clk(clk), .a_we({LANES{1'b0}}), .a_addr(act_raddr), .a_din({AW8{1'b0}}), .a_dout(act_rdata),
                                   .b_we(ab_we), .b_addr(ab_a), .b_din(ab_d), .b_dout(ab_q));

    // box RAM: engine writes (A), post reads (B)
    wire [15:0] box_unused;
    ram_tdp #(16, BAW, D_BOX) u_box (.clk(clk), .a_we({2{box_we}}), .a_addr(box_waddr), .a_din(box_wdata), .a_dout(box_unused),
                              .b_we(2'b0), .b_addr(box_raddr), .b_din(16'd0), .b_dout(box_rdata));

    // output RAM: post writes (A), host reads (B)
    wire [OUTAW-1:0] ob_a = h_waddr[OUTAW-1:0];
    wire [31:0] ob_q;
    ram_tdp #(32, OUTAW) u_out (.clk(clk), .a_we({4{out_we}}), .a_addr(out_waddr), .a_din(out_wdata), .a_dout(),
                                .b_we(4'b0), .b_addr(ob_a), .b_din(32'd0), .b_dout(ob_q));
    assign rd_out = ob_q;

    // ---------------- engines
    reg eng_start, post_start;
    wire eng_done, post_done;

    conv_engine #(.LANES(LANES), .LOGL(LOGL), .AAW(AAW), .WAW(WAW), .PAW(PAW), .LAW(LAW), .PRW(PRW), .BAW(BAW)) u_eng (
        .clk(clk), .rst_n(rst_n), .start(eng_start), .done(eng_done),
        .prog_addr(e_prog_addr), .prog_rdata(prog_rdata),
        .act_raddr(e_act_raddr), .act_rdata(act_rdata),
        .act_we(act_we), .act_waddr(act_waddr), .act_wdata(act_wdata),
        .box_we(box_we), .box_waddr(box_waddr), .box_wdata(box_wdata),
        .w_raddr(w_raddr), .w_rdata(w_rdata), .p_raddr(p_raddr), .p_rdata(p_rdata),
        .l_raddr(l_raddr), .l_rdata(l_rdata));

    det_post #(.LANES(LANES), .LOGL(LOGL), .K(K), .IW(IW), .AAW(AAW), .BAW(BAW), .PRW(PRW), .YXW(YXW), .CW(CW), .OUTAW(OUTAW)) u_post (
        .clk(clk), .rst_n(rst_n), .start(post_start), .done(post_done),
        .prog_addr(p_prog_addr), .prog_rdata(prog_rdata),
        .act_raddr(p_act_raddr), .act_rdata(act_rdata),
        .box_raddr(box_raddr), .box_rdata(box_rdata),
        .sig_raddr(sig_raddr), .sig_rdata(sig_rdata),
        .out_we(out_we), .out_waddr(out_waddr), .out_wdata(out_wdata), .n_rows(n_rows));

    always @(posedge clk) begin
        if (!rst_n) begin
            ts <= T_IDLE; eng_start <= 1'b0; post_start <= 1'b0; done <= 1'b0; cycles <= 0;
        end else begin
            eng_start <= 1'b0; post_start <= 1'b0; done <= 1'b0;
            if (busy) cycles <= cycles + 1'b1;
            case (ts)
                T_IDLE: if (start) begin ts <= T_ENG; eng_start <= 1'b1; cycles <= 0; end
                T_ENG:  if (eng_done)  begin ts <= T_POST; post_start <= 1'b1; end
                T_POST: if (post_done) begin ts <= T_IDLE; done <= 1'b1; end
                default: ts <= T_IDLE;
            endcase
        end
    end
endmodule
