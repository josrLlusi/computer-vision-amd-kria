`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// yolo26_head_top : YOLO26 one2one Detect head.
//   input  : neck feature maps P3/P4/P5 (int8 in 16-bit words) preloaded in act RAM
//   program: layer descriptors in prog RAM, weights/params/LUTs in their RAMs
//   output : n_rows x 6 words in out RAM   [x1,y1,x2,y2 (Q.2 pixels), score(u16), class]
// Phase 1 runs conv_engine over all box/cls chains of all 3 scales,
// phase 2 runs det_post (decode + sigmoid + two-stage top-k).
// -----------------------------------------------------------------------------
module yolo26_head_top #(
    parameter AAW   = 16,     // activation RAM address bits
    parameter WAW   = 16,     // weight RAM
    parameter PAW   = 14,     // bias/multiplier RAM
    parameter LAW   = 12,     // SiLU LUT RAM (256 words per activation layer)
    parameter PRW   = 10,     // program RAM
    parameter K     = 300,    // max detections (top-k size)
    parameter IW    = 9,      // 2**IW >= K
    parameter OUTAW = 12      // output RAM, needs >= 6*K words
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    output reg              done,
    input  wire [OUTAW-1:0] host_raddr,
    output wire [15:0]      host_rdata,
    output wire [IW:0]      n_rows
);
    // ---- memories
    wire [PRW-1:0] prog_addr, e_prog_addr, p_prog_addr;
    wire [31:0]    prog_rdata;
    wire [AAW-1:0] act_raddr, e_act_raddr, p_act_raddr, act_waddr;
    wire signed [15:0] act_rdata, act_wdata;
    wire           act_we;
    wire [WAW-1:0] w_raddr;  wire signed [7:0] w_rdata;
    wire [PAW-1:0] p_raddr;  wire [31:0] p_rdata;
    wire [LAW-1:0] l_raddr;  wire signed [7:0] l_rdata;
    wire [7:0]     sig_raddr; wire [15:0] sig_rdata;
    wire           out_we;   wire [OUTAW-1:0] out_waddr; wire [15:0] out_wdata;

    reg [1:0] ts;  localparam T_IDLE = 0, T_ENG = 1, T_POST = 2;
    wire post_phase = (ts == T_POST);
    assign prog_addr = post_phase ? p_prog_addr : e_prog_addr;
    assign act_raddr = post_phase ? p_act_raddr : e_act_raddr;

    ram_sync #(.DW(32), .AW(PRW)) u_prog (.clk(clk), .we(1'b0), .waddr({PRW{1'b0}}), .wdata(32'd0),
                                          .raddr(prog_addr), .rdata(prog_rdata));
    ram_sync #(.DW(16), .AW(AAW)) u_act  (.clk(clk), .we(act_we), .waddr(act_waddr), .wdata(act_wdata),
                                          .raddr(act_raddr), .rdata(act_rdata));
    ram_sync #(.DW(8),  .AW(WAW)) u_wgt  (.clk(clk), .we(1'b0), .waddr({WAW{1'b0}}), .wdata(8'd0),
                                          .raddr(w_raddr), .rdata(w_rdata));
    ram_sync #(.DW(32), .AW(PAW)) u_par  (.clk(clk), .we(1'b0), .waddr({PAW{1'b0}}), .wdata(32'd0),
                                          .raddr(p_raddr), .rdata(p_rdata));
    ram_sync #(.DW(8),  .AW(LAW)) u_lut  (.clk(clk), .we(1'b0), .waddr({LAW{1'b0}}), .wdata(8'd0),
                                          .raddr(l_raddr), .rdata(l_rdata));
    ram_sync #(.DW(16), .AW(8))   u_sig  (.clk(clk), .we(1'b0), .waddr(8'd0), .wdata(16'd0),
                                          .raddr(sig_raddr), .rdata(sig_rdata));
    ram_sync #(.DW(16), .AW(OUTAW)) u_out(.clk(clk), .we(out_we), .waddr(out_waddr), .wdata(out_wdata),
                                          .raddr(host_raddr), .rdata(host_rdata));

    // ---- engines
    reg eng_start, post_start;
    wire eng_done, post_done;

    conv_engine #(.AAW(AAW), .WAW(WAW), .PAW(PAW), .LAW(LAW), .PRW(PRW)) u_eng (
        .clk(clk), .rst_n(rst_n), .start(eng_start), .done(eng_done),
        .prog_addr(e_prog_addr), .prog_rdata(prog_rdata),
        .act_raddr(e_act_raddr), .act_rdata(act_rdata),
        .act_we(act_we), .act_waddr(act_waddr), .act_wdata(act_wdata),
        .w_raddr(w_raddr), .w_rdata(w_rdata),
        .p_raddr(p_raddr), .p_rdata(p_rdata),
        .l_raddr(l_raddr), .l_rdata(l_rdata));

    det_post #(.K(K), .IW(IW), .AAW(AAW), .PRW(PRW), .OUTAW(OUTAW)) u_post (
        .clk(clk), .rst_n(rst_n), .start(post_start), .done(post_done),
        .prog_addr(p_prog_addr), .prog_rdata(prog_rdata),
        .act_raddr(p_act_raddr), .act_rdata(act_rdata),
        .sig_raddr(sig_raddr), .sig_rdata(sig_rdata),
        .out_we(out_we), .out_waddr(out_waddr), .out_wdata(out_wdata),
        .n_rows(n_rows));

    always @(posedge clk) begin
        if (!rst_n) begin
            ts <= T_IDLE; eng_start <= 1'b0; post_start <= 1'b0; done <= 1'b0;
        end else begin
            eng_start <= 1'b0; post_start <= 1'b0; done <= 1'b0;
            case (ts)
                T_IDLE: if (start) begin ts <= T_ENG; eng_start <= 1'b1; end
                T_ENG:  if (eng_done)  begin ts <= T_POST; post_start <= 1'b1; end
                T_POST: if (post_done) begin ts <= T_IDLE; done <= 1'b1; end
                default: ts <= T_IDLE;
            endcase
        end
    end
endmodule
