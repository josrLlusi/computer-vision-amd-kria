`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// det_post (Kria / v2): YOLO26 one2one post-processing, NMS-free.
//  stage 1 : best class logit (int8) per cell over the 3 scales -> sorter #1 keeps the k best cells
//  stage 2 : every (cell,class) logit of those k cells -> sorter #2 keeps the k best
//  stage 3 : per winner: 4 box distances (16-bit RAM) -> corners, sigmoid LUT, write 6 words
//            row = [x1,y1,x2,y2 (pixels*4), score (u16), class]   each stored sign-extended in a 32-bit word
//  k = min(K, number of cells). Ties -> earlier candidate wins. Ranking uses logits (monotonic with sigmoid).
// scale table in prog ROM header, 6 words per scale starting at word 4:
//   cls_base(word) box_base(box-RAM word) H W stride_shift cls_words_per_pixel
// -----------------------------------------------------------------------------
module det_post #(
    parameter LANES = 16, LOGL = 4,
    parameter K = 300, IW = 9,
    parameter AAW = 16, BAW = 14, PRW = 10, YXW = 10, CW = 10, OUTAW = 11
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    output reg                  done,

    output wire [PRW-1:0]       prog_addr,
    input  wire [31:0]          prog_rdata,

    output wire [AAW-1:0]       act_raddr,
    input  wire [8*LANES-1:0]   act_rdata,

    output wire [BAW-1:0]       box_raddr,
    input  wire signed [15:0]   box_rdata,

    output wire [7:0]           sig_raddr,
    input  wire [31:0]          sig_rdata,

    output wire                 out_we,
    output wire [OUTAW-1:0]     out_waddr,
    output reg  [31:0]          out_wdata,
    output reg  [IW:0]          n_rows
);
    localparam PW1 = 2 + 2*YXW;
    localparam PW2 = IW + CW;

    localparam [5:0]
        P_IDLE = 0,  P_HREQ = 1, P_HCAP = 2, P_INIT0 = 3, P_INIT1 = 4, P_INIT2 = 5,
        S1_SETUP = 6, S1_REQ = 7, S1_CAP = 8,
        S2_J = 9, S2_J2 = 10, S2_J3 = 11, S2_REQ = 12, S2_CAP = 13,
        S3_R = 14, S3_R2 = 15, S3_B0 = 16, S3_B1 = 17, S3_BREQ = 18, S3_BCAP = 19,
        S3_SIG = 20, S3_SIGC = 21, S3_CALC = 22, S3_W = 23, P_FINISH = 24;

    reg [5:0] st;

    // ---- header
    reg [5:0]  hi;
    reg [15:0] nc;
    reg [7:0]  fb;
    reg [31:0] cb [0:2];
    reg [31:0] bb [0:2];
    reg [31:0] hh [0:2];
    reg [31:0] ww [0:2];
    reg [31:0] sh [0:2];
    reg [31:0] cw [0:2];
    reg [31:0] hw [0:2];
    reg [31:0] total;
    reg [IW:0] kreg;

    // ---- loop state
    reg [1:0]  s;
    reg [15:0] y, x, c;
    reg [31:0] cbase;
    reg signed [15:0] best;
    reg [15:0] j;
    reg [1:0]  s2s;
    reg [15:0] cy, cx;
    reg [31:0] cellr;
    reg        in_stage3;
    reg [15:0] r;
    reg signed [7:0] lg;
    reg [IW-1:0] jj;
    reg [CW-1:0] cc;
    reg [1:0]  ps;
    reg [15:0] py, px;
    reg [31:0] bcaddr, cellb;
    reg [2:0]  ch;
    reg signed [15:0] bv0, bv1, bv2, bv3;
    reg [15:0] scr;
    reg signed [15:0] o0, o1, o2, o3;
    reg [2:0]  wf;

    // ---- byte select from the wide activation word
    wire [LOGL-1:0] lane = c[LOGL-1:0];
    wire signed [7:0] cv = act_rdata[8*lane +: 8];
    wire signed [15:0] cv16 = {{8{cv[7]}}, cv};

    // ---- sorters
    wire [IW:0] cnt1, cnt2;
    wire signed [7:0] rd1_score, rd2_score;
    wire [PW1-1:0] rd1_payload;
    wire [PW2-1:0] rd2_payload;
    wire [IW-1:0] rd1_idx = in_stage3 ? jj : j[IW-1:0];
    wire [IW-1:0] rd2_idx = r[IW-1:0];

    wire signed [15:0] bnew = (c == 0 || cv16 > best) ? cv16 : best;
    wire push1 = (st == S1_CAP) && (c == nc - 1);
    wire push2 = (st == S2_CAP);
    wire clr   = (st == P_INIT2);

    topk_sorter #(.K(K), .SW(8), .PW(PW1), .IW(IW)) u_s1 (
        .clk(clk), .clear(clr), .limit(kreg), .push(push1),
        .score(bnew[7:0]), .payload({s, y[YXW-1:0], x[YXW-1:0]}),
        .count(cnt1), .rd_idx(rd1_idx), .rd_score(rd1_score), .rd_payload(rd1_payload));

    topk_sorter #(.K(K), .SW(8), .PW(PW2), .IW(IW)) u_s2 (
        .clk(clk), .clear(clr), .limit(kreg), .push(push2),
        .score(cv), .payload({j[IW-1:0], c[CW-1:0]}),
        .count(cnt2), .rd_idx(rd2_idx), .rd_score(rd2_score), .rd_payload(rd2_payload));

    // ---- memory addressing
    assign prog_addr = {{(PRW-6){1'b0}}, hi};
    wire [31:0] ra = cbase + (c >> LOGL);
    assign act_raddr = ra[AAW-1:0];
    assign box_raddr = bcaddr[BAW-1:0];
    assign sig_raddr = {~lg[7], lg[6:0]};

    assign out_we    = (st == S3_W);
    wire [31:0] ow   = r * 6 + wf;
    assign out_waddr = ow[OUTAW-1:0];
    always @* begin
        case (wf)
            3'd0: out_wdata = {{16{o0[15]}}, o0};
            3'd1: out_wdata = {{16{o1[15]}}, o1};
            3'd2: out_wdata = {{16{o2[15]}}, o2};
            3'd3: out_wdata = {{16{o3[15]}}, o3};
            3'd4: out_wdata = {16'b0, scr};
            default: out_wdata = {{(32-CW){1'b0}}, cc};
        endcase
    end

    // ---- box decode
    wire signed [31:0] axf = $signed((({16'b0, px} << 1) + 32'd1) << (fb - 1));
    wire signed [31:0] ayf = $signed((({16'b0, py} << 1) + 32'd1) << (fb - 1));
    wire signed [31:0] x1f = axf - bv0;
    wire signed [31:0] y1f = ayf - bv1;
    wire signed [31:0] x2f = axf + bv2;
    wire signed [31:0] y2f = ayf + bv3;
    wire [31:0] shv = sh[ps];

    function signed [15:0] topix(input signed [31:0] v, input [2:0] shamt, input [7:0] f);
        reg signed [31:0] t;
        begin
            t = v <<< shamt;
            t = t >>> (f - 2);
            topix = (t > 32'sd32767) ? 16'sd32767 : (t < -32'sd32768) ? -16'sh8000 : t[15:0];
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= P_IDLE; done <= 1'b0; in_stage3 <= 1'b0; n_rows <= 0;
        end else begin
            done <= 1'b0;
            case (st)
            P_IDLE: if (start) begin hi <= 6'd1; in_stage3 <= 1'b0; st <= P_HREQ; end

            P_HREQ: st <= P_HCAP;
            P_HCAP: begin
                case (hi)
                    6'd1:  nc <= prog_rdata[15:0];
                    6'd2:  fb <= prog_rdata[7:0];
                    6'd4:  cb[0] <= prog_rdata;  6'd5:  bb[0] <= prog_rdata;
                    6'd6:  hh[0] <= prog_rdata;  6'd7:  ww[0] <= prog_rdata;
                    6'd8:  sh[0] <= prog_rdata;  6'd9:  cw[0] <= prog_rdata;
                    6'd10: cb[1] <= prog_rdata;  6'd11: bb[1] <= prog_rdata;
                    6'd12: hh[1] <= prog_rdata;  6'd13: ww[1] <= prog_rdata;
                    6'd14: sh[1] <= prog_rdata;  6'd15: cw[1] <= prog_rdata;
                    6'd16: cb[2] <= prog_rdata;  6'd17: bb[2] <= prog_rdata;
                    6'd18: hh[2] <= prog_rdata;  6'd19: ww[2] <= prog_rdata;
                    6'd20: sh[2] <= prog_rdata;  6'd21: cw[2] <= prog_rdata;
                    default: ;
                endcase
                if (hi == 6'd21) st <= P_INIT0;
                else begin hi <= hi + 1'b1; st <= P_HREQ; end
            end
            P_INIT0: begin
                hw[0] <= hh[0] * ww[0]; hw[1] <= hh[1] * ww[1]; hw[2] <= hh[2] * ww[2];
                st <= P_INIT1;
            end
            P_INIT1: begin
                total <= hw[0] + hw[1] + hw[2];
                kreg  <= ((hw[0] + hw[1] + hw[2]) < K) ? (hw[0] + hw[1] + hw[2]) : K;
                st <= P_INIT2;
            end
            P_INIT2: begin
                s <= 0; y <= 0; x <= 0; cbase <= cb[0];
                st <= S1_SETUP;
            end

            // ---------------- stage 1
            S1_SETUP: begin c <= 0; st <= S1_REQ; end
            S1_REQ:   st <= S1_CAP;
            S1_CAP: begin
                best <= bnew;
                if (c == nc - 1) begin
                    if (x == ww[s] - 1) begin
                        x <= 0;
                        if (y == hh[s] - 1) begin
                            y <= 0;
                            if (s == 2) begin j <= 0; st <= S2_J; end
                            else begin s <= s + 1'b1; cbase <= cb[s+1]; st <= S1_SETUP; end
                        end else begin y <= y + 1'b1; cbase <= cbase + cw[s]; st <= S1_SETUP; end
                    end else begin x <= x + 1'b1; cbase <= cbase + cw[s]; st <= S1_SETUP; end
                end else begin
                    c <= c + 1'b1; st <= S1_REQ;
                end
            end

            // ---------------- stage 2
            S2_J: begin
                s2s <= rd1_payload[2*YXW+1:2*YXW];
                cy  <= rd1_payload[2*YXW-1:YXW];
                cx  <= rd1_payload[YXW-1:0];
                st  <= S2_J2;
            end
            S2_J2: begin cellr <= cy * ww[s2s] + cx; st <= S2_J3; end
            S2_J3: begin cbase <= cb[s2s] + cellr * cw[s2s]; c <= 0; st <= S2_REQ; end
            S2_REQ: st <= S2_CAP;
            S2_CAP: begin
                if (c == nc - 1) begin
                    if (j == cnt1 - 1) begin r <= 0; in_stage3 <= 1'b1; st <= S3_R; end
                    else begin j <= j + 1'b1; st <= S2_J; end
                end else begin c <= c + 1'b1; st <= S2_REQ; end
            end

            // ---------------- stage 3
            S3_R: begin
                lg <= rd2_score;
                jj <= rd2_payload[IW+CW-1:CW];
                cc <= rd2_payload[CW-1:0];
                st <= S3_R2;
            end
            S3_R2: begin
                ps <= rd1_payload[2*YXW+1:2*YXW];
                py <= rd1_payload[2*YXW-1:YXW];
                px <= rd1_payload[YXW-1:0];
                st <= S3_B0;
            end
            S3_B0: begin cellb <= py * ww[ps] + px; st <= S3_B1; end
            S3_B1: begin bcaddr <= bb[ps] + {cellb[29:0], 2'b00}; ch <= 0; st <= S3_BREQ; end
            S3_BREQ: st <= S3_BCAP;
            S3_BCAP: begin
                case (ch[1:0])
                    2'd0: bv0 <= box_rdata;
                    2'd1: bv1 <= box_rdata;
                    2'd2: bv2 <= box_rdata;
                    default: bv3 <= box_rdata;
                endcase
                bcaddr <= bcaddr + 1'b1;
                if (ch == 3) st <= S3_SIG;
                else begin ch <= ch + 1'b1; st <= S3_BREQ; end
            end
            S3_SIG:  st <= S3_SIGC;
            S3_SIGC: begin scr <= sig_rdata[15:0]; st <= S3_CALC; end
            S3_CALC: begin
                o0 <= topix(x1f, shv[2:0], fb);
                o1 <= topix(y1f, shv[2:0], fb);
                o2 <= topix(x2f, shv[2:0], fb);
                o3 <= topix(y2f, shv[2:0], fb);
                wf <= 0; st <= S3_W;
            end
            S3_W: begin
                if (wf == 3'd5) begin
                    if (r == cnt2 - 1) begin n_rows <= cnt2; st <= P_FINISH; end
                    else begin r <= r + 1'b1; st <= S3_R; end
                end else wf <= wf + 1'b1;
            end

            P_FINISH: begin done <= 1'b1; in_stage3 <= 1'b0; st <= P_IDLE; end
            default:  st <= P_IDLE;
            endcase
        end
    end
endmodule
