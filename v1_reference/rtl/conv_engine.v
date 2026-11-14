`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// conv_engine : sequential layer processor for the YOLO26 head conv chains.
//
// Reads a layer program (prog ROM), then for every layer computes
//     out[co][y][x] = act( requant( bias[co] + sum_taps in*w ) )
// Supported layer kinds (all stride 1, zero padding k/2):
//     dense  k x k   (k = 1 or 3)           flags.dw = 0
//     depthwise k x k                       flags.dw = 1  (cin == cout)
// 1 MAC per cycle (taps issued every cycle through a 1-deep read pipeline).
//
// prog ROM layout (32-bit words):
//   [0] NLAYERS   [1] NC   [2] FB   [4..18] scale table   (header = 32 words)
//   per layer, 8 words starting at 32 + 8*layer:
//     0 in_base   1 out_base   2 w_base   3 par_base   4 lut_base
//     5 {H,W}     6 {Cin,Cout} 7 flags: [0]dw [1]act [2]sat16 [6:3]k [15:8]shift
// activation memory is CHW: addr = base + (c*H + y)*W + x, 16-bit words.
// weights: dense [co][ci][ky][kx], depthwise [c][ky][kx]  (int8)
// params : word 2*co = bias (int32), 2*co+1 = requant multiplier (>=0, < 2^31)
// -----------------------------------------------------------------------------
module conv_engine #(
    parameter AAW = 20,
    parameter WAW = 20,
    parameter PAW = 20,
    parameter LAW = 16,
    parameter PRW = 12
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    output reg                  done,

    output wire [PRW-1:0]       prog_addr,
    input  wire [31:0]          prog_rdata,

    output wire [AAW-1:0]       act_raddr,
    input  wire signed [15:0]   act_rdata,
    output wire                 act_we,
    output wire [AAW-1:0]       act_waddr,
    output wire signed [15:0]   act_wdata,

    output wire [WAW-1:0]       w_raddr,
    input  wire signed [7:0]    w_rdata,

    output wire [PAW-1:0]       p_raddr,
    input  wire [31:0]          p_rdata,

    output wire [LAW-1:0]       l_raddr,
    input  wire signed [7:0]    l_rdata
);
    localparam HDR_WORDS = 32;

    localparam [4:0]
        S_IDLE = 0,  S_H_REQ = 1, S_H_CAP = 2, S_LD_REQ = 3, S_LD_CAP = 4,
        S_LI0  = 5,  S_LI1   = 6, S_CO0   = 7, S_P0 = 8,     S_P1 = 9,   S_P2 = 10,
        S_OUT_INIT = 11, S_TAP = 12, S_FLUSH = 13, S_RQ = 14,
        S_L0 = 15,   S_L1 = 16,   S_WR = 17,  S_FINISH = 18;

    reg [4:0]  st;
    reg [15:0] nl, lay;
    reg [3:0]  ld_i;

    // layer descriptor
    reg [31:0] f_in, f_out, f_w, f_par, f_lut;
    reg [15:0] H, W, cin, cout;
    reg        dw, act_en, sat16;
    reg [3:0]  k, pad;
    reg [7:0]  shift;

    // derived per layer
    reg [31:0] kk, hw, taps;
    // loop state
    reg [31:0] wcob;                    // weight base of current output channel
    reg [15:0] co, y, x, ci;
    reg [3:0]  ky, kx;
    reg [31:0] tapi;
    // datapath
    reg signed [31:0] acc, bias, mult;
    reg signed [15:0] qreg, wval;
    reg               v1, inb1;

    // ---------------------------------------------------------------- program ROM address
    wire [31:0] pa = (st == S_H_REQ || st == S_H_CAP) ? 32'd0
                                                      : (HDR_WORDS + {lay, 3'b000} + ld_i);
    assign prog_addr = pa[PRW-1:0];

    // ---------------------------------------------------------------- tap addressing
    wire signed [31:0] s_y   = {16'b0, y};
    wire signed [31:0] s_x   = {16'b0, x};
    wire signed [31:0] s_ky  = {28'b0, ky};
    wire signed [31:0] s_kx  = {28'b0, kx};
    wire signed [31:0] s_pad = {28'b0, pad};
    wire signed [31:0] s_H   = {16'b0, H};
    wire signed [31:0] s_W   = {16'b0, W};
    wire signed [31:0] iy    = s_y + s_ky - s_pad;
    wire signed [31:0] ix    = s_x + s_kx - s_pad;
    wire               inb   = (iy >= 0) && (iy < s_H) && (ix >= 0) && (ix < s_W);
    wire        [31:0] c_idx = dw ? {16'b0, co} : {16'b0, ci};
    wire        [31:0] a_adr = f_in + c_idx * hw + iy * s_W + ix;     // modulo-2^32, valid when inb
    assign act_raddr = inb ? a_adr[AAW-1:0] : {AAW{1'b0}};

    wire [31:0] w_adr = wcob + tapi;
    assign w_raddr = w_adr[WAW-1:0];

    wire is_last = (kx == k - 1) && (ky == k - 1) && (dw || (ci == cin - 1));

    // ---------------------------------------------------------------- MAC
    wire signed [31:0] prod = act_rdata * w_rdata;
    always @(posedge clk) begin
        if (st == S_OUT_INIT)       acc <= 32'sd0;
        else if (v1 && inb1)        acc <= acc + prod;
    end

    // ---------------------------------------------------------------- requantise
    wire signed [63:0] acc64  = {{32{acc[31]}},  acc};
    wire signed [63:0] bias64 = {{32{bias[31]}}, bias};
    wire signed [63:0] mult64 = {{32{mult[31]}}, mult};
    wire signed [63:0] sum64  = acc64 + bias64;
    wire signed [63:0] prod64 = sum64 * mult64;
    wire signed [63:0] rnd64  = prod64 + (64'sd1 <<< (shift - 1));
    wire signed [63:0] shr64  = rnd64 >>> shift;
    wire signed [15:0] q8     = (shr64 >  64'sd127)   ? 16'sd127   :
                                (shr64 < -64'sd128)   ? -16'sd128  : shr64[15:0];
    wire signed [15:0] q16    = (shr64 >  64'sd32767) ? 16'sd32767 :
                                (shr64 < -64'sd32768) ? -16'sh8000 : shr64[15:0];
    wire signed [15:0] qsel   = sat16 ? q16 : q8;

    // ---------------------------------------------------------------- params / lut / write
    wire [31:0] p_adr = f_par + {co, 1'b0} + ((st == S_P1) ? 32'd1 : 32'd0);
    assign p_raddr = p_adr[PAW-1:0];

    wire [31:0] l_adr = f_lut + {24'b0, ~qreg[7], qreg[6:0]};         // q + 128
    assign l_raddr = l_adr[LAW-1:0];

    wire [31:0] o_adr = f_out + co * hw + y * {16'b0, W} + x;
    assign act_we    = (st == S_WR);
    assign act_waddr = o_adr[AAW-1:0];
    assign act_wdata = wval;

    // ---------------------------------------------------------------- FSM
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 1'b0; v1 <= 1'b0; inb1 <= 1'b0;
            nl <= 0; lay <= 0; ld_i <= 0;
        end else begin
            done <= 1'b0;
            case (st)
            S_IDLE: if (start) begin lay <= 0; st <= S_H_REQ; end

            S_H_REQ: st <= S_H_CAP;
            S_H_CAP: begin
                nl <= prog_rdata[15:0];
                ld_i <= 0;
                st <= (prog_rdata[15:0] == 0) ? S_FINISH : S_LD_REQ;
            end

            S_LD_REQ: st <= S_LD_CAP;
            S_LD_CAP: begin
                case (ld_i)
                    4'd0: f_in  <= prog_rdata;
                    4'd1: f_out <= prog_rdata;
                    4'd2: f_w   <= prog_rdata;
                    4'd3: f_par <= prog_rdata;
                    4'd4: f_lut <= prog_rdata;
                    4'd5: begin H <= prog_rdata[31:16];   W    <= prog_rdata[15:0]; end
                    4'd6: begin cin <= prog_rdata[31:16]; cout <= prog_rdata[15:0]; end
                    4'd7: begin
                        dw <= prog_rdata[0]; act_en <= prog_rdata[1]; sat16 <= prog_rdata[2];
                        k <= prog_rdata[6:3]; pad <= {1'b0, prog_rdata[6:4]};   // pad = k>>1
                        shift <= prog_rdata[15:8];
                    end
                    default: ;
                endcase
                if (ld_i == 4'd7) st <= S_LI0;
                else begin ld_i <= ld_i + 1'b1; st <= S_LD_REQ; end
            end

            S_LI0: begin kk <= k * k; hw <= H * W; st <= S_LI1; end
            S_LI1: begin
                taps <= dw ? kk : cin * kk;
                co <= 0; y <= 0; x <= 0;
                st <= S_CO0;
            end

            S_CO0: begin wcob <= f_w + co * taps; st <= S_P0; end
            S_P0:  st <= S_P1;
            S_P1:  begin bias <= p_rdata; st <= S_P2; end
            S_P2:  begin mult <= p_rdata; st <= S_OUT_INIT; end

            S_OUT_INIT: begin
                ci <= 0; ky <= 0; kx <= 0; tapi <= 0;
                st <= S_TAP;
            end

            S_TAP: begin
                v1   <= 1'b1;
                inb1 <= inb;
                tapi <= tapi + 1'b1;
                if (kx == k - 1) begin
                    kx <= 0;
                    if (ky == k - 1) begin ky <= 0; ci <= ci + 1'b1; end
                    else ky <= ky + 1'b1;
                end else kx <= kx + 1'b1;
                if (is_last) st <= S_FLUSH;
            end

            S_FLUSH: begin v1 <= 1'b0; st <= S_RQ; end

            S_RQ: begin
                qreg <= qsel;
                if (act_en) st <= S_L0;
                else begin wval <= qsel; st <= S_WR; end
            end
            S_L0: st <= S_L1;
            S_L1: begin wval <= {{8{l_rdata[7]}}, l_rdata}; st <= S_WR; end

            S_WR: begin
                if (x == W - 1) begin
                    x <= 0;
                    if (y == H - 1) begin
                        y <= 0;
                        if (co == cout - 1) begin
                            if (lay + 1 == nl) st <= S_FINISH;
                            else begin lay <= lay + 1'b1; ld_i <= 0; st <= S_LD_REQ; end
                        end else begin co <= co + 1'b1; st <= S_CO0; end
                    end else begin y <= y + 1'b1; st <= S_OUT_INIT; end
                end else begin x <= x + 1'b1; st <= S_OUT_INIT; end
            end

            S_FINISH: begin done <= 1'b1; st <= S_IDLE; end
            default:  st <= S_IDLE;
            endcase
        end
    end
endmodule
