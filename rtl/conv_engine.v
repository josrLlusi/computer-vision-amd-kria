`timescale 1ns/1ps
// -----------------------------------------------------------------------------
// conv_engine (Kria / v2): pipelined layer processor, LANES int8 MACs per cycle.
//
//  dense layers (k=1/3): LANES *input* channels in parallel. Per output channel and pixel the
//      taps (ky,kx,chunk) stream one per cycle into a 16-wide multiply + adder tree; one output byte
//      per (pixel,cout) leaves the single requantiser. Fully pipelined, no per-output bubbles.
//  depthwise layers: LANES channels in parallel (one accumulator per lane); results drained
//      through the same requantiser, LANES per (pixel,chunk).
//  Pipeline (dense): issue t | read+mult t+1 | partial sums t+2 | total t+3 | accumulate t+4 |
//      +bias t+5 | *mult t+6 | round/shift/clamp t+7 | LUT read t+8 | byte assemble + write t+9
//
// activation RAM word = LANES int8, layout HWC (see py/golden.py).  box (sat16) results go to a
// separate 16-bit RAM.  Parameters per output channel (64-bit): bias[31:0] mult[47:32] shift[53:48].
//
// prog ROM descriptor (8 words / layer, after a 32-word header):
//   in_base out_base w_base par_base lut_base {H,W} {Cw_in,Cout} flags[0]=dw [1]=act [2]=sat16 [6:3]=k
// -----------------------------------------------------------------------------
module conv_engine #(
    parameter LANES = 16,
    parameter LOGL  = 4,
    parameter AAW = 16, WAW = 16, PAW = 14, LAW = 12, PRW = 10, BAW = 14
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 start,
    output reg                  done,

    output wire [PRW-1:0]       prog_addr,
    input  wire [31:0]          prog_rdata,

    output wire [AAW-1:0]       act_raddr,
    input  wire [8*LANES-1:0]   act_rdata,
    output wire                 act_we,
    output wire [AAW-1:0]       act_waddr,
    output wire [8*LANES-1:0]   act_wdata,

    output wire                 box_we,
    output wire [BAW-1:0]       box_waddr,
    output wire [15:0]          box_wdata,

    output wire [WAW-1:0]       w_raddr,
    input  wire [8*LANES-1:0]   w_rdata,

    output wire [PAW-1:0]       p_raddr,
    input  wire [63:0]          p_rdata,

    output wire [LAW-1:0]       l_raddr,
    input  wire [31:0]          l_rdata
);
    localparam HDR_WORDS = 32;
    localparam NG = LANES / 4;

    localparam [4:0]
        S_IDLE = 0, S_H_REQ = 1, S_H_CAP = 2, S_LD_REQ = 3, S_LD_CAP = 4,
        S_LI0 = 5, S_LI1 = 6, S_LI2 = 7, S_RUN_D = 8, S_RUN_W = 9,
        S_W1 = 10, S_W2 = 11, S_DRAIN = 12, S_WAIT = 13, S_NEXT = 14, S_FINISH = 15;

    reg [4:0]  st;
    reg [15:0] nl, lay;
    reg [3:0]  ld_i;

    // ---- layer descriptor
    reg [31:0] f_in, f_out, f_w, f_par, f_lut;
    reg [15:0] H, W, cwi, cout;
    reg        dw, act_en, sat16;
    reg [3:0]  k, pad;
    // ---- derived
    reg [31:0] kk, npix, cwo, taps, rowstride, cout_eff, tot_out;
    // ---- issue state
    reg [31:0] tw, tb, tpx, rowstart, wb, tapi;
    reg [15:0] y, x, co, ch, chunk;
    reg [3:0]  ky, kx;
    reg [7:0]  dl;
    // ---- output-side counters (declared early: used by the FSM)
    reg  [8*LANES-1:0] obuf;
    reg  [31:0] oco, opix_w, bidx, outs_done;

    // ---------------------------------------------------------------- program ROM
    wire [31:0] pa = (st == S_H_REQ || st == S_H_CAP) ? 32'd0 : (HDR_WORDS + {lay, 3'b000} + ld_i);
    assign prog_addr = pa[PRW-1:0];

    // ---------------------------------------------------------------- tap addressing
    wire signed [31:0] iy = $signed({16'b0, y}) + $signed({28'b0, ky}) - $signed({28'b0, pad});
    wire signed [31:0] ix = $signed({16'b0, x}) + $signed({28'b0, kx}) - $signed({28'b0, pad});
    wire inb = (iy >= 0) && (iy < $signed({16'b0, H})) && (ix >= 0) && (ix < $signed({16'b0, W}));
    wire [31:0] a_adr = f_in + tw;
    assign act_raddr = a_adr[AAW-1:0];
    wire [31:0] w_adr = wb + tapi;
    assign w_raddr = w_adr[WAW-1:0];
    wire [31:0] p_adr = (st == S_DRAIN) ? (f_par + {chunk, {LOGL{1'b0}}} + dl) : (f_par + co);
    assign p_raddr = p_adr[PAW-1:0];

    wire iss      = (st == S_RUN_D) || (st == S_RUN_W);
    wire iss_last = (tapi == taps - 1);
    wire iss_first= (tapi == 0);

    // ---------------------------------------------------------------- issue FSM
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 1'b0; nl <= 0; lay <= 0; ld_i <= 0;
        end else begin
            done <= 1'b0;
            case (st)
            S_IDLE: if (start) begin lay <= 0; st <= S_H_REQ; end
            S_H_REQ: st <= S_H_CAP;
            S_H_CAP: begin
                nl <= prog_rdata[15:0]; ld_i <= 0;
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
                    4'd6: begin cwi <= prog_rdata[31:16]; cout <= prog_rdata[15:0]; end
                    4'd7: begin
                        dw <= prog_rdata[0]; act_en <= prog_rdata[1]; sat16 <= prog_rdata[2];
                        k <= prog_rdata[6:3]; pad <= {1'b0, prog_rdata[6:4]};
                    end
                    default: ;
                endcase
                if (ld_i == 4'd7) st <= S_LI0;
                else begin ld_i <= ld_i + 1'b1; st <= S_LD_REQ; end
            end
            S_LI0: begin
                kk   <= k * k;
                npix <= H * W;
                cwo  <= (cout + LANES - 1) >> LOGL;
                st <= S_LI1;
            end
            S_LI1: begin
                taps      <= dw ? kk : kk * cwi;
                rowstride <= W * cwi;
                cout_eff  <= dw ? (cwo << LOGL) : cout;
                st <= S_LI2;
            end
            S_LI2: begin
                tot_out <= npix * cout_eff;
                tpx <= 32'd0 - pad * (rowstride + cwi);
                tb  <= 32'd0 - pad * (rowstride + cwi);
                tw  <= 32'd0 - pad * (rowstride + cwi);
                rowstart <= 32'd0 - pad * (rowstride + cwi);
                wb <= f_w;
                y <= 0; x <= 0; co <= 0; ch <= 0; chunk <= 0; ky <= 0; kx <= 0; tapi <= 0;
                st <= dw ? S_RUN_W : S_RUN_D;
            end

            S_RUN_D: begin
                tapi <= tapi + 1'b1;
                if (!iss_last) begin
                    if (ch != cwi - 1) begin ch <= ch + 1'b1; tw <= tw + 1'b1; end
                    else begin
                        ch <= 0;
                        if (kx != k - 1) begin kx <= kx + 1'b1; tw <= tw + 1'b1; end
                        else begin
                            kx <= 0; ky <= ky + 1'b1;
                            tw <= rowstart + rowstride; rowstart <= rowstart + rowstride;
                        end
                    end
                end else begin
                    tapi <= 0; ch <= 0; kx <= 0; ky <= 0; tw <= tb; rowstart <= tb;
                    wb <= wb + taps;
                    if (co == cout - 1) begin
                        co <= 0; wb <= f_w;
                        tb <= tb + cwi; tw <= tb + cwi; rowstart <= tb + cwi;
                        if (x == W - 1) begin
                            x <= 0;
                            if (y == H - 1) begin y <= 0; st <= S_WAIT; end
                            else y <= y + 1'b1;
                        end else x <= x + 1'b1;
                    end else co <= co + 1'b1;
                end
            end

            S_RUN_W: begin
                tapi <= tapi + 1'b1;
                if (!iss_last) begin
                    if (kx != k - 1) begin kx <= kx + 1'b1; tw <= tw + cwi; end
                    else begin
                        kx <= 0; ky <= ky + 1'b1;
                        tw <= rowstart + rowstride; rowstart <= rowstart + rowstride;
                    end
                end else begin
                    tapi <= 0; kx <= 0; ky <= 0; st <= S_W1;
                end
            end
            S_W1: st <= S_W2;
            S_W2: begin dl <= 0; st <= S_DRAIN; end
            S_DRAIN: begin
                dl <= dl + 1'b1;
                if (dl == LANES - 1) begin
                    if (chunk != cwi - 1) begin
                        chunk <= chunk + 1'b1; tb <= tb + 1'b1; tw <= tb + 1'b1; rowstart <= tb + 1'b1;
                        wb <= wb + kk; st <= S_RUN_W;
                    end else begin
                        chunk <= 0; wb <= f_w;
                        tpx <= tpx + cwi; tb <= tpx + cwi; tw <= tpx + cwi; rowstart <= tpx + cwi;
                        if (x == W - 1) begin
                            x <= 0;
                            if (y == H - 1) begin y <= 0; st <= S_WAIT; end
                            else begin y <= y + 1'b1; st <= S_RUN_W; end
                        end else begin x <= x + 1'b1; st <= S_RUN_W; end
                    end
                end
            end

            S_WAIT: if (outs_done == tot_out) st <= S_NEXT;
            S_NEXT: begin
                if (lay + 1 == nl) st <= S_FINISH;
                else begin lay <= lay + 1'b1; ld_i <= 0; st <= S_LD_REQ; end
            end
            S_FINISH: begin done <= 1'b1; st <= S_IDLE; end
            default: st <= S_IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------- datapath pipeline
    reg v1, inb1, f1, l1;                         // tags aligned with RAM read data
    always @(posedge clk) begin v1 <= iss; inb1 <= inb; f1 <= iss_first; l1 <= iss_last; end

    // stage 1: LANES multipliers; depthwise accumulators live in the same generate
    wire [LANES*16-1:0]   pr_flat;
    wire [LANES*32-1:0]   accl_flat;
    reg v2, f2, l2;
    always @(posedge clk) begin v2 <= v1; f2 <= f1; l2 <= l1; end

    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl = gl + 1) begin : G_MAC
            wire signed [7:0] a  = inb1 ? act_rdata[8*gl +: 8] : 8'sd0;
            wire signed [7:0] wv = w_rdata[8*gl +: 8];
            reg  signed [15:0] p;
            reg  signed [31:0] accl;
            always @(posedge clk) p <= a * wv;
            always @(posedge clk) if (v2 && dw) accl <= f2 ? {{16{p[15]}}, p} : accl + {{16{p[15]}}, p};
            assign pr_flat[16*gl +: 16]   = p;
            assign accl_flat[32*gl +: 32] = accl;
        end
    endgenerate

    // stage 2/3: adder tree (groups of 4, then sum of groups)
    reg [NG*20-1:0] part;
    reg v3, f3, l3;
    always @(posedge clk) begin v3 <= v2; f3 <= f2; l3 <= l2; end
    genvar gg;
    generate
        for (gg = 0; gg < NG; gg = gg + 1) begin : G_PART
            wire signed [15:0] p0 = pr_flat[16*(4*gg+0) +: 16];
            wire signed [15:0] p1 = pr_flat[16*(4*gg+1) +: 16];
            wire signed [15:0] p2 = pr_flat[16*(4*gg+2) +: 16];
            wire signed [15:0] p3 = pr_flat[16*(4*gg+3) +: 16];
            always @(posedge clk) part[20*gg +: 20] <= p0 + p1 + p2 + p3;
        end
    endgenerate
    reg signed [31:0] total;
    reg v4, f4, l4;
    integer gi;
    reg signed [31:0] tsum;
    always @(posedge clk) begin
        tsum = 0;
        for (gi = 0; gi < NG; gi = gi + 1) tsum = tsum + $signed(part[20*gi +: 20]);
        total <= tsum;
        v4 <= v3; f4 <= f3; l4 <= l3;
    end
    // stage 4: accumulate over taps
    reg signed [31:0] acc;
    reg v5, l5;
    always @(posedge clk) begin
        if (v4 && !dw) acc <= f4 ? total : acc + total;
        v5 <= v4; l5 <= l4;
    end
    // params travel with the data: RAM output at t+1, so 4 delay registers reach t+5
    reg [63:0] par_d1, par_d2, par_d3, par_d4;
    always @(posedge clk) begin par_d1 <= p_rdata; par_d2 <= par_d1; par_d3 <= par_d2; par_d4 <= par_d3; end

    // depthwise drain: one lane per cycle
    reg signed [31:0] acc_r;
    reg               rv;
    always @(posedge clk) begin
        acc_r <= accl_flat[32*dl[LOGL-1:0] +: 32];
        rv    <= (st == S_DRAIN);
    end

    // requantiser input mux
    wire        rq_v   = dw ? rv : (v5 && l5);
    wire signed [31:0] rq_acc = dw ? acc_r : acc;
    wire [63:0] rq_par = dw ? p_rdata : par_d4;

    // R1: acc + bias      R2: * mult      R3: round, shift, clamp
    reg signed [33:0] sum1;  reg [15:0] mult1; reg [5:0] sh1; reg rv1;
    always @(posedge clk) begin
        sum1  <= $signed({{2{rq_acc[31]}}, rq_acc}) + $signed({{2{rq_par[31]}}, rq_par[31:0]});
        mult1 <= rq_par[47:32]; sh1 <= rq_par[53:48]; rv1 <= rq_v;
    end
    reg signed [50:0] prod2; reg [5:0] sh2; reg rv2;
    always @(posedge clk) begin
        prod2 <= sum1 * $signed({1'b0, mult1});
        sh2 <= sh1; rv2 <= rv1;
    end
    wire signed [50:0] rnd  = prod2 + (51'sd1 <<< (sh2 - 1));
    wire signed [50:0] shr  = rnd >>> sh2;
    wire signed [15:0] q8   = (shr > 51'sd127)   ? 16'sd127   : (shr < -51'sd128)   ? -16'sd128 : shr[15:0];
    wire signed [15:0] q16  = (shr > 51'sd32767) ? 16'sd32767 : (shr < -51'sd32768) ? -16'sh8000 : shr[15:0];
    reg signed [15:0] q3; reg rv3;
    always @(posedge clk) begin
        q3  <= sat16 ? q16 : q8;
        rv3 <= rv2;
    end
    // R4: LUT read
    wire [31:0] l_adr = f_lut + {24'b0, ~q3[7], q3[6:0]};
    assign l_raddr = l_adr[LAW-1:0];
    reg signed [15:0] q4; reg rv4;
    always @(posedge clk) begin q4 <= q3; rv4 <= rv3; end

    // ---------------------------------------------------------------- output assembly
    wire [7:0] obyte = act_en ? l_rdata[7:0] : q4[7:0];
    wire [LOGL-1:0] lane = oco[LOGL-1:0];
    wire last_in_word = (lane == LANES - 1) || (oco == cout_eff - 1);

    reg [8*LANES-1:0] wd;
    integer wl;
    always @* begin
        for (wl = 0; wl < LANES; wl = wl + 1)
            wd[8*wl +: 8] = (wl < lane) ? obuf[8*wl +: 8] : (wl == lane) ? obyte : 8'h00;
    end
    wire [31:0] aw_adr = f_out + opix_w + (oco >> LOGL);
    assign act_we    = rv4 && !sat16 && last_in_word;
    assign act_waddr = aw_adr[AAW-1:0];
    assign act_wdata = wd;
    wire [31:0] bw_adr = f_out + bidx;
    assign box_we    = rv4 && sat16;
    assign box_waddr = bw_adr[BAW-1:0];
    assign box_wdata = q4;

    always @(posedge clk) begin
        if (st == S_LI2) begin
            oco <= 0; opix_w <= 0; bidx <= 0; outs_done <= 0;
        end else if (rv4) begin
            outs_done <= outs_done + 1'b1;
            if (sat16) bidx <= bidx + 1'b1;
            else obuf[8*lane +: 8] <= obyte;
            if (oco == cout_eff - 1) begin oco <= 0; opix_w <= opix_w + cwo; end
            else oco <= oco + 1'b1;
        end
    end
endmodule
