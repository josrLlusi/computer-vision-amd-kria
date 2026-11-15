`timescale 1ns / 1ps

module yolo_top #(
    parameter int IMG          = 640,
    parameter int NECK_DW      = 16,  // Q8.8
    parameter int P3_CH        = 128,
    parameter int P4_CH        = 128,
    parameter int P5_CH        = 256,
    parameter     SIGMOID_FILE = "sigmoid_lut.hex",

    parameter logic [17:0] P3_ACT_BASE = 18'd0,
    parameter logic [17:0] P4_ACT_BASE = 18'd204800,
    parameter logic [17:0] P5_ACT_BASE = 18'd256000,

    parameter int LANES  = 16,
    parameter int LOGL   = 4,
    parameter int AAW    = 16,
    parameter int WAW    = 14,
    parameter int PAW    = 11,
    parameter int LAW    = 13,
    parameter int PRW    = 8,
    parameter int BAW    = 14,
    parameter int K      = 100,
    parameter int IW     = 7,
    parameter int OUTAW  = 10,
    parameter int HAW    = 18,
    parameter int YXW    = 8,
    parameter int CW     = 7,
    parameter int D_PROG = 224,
    parameter int D_PAR  = 1516,
    parameter int D_WGT  = 9503,
    parameter int D_LUT  = 4864,
    parameter int D_ACT  = 37700,
    parameter int D_BOX  = 8400
)(
    input  logic                 clk,
    input  logic                 rst_n,

    input  logic [(3*32)-1:0]    pixel_in,
    input  logic                 valid_in,
    output logic                 ready_in,

    input  logic                 host_en,
    input  logic                 host_we,
    input  logic [2:0]           host_win,
    input  logic [HAW-1:0]       host_waddr,
    input  logic [31:0]          host_wdata,
    output logic [31:0]          host_rdata,

    output logic                 head_busy,
    output logic                 head_done,
    output logic [IW:0]          n_rows,
    output logic [31:0]          head_cycles
);

    localparam int P3_DIM = IMG / 8;
    localparam int P4_DIM = IMG / 16;
    localparam int P5_DIM = IMG / 32;

    function automatic logic signed [15:0] q16_16_to_q8_8(input logic signed [31:0] x);
        logic signed [31:0] shifted;
        begin
            shifted = x >>> 8;
            if (shifted > 32'sd32767)       q16_16_to_q8_8 = 16'sh7FFF;
            else if (shifted < -32'sd32768) q16_16_to_q8_8 = 16'sh8000;
            else                            q16_16_to_q8_8 = shifted[15:0];
        end
    endfunction

    function automatic logic signed [7:0] q8_8_to_int8(input logic signed [15:0] x);
        logic signed [15:0] shifted;
        begin
            shifted = x >>> 8;
            if (shifted > 16'sd127)       q8_8_to_int8 = 8'sh7F;
            else if (shifted < -16'sd128) q8_8_to_int8 = 8'sh80;
            else                          q8_8_to_int8 = shifted[7:0];
        end
    endfunction

    // =========================================================================
    // 1. Backbone L0..L5 (Jose, Q16.16)
    // =========================================================================
    logic [(P3_CH*32)-1:0] p3_bb_q16;
    logic                  p3_bb_valid, p3_bb_ready;

    logic [(P4_CH*32)-1:0] l5_bb_q16;
    logic                  l5_bb_valid, l5_bb_ready;

    backbone_l0_l5 #(
        .IMG(IMG),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_backbone_l0_l5 (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .ready_in(ready_in),
        .p3_data(p3_bb_q16),
        .p3_valid(p3_bb_valid),
        .p3_ready(p3_bb_ready),
        .l5_data(l5_bb_q16),
        .l5_valid(l5_bb_valid),
        .l5_ready(l5_bb_ready)
    );

    // =========================================================================
    // 2. Backbone Tail L6..L10 (Menghasilkan P4 & P5)
    // =========================================================================
    logic [(P4_CH*32)-1:0] p4_bb_q16;
    logic                  p4_bb_valid, p4_bb_ready;

    c3k2 #(
        .IMAGE_WIDTH(P4_DIM),
        .IMAGE_HEIGHT(P4_DIM),
        .C1(P4_CH),
        .C2(P4_CH),
        .C(P4_CH / 4),
        .CB(P4_CH / 8),
        .PREFIX("L6"),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_l6_c3k2 (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(l5_bb_q16),
        .valid_in(l5_bb_valid),
        .ready_in(l5_bb_ready),
        .pixel_out(p4_bb_q16),
        .valid_out(p4_bb_valid),
        .ready_out(p4_bb_ready)
    );

    logic p4_fifo_in_ready, l7_in_ready;
    logic p4_fire;
    assign p4_fire     = p4_bb_valid && p4_fifo_in_ready && l7_in_ready;
    assign p4_bb_ready = p4_fifo_in_ready && l7_in_ready;

    logic [(P5_CH*32)-1:0] l7_q16;
    logic                  l7_valid, l7_ready;

    conv_kxk #(
        .IMAGE_WIDTH(P4_DIM),
        .IMAGE_HEIGHT(P4_DIM),
        .IN_CHANNELS(P4_CH),
        .OUT_CHANNELS(P5_CH),
        .KERNEL(3),
        .STRIDE(2),
        .PAR_IC(4),
        .PAR_OC(4),
        .WEIGHT_FILE("L7.w.hex"),
        .BIAS_FILE("L7.b.hex"),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_l7_conv (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(p4_bb_q16),
        .valid_in(p4_fire),
        .ready_in(l7_in_ready),
        .pixel_out(l7_q16),
        .valid_out(l7_valid),
        .ready_out(l7_ready)
    );

    logic [(P5_CH*32)-1:0] l8_q16;
    logic                  l8_valid, l8_ready;

    c3k2 #(
        .IMAGE_WIDTH(P5_DIM),
        .IMAGE_HEIGHT(P5_DIM),
        .C1(P5_CH),
        .C2(P5_CH),
        .C(P5_CH / 4),
        .CB(P5_CH / 8),
        .PREFIX("L8"),
        .SIGMOID_FILE(SIGMOID_FILE)
    ) u_l8_c3k2 (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(l7_q16),
        .valid_in(l7_valid),
        .ready_in(l7_ready),
        .pixel_out(l8_q16),
        .valid_out(l8_valid),
        .ready_out(l8_ready)
    );

    // =========================================================================
    // 3. Konversi Q16.16 -> Q8.8 + Buffer Skip P3/P4/P5 + L9 SPPF + L10 C2PSA
    // =========================================================================
    logic [(P3_CH*NECK_DW)-1:0] p3_neck_data;
    logic [(P4_CH*NECK_DW)-1:0] p4_neck_data;
    logic [(P5_CH*NECK_DW)-1:0] l8_q8_data;

    genvar g3, g4, g5;
    generate
        for (g3 = 0; g3 < P3_CH; g3++) begin : GEN_P3_Q8
            assign p3_neck_data[g3*NECK_DW +: NECK_DW] = q16_16_to_q8_8(p3_bb_q16[g3*32 +: 32]);
        end
        for (g4 = 0; g4 < P4_CH; g4++) begin : GEN_P4_Q8
            assign p4_neck_data[g4*NECK_DW +: NECK_DW] = q16_16_to_q8_8(p4_bb_q16[g4*32 +: 32]);
        end
        for (g5 = 0; g5 < P5_CH; g5++) begin : GEN_P5_Q8
            assign l8_q8_data[g5*NECK_DW +: NECK_DW] = q16_16_to_q8_8(l8_q16[g5*32 +: 32]);
        end
    endgenerate

    logic [(P3_CH*NECK_DW)-1:0] p3_fifo_out_data;
    logic                       p3_fifo_out_valid;
    logic                       p3_fifo_out_ready;

    stream_fifo #(
        .WIDTH(P3_CH * NECK_DW),
        .DEPTH((P3_DIM * P3_DIM) + 8)
    ) u_fifo_p3_neck (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(p3_neck_data),
        .in_valid(p3_bb_valid),
        .in_ready(p3_bb_ready),
        .out_data(p3_fifo_out_data),
        .out_valid(p3_fifo_out_valid),
        .out_ready(p3_fifo_out_ready)
    );

    logic [(P4_CH*NECK_DW)-1:0] p4_fifo_out_data;
    logic                       p4_fifo_out_valid;
    logic                       p4_fifo_out_ready;

    stream_fifo #(
        .WIDTH(P4_CH * NECK_DW),
        .DEPTH((P4_DIM * P4_DIM) + 8)
    ) u_fifo_p4_neck (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(p4_neck_data),
        .in_valid(p4_fire),
        .in_ready(p4_fifo_in_ready),
        .out_data(p4_fifo_out_data),
        .out_valid(p4_fifo_out_valid),
        .out_ready(p4_fifo_out_ready)
    );

    // L9: SPPF (valid-only)
    logic [(P5_CH*NECK_DW)-1:0] l9_sppf_data;
    logic                       l9_sppf_valid;

    yolo_sppf #(
        .DATA_WIDTH(NECK_DW),
        .FRAC_BITS(8),
        .IN_CHANNELS(P5_CH),
        .OUT_CHANNELS(P5_CH),
        .POOL_SIZE(5),
        .IMG_WIDTH(P5_DIM),
        .IMG_HEIGHT(P5_DIM)
    ) u_l9_sppf (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(l8_valid),
        .in_data(l8_q8_data),
        .out_valid(l9_sppf_valid),
        .out_data(l9_sppf_data)
    );

    // L10: C2PSA (valid-only -> buffered by u_fifo_p5_neck)
    logic [(P5_CH*NECK_DW)-1:0] p5_raw_data;
    logic                       p5_raw_valid;
    logic                       p5_fifo_in_ready;

    yolo_c2psa #(
        .DATA_WIDTH(NECK_DW),
        .FRAC_BITS(8),
        .IN_CHANNELS(P5_CH),
        .OUT_CHANNELS(P5_CH),
        .NUM_PSA(2),
        .NUM_HEADS(2),
        .IMG_WIDTH(P5_DIM),
        .IMG_HEIGHT(P5_DIM)
    ) u_l10_c2psa (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(l9_sppf_valid),
        .in_data(l9_sppf_data),
        .out_valid(p5_raw_valid),
        .out_data(p5_raw_data)
    );
    assign l8_ready = p5_fifo_in_ready;

    logic [(P5_CH*NECK_DW)-1:0] p5_neck_data;
    logic                       p5_neck_valid;
    logic                       p5_neck_ready;

    stream_fifo #(
        .WIDTH(P5_CH * NECK_DW),
        .DEPTH((P5_DIM * P5_DIM) + 8)
    ) u_fifo_p5_neck (
        .clk(clk),
        .rst_n(rst_n),
        .in_data(p5_raw_data),
        .in_valid(p5_raw_valid),
        .in_ready(p5_fifo_in_ready),
        .out_data(p5_neck_data),
        .out_valid(p5_neck_valid),
        .out_ready(p5_neck_ready)
    );

    // =========================================================================
    // 4. Neck Fusion (Marco, Q8.8)
    // =========================================================================
    logic [(P3_CH*NECK_DW)-1:0] head_small_data;
    logic                       head_small_valid, head_small_ready;

    logic [(P4_CH*NECK_DW)-1:0] head_medium_data;
    logic                       head_medium_valid, head_medium_ready;

    logic [(P5_CH*NECK_DW)-1:0] head_large_data;
    logic                       head_large_valid, head_large_ready;

    yolo_neck_fusion #(
        .DATA_WIDTH(NECK_DW),
        .FRAC_BITS(8),
        .P3_CH(P3_CH),
        .P4_CH(P4_CH),
        .P5_CH(P5_CH),
        .IMG_WIDTH(P3_DIM),
        .IMG_HEIGHT(P3_DIM)
    ) u_neck_fusion (
        .clk(clk),
        .rst_n(rst_n),
        .p3_data_in(p3_fifo_out_data),
        .p3_valid_in(p3_fifo_out_valid),
        .p3_ready_out(p3_fifo_out_ready),
        .p4_data_in(p4_fifo_out_data),
        .p4_valid_in(p4_fifo_out_valid),
        .p4_ready_out(p4_fifo_out_ready),
        .p5_data_in(p5_neck_data),
        .p5_valid_in(p5_neck_valid),
        .p5_ready_out(p5_neck_ready),
        .head_small_data(head_small_data),
        .head_small_valid(head_small_valid),
        .head_small_ready(head_small_ready),
        .head_medium_data(head_medium_data),
        .head_medium_valid(head_medium_valid),
        .head_medium_ready(head_medium_ready),
        .head_large_data(head_large_data),
        .head_large_valid(head_large_valid),
        .head_large_ready(head_large_ready)
    );

    // =========================================================================
    // 5. Bridge Neck (Q8.8 Stream) -> Detect Head ACT RAM (INT8 32-bit Words)
    // =========================================================================
    localparam int P3_PIXELS = P3_DIM * P3_DIM;
    localparam int P4_PIXELS = P4_DIM * P4_DIM;
    localparam int P5_PIXELS = P5_DIM * P5_DIM;
    localparam int P3_WORDS  = P3_CH / 4;
    localparam int P4_WORDS  = P4_CH / 4;
    localparam int P5_WORDS  = P5_CH / 4;

    localparam logic [2:0] B_IDLE      = 3'd0;
    localparam logic [2:0] B_WR_SMALL  = 3'd1;
    localparam logic [2:0] B_WR_MEDIUM = 3'd2;
    localparam logic [2:0] B_WR_LARGE  = 3'd3;
    localparam logic [2:0] B_TRIGGER   = 3'd4;

    logic [2:0]                  b_state;
    logic [15:0]                 pix_cnt;
    logic [7:0]                  word_idx;
    logic [(P5_CH*8)-1:0]        quant_buf;
    logic                        bridge_active;
    logic                        bridge_we;
    logic [HAW-1:0]              bridge_waddr;
    logic [31:0]                 bridge_wdata;
    logic                        head_start_pulse;

    assign bridge_active = (b_state != B_IDLE);

    assign head_small_ready  = (b_state == B_IDLE) && !head_busy;
    assign head_medium_ready = (b_state == B_IDLE) && !head_busy && !head_small_valid;
    assign head_large_ready  = (b_state == B_IDLE) && !head_busy && !head_small_valid && !head_medium_valid;

    integer qi;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            b_state          <= B_IDLE;
            pix_cnt          <= '0;
            word_idx         <= '0;
            quant_buf        <= '0;
            bridge_we        <= 1'b0;
            bridge_waddr     <= '0;
            bridge_wdata     <= 32'd0;
            head_start_pulse <= 1'b0;
        end else begin
            bridge_we        <= 1'b0;
            head_start_pulse <= 1'b0;

            case (b_state)
                B_IDLE: begin
                    word_idx <= '0;
                    if (!head_busy) begin
                        if (head_small_valid) begin
                            for (qi = 0; qi < P3_CH; qi = qi + 1)
                                quant_buf[qi*8 +: 8] <= q8_8_to_int8(head_small_data[qi*NECK_DW +: NECK_DW]);
                            b_state <= B_WR_SMALL;
                        end else if (head_medium_valid) begin
                            for (qi = 0; qi < P4_CH; qi = qi + 1)
                                quant_buf[qi*8 +: 8] <= q8_8_to_int8(head_medium_data[qi*NECK_DW +: NECK_DW]);
                            b_state <= B_WR_MEDIUM;
                        end else if (head_large_valid) begin
                            for (qi = 0; qi < P5_CH; qi = qi + 1)
                                quant_buf[qi*8 +: 8] <= q8_8_to_int8(head_large_data[qi*NECK_DW +: NECK_DW]);
                            b_state <= B_WR_LARGE;
                        end
                    end
                end

                B_WR_SMALL: begin
                    bridge_we    <= 1'b1;
                    bridge_waddr <= P3_ACT_BASE + (pix_cnt * P3_WORDS) + word_idx;
                    bridge_wdata <= quant_buf[word_idx*32 +: 32];
                    if (word_idx == P3_WORDS - 1) begin
                        word_idx <= '0;
                        if (pix_cnt == P3_PIXELS - 1) begin
                            pix_cnt <= '0;
                            b_state <= B_IDLE;
                        end else begin
                            pix_cnt <= pix_cnt + 1'b1;
                            b_state <= B_IDLE;
                        end
                    end else begin
                        word_idx <= word_idx + 1'b1;
                    end
                end

                B_WR_MEDIUM: begin
                    bridge_we    <= 1'b1;
                    bridge_waddr <= P4_ACT_BASE + (pix_cnt * P4_WORDS) + word_idx;
                    bridge_wdata <= quant_buf[word_idx*32 +: 32];
                    if (word_idx == P4_WORDS - 1) begin
                        word_idx <= '0;
                        if (pix_cnt == P4_PIXELS - 1) begin
                            pix_cnt <= '0;
                            b_state <= B_IDLE;
                        end else begin
                            pix_cnt <= pix_cnt + 1'b1;
                            b_state <= B_IDLE;
                        end
                    end else begin
                        word_idx <= word_idx + 1'b1;
                    end
                end

                B_WR_LARGE: begin
                    bridge_we    <= 1'b1;
                    bridge_waddr <= P5_ACT_BASE + (pix_cnt * P5_WORDS) + word_idx;
                    bridge_wdata <= quant_buf[word_idx*32 +: 32];
                    if (word_idx == P5_WORDS - 1) begin
                        word_idx <= '0;
                        if (pix_cnt == P5_PIXELS - 1) begin
                            pix_cnt <= '0;
                            b_state <= B_TRIGGER;
                        end else begin
                            pix_cnt <= pix_cnt + 1'b1;
                            b_state <= B_IDLE;
                        end
                    end else begin
                        word_idx <= word_idx + 1'b1;
                    end
                end

                B_TRIGGER: begin
                    head_start_pulse <= 1'b1;
                    b_state          <= B_IDLE;
                end

                default: b_state <= B_IDLE;
            endcase
        end
    end

    logic           mux_h_en;
    logic           mux_h_we;
    logic [2:0]     mux_h_win;
    logic [HAW-1:0] mux_h_waddr;
    logic [31:0]    mux_h_wdata;

    assign mux_h_en    = bridge_active ? bridge_we    : host_en;
    assign mux_h_we    = bridge_active ? bridge_we    : host_we;
    assign mux_h_win   = bridge_active ? 3'd5         : host_win;
    assign mux_h_waddr = bridge_active ? bridge_waddr : host_waddr;
    assign mux_h_wdata = bridge_active ? bridge_wdata : host_wdata;

    // =========================================================================
    // 6. Detect Head Core (Danny, INT8 + NMS-free Post-Processing)
    // =========================================================================
    yolo26_head_core #(
        .LANES(LANES),
        .LOGL(LOGL),
        .AAW(AAW),
        .WAW(WAW),
        .PAW(PAW),
        .LAW(LAW),
        .PRW(PRW),
        .BAW(BAW),
        .K(K),
        .IW(IW),
        .OUTAW(OUTAW),
        .HAW(HAW),
        .YXW(YXW),
        .CW(CW),
        .D_PROG(D_PROG),
        .D_PAR(D_PAR),
        .D_WGT(D_WGT),
        .D_LUT(D_LUT),
        .D_ACT(D_ACT),
        .D_BOX(D_BOX)
    ) u_head_core (
        .clk(clk),
        .rst_n(rst_n),
        .start(head_start_pulse),
        .busy(head_busy),
        .done(head_done),
        .n_rows(n_rows),
        .cycles(head_cycles),
        .h_en(mux_h_en),
        .h_we(mux_h_we),
        .h_win(mux_h_win),
        .h_waddr(mux_h_waddr),
        .h_wdata(mux_h_wdata),
        .h_rdata(host_rdata)
    );

endmodule
