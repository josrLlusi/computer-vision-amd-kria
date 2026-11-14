// ============================================================================
// Module: yolo_psa_block
// Description: Position-Sensitive Attention Block (PSABlock) for YOLO26
//              Architecture:
//                x1 = x + Attention(x)
//                x2 = x1 + FFN(x1)
//              where FFN = Conv1x1(c -> 2c, SiLU) -> Conv1x1(2c -> c, Linear)
//              Targeted for AMD Kria KV260/KR260.
// ============================================================================

`timescale 1ns / 1ps

module yolo_psa_block #(
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS  = 8,
    parameter CHANNELS   = 4,
    parameter NUM_HEADS  = 2,
    parameter SHORTCUT   = 1,
    parameter IMG_WIDTH  = 16,
    parameter IMG_HEIGHT = 16
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              in_valid,
    input  wire [CHANNELS*DATA_WIDTH-1:0]    in_data,
    output wire                              out_valid,
    output wire [CHANNELS*DATA_WIDTH-1:0]    out_data
);

    localparam FFN_HIDDEN = 2 * CHANNELS;

    // ------------------------------------------------------------------------
    // Part 1: Attention Sub-layer
    // ------------------------------------------------------------------------
    wire                              attn_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]    attn_out;

    yolo_attention #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .CHANNELS(CHANNELS),
        .NUM_HEADS(NUM_HEADS),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_attn (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .out_valid(attn_valid),
        .out_data(attn_out)
    );

    // Residual 1 Alignment FIFO
    localparam FIFO_DEPTH = 256;
    reg [CHANNELS*DATA_WIDTH-1:0] fifo1 [0:FIFO_DEPTH-1];
    reg [7:0] wr1, rd1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr1 <= 8'd0;
            rd1 <= 8'd0;
        end else begin
            if (in_valid) begin
                fifo1[wr1] <= in_data;
                wr1 <= wr1 + 8'd1;
            end
            if (attn_valid) begin
                rd1 <= rd1 + 8'd1;
            end
        end
    end

    wire [CHANNELS*DATA_WIDTH-1:0] delayed_in1 = fifo1[rd1];

    // Unpack for Residual 1
    wire signed [DATA_WIDTH-1:0] attn_out_chan   [0:CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] delayed_in1_chan [0:CHANNELS-1];
    genvar ui;
    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : UNPACK_R1
            assign attn_out_chan[ui]   = attn_out[ui*DATA_WIDTH +: DATA_WIDTH];
            assign delayed_in1_chan[ui] = delayed_in1[ui*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Residual 1 Addition: x1 = x + Attention(x)
    reg signed [DATA_WIDTH-1:0] x1_arr [0:CHANNELS-1];
    reg                         x1_valid;
    integer c1;
    reg signed [DATA_WIDTH:0] sum1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x1_valid <= 1'b0;
            for (c1 = 0; c1 < CHANNELS; c1 = c1 + 1) begin
                x1_arr[c1] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            x1_valid <= attn_valid;
            if (attn_valid) begin
                for (c1 = 0; c1 < CHANNELS; c1 = c1 + 1) begin
                    if (SHORTCUT) begin
                        sum1 = $signed(attn_out_chan[c1]) + $signed(delayed_in1_chan[c1]);
                        if (sum1 > 17'sh07FFF)
                            x1_arr[c1] <= 16'sh7FFF;
                        else if (sum1 < -17'sh08000)
                            x1_arr[c1] <= 16'sh8000;
                        else
                            x1_arr[c1] <= sum1[DATA_WIDTH-1:0];
                    end else begin
                        x1_arr[c1] <= attn_out_chan[c1];
                    end
                end
            end
        end
    end

    wire [CHANNELS*DATA_WIDTH-1:0] x1_vec;
    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : PACK_X1
            assign x1_vec[ui*DATA_WIDTH +: DATA_WIDTH] = x1_arr[ui];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Part 2: Feed-Forward Network (FFN)
    // Layer 1: Conv1x1 (CHANNELS -> FFN_HIDDEN, SiLU)
    // Layer 2: Conv1x1 (FFN_HIDDEN -> CHANNELS, Linear)
    // ------------------------------------------------------------------------
    wire                                ffn1_valid;
    wire [FFN_HIDDEN*DATA_WIDTH-1:0]    ffn1_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(FFN_HIDDEN),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(1) // SiLU
    ) u_ffn_layer1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(x1_valid),
        .in_data(x1_vec),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(ffn1_valid),
        .out_data(ffn1_data)
    );

    wire                                ffn2_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]      ffn2_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(FFN_HIDDEN),
        .OUT_CHANNELS(CHANNELS),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(0) // Linear
    ) u_ffn_layer2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(ffn1_valid),
        .in_data(ffn1_data),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(ffn2_valid),
        .out_data(ffn2_data)
    );

    // Residual 2 Alignment FIFO
    reg [CHANNELS*DATA_WIDTH-1:0] fifo2 [0:FIFO_DEPTH-1];
    reg [7:0] wr2, rd2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr2 <= 8'd0;
            rd2 <= 8'd0;
        end else begin
            if (x1_valid) begin
                fifo2[wr2] <= x1_vec;
                wr2 <= wr2 + 8'd1;
            end
            if (ffn2_valid) begin
                rd2 <= rd2 + 8'd1;
            end
        end
    end

    wire [CHANNELS*DATA_WIDTH-1:0] delayed_x1 = fifo2[rd2];

    // Unpack for Residual 2
    wire signed [DATA_WIDTH-1:0] ffn2_chan       [0:CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] delayed_x1_chan  [0:CHANNELS-1];
    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : UNPACK_R2
            assign ffn2_chan[ui]       = ffn2_data[ui*DATA_WIDTH +: DATA_WIDTH];
            assign delayed_x1_chan[ui] = delayed_x1[ui*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Residual 2 Addition: x2 = x1 + FFN(x1)
    reg signed [DATA_WIDTH-1:0] x2_arr [0:CHANNELS-1];
    reg                         x2_valid;
    integer c2;
    reg signed [DATA_WIDTH:0] sum2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x2_valid <= 1'b0;
            for (c2 = 0; c2 < CHANNELS; c2 = c2 + 1) begin
                x2_arr[c2] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            x2_valid <= ffn2_valid;
            if (ffn2_valid) begin
                for (c2 = 0; c2 < CHANNELS; c2 = c2 + 1) begin
                    if (SHORTCUT) begin
                        sum2 = $signed(ffn2_chan[c2]) + $signed(delayed_x1_chan[c2]);
                        if (sum2 > 17'sh07FFF)
                            x2_arr[c2] <= 16'sh7FFF;
                        else if (sum2 < -17'sh08000)
                            x2_arr[c2] <= 16'sh8000;
                        else
                            x2_arr[c2] <= sum2[DATA_WIDTH-1:0];
                    end else begin
                        x2_arr[c2] <= ffn2_chan[c2];
                    end
                end
            end
        end
    end

    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : PACK_X2
            assign out_data[ui*DATA_WIDTH +: DATA_WIDTH] = x2_arr[ui];
        end
    endgenerate

    assign out_valid = x2_valid;

endmodule
