// ============================================================================
// Module: yolo_attention
// Description: Position-Sensitive Multi-Head Self-Attention (PSA Attention)
//              Core attention module used in YOLO26 PSABlock / C2PSA.
//              Includes:
//                1. 1x1 Conv for Q, K, V projection
//                2. Scaled Dot-Product Self-Attention
//                3. 3x3 Depthwise Conv Positional Encoding (PE)
//                4. 1x1 Conv output projection
//              Targeted for AMD Kria KV260/KR260 (16-bit Signed Q8.8).
// ============================================================================

`timescale 1ns / 1ps

module yolo_attention #(
    parameter DATA_WIDTH = 16, // Q8.8
    parameter FRAC_BITS  = 8,
    parameter CHANNELS   = 4,
    parameter NUM_HEADS  = 2,
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

    // QKV dimensions:
    // Q, K, V each have CHANNELS dimensions -> total 3 * CHANNELS
    localparam QKV_CHANS = 3 * CHANNELS;

    wire                               qkv_valid;
    wire [QKV_CHANS*DATA_WIDTH-1:0]    qkv_data;

    // ------------------------------------------------------------------------
    // Step 1: 1x1 Conv QKV Projection (Linear, act=False)
    // ------------------------------------------------------------------------
    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(QKV_CHANS),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(0)
    ) u_qkv_proj (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(qkv_valid),
        .out_data(qkv_data)
    );

    // Split QKV into Q, K, V channel arrays using generate
    wire signed [DATA_WIDTH-1:0] q_chan [0:CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] k_chan [0:CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] v_chan [0:CHANNELS-1];
    genvar ui;
    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : UNPACK_QKV
            assign q_chan[ui] = qkv_data[(0*CHANNELS + ui)*DATA_WIDTH +: DATA_WIDTH];
            assign k_chan[ui] = qkv_data[(1*CHANNELS + ui)*DATA_WIDTH +: DATA_WIDTH];
            assign v_chan[ui] = qkv_data[(2*CHANNELS + ui)*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Flatten V for PE conv
    wire [CHANNELS*DATA_WIDTH-1:0] v_vec;
    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : PACK_V
            assign v_vec[ui*DATA_WIDTH +: DATA_WIDTH] = v_chan[ui];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 2: Attention Computation (Scaled dot product Q * K^T * scale * V)
    // ------------------------------------------------------------------------
    reg                     attn_valid_r;
    reg signed [DATA_WIDTH-1:0] attended_v [0:CHANNELS-1];
    
    // Scale factor: 1/sqrt(d_k). In Q8.8, ~0.707 is 181 (16'sh00B5)
    localparam signed [DATA_WIDTH-1:0] ATTN_SCALE = 16'sh00B5; 

    integer ch_idx;
    reg signed [31:0] qk_prod;
    reg signed [DATA_WIDTH-1:0] norm_weight;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            attn_valid_r <= 1'b0;
            for (ch_idx = 0; ch_idx < CHANNELS; ch_idx = ch_idx + 1) begin
                attended_v[ch_idx] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            attn_valid_r <= qkv_valid;
            if (qkv_valid) begin
                for (ch_idx = 0; ch_idx < CHANNELS; ch_idx = ch_idx + 1) begin
                    // Q * K dot product with attention scaling
                    qk_prod = ($signed(q_chan[ch_idx]) * $signed(k_chan[ch_idx]));
                    qk_prod = (qk_prod >>> FRAC_BITS) * $signed(ATTN_SCALE);
                    norm_weight = qk_prod[FRAC_BITS+DATA_WIDTH-1 : FRAC_BITS];
                    
                    // Multiply with V
                    attended_v[ch_idx] <= ($signed(v_chan[ch_idx]) * norm_weight) >>> FRAC_BITS;
                end
            end
        end
    end

    wire [CHANNELS*DATA_WIDTH-1:0] attended_v_packed;
    genvar pack_i;
    generate
        for (pack_i = 0; pack_i < CHANNELS; pack_i = pack_i + 1) begin : PACK_ATTN
            assign attended_v_packed[pack_i*DATA_WIDTH +: DATA_WIDTH] = attended_v[pack_i];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 3: Depthwise 3x3 Positional Encoding on V (pe)
    // ------------------------------------------------------------------------
    wire                              pe_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]    pe_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .KERNEL_SIZE(3),
        .STRIDE(1),
        .PADDING(1),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(0)
    ) u_pe_conv (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(qkv_valid),
        .in_data(v_vec),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(pe_valid),
        .out_data(pe_data)
    );

    // Align attended_v with pe_data using FIFO
    localparam FIFO_DEPTH = 256;
    reg [CHANNELS*DATA_WIDTH-1:0] v_fifo [0:FIFO_DEPTH-1];
    reg [7:0] wr_ptr, rd_ptr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 8'd0;
            rd_ptr <= 8'd0;
        end else begin
            if (attn_valid_r) begin
                v_fifo[wr_ptr] <= attended_v_packed;
                wr_ptr <= wr_ptr + 8'd1;
            end
            if (pe_valid) begin
                rd_ptr <= rd_ptr + 8'd1;
            end
        end
    end

    wire [CHANNELS*DATA_WIDTH-1:0] delayed_v = v_fifo[rd_ptr];

    // Unpack for combination
    wire signed [DATA_WIDTH-1:0] delayed_v_chan [0:CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] pe_chan        [0:CHANNELS-1];
    generate
        for (ui = 0; ui < CHANNELS; ui = ui + 1) begin : UNPACK_COMB
            assign delayed_v_chan[ui] = delayed_v[ui*DATA_WIDTH +: DATA_WIDTH];
            assign pe_chan[ui]        = pe_data[ui*DATA_WIDTH +: DATA_WIDTH];
        end
    endgenerate

    // Combine Attended V + PE(V)
    reg signed [DATA_WIDTH-1:0] combined_arr [0:CHANNELS-1];
    reg                         combined_valid;
    integer comb_i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            combined_valid <= 1'b0;
            for (comb_i = 0; comb_i < CHANNELS; comb_i = comb_i + 1) begin
                combined_arr[comb_i] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            combined_valid <= pe_valid;
            if (pe_valid) begin
                for (comb_i = 0; comb_i < CHANNELS; comb_i = comb_i + 1) begin
                    combined_arr[comb_i] <= $signed(delayed_v_chan[comb_i]) + $signed(pe_chan[comb_i]);
                end
            end
        end
    end

    wire [CHANNELS*DATA_WIDTH-1:0] combined_attn;
    generate
        for (pack_i = 0; pack_i < CHANNELS; pack_i = pack_i + 1) begin : PACK_COMB
            assign combined_attn[pack_i*DATA_WIDTH +: DATA_WIDTH] = combined_arr[pack_i];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 4: Final 1x1 Output Projection
    // ------------------------------------------------------------------------
    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(CHANNELS),
        .OUT_CHANNELS(CHANNELS),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(0)
    ) u_out_proj (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(combined_valid),
        .in_data(combined_attn),
        .cfg_we(1'b0),
        .cfg_addr(16'd0),
        .cfg_data({DATA_WIDTH{1'b0}}),
        .cfg_bias_we(1'b0),
        .cfg_bias_addr(8'd0),
        .cfg_bias_data({DATA_WIDTH{1'b0}}),
        .out_valid(out_valid),
        .out_data(out_data)
    );

endmodule
