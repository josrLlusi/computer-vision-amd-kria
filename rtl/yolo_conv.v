// ============================================================================
// Module: yolo_conv
// Description: Primary YOLO26 Convolution Block with Fused Bias/BatchNorm & SiLU.
//              Supports 1x1 Conv (Pointwise) and 3x3 Conv with Stride 1 or 2.
//              Uses 16-bit Signed Q8.8 Fixed-Point representation.
//              Targeted for AMD Kria KV260/KR260 (DSP48E2 slices).
// ============================================================================

`timescale 1ns / 1ps

module yolo_conv #(
    parameter DATA_WIDTH   = 16, // Q8.8 signed
    parameter FRAC_BITS    = 8,
    parameter IN_CHANNELS  = 4,
    parameter OUT_CHANNELS = 4,
    parameter KERNEL_SIZE  = 3,  // 1 or 3
    parameter STRIDE       = 1,  // 1 or 2
    parameter PADDING      = 1,  // 0 or 1
    parameter IMG_WIDTH    = 16,
    parameter IMG_HEIGHT   = 16,
    parameter USE_ACT      = 1   // 1: SiLU, 0: Linear
)(
    input  wire                               clk,
    input  wire                               rst_n,
    
    // Feature map streaming input
    input  wire                               in_valid,
    input  wire [IN_CHANNELS*DATA_WIDTH-1:0]  in_data,
    
    // Weight configuration interface (optional dynamic loading)
    input  wire                               cfg_we,
    input  wire [15:0]                        cfg_addr,
    input  wire signed [DATA_WIDTH-1:0]       cfg_data,
    input  wire                               cfg_bias_we,
    input  wire [7:0]                         cfg_bias_addr,
    input  wire signed [DATA_WIDTH-1:0]       cfg_bias_data,

    // Feature map streaming output
    output wire                               out_valid,
    output wire [OUT_CHANNELS*DATA_WIDTH-1:0] out_data
);

    localparam K_SQ = (KERNEL_SIZE == 3) ? 9 : 1;
    localparam TOTAL_WEIGHTS = OUT_CHANNELS * IN_CHANNELS * K_SQ;

    // Weight and bias storage
    reg signed [DATA_WIDTH-1:0] weights [0:TOTAL_WEIGHTS-1];
    reg signed [DATA_WIDTH-1:0] biases  [0:OUT_CHANNELS-1];

    integer wi, bi;
    initial begin
        // Default deterministic initialization (e.g. Identity or scaled uniform weights)
        for (wi = 0; wi < TOTAL_WEIGHTS; wi = wi + 1) begin
            // Non-zero default weights (0.25 to 0.75 in Q8.8)
            weights[wi] = 16'sh0040 + ((wi % 5) * 16'sh0010); // ~0.25 to 0.5
        end
        for (bi = 0; bi < OUT_CHANNELS; bi = bi + 1) begin
            biases[bi] = 16'sh0010 * (bi + 1); // Small positive bias ~0.0625
        end
    end

    // Configuration writes
    always @(posedge clk) begin
        if (cfg_we && (cfg_addr < TOTAL_WEIGHTS))
            weights[cfg_addr] <= cfg_data;
        if (cfg_bias_we && (cfg_bias_addr < OUT_CHANNELS))
            biases[cfg_bias_addr] <= cfg_bias_data;
    end

    // Unpack input channels
    wire signed [DATA_WIDTH-1:0] in_chan [0:IN_CHANNELS-1];
    genvar ci;
    generate
        for (ci = 0; ci < IN_CHANNELS; ci = ci + 1) begin : UNPACK_IN
            assign in_chan[ci] = in_data[(ci+1)*DATA_WIDTH-1 : ci*DATA_WIDTH];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Spatial buffering for K=3 vs K=1
    // ------------------------------------------------------------------------
    wire line_valid [0:IN_CHANNELS-1];
    wire signed [DATA_WIDTH-1:0] win_patches [0:IN_CHANNELS-1][0:8];
    wire [15:0] cur_row, cur_col;

    generate
        if (KERNEL_SIZE == 3) begin : GEN_3X3_BUF
            for (ci = 0; ci < IN_CHANNELS; ci = ci + 1) begin : LINE_BUFS
                yolo_line_buffer #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .IMG_WIDTH(IMG_WIDTH),
                    .IMG_HEIGHT(IMG_HEIGHT)
                ) u_line_buf (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(in_valid),
                    .in_pixel(in_chan[ci]),
                    .win_valid(line_valid[ci]),
                    .win_00(win_patches[ci][0]), .win_01(win_patches[ci][1]), .win_02(win_patches[ci][2]),
                    .win_10(win_patches[ci][3]), .win_11(win_patches[ci][4]), .win_12(win_patches[ci][5]),
                    .win_20(win_patches[ci][6]), .win_21(win_patches[ci][7]), .win_22(win_patches[ci][8]),
                    .cur_row(cur_row),
                    .cur_col(cur_col)
                );
            end
        end else begin : GEN_1X1_DIRECT
            // For 1x1 Conv, window patch is just the single current pixel
            for (ci = 0; ci < IN_CHANNELS; ci = ci + 1) begin : WIRE_1X1
                assign win_patches[ci][0] = in_chan[ci];
                assign line_valid[ci]     = in_valid;
            end
            assign cur_row = 16'd0;
            assign cur_col = 16'd0;
        end
    endgenerate

    // ------------------------------------------------------------------------
    // MAC Array: Compute dot-product for all OUT_CHANNELS in parallel
    // Pipelined Stage 1: Multiply and Accumulate
    // ------------------------------------------------------------------------
    wire mac_trigger = (KERNEL_SIZE == 3) ? line_valid[0] : in_valid;
    wire stride_enable = (STRIDE == 2) ? (cur_row[0] & cur_col[0]) : 1'b1;

    reg                     pipe1_valid;
    reg signed [31:0]       pipe1_acc [0:OUT_CHANNELS-1];

    integer oc, ic, ki;
    reg signed [31:0] accum_sum;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe1_valid <= 1'b0;
            for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin
                pipe1_acc[oc] <= 32'sd0;
            end
        end else begin
            pipe1_valid <= mac_trigger & stride_enable;
            if (mac_trigger & stride_enable) begin
                for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin
                    accum_sum = $signed({{16{biases[oc][DATA_WIDTH-1]}}, biases[oc]}) <<< FRAC_BITS;
                    for (ic = 0; ic < IN_CHANNELS; ic = ic + 1) begin
                        for (ki = 0; ki < K_SQ; ki = ki + 1) begin
                            accum_sum = accum_sum + ($signed(win_patches[ic][ki]) * $signed(weights[(oc * IN_CHANNELS + ic) * K_SQ + ki]));
                        end
                    end
                    pipe1_acc[oc] <= accum_sum;
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 2: Scaling (Q8.8 Shift) and Saturation
    // ------------------------------------------------------------------------
    reg                     pipe2_valid;
    reg signed [DATA_WIDTH-1:0] pipe2_scaled [0:OUT_CHANNELS-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipe2_valid <= 1'b0;
            for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin
                pipe2_scaled[oc] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            pipe2_valid <= pipe1_valid;
            for (oc = 0; oc < OUT_CHANNELS; oc = oc + 1) begin
                // Shift down by FRAC_BITS (8)
                // Check saturation for 16-bit signed
                if (pipe1_acc[oc] >>> FRAC_BITS > 32'sd32767)
                    pipe2_scaled[oc] <= 16'sh7FFF;
                else if (pipe1_acc[oc] >>> FRAC_BITS < -32'sd32768)
                    pipe2_scaled[oc] <= 16'sh8000;
                else
                    pipe2_scaled[oc] <= pipe1_acc[oc][FRAC_BITS + DATA_WIDTH - 1 : FRAC_BITS];
            end
        end
    end

    // ------------------------------------------------------------------------
    // Pipeline Stage 3: Activation (SiLU or Pass-Through)
    // ------------------------------------------------------------------------
    wire signed [DATA_WIDTH-1:0] silu_out [0:OUT_CHANNELS-1];
    wire [OUT_CHANNELS-1:0]      silu_val;

    genvar oci;
    generate
        for (oci = 0; oci < OUT_CHANNELS; oci = oci + 1) begin : GEN_ACT
            if (USE_ACT == 1) begin : WITH_SILU
                yolo_silu #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .FRAC_BITS(FRAC_BITS)
                ) u_silu (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(pipe2_valid),
                    .in_data(pipe2_scaled[oci]),
                    .out_valid(silu_val[oci]),
                    .out_data(silu_out[oci])
                );
                assign out_data[(oci+1)*DATA_WIDTH-1 : oci*DATA_WIDTH] = silu_out[oci];
            end else begin : NO_ACT
                assign out_data[(oci+1)*DATA_WIDTH-1 : oci*DATA_WIDTH] = pipe2_scaled[oci];
            end
        end
    endgenerate

    assign out_valid = (USE_ACT == 1) ? silu_val[0] : pipe2_valid;

endmodule
