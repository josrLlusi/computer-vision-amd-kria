// ============================================================================
// Module: yolo_c3k2
// Description: Cross Stage Partial Block with 2 Convolutions (C3k2) for YOLO26.
//              Used extensively in Backbone (P3, P4, P5) and Neck.
//              Architecture (from YOLO26 Paper Fig S2 & Roadmap):
//                Input -> cv1 (1x1 Conv, c_in -> 2*c_h) -> Chunk into y0, y1
//                y1 -> [Inner Block] x N -> y2, ..., y(N+1)
//                where Inner Block can be:
//                  - Standard Bottleneck (C3K_MODE=0, ATTN_MODE=0)
//                  - C3k: Two stacked bottlenecks with 3x3 path (C3K_MODE=1)
//                  - Bottleneck + PSABlock (ATTN_MODE=1)
//                Concat [y0, y1, y2, ..., y(N+1)] -> cv2 (1x1 Conv -> c_out)
//              Targeted for AMD Kria KV260/KR260.
// ============================================================================

`timescale 1ns / 1ps

module yolo_c3k2 #(
    parameter DATA_WIDTH   = 16, // Q8.8
    parameter FRAC_BITS    = 8,
    parameter IN_CHANNELS  = 4,
    parameter OUT_CHANNELS = 4,
    parameter NUM_BLOCKS   = 2,  // Default 2 repeats (C3k2 x2)
    parameter C3K_MODE     = 0,  // 1: C3k blocks, 0: Standard bottlenecks
    parameter ATTN_MODE    = 0,  // 1: Bottleneck + PSABlock (attn=True)
    parameter IMG_WIDTH    = 16,
    parameter IMG_HEIGHT   = 16
)(
    input  wire                               clk,
    input  wire                               rst_n,
    input  wire                               in_valid,
    input  wire [IN_CHANNELS*DATA_WIDTH-1:0]  in_data,
    output wire                               out_valid,
    output wire [OUT_CHANNELS*DATA_WIDTH-1:0] out_data
);

    // Hidden channels: c_h = OUT_CHANNELS / 2 (expansion ratio e = 0.5)
    // If OUT_CHANNELS=4, c_h=2. 2*c_h = 4.
    localparam C_HIDDEN   = OUT_CHANNELS / 2;
    localparam C_SPLIT    = 2 * C_HIDDEN;
    localparam C_CONCAT   = (2 + NUM_BLOCKS) * C_HIDDEN;

    // ------------------------------------------------------------------------
    // Step 1: Initial 1x1 Conv (cv1: IN_CHANNELS -> 2 * C_HIDDEN)
    // ------------------------------------------------------------------------
    wire                             cv1_valid;
    wire [C_SPLIT*DATA_WIDTH-1:0]    cv1_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(C_SPLIT),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(1) // SiLU
    ) u_cv1 (
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
        .out_valid(cv1_valid),
        .out_data(cv1_data)
    );

    // Split into y0 (bypass) and y1 (processing branch)
    wire [C_HIDDEN*DATA_WIDTH-1:0] y_branch0 = cv1_data[1*C_HIDDEN*DATA_WIDTH-1 : 0*C_HIDDEN*DATA_WIDTH];
    wire [C_HIDDEN*DATA_WIDTH-1:0] y_branch1 = cv1_data[2*C_HIDDEN*DATA_WIDTH-1 : 1*C_HIDDEN*DATA_WIDTH];

    // ------------------------------------------------------------------------
    // Step 2: Inner Blocks Cascade (y1 -> Block 1 -> y2 -> Block 2 -> ... -> y(N+1))
    // ------------------------------------------------------------------------
    wire [C_HIDDEN*DATA_WIDTH-1:0] block_out [0:NUM_BLOCKS];
    wire                          block_val [0:NUM_BLOCKS];

    assign block_out[0] = y_branch1;
    assign block_val[0] = cv1_valid;

    genvar bi;
    generate
        for (bi = 0; bi < NUM_BLOCKS; bi = bi + 1) begin : GEN_INNER_BLOCKS
            if (ATTN_MODE == 1) begin : MODE_ATTN
                // Bottleneck followed by PSABlock
                wire [C_HIDDEN*DATA_WIDTH-1:0] btnk_data;
                wire                           btnk_val;

                yolo_bottleneck #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .FRAC_BITS(FRAC_BITS),
                    .IN_CHANNELS(C_HIDDEN),
                    .OUT_CHANNELS(C_HIDDEN),
                    .HIDDEN_CHANS(C_HIDDEN),
                    .KERNEL_SIZE1(3),
                    .KERNEL_SIZE2(3),
                    .SHORTCUT(1),
                    .IMG_WIDTH(IMG_WIDTH),
                    .IMG_HEIGHT(IMG_HEIGHT)
                ) u_btnk_attn (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(block_val[bi]),
                    .in_data(block_out[bi]),
                    .out_valid(btnk_val),
                    .out_data(btnk_data)
                );

                yolo_psa_block #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .FRAC_BITS(FRAC_BITS),
                    .CHANNELS(C_HIDDEN),
                    .NUM_HEADS(1),
                    .SHORTCUT(1),
                    .IMG_WIDTH(IMG_WIDTH),
                    .IMG_HEIGHT(IMG_HEIGHT)
                ) u_psa_inner (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(btnk_val),
                    .in_data(btnk_data),
                    .out_valid(block_val[bi+1]),
                    .out_data(block_out[bi+1])
                );
            end else if (C3K_MODE == 1) begin : MODE_C3K
                // C3k module (two stacked bottlenecks with 3x3 path)
                yolo_c3k #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .FRAC_BITS(FRAC_BITS),
                    .CHANNELS(C_HIDDEN),
                    .SHORTCUT(1),
                    .IMG_WIDTH(IMG_WIDTH),
                    .IMG_HEIGHT(IMG_HEIGHT)
                ) u_c3k (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(block_val[bi]),
                    .in_data(block_out[bi]),
                    .out_valid(block_val[bi+1]),
                    .out_data(block_out[bi+1])
                );
            end else begin : MODE_BOTTLENECK
                // Standard Bottleneck
                yolo_bottleneck #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .FRAC_BITS(FRAC_BITS),
                    .IN_CHANNELS(C_HIDDEN),
                    .OUT_CHANNELS(C_HIDDEN),
                    .HIDDEN_CHANS(C_HIDDEN),
                    .KERNEL_SIZE1(3),
                    .KERNEL_SIZE2(3),
                    .SHORTCUT(1),
                    .IMG_WIDTH(IMG_WIDTH),
                    .IMG_HEIGHT(IMG_HEIGHT)
                ) u_bottleneck (
                    .clk(clk),
                    .rst_n(rst_n),
                    .in_valid(block_val[bi]),
                    .in_data(block_out[bi]),
                    .out_valid(block_val[bi+1]),
                    .out_data(block_out[bi+1])
                );
            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 3: Alignment Buffers for all intermediate taps: y0, y1, ..., y(N)
    // Synchronizes all taps with block_val[NUM_BLOCKS]
    // ------------------------------------------------------------------------
    localparam FIFO_DEPTH = 512;
    wire final_valid = block_val[NUM_BLOCKS];

    // Delay FIFO for y0 (bypass)
    reg [C_HIDDEN*DATA_WIDTH-1:0] fifo_y0 [0:FIFO_DEPTH-1];
    reg [8:0] wr_y0, rd_y0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_y0 <= 9'd0;
            rd_y0 <= 9'd0;
        end else begin
            if (cv1_valid) begin
                fifo_y0[wr_y0] <= y_branch0;
                wr_y0 <= wr_y0 + 9'd1;
            end
            if (final_valid) begin
                rd_y0 <= rd_y0 + 9'd1;
            end
        end
    end
    wire [C_HIDDEN*DATA_WIDTH-1:0] delayed_y0 = fifo_y0[rd_y0];

    // Delay FIFOs for intermediate block outputs: y1, ..., y(NUM_BLOCKS-1)
    wire [C_HIDDEN*DATA_WIDTH-1:0] delayed_taps [0:NUM_BLOCKS-1];
    genvar ti;
    generate
        for (ti = 0; ti < NUM_BLOCKS; ti = ti + 1) begin : TAP_FIFOS
            reg [C_HIDDEN*DATA_WIDTH-1:0] fifo_tap [0:FIFO_DEPTH-1];
            reg [8:0] wr_tap, rd_tap;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    wr_tap <= 9'd0;
                    rd_tap <= 9'd0;
                end else begin
                    if (block_val[ti]) begin
                        fifo_tap[wr_tap] <= block_out[ti];
                        wr_tap <= wr_tap + 9'd1;
                    end
                    if (final_valid) begin
                        rd_tap <= rd_tap + 9'd1;
                    end
                end
            end
            assign delayed_taps[ti] = fifo_tap[rd_tap];
        end
    endgenerate

    // Concatenate [delayed_y0, delayed_taps[0], ..., block_out[NUM_BLOCKS]]
    wire [C_CONCAT*DATA_WIDTH-1:0] concat_all;
    assign concat_all[1*C_HIDDEN*DATA_WIDTH-1 : 0*C_HIDDEN*DATA_WIDTH] = delayed_y0;
    generate
        for (ti = 0; ti < NUM_BLOCKS; ti = ti + 1) begin : CONCAT_TAPS
            assign concat_all[(ti+2)*C_HIDDEN*DATA_WIDTH-1 : (ti+1)*C_HIDDEN*DATA_WIDTH] = delayed_taps[ti];
        end
    endgenerate
    assign concat_all[C_CONCAT*DATA_WIDTH-1 : (NUM_BLOCKS+1)*C_HIDDEN*DATA_WIDTH] = block_out[NUM_BLOCKS];

    // ------------------------------------------------------------------------
    // Step 4: Final 1x1 Conv (cv2: C_CONCAT -> OUT_CHANNELS)
    // ------------------------------------------------------------------------
    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(C_CONCAT),
        .OUT_CHANNELS(OUT_CHANNELS),
        .KERNEL_SIZE(1),
        .STRIDE(1),
        .PADDING(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .USE_ACT(1) // SiLU
    ) u_cv2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(final_valid),
        .in_data(concat_all),
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
