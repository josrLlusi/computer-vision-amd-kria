// ============================================================================
// Module: yolo_c2psa
// Description: Cross Stage Partial Block with Position-Sensitive Attention (C2PSA)
//              Used in YOLO26 at P5 stage (1024 channels) and feature refinement.
//              Architecture (from YOLO26 Paper Fig S2):
//                Input -> 1x1 Conv (c_in -> 2*c_h) -> Split into Branch A and Branch B
//                Branch A: Identity bypass
//                Branch B: PSABlock x N (Attention + FFN)
//                Concat [Branch A, Branch B] -> 1x1 Conv (2*c_h -> c_out)
//              Targeted for AMD Kria KV260/KR260.
// ============================================================================

`timescale 1ns / 1ps

module yolo_c2psa #(
    parameter DATA_WIDTH   = 16, // Q8.8
    parameter FRAC_BITS    = 8,
    parameter IN_CHANNELS  = 4,
    parameter OUT_CHANNELS = 4,
    parameter NUM_PSA      = 2,  // Default 2 PSABlocks (C2PSA x2)
    parameter NUM_HEADS    = 2,
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

    // Hidden channels per branch: c_h = IN_CHANNELS / 2
    localparam C_HIDDEN   = IN_CHANNELS / 2;
    localparam C_EXPANDED = 2 * C_HIDDEN;

    // ------------------------------------------------------------------------
    // Step 1: Initial 1x1 Conv (cv1: IN_CHANNELS -> 2 * C_HIDDEN)
    // ------------------------------------------------------------------------
    wire                               cv1_valid;
    wire [C_EXPANDED*DATA_WIDTH-1:0]   cv1_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(C_EXPANDED),
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

    // Split cv1 output into Branch A and Branch B
    wire [C_HIDDEN*DATA_WIDTH-1:0] branch_a = cv1_data[1*C_HIDDEN*DATA_WIDTH-1 : 0*C_HIDDEN*DATA_WIDTH];
    wire [C_HIDDEN*DATA_WIDTH-1:0] branch_b = cv1_data[2*C_HIDDEN*DATA_WIDTH-1 : 1*C_HIDDEN*DATA_WIDTH];

    // ------------------------------------------------------------------------
    // Step 2: Branch B - Cascade of N PSABlocks
    // ------------------------------------------------------------------------
    wire [C_HIDDEN*DATA_WIDTH-1:0] psa_data [0:NUM_PSA];
    wire                          psa_valid [0:NUM_PSA];

    assign psa_data[0]  = branch_b;
    assign psa_valid[0] = cv1_valid;

    genvar pi;
    generate
        for (pi = 0; pi < NUM_PSA; pi = pi + 1) begin : GEN_PSA
            yolo_psa_block #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS),
                .CHANNELS(C_HIDDEN),
                .NUM_HEADS(NUM_HEADS),
                .SHORTCUT(1),
                .IMG_WIDTH(IMG_WIDTH),
                .IMG_HEIGHT(IMG_HEIGHT)
            ) u_psa (
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(psa_valid[pi]),
                .in_data(psa_data[pi]),
                .out_valid(psa_valid[pi+1]),
                .out_data(psa_data[pi+1])
            );
        end
    endgenerate

    // ------------------------------------------------------------------------
    // Step 3: Branch A Alignment FIFO
    // Buffers Branch A until Branch B finishes all PSABlocks
    // ------------------------------------------------------------------------
    localparam FIFO_DEPTH = 512;
    reg [C_HIDDEN*DATA_WIDTH-1:0] fifo_a [0:FIFO_DEPTH-1];
    reg [8:0] wr_a, rd_a;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_a <= 9'd0;
            rd_a <= 9'd0;
        end else begin
            if (cv1_valid) begin
                fifo_a[wr_a] <= branch_a;
                wr_a <= wr_a + 9'd1;
            end
            if (psa_valid[NUM_PSA]) begin
                rd_a <= rd_a + 9'd1;
            end
        end
    end

    wire [C_HIDDEN*DATA_WIDTH-1:0] delayed_branch_a = fifo_a[rd_a];

    // Concatenate [Branch A, Branch B]
    wire [C_EXPANDED*DATA_WIDTH-1:0] concat_data = {psa_data[NUM_PSA], delayed_branch_a};
    wire                             concat_valid = psa_valid[NUM_PSA];

    // ------------------------------------------------------------------------
    // Step 4: Final 1x1 Conv (cv2: 2 * C_HIDDEN -> OUT_CHANNELS)
    // ------------------------------------------------------------------------
    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(C_EXPANDED),
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
        .in_valid(concat_valid),
        .in_data(concat_data),
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
