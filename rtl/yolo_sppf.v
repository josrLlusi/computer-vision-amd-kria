// ============================================================================
// Module: yolo_sppf
// Description: Spatial Pyramid Pooling - Fast (SPPF) Block for YOLO26.
//              Located at the transition between Backbone P5 stage and C2PSA.
//              Architecture (from YOLO26 Paper Fig S2 & Roadmap):
//                Input -> cv1 (1x1 Conv, c_in -> c_h, where c_h = c_in / 2)
//                y0 = cv1_out
//                y1 = MaxPool(y0, 5x5)
//                y2 = MaxPool(y1, 5x5)
//                y3 = MaxPool(y2, 5x5)
//                Concat [y0, y1, y2, y3] (4 * c_h channels)
//                cv2 (1x1 Conv, 4*c_h -> c_out) -> Output
//              Targeted for AMD Kria KV260/KR260.
// ============================================================================

`timescale 1ns / 1ps

module yolo_sppf #(
    parameter DATA_WIDTH   = 16, // Q8.8
    parameter FRAC_BITS    = 8,
    parameter IN_CHANNELS  = 4,
    parameter OUT_CHANNELS = 4,
    parameter POOL_SIZE    = 5,  // 5x5 pooling
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

    localparam C_HIDDEN = IN_CHANNELS / 2;
    localparam C_CONCAT = 4 * C_HIDDEN;

    // ------------------------------------------------------------------------
    // Step 1: Initial 1x1 Conv (cv1: IN_CHANNELS -> C_HIDDEN)
    // ------------------------------------------------------------------------
    wire                             cv1_valid;
    wire [C_HIDDEN*DATA_WIDTH-1:0]   cv1_data;

    yolo_conv #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(C_HIDDEN),
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

    wire [C_HIDDEN*DATA_WIDTH-1:0] y0 = cv1_data;
    wire                           y0_valid = cv1_valid;

    // ------------------------------------------------------------------------
    // Step 2: Three Cascaded 5x5 MaxPool Stages
    // ------------------------------------------------------------------------
    wire [C_HIDDEN*DATA_WIDTH-1:0] y1;
    wire                           y1_valid;

    yolo_maxpool2d #(
        .DATA_WIDTH(DATA_WIDTH),
        .CHANNELS(C_HIDDEN),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_pool1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(y0_valid),
        .in_data(y0),
        .out_valid(y1_valid),
        .out_data(y1)
    );

    wire [C_HIDDEN*DATA_WIDTH-1:0] y2;
    wire                           y2_valid;

    yolo_maxpool2d #(
        .DATA_WIDTH(DATA_WIDTH),
        .CHANNELS(C_HIDDEN),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_pool2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(y1_valid),
        .in_data(y1),
        .out_valid(y2_valid),
        .out_data(y2)
    );

    wire [C_HIDDEN*DATA_WIDTH-1:0] y3;
    wire                           y3_valid;

    yolo_maxpool2d #(
        .DATA_WIDTH(DATA_WIDTH),
        .CHANNELS(C_HIDDEN),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_pool3 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(y2_valid),
        .in_data(y2),
        .out_valid(y3_valid),
        .out_data(y3)
    );

    // ------------------------------------------------------------------------
    // Step 3: Synchronization FIFOs for y0, y1, y2
    // Aligns y0, y1, y2 with y3_valid
    // ------------------------------------------------------------------------
    localparam FIFO_DEPTH = 256;

    reg [C_HIDDEN*DATA_WIDTH-1:0] fifo0 [0:FIFO_DEPTH-1];
    reg [7:0] wr0, rd0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr0 <= 8'd0; rd0 <= 8'd0;
        end else begin
            if (y0_valid) begin
                fifo0[wr0] <= y0;
                wr0 <= wr0 + 8'd1;
            end
            if (y3_valid) begin
                rd0 <= rd0 + 8'd1;
            end
        end
    end
    wire [C_HIDDEN*DATA_WIDTH-1:0] delayed_y0 = fifo0[rd0];

    reg [C_HIDDEN*DATA_WIDTH-1:0] fifo1 [0:FIFO_DEPTH-1];
    reg [7:0] wr1, rd1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr1 <= 8'd0; rd1 <= 8'd0;
        end else begin
            if (y1_valid) begin
                fifo1[wr1] <= y1;
                wr1 <= wr1 + 8'd1;
            end
            if (y3_valid) begin
                rd1 <= rd1 + 8'd1;
            end
        end
    end
    wire [C_HIDDEN*DATA_WIDTH-1:0] delayed_y1 = fifo1[rd1];

    reg [C_HIDDEN*DATA_WIDTH-1:0] fifo2 [0:FIFO_DEPTH-1];
    reg [7:0] wr2, rd2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr2 <= 8'd0; rd2 <= 8'd0;
        end else begin
            if (y2_valid) begin
                fifo2[wr2] <= y2;
                wr2 <= wr2 + 8'd1;
            end
            if (y3_valid) begin
                rd2 <= rd2 + 8'd1;
            end
        end
    end
    wire [C_HIDDEN*DATA_WIDTH-1:0] delayed_y2 = fifo2[rd2];

    // Concatenate [delayed_y0, delayed_y1, delayed_y2, y3]
    wire [C_CONCAT*DATA_WIDTH-1:0] concat_data;
    assign concat_data[1*C_HIDDEN*DATA_WIDTH-1 : 0*C_HIDDEN*DATA_WIDTH] = delayed_y0;
    assign concat_data[2*C_HIDDEN*DATA_WIDTH-1 : 1*C_HIDDEN*DATA_WIDTH] = delayed_y1;
    assign concat_data[3*C_HIDDEN*DATA_WIDTH-1 : 2*C_HIDDEN*DATA_WIDTH] = delayed_y2;
    assign concat_data[4*C_HIDDEN*DATA_WIDTH-1 : 3*C_HIDDEN*DATA_WIDTH] = y3;

    wire concat_valid = y3_valid;

    // ------------------------------------------------------------------------
    // Step 4: Final 1x1 Conv (cv2: 4 * C_HIDDEN -> OUT_CHANNELS)
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
