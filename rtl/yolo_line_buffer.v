// ============================================================================
// Module: yolo_line_buffer
// Description: Streaming 2D Line Buffer and 3x3 Sliding Window Generator.
//              Buffers two image rows in dual-port circular BRAM/registers
//              to provide a concurrent 3x3 spatial neighborhood for each pixel.
//              Supports zero-padding at image boundaries.
//              Guarantees 1-to-1 streaming: N input pixels -> N valid 3x3 windows.
// Parameters:
//   DATA_WIDTH : Width of each feature element (e.g. 16-bit Q8.8)
//   IMG_WIDTH  : Width of feature map in pixels
//   IMG_HEIGHT : Height of feature map in pixels
// ============================================================================

`timescale 1ns / 1ps

module yolo_line_buffer #(
    parameter DATA_WIDTH = 16,
    parameter IMG_WIDTH  = 32,
    parameter IMG_HEIGHT = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,
    
    // Pixel stream input
    input  wire                   in_valid,
    input  wire signed [DATA_WIDTH-1:0] in_pixel,
    
    // 3x3 window output
    output reg                    win_valid,
    output reg  signed [DATA_WIDTH-1:0] win_00, win_01, win_02,
    output reg  signed [DATA_WIDTH-1:0] win_10, win_11, win_12,
    output reg  signed [DATA_WIDTH-1:0] win_20, win_21, win_22,
    
    // Coordinates
    output reg  [15:0]            cur_row,
    output reg  [15:0]            cur_col
);

    // Coordinate counters
    reg [15:0] col_cnt;
    reg [15:0] row_cnt;

    // Dual line buffers storing previous two rows
    reg signed [DATA_WIDTH-1:0] line_buf0 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] line_buf1 [0:IMG_WIDTH-1];

    // Shift registers for 3 adjacent columns (taps)
    reg signed [DATA_WIDTH-1:0] tap0_0, tap0_1, tap0_2;
    reg signed [DATA_WIDTH-1:0] tap1_0, tap1_1, tap1_2;
    reg signed [DATA_WIDTH-1:0] tap2_0, tap2_1, tap2_2;

    // Read pointers and fetched data
    wire signed [DATA_WIDTH-1:0] buf0_out = line_buf0[col_cnt];
    wire signed [DATA_WIDTH-1:0] buf1_out = line_buf1[col_cnt];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col_cnt   <= 16'd0;
            row_cnt   <= 16'd0;
            cur_col   <= 16'd0;
            cur_row   <= 16'd0;
            win_valid <= 1'b0;

            for (i = 0; i < IMG_WIDTH; i = i + 1) begin
                line_buf0[i] <= {DATA_WIDTH{1'b0}};
                line_buf1[i] <= {DATA_WIDTH{1'b0}};
            end
            
            tap0_0 <= {DATA_WIDTH{1'b0}}; tap0_1 <= {DATA_WIDTH{1'b0}}; tap0_2 <= {DATA_WIDTH{1'b0}};
            tap1_0 <= {DATA_WIDTH{1'b0}}; tap1_1 <= {DATA_WIDTH{1'b0}}; tap1_2 <= {DATA_WIDTH{1'b0}};
            tap2_0 <= {DATA_WIDTH{1'b0}}; tap2_1 <= {DATA_WIDTH{1'b0}}; tap2_2 <= {DATA_WIDTH{1'b0}};
            
            win_00 <= {DATA_WIDTH{1'b0}}; win_01 <= {DATA_WIDTH{1'b0}}; win_02 <= {DATA_WIDTH{1'b0}};
            win_10 <= {DATA_WIDTH{1'b0}}; win_11 <= {DATA_WIDTH{1'b0}}; win_12 <= {DATA_WIDTH{1'b0}};
            win_20 <= {DATA_WIDTH{1'b0}}; win_21 <= {DATA_WIDTH{1'b0}}; win_22 <= {DATA_WIDTH{1'b0}};
        end else if (in_valid) begin
            // Update line buffers
            line_buf0[col_cnt] <= buf1_out;
            line_buf1[col_cnt] <= in_pixel;

            // Shift column taps
            tap0_2 <= tap0_1;
            tap0_1 <= tap0_0;
            tap0_0 <= buf0_out;

            tap1_2 <= tap1_1;
            tap1_1 <= tap1_0;
            tap1_0 <= buf1_out;

            tap2_2 <= tap2_1;
            tap2_1 <= tap2_0;
            tap2_0 <= in_pixel;

            // Output coordinate tracking
            cur_col <= col_cnt;
            cur_row <= row_cnt;

            // 3x3 Window mapping with boundary zero-padding
            win_00 <= (row_cnt < 2 || col_cnt < 2) ? {DATA_WIDTH{1'b0}} : tap0_2;
            win_01 <= (row_cnt < 2 || col_cnt < 1) ? {DATA_WIDTH{1'b0}} : tap0_1;
            win_02 <= (row_cnt < 2)                ? {DATA_WIDTH{1'b0}} : tap0_0;

            win_10 <= (row_cnt < 1 || col_cnt < 2) ? {DATA_WIDTH{1'b0}} : tap1_2;
            win_11 <= (row_cnt < 1 || col_cnt < 1) ? {DATA_WIDTH{1'b0}} : tap1_1;
            win_12 <= (row_cnt < 1)                ? {DATA_WIDTH{1'b0}} : tap1_0;

            win_20 <= (col_cnt < 2)                ? {DATA_WIDTH{1'b0}} : tap2_2;
            win_21 <= (col_cnt < 1)                ? {DATA_WIDTH{1'b0}} : tap2_1;
            win_22 <= in_pixel;

            // Continuous 1-to-1 streaming valid
            win_valid <= 1'b1;

            // Raster increment
            if (col_cnt == IMG_WIDTH - 1) begin
                col_cnt <= 16'd0;
                if (row_cnt == IMG_HEIGHT - 1)
                    row_cnt <= 16'd0;
                else
                    row_cnt <= row_cnt + 16'd1;
            end else begin
                col_cnt <= col_cnt + 16'd1;
            end
        end else begin
            win_valid <= 1'b0;
        end
    end

endmodule
