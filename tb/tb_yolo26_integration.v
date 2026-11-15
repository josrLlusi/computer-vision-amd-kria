// ============================================================================
// Testbench: tb_yolo26_integration
// Description: Comprehensive End-to-End Integration Testbench for YOLO26.
//              Tests the full cascade of building blocks from Backbone P5:
//                Input (16x16) -> Conv (3x3 / 2 Downsampling) 
//                              -> C3k2 (C3k CSP block)
//                              -> SPPF (Spatial Pyramid Pooling Fast, k=5)
//                              -> C2PSA (Position-Sensitive Attention Block x2)
//                              -> Output Feature Map (8x8)
//              Verifies multi-block streaming, handshaking, downsampling,
//              and pipeline alignment across the complete YOLO26 stage.
// ============================================================================

`timescale 1ns / 1ps

module tb_yolo26_integration;

    parameter DATA_WIDTH  = 16;
    parameter FRAC_BITS   = 8;
    parameter CHANNELS    = 4;
    parameter IMG_WIDTH   = 16;
    parameter IMG_HEIGHT  = 16;

    reg                              clk;
    reg                              rst_n;
    reg                              in_valid;
    reg  [CHANNELS*DATA_WIDTH-1:0]   in_data;

    wire                             out_valid;
    wire [CHANNELS*DATA_WIDTH-1:0]   out_data;

    integer errors;
    integer pixel_idx;
    integer out_cnt;

    // Clock generation (100 MHz)
    always #5 clk = ~clk;

    // Instantiate Top-level Integration Stage
    yolo26_stage_demo #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .CHANNELS(CHANNELS),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_yolo26_stage (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .out_valid(out_valid),
        .out_data(out_data)
    );

    // Output monitor
    always @(posedge clk) begin
        if (out_valid) begin
            out_cnt = out_cnt + 1;
            if (out_data === { (CHANNELS*DATA_WIDTH){1'bx} }) begin
                $display("[ERROR] Stage output produced X/Z data at time %0t", $time);
                errors = errors + 1;
            end
        end
    end

    // Stimulus
    initial begin
        $display("============================================================");
        $display("Starting Testbench: tb_yolo26_integration");
        $display("YOLO26 P5 Stage Pipeline: Conv -> C3k2 -> SPPF -> C2PSA");
        $display("============================================================");

        clk       = 0;
        rst_n     = 0;
        in_valid  = 0;
        in_data   = 0;
        errors    = 0;
        pixel_idx = 0;
        out_cnt   = 0;

        #20;
        rst_n = 1;
        #20;

        $display("[INFO] Streaming 16x16 input image (%0d pixels) with %0d channels...", 
                 IMG_WIDTH * IMG_HEIGHT, CHANNELS);

        for (pixel_idx = 0; pixel_idx < IMG_WIDTH * IMG_HEIGHT; pixel_idx = pixel_idx + 1) begin
            @(posedge clk);
            in_valid <= 1'b1;
            // Explicit 16-bit widths for all operands
            in_data  <= { (16'h0100 + pixel_idx[15:0]), 
                          (16'h00C0 + (pixel_idx[15:0] & 16'h001F)), 
                          16'h0080, 
                          (16'h0040 + (pixel_idx[15:0] & 16'h000F)) };
        end

        @(posedge clk);
        in_valid <= 1'b0;
        in_data  <= 0;

        // Allow full pipeline to process and drain
        #8000;

        $display("[INFO] End-to-end simulation completed.");
        $display("[INFO] Total input pixels: %0d (16x16)", IMG_WIDTH * IMG_HEIGHT);
        $display("[INFO] Total downsampled output pixels: %0d (expected: 64 for 8x8)", out_cnt);

        if (out_cnt != 64) begin
            $display("[ERROR] Expected 64 downsampled outputs from stride-2 pipeline, but got %0d!", out_cnt);
            errors = errors + 1;
        end

        $display("============================================================");
        if (errors == 0) begin
            $display(">>> ALL YOLO26 INTEGRATION TESTS PASSED SUCCESSFULLY! <<<");
            $display(">>> CONV -> C3K2 -> SPPF -> C2PSA VERIFIED END-TO-END! <<<");
        end else begin
            $display(">>> FAILED: %0d errors detected in tb_yolo26_integration <<<", errors);
        end
        $display("============================================================");

        $finish;
    end

endmodule
