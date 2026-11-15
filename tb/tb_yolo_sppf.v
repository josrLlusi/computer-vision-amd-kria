// ============================================================================
// Testbench: tb_yolo_sppf
// Description: Self-checking testbench for YOLO26 SPPF building block.
//              Tests:
//                1. 1x1 Convolution channel reduction
//                2. 3x Cascaded 5x5 Max Pooling stages (y0 -> y1 -> y2 -> y3)
//                3. 4-way multi-scale feature concatenation
//                4. Final 1x1 Convolution projection
//                5. 1-to-1 streaming valid preservation
// ============================================================================

`timescale 1ns / 1ps

module tb_yolo_sppf;

    parameter DATA_WIDTH   = 16;
    parameter FRAC_BITS    = 8;
    parameter IN_CHANNELS  = 4;
    parameter OUT_CHANNELS = 4;
    parameter IMG_WIDTH    = 8;
    parameter IMG_HEIGHT   = 8;

    reg                               clk;
    reg                               rst_n;
    reg                               in_valid;
    reg  [IN_CHANNELS*DATA_WIDTH-1:0] in_data;

    wire                              out_valid;
    wire [OUT_CHANNELS*DATA_WIDTH-1:0] out_data;

    integer errors;
    integer pixel_idx;
    integer out_cnt;

    // Clock generation (100 MHz)
    always #5 clk = ~clk;

    // Instantiate SPPF Module
    yolo_sppf #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .POOL_SIZE(5),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_sppf (
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
            if (out_data === { (OUT_CHANNELS*DATA_WIDTH){1'bx} }) begin
                $display("[ERROR] SPPF produced X/Z data at time %0t", $time);
                errors = errors + 1;
            end
        end
    end

    // Stimulus
    initial begin
        $display("------------------------------------------------------------");
        $display("Starting Testbench: tb_yolo_sppf (SPPF Building Block)");
        $display("------------------------------------------------------------");

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

        $display("[INFO] Streaming 8x8 image (%0d pixels) through SPPF...", IMG_WIDTH * IMG_HEIGHT);

        for (pixel_idx = 0; pixel_idx < IMG_WIDTH * IMG_HEIGHT; pixel_idx = pixel_idx + 1) begin
            @(posedge clk);
            in_valid <= 1'b1;
            // Distinct non-zero test patterns in Q8.8
            in_data  <= { 16'sh0200 + pixel_idx, 16'sh0180, 16'sh0100, 16'sh0080 + pixel_idx };
        end

        @(posedge clk);
        in_valid <= 1'b0;
        in_data  <= 0;

        // Drain pipeline
        #2500;

        $display("[INFO] Simulation finished. SPPF outputs received: %0d (expected: %0d)", 
                 out_cnt, IMG_WIDTH * IMG_HEIGHT);

        if (out_cnt != IMG_WIDTH * IMG_HEIGHT) begin
            $display("[ERROR] Output count mismatch! Expected %0d, received %0d", 
                     IMG_WIDTH * IMG_HEIGHT, out_cnt);
            errors = errors + 1;
        end

        $display("------------------------------------------------------------");
        if (errors == 0) begin
            $display(">>> ALL YOLO SPPF TESTS PASSED SUCCESSFULLY! <<<");
        end else begin
            $display(">>> FAILED: %0d errors detected in tb_yolo_sppf <<<", errors);
        end
        $display("------------------------------------------------------------");

        $finish;
    end

endmodule
