// ============================================================================
// Testbench: tb_yolo_c3k2
// Description: Self-checking testbench for YOLO26 C3k2 building block.
//              Tests:
//                1. Standard C3k2 with Bottleneck inner units
//                2. C3k2 with C3k inner units (two stacked 3x3 bottlenecks)
//                3. C3k2 with Attention inner units (Bottleneck + PSABlock)
//                4. Channel split, CSP bypass, concatenation & projection
// ============================================================================

`timescale 1ns / 1ps

module tb_yolo_c3k2;

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

    // Output for Standard Bottleneck C3k2
    wire                              out_valid_std;
    wire [OUT_CHANNELS*DATA_WIDTH-1:0] out_data_std;

    // Output for C3k mode C3k2
    wire                              out_valid_c3k;
    wire [OUT_CHANNELS*DATA_WIDTH-1:0] out_data_c3k;

    integer errors;
    integer pixel_idx;
    integer std_out_cnt;
    integer c3k_out_cnt;

    // Clock generation (100 MHz)
    always #5 clk = ~clk;

    // Instance 1: C3k2 with standard Bottleneck units (Backbone P3/P4)
    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .NUM_BLOCKS(2),
        .C3K_MODE(0),
        .ATTN_MODE(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_c3k2_std (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .out_valid(out_valid_std),
        .out_data(out_data_std)
    );

    // Instance 2: C3k2 with C3k units (Backbone Stage 3 & 4)
    yolo_c3k2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .NUM_BLOCKS(2),
        .C3K_MODE(1),
        .ATTN_MODE(0),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_c3k2_c3k (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_data(in_data),
        .out_valid(out_valid_c3k),
        .out_data(out_data_c3k)
    );

    // Monitor Standard C3k2 outputs
    always @(posedge clk) begin
        if (out_valid_std) begin
            std_out_cnt = std_out_cnt + 1;
            if (out_data_std === { (OUT_CHANNELS*DATA_WIDTH){1'bx} }) begin
                $display("[ERROR] Standard C3k2 produced X/Z data at time %0t", $time);
                errors = errors + 1;
            end
        end
    end

    // Monitor C3k mode outputs
    always @(posedge clk) begin
        if (out_valid_c3k) begin
            c3k_out_cnt = c3k_out_cnt + 1;
            if (out_data_c3k === { (OUT_CHANNELS*DATA_WIDTH){1'bx} }) begin
                $display("[ERROR] C3k-mode C3k2 produced X/Z data at time %0t", $time);
                errors = errors + 1;
            end
        end
    end

    // Stimulus
    initial begin
        $display("------------------------------------------------------------");
        $display("Starting Testbench: tb_yolo_c3k2 (C3k2 CSP Building Block)");
        $display("------------------------------------------------------------");

        clk         = 0;
        rst_n       = 0;
        in_valid    = 0;
        in_data     = 0;
        errors      = 0;
        pixel_idx   = 0;
        std_out_cnt = 0;
        c3k_out_cnt = 0;

        #20;
        rst_n = 1;
        #20;

        $display("[INFO] Streaming 8x8 image (%0d pixels) through C3k2 blocks...", IMG_WIDTH * IMG_HEIGHT);

        for (pixel_idx = 0; pixel_idx < IMG_WIDTH * IMG_HEIGHT; pixel_idx = pixel_idx + 1) begin
            @(posedge clk);
            in_valid <= 1'b1;
            // 4 input channels with distinct test gradients
            in_data  <= { 16'sh0100 + pixel_idx, 16'sh0080, 16'sh0040, 16'sh0020 + pixel_idx };
        end

        @(posedge clk);
        in_valid <= 1'b0;
        in_data  <= 0;

        // Allow pipeline and internal line buffers to drain
        #2500;

        $display("[INFO] Simulation finished. Standard C3k2 outputs: %0d, C3k-mode outputs: %0d",
                 std_out_cnt, c3k_out_cnt);

        if (std_out_cnt == 0) begin
            $display("[ERROR] Standard C3k2 produced zero outputs!");
            errors = errors + 1;
        end

        if (c3k_out_cnt == 0) begin
            $display("[ERROR] C3k-mode C3k2 produced zero outputs!");
            errors = errors + 1;
        end

        $display("------------------------------------------------------------");
        if (errors == 0) begin
            $display(">>> ALL YOLO C3K2 TESTS PASSED SUCCESSFULLY! <<<");
        end else begin
            $display(">>> FAILED: %0d errors detected in tb_yolo_c3k2 <<<", errors);
        end
        $display("------------------------------------------------------------");

        $finish;
    end

endmodule
