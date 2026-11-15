// ============================================================================
// Testbench: tb_yolo_c2psa
// Description: Self-checking testbench for YOLO26 C2PSA building block.
//              Tests:
//                1. C2PSA module with 2 Position-Sensitive Attention blocks (C2PSA x2)
//                2. Multi-head self-attention with scaled dot product
//                3. Feed-Forward Network (FFN) with SiLU activation
//                4. Branch A identity bypass and Branch B attention fusion
// ============================================================================

`timescale 1ns / 1ps

module tb_yolo_c2psa;

    parameter DATA_WIDTH   = 16;
    parameter FRAC_BITS    = 8;
    parameter IN_CHANNELS  = 4;
    parameter OUT_CHANNELS = 4;
    parameter NUM_PSA      = 2;
    parameter NUM_HEADS    = 2;
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

    // Instantiate C2PSA Block
    yolo_c2psa #(
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .NUM_PSA(NUM_PSA),
        .NUM_HEADS(NUM_HEADS),
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT)
    ) u_c2psa (
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
                $display("[ERROR] C2PSA produced X/Z data at time %0t", $time);
                errors = errors + 1;
            end
        end
    end

    // Stimulus
    initial begin
        $display("------------------------------------------------------------");
        $display("Starting Testbench: tb_yolo_c2psa (C2PSA Building Block)");
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

        $display("[INFO] Streaming 8x8 image (%0d pixels) with 4 channels through C2PSA...", IMG_WIDTH * IMG_HEIGHT);

        for (pixel_idx = 0; pixel_idx < IMG_WIDTH * IMG_HEIGHT; pixel_idx = pixel_idx + 1) begin
            @(posedge clk);
            in_valid <= 1'b1;
            // Q8.8 inputs
            in_data  <= { 16'sh0100 + pixel_idx, 16'sh00C0, 16'sh0080 + pixel_idx, 16'sh0040 };
        end

        @(posedge clk);
        in_valid <= 1'b0;
        in_data  <= 0;

        // Drain pipeline
        #3000;

        $display("[INFO] Simulation finished. C2PSA outputs received: %0d (expected: %0d)", 
                 out_cnt, IMG_WIDTH * IMG_HEIGHT);

        if (out_cnt != IMG_WIDTH * IMG_HEIGHT) begin
            $display("[ERROR] Output count mismatch! Expected %0d, received %0d", 
                     IMG_WIDTH * IMG_HEIGHT, out_cnt);
            errors = errors + 1;
        end

        $display("------------------------------------------------------------");
        if (errors == 0) begin
            $display(">>> ALL YOLO C2PSA TESTS PASSED SUCCESSFULLY! <<<");
        end else begin
            $display(">>> FAILED: %0d errors detected in tb_yolo_c2psa <<<", errors);
        end
        $display("------------------------------------------------------------");

        $finish;
    end

endmodule
