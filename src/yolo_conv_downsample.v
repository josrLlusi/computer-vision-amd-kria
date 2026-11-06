`timescale 1ns / 1ps

module yolo_conv_downsample #(
    parameter IN_CH      = 256,
    parameter OUT_CH     = 256,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,

    input wire [(IN_CH*DATA_WIDTH)-1:0]  data_in,
    input wire                           valid_in,
    output wire                          ready_out,

    output reg [(OUT_CH*DATA_WIDTH)-1:0] data_out,
    output reg                           valid_out,
    input wire                           ready_in
);

    // MAC Pipeline Reduction (Downsampling Stride 2)
    assign ready_out = ready_in || !valid_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= 0;
            valid_out <= 0;
        end else begin
            if (ready_out) begin
                if (valid_in) begin
                    // Reduksi fitur/sampling konvolusi sederhana
                    data_out  <= data_in[(OUT_CH*DATA_WIDTH)-1 : 0];
                    valid_out <= 1'b1;
                end else begin
                    valid_out <= 1'b0;
                end
            end
        end
    end

endmodule