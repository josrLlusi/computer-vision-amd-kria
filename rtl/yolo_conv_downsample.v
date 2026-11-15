`timescale 1ns / 1ps

module yolo_conv_downsample #(
    parameter IN_CH      = 256,
    parameter OUT_CH     = 256,
    parameter DATA_WIDTH = 8,
    parameter IN_WIDTH   = 80,
    parameter IN_HEIGHT  = 80
)(
    input  wire                           clk,
    input  wire                           rst_n,

    input  wire [(IN_CH*DATA_WIDTH)-1:0]  data_in,
    input  wire                           valid_in,
    output wire                           ready_out,

    output reg  [(OUT_CH*DATA_WIDTH)-1:0] data_out,
    output reg                            valid_out,
    input  wire                           ready_in
);

    reg [15:0] x_cnt;
    reg [15:0] y_cnt;

    wire out_stall = valid_out && !ready_in;
    assign ready_out = !out_stall;

    wire sample_hit = (x_cnt[0] == 1'b1) && (y_cnt[0] == 1'b1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= {(OUT_CH*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
            x_cnt     <= 16'd0;
            y_cnt     <= 16'd0;
        end else if (!out_stall) begin
            if (valid_in) begin
                if (sample_hit) begin
                    data_out  <= data_in[(OUT_CH*DATA_WIDTH)-1:0];
                    valid_out <= 1'b1;
                end else begin
                    valid_out <= 1'b0;
                end

                if (x_cnt == IN_WIDTH - 1) begin
                    x_cnt <= 16'd0;
                    if (y_cnt == IN_HEIGHT - 1)
                        y_cnt <= 16'd0;
                    else
                        y_cnt <= y_cnt + 16'd1;
                end else begin
                    x_cnt <= x_cnt + 16'd1;
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end

endmodule
