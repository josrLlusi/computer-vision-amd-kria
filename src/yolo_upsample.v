`timescale 1ns / 1ps

module yolo_upsample #(
    parameter CHANNELS   = 512,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,

    input wire [(CHANNELS*DATA_WIDTH)-1:0] data_in,
    input wire                            valid_in,
    output wire                           ready_out,

    output reg [(CHANNELS*DATA_WIDTH)-1:0] data_out,
    output reg                            valid_out,
    input wire                            ready_in
);

    reg [1:0] repeat_cnt;

    assign ready_out = ready_in || !valid_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out   <= 0;
            valid_out  <= 0;
            repeat_cnt <= 0;
        end else begin
            if (ready_out) begin
                if (valid_in) begin
                    data_out   <= data_in;
                    valid_out  <= 1'b1;
                    repeat_cnt <= repeat_cnt + 1'b1;
                end else begin
                    valid_out  <= 1'b0;
                end
            end
        end
    end

endmodule