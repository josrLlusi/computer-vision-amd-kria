`timescale 1ns / 1ps

module yolo_concat #(
    parameter CH_A       = 512,
    parameter CH_B       = 256,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,

    input wire [(CH_A*DATA_WIDTH)-1:0] data_a,
    input wire                         valid_a,
    output wire                        ready_a,

    input wire [(CH_B*DATA_WIDTH)-1:0] data_b,
    input wire                         valid_b,
    output wire                        ready_b,

    output reg [((CH_A+CH_B)*DATA_WIDTH)-1:0] data_out,
    output reg                                valid_out,
    input wire                                ready_in
);

    wire both_valid = valid_a && valid_b;
    assign ready_a  = (ready_in || !valid_out) && both_valid;
    assign ready_b  = ready_a;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= 0;
            valid_out <= 0;
        end else begin
            if (ready_in || !valid_out) begin
                if (both_valid) begin
                    data_out  <= {data_a, data_b};
                    valid_out <= 1'b1;
                end else begin
                    valid_out <= 1'b0;
                end
            end
        end
    end

endmodule