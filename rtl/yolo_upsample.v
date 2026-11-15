`timescale 1ns / 1ps

module yolo_upsample #(
    parameter CHANNELS   = 512,
    parameter DATA_WIDTH = 8,
    parameter IN_WIDTH   = 20,
    parameter IN_HEIGHT  = 20
)(
    input  wire                             clk,
    input  wire                             rst_n,

    input  wire [(CHANNELS*DATA_WIDTH)-1:0] data_in,
    input  wire                             valid_in,
    output wire                             ready_out,

    output reg  [(CHANNELS*DATA_WIDTH)-1:0] data_out,
    output reg                              valid_out,
    input  wire                             ready_in
);

    localparam int MAX_W = (IN_WIDTH > 64) ? IN_WIDTH : 64;

    reg [(CHANNELS*DATA_WIDTH)-1:0] line_buf [0:MAX_W-1];
    reg [15:0] x_cnt;
    reg [15:0] y_cnt;
    reg        rep_x;
    reg        rep_y;

    wire out_stall = valid_out && !ready_in;
    // Hanya terima piksel input baru pada pass baris pertama (rep_y == 0) dan duplikasi horizontal pertama (rep_x == 0)
    assign ready_out = !out_stall && (rep_y == 1'b0) && (rep_x == 1'b0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out  <= {(CHANNELS*DATA_WIDTH){1'b0}};
            valid_out <= 1'b0;
            x_cnt     <= 16'd0;
            y_cnt     <= 16'd0;
            rep_x     <= 1'b0;
            rep_y     <= 1'b0;
        end else if (!out_stall) begin
            if (rep_y == 1'b0) begin
                if (rep_x == 1'b0) begin
                    if (valid_in) begin
                        line_buf[x_cnt] <= data_in;
                        data_out        <= data_in;
                        valid_out       <= 1'b1;
                        rep_x           <= 1'b1;
                    end else begin
                        valid_out <= 1'b0;
                    end
                end else begin
                    // Duplikasi horizontal kedua dari piksel yang sama
                    data_out  <= line_buf[x_cnt];
                    valid_out <= 1'b1;
                    rep_x     <= 1'b0;
                    if (x_cnt == IN_WIDTH - 1) begin
                        x_cnt <= 16'd0;
                        rep_y <= 1'b1;
                    end else begin
                        x_cnt <= x_cnt + 16'd1;
                    end
                end
            end else begin
                // Replay baris dari line_buf untuk duplikasi vertikal kedua
                data_out  <= line_buf[x_cnt];
                valid_out <= 1'b1;
                if (rep_x == 1'b0) begin
                    rep_x <= 1'b1;
                end else begin
                    rep_x <= 1'b0;
                    if (x_cnt == IN_WIDTH - 1) begin
                        x_cnt <= 16'd0;
                        rep_y <= 1'b0;
                        if (y_cnt == IN_HEIGHT - 1)
                            y_cnt <= 16'd0;
                        else
                            y_cnt <= y_cnt + 16'd1;
                    end else begin
                        x_cnt <= x_cnt + 16'd1;
                    end
                end
            end
        end
    end

endmodule
