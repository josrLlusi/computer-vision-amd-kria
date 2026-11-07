`timescale 1ns / 1ps
// ============================================================================
// stream_fifo.sv - FIFO sinkron dengan handshake valid/ready di kedua sisi.
// Dipakai di C3k2 untuk menahan cabang y0/y1 selama Bottleneck menghitung
// (latensi ~2 baris), sehingga residual & concat sejajar tanpa frame buffer.
// Baca "show-ahead": out_data selalu = elemen terdepan saat out_valid = 1.
// ============================================================================
module stream_fifo #(
    parameter int WIDTH = 32,
    parameter int DEPTH = 16
)(
    input  logic             clk,
    input  logic             rst_n,

    input  logic [WIDTH-1:0] in_data,
    input  logic             in_valid,
    output logic             in_ready,

    output logic [WIDTH-1:0] out_data,
    output logic             out_valid,
    input  logic             out_ready
);
    localparam int AW = (DEPTH > 1) ? $clog2(DEPTH) : 1;

    logic [WIDTH-1:0] mem [0:DEPTH-1];
    logic [AW-1:0]    wr_ptr, rd_ptr;
    logic [AW:0]      count;

    logic do_wr, do_rd;
    assign in_ready  = (count != (AW+1)'(DEPTH));
    assign out_valid = (count != '0);
    assign out_data  = mem[rd_ptr];
    assign do_wr     = in_valid  && in_ready;
    assign do_rd     = out_valid && out_ready;

    always_ff @(posedge clk) begin
        if (do_wr) mem[wr_ptr] <= in_data;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            if (do_wr) wr_ptr <= (wr_ptr == AW'(DEPTH - 1)) ? '0 : wr_ptr + 1'b1;
            if (do_rd) rd_ptr <= (rd_ptr == AW'(DEPTH - 1)) ? '0 : rd_ptr + 1'b1;
            case ({do_wr, do_rd})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end

    // Pemantau simulasi: berapa isi maksimum yang pernah dipakai
    // synthesis translate_off
    int max_fill = 0;
    always @(posedge clk) if (int'(count) > max_fill) max_fill = int'(count);
    final $display("  [stream_fifo %m] isi maksimum %0d dari %0d entri", max_fill, DEPTH);
    // synthesis translate_on

endmodule
