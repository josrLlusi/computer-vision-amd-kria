`timescale 1ns/1ps
// Simple dual-port RAM: 1 write port, 1 read port, synchronous read (1-cycle latency).
// Infers block RAM. In simulation the testbench preloads it with $readmemh on `mem`.
module ram_sync #(
    parameter DW = 16,
    parameter AW = 16
)(
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata
);
    reg [DW-1:0] mem [0:(1<<AW)-1];
    integer i;
    initial begin
        for (i = 0; i < (1<<AW); i = i + 1) mem[i] = {DW{1'b0}};
        rdata = {DW{1'b0}};
    end
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[raddr];
    end
endmodule
