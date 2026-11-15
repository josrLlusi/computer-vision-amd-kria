`timescale 1ns/1ps
// True dual-port RAM, synchronous read (read-first), per-byte write enables on both ports.
// Written in the standard Vivado TDP template so it infers RAMB36 (or URAM288 with ram_style "ultra").
// W must be a multiple of 8.
module ram_tdp #(
    parameter W  = 32,
    parameter AW = 10,
    parameter DEPTH = (1<<AW)     // exact depth: Vivado sizes the RAM by DEPTH, not by 2**AW
)(
    input  wire             clk,
    input  wire [W/8-1:0]   a_we,
    input  wire [AW-1:0]    a_addr,
    input  wire [W-1:0]     a_din,
    output reg  [W-1:0]     a_dout,
    input  wire [W/8-1:0]   b_we,
    input  wire [AW-1:0]    b_addr,
    input  wire [W-1:0]     b_din,
    output reg  [W-1:0]     b_dout
);
    (* ram_style = "block" *) reg [W-1:0] mem [0:DEPTH-1];
    integer i, ia, ib;
`ifndef SYNTHESIS   // simulation only: avoid X on never-written locations
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) mem[i] = {W{1'b0}};
        a_dout = {W{1'b0}}; b_dout = {W{1'b0}};
    end
`endif
    always @(posedge clk) begin
        for (ia = 0; ia < W/8; ia = ia + 1)
            if (a_we[ia]) mem[a_addr][8*ia +: 8] <= a_din[8*ia +: 8];
        a_dout <= mem[a_addr];
    end
    always @(posedge clk) begin
        for (ib = 0; ib < W/8; ib = ib + 1)
            if (b_we[ib]) mem[b_addr][8*ib +: 8] <= b_din[8*ib +: 8];
        b_dout <= mem[b_addr];
    end
endmodule

// Same RAM for the big activation memory: UltraRAM by default (override with -define ACT_RAM_STYLE="block").
`ifndef ACT_RAM_STYLE
 `define ACT_RAM_STYLE "ultra"
`endif
module ram_tdp_act #(
    parameter W  = 128,
    parameter AW = 14,
    parameter DEPTH = (1<<AW)
)(
    input  wire             clk,
    input  wire [W/8-1:0]   a_we,
    input  wire [AW-1:0]    a_addr,
    input  wire [W-1:0]     a_din,
    output reg  [W-1:0]     a_dout,
    input  wire [W/8-1:0]   b_we,
    input  wire [AW-1:0]    b_addr,
    input  wire [W-1:0]     b_din,
    output reg  [W-1:0]     b_dout
);
    (* ram_style = `ACT_RAM_STYLE *) reg [W-1:0] mem [0:DEPTH-1];
    integer i, ia, ib;
`ifndef SYNTHESIS   // simulation only: avoid X on never-written locations
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) mem[i] = {W{1'b0}};
        a_dout = {W{1'b0}}; b_dout = {W{1'b0}};
    end
`endif
    always @(posedge clk) begin
        for (ia = 0; ia < W/8; ia = ia + 1)
            if (a_we[ia]) mem[a_addr][8*ia +: 8] <= a_din[8*ia +: 8];
        a_dout <= mem[a_addr];
    end
    always @(posedge clk) begin
        for (ib = 0; ib < W/8; ib = ib + 1)
            if (b_we[ib]) mem[b_addr][8*ib +: 8] <= b_din[8*ib +: 8];
        b_dout <= mem[b_addr];
    end
endmodule
