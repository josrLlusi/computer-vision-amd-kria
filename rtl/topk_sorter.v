`timescale 1ns/1ps
// Streaming top-K insertion sorter (replaces NMS in the YOLO26 end-to-end head).
//  * keeps the K largest scores seen since `clear`, sorted descending
//  * one push per cycle; every entry compares against the new score in parallel
//  * ties: an equal score is placed AFTER existing equal ones (earlier push wins)
//  * `limit` (<= K) is a run-time cap, used because k = min(max_det, number_of_candidates)
module topk_sorter #(
    parameter K  = 300,
    parameter SW = 8,      // score width (signed)
    parameter PW = 16,     // payload width
    parameter IW = 9       // index width, must satisfy 2**IW >= K
)(
    input  wire                 clk,
    input  wire                 clear,
    input  wire [IW:0]          limit,
    input  wire                 push,
    input  wire signed [SW-1:0] score,
    input  wire [PW-1:0]        payload,
    output reg  [IW:0]          count,
    input  wire [IW-1:0]        rd_idx,
    output wire signed [SW-1:0] rd_score,
    output wire [PW-1:0]        rd_payload
);
    reg signed [SW-1:0] sc [0:K-1];
    reg        [PW-1:0] pl [0:K-1];
    reg        [K-1:0]  vd;
    wire       [K-1:0]  lt;      // lt[i]: new score belongs at, or before, slot i

    genvar g;
    generate
        for (g = 0; g < K; g = g + 1) begin : G_CMP
            assign lt[g] = (g < limit) && (!vd[g] || (sc[g] < score));
        end
    endgenerate

    integer i;
`ifndef SYNTHESIS
    initial begin
        for (i = 0; i < K; i = i + 1) begin sc[i] = 0; pl[i] = 0; end
        vd = {K{1'b0}};
        count = 0;
    end
`endif

    always @(posedge clk) begin
        if (clear) begin
            vd    <= {K{1'b0}};
            count <= 0;
        end else if (push) begin
            for (i = 0; i < K; i = i + 1) begin
                if (i < limit) begin
                    if (i > 0 && lt[i-1]) begin         // insertion happened above me: shift down
                        sc[i] <= sc[i-1]; pl[i] <= pl[i-1]; vd[i] <= vd[i-1];
                    end else if (lt[i]) begin           // I am the insertion point
                        sc[i] <= score;   pl[i] <= payload; vd[i] <= 1'b1;
                    end
                end
            end
            if (count < limit) count <= count + 1'b1;
        end
    end

    assign rd_score   = sc[rd_idx];
    assign rd_payload = pl[rd_idx];
endmodule
