// ============================================================================
// Module: yolo_silu
// Description: Fixed-Point SiLU (Swish-1) Activation Function for YOLO26
//              Computes: y = x * sigmoid(x) = x / (1 + exp(-x))
//              Uses 16-bit Signed Q8.8 Fixed-Point format.
//              Optimized for AMD Kria KV260/KR260 (Zynq UltraScale+).
// Latency: 2 clock cycles (pipelined)
// ============================================================================

`timescale 1ns / 1ps

module yolo_silu #(
    parameter DATA_WIDTH = 16,  // Fixed-point signed width (Q8.8)
    parameter FRAC_BITS  = 8    // Fractional bits
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   in_valid,
    input  wire signed [DATA_WIDTH-1:0] in_data,
    output reg                    out_valid,
    output reg  signed [DATA_WIDTH-1:0] out_data
);

    // Q8.8 thresholds:
    // +4.0 is 4 * 256 = +1024 (16'sh0400)
    // -4.0 is -4 * 256 = -1024 (-16'sh0400 = 16'shFC00)
    localparam signed [DATA_WIDTH-1:0] UPPER_BOUND = 16'sh0400; // +4.0
    localparam signed [DATA_WIDTH-1:0] LOWER_BOUND = -16'sh0400; // -4.0

    // Stage 1 registers
    reg                    s1_valid;
    reg signed [DATA_WIDTH-1:0] s1_in_data;
    reg signed [DATA_WIDTH-1:0] s1_lut_val;
    reg [1:0]              s1_region; // 0: <= -4.0, 1: [-4.0, +4.0], 2: >= +4.0

    // Intermediate calculation for LUT index
    wire signed [DATA_WIDTH-1:0] shifted_x = in_data + UPPER_BOUND;
    wire [5:0] raw_idx = shifted_x[DATA_WIDTH-1] ? 6'd0 : shifted_x[10:5];

    // 64-entry LUT covering [-4.0, +4.0] with step 0.125 in Q8.8
    reg signed [DATA_WIDTH-1:0] lut_val;
    always @(*) begin
        case (raw_idx)
            6'd 0: lut_val = 16'shFFED; // x=-3.938, y=-0.075
            6'd 1: lut_val = 16'shFFEB; // x=-3.812, y=-0.082
            6'd 2: lut_val = 16'shFFE9; // x=-3.688, y=-0.090
            6'd 3: lut_val = 16'shFFE7; // x=-3.562, y=-0.098
            6'd 4: lut_val = 16'shFFE5; // x=-3.438, y=-0.107
            6'd 5: lut_val = 16'shFFE2; // x=-3.312, y=-0.116
            6'd 6: lut_val = 16'shFFE0; // x=-3.188, y=-0.126
            6'd 7: lut_val = 16'shFFDD; // x=-3.062, y=-0.137
            6'd 8: lut_val = 16'shFFDA; // x=-2.938, y=-0.148
            6'd 9: lut_val = 16'shFFD7; // x=-2.812, y=-0.159
            6'd10: lut_val = 16'shFFD4; // x=-2.688, y=-0.171
            6'd11: lut_val = 16'shFFD1; // x=-2.562, y=-0.183
            6'd12: lut_val = 16'shFFCE; // x=-2.438, y=-0.196
            6'd13: lut_val = 16'shFFCB; // x=-2.312, y=-0.208
            6'd14: lut_val = 16'shFFC8; // x=-2.188, y=-0.221
            6'd15: lut_val = 16'shFFC4; // x=-2.062, y=-0.233
            6'd16: lut_val = 16'shFFC2; // x=-1.938, y=-0.244
            6'd17: lut_val = 16'shFFBF; // x=-1.812, y=-0.254
            6'd18: lut_val = 16'shFFBD; // x=-1.688, y=-0.263
            6'd19: lut_val = 16'shFFBB; // x=-1.562, y=-0.271
            6'd20: lut_val = 16'shFFB9; // x=-1.438, y=-0.276
            6'd21: lut_val = 16'shFFB9; // x=-1.312, y=-0.278
            6'd22: lut_val = 16'shFFB9; // x=-1.188, y=-0.278
            6'd23: lut_val = 16'shFFBA; // x=-1.062, y=-0.273
            6'd24: lut_val = 16'shFFBC; // x=-0.938, y=-0.264
            6'd25: lut_val = 16'shFFC0; // x=-0.812, y=-0.250
            6'd26: lut_val = 16'shFFC5; // x=-0.688, y=-0.230
            6'd27: lut_val = 16'shFFCC; // x=-0.562, y=-0.204
            6'd28: lut_val = 16'shFFD4; // x=-0.438, y=-0.172
            6'd29: lut_val = 16'shFFDE; // x=-0.312, y=-0.132
            6'd30: lut_val = 16'shFFEA; // x=-0.188, y=-0.085
            6'd31: lut_val = 16'shFFF8; // x=-0.062, y=-0.030
            6'd32: lut_val = 16'sh0008; // x=+0.062, y=+0.032
            6'd33: lut_val = 16'sh001A; // x=+0.188, y=+0.103
            6'd34: lut_val = 16'sh002E; // x=+0.312, y=+0.180
            6'd35: lut_val = 16'sh0044; // x=+0.438, y=+0.266
            6'd36: lut_val = 16'sh005C; // x=+0.562, y=+0.358
            6'd37: lut_val = 16'sh0075; // x=+0.688, y=+0.457
            6'd38: lut_val = 16'sh0090; // x=+0.812, y=+0.563
            6'd39: lut_val = 16'sh00AC; // x=+0.938, y=+0.674
            6'd40: lut_val = 16'sh00CA; // x=+1.062, y=+0.790
            6'd41: lut_val = 16'sh00E9; // x=+1.188, y=+0.910
            6'd42: lut_val = 16'sh0109; // x=+1.312, y=+1.034
            6'd43: lut_val = 16'sh0129; // x=+1.438, y=+1.162
            6'd44: lut_val = 16'sh014B; // x=+1.562, y=+1.292
            6'd45: lut_val = 16'sh016D; // x=+1.688, y=+1.424
            6'd46: lut_val = 16'sh018F; // x=+1.812, y=+1.558
            6'd47: lut_val = 16'sh01B2; // x=+1.938, y=+1.694
            6'd48: lut_val = 16'sh01D4; // x=+2.062, y=+1.830
            6'd49: lut_val = 16'sh01F8; // x=+2.188, y=+1.967
            6'd50: lut_val = 16'sh021B; // x=+2.312, y=+2.104
            6'd51: lut_val = 16'sh023E; // x=+2.438, y=+2.242
            6'd52: lut_val = 16'sh0261; // x=+2.562, y=+2.379
            6'd53: lut_val = 16'sh0284; // x=+2.688, y=+2.516
            6'd54: lut_val = 16'sh02A7; // x=+2.812, y=+2.653
            6'd55: lut_val = 16'sh02CA; // x=+2.938, y=+2.790
            6'd56: lut_val = 16'sh02ED; // x=+3.062, y=+2.926
            6'd57: lut_val = 16'sh0310; // x=+3.188, y=+3.061
            6'd58: lut_val = 16'sh0332; // x=+3.312, y=+3.196
            6'd59: lut_val = 16'sh0355; // x=+3.438, y=+3.330
            6'd60: lut_val = 16'sh0377; // x=+3.562, y=+3.464
            6'd61: lut_val = 16'sh0399; // x=+3.688, y=+3.597
            6'd62: lut_val = 16'sh03BB; // x=+3.812, y=+3.730
            6'd63: lut_val = 16'sh03DD; // x=+3.938, y=+3.862
            default: lut_val = 16'sh0000;
        endcase
    end

    // Pipeline Stage 1: Register input and LUT lookup
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid   <= 1'b0;
            s1_in_data <= {DATA_WIDTH{1'b0}};
            s1_lut_val <= {DATA_WIDTH{1'b0}};
            s1_region  <= 2'd0;
        end else begin
            s1_valid   <= in_valid;
            s1_in_data <= in_data;
            s1_lut_val <= lut_val;
            if (in_data <= LOWER_BOUND)
                s1_region <= 2'd0; // Saturation to 0
            else if (in_data >= UPPER_BOUND)
                s1_region <= 2'd2; // Pass-through x
            else
                s1_region <= 2'd1; // LUT interpolation
        end
    end

    // Pipeline Stage 2: Mux and register output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= {DATA_WIDTH{1'b0}};
        end else begin
            out_valid <= s1_valid;
            case (s1_region)
                2'd0: out_data <= {DATA_WIDTH{1'b0}}; // 0 for x <= -4.0
                2'd1: out_data <= s1_lut_val;        // LUT value for -4.0 < x < +4.0
                2'd2: out_data <= s1_in_data;        // x for x >= +4.0
                default: out_data <= {DATA_WIDTH{1'b0}};
            endcase
        end
    end

endmodule
