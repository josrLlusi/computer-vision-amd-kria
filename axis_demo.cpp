#include <yolo_types.hpp>

void axis_demo(
    hls::stream<axis8_t> &s_in,
    hls::stream<axis8_t> &s_out,
    int N
) {
#pragma HLS INTERFACE axis port=s_in
#pragma HLS INTERFACE axis port=s_out
#pragma HLS INTERFACE s_axilite port=N bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1

        axis8_t input = s_in.read();

        ap_int<8> x = input.data;
        ap_int<8> y = x << 1;

        axis8_t output;
        output.data = y;
        output.keep = -1;
        output.strb = -1;
        output.last = (i == N - 1);

        s_out.write(output);
    }
}