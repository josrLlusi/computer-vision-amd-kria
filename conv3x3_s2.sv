`timescale 1ns / 1ps

module conv3x3_s2 (
    input  logic               clk,
    input  logic               rst_n,
    
    // Antarmuka Aliran Masukan (Mirip AXI-Stream)
    input  logic signed [31:0] pixel_in,   // Data gambar format Q16.16
    input  logic               valid_in,
    output logic               ready_out,
    
    // Antarmuka Aliran Keluaran
    output logic signed [31:0] pixel_out,  // Hasil konvolusi format Q16.16
    output logic               valid_out
);

    // Parameter konfigurasi arsitektur
    localparam IMAGE_WIDTH = 640;
    localparam CHANNELS    = 3;
    
// --------------------------------------------------------
    // TODO 1: Deklarasi Line Buffers (BRAM Inference)
    // --------------------------------------------------------
    // Kedalaman 640 menyesuaikan lebar gambar. Sintesis FPGA (Quartus/Vivado) 
    // akan otomatis memetakan array ini ke dalam Block RAM (BRAM).
    logic signed [31:0] line_buf_0 [0:IMAGE_WIDTH-1];
    logic signed [31:0] line_buf_1 [0:IMAGE_WIDTH-1];
    
    // Pointer penulisan untuk sirkuit Circular Buffer
    logic [9:0] col_ptr; 
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            col_ptr <= '0;
        end else if (valid_in) begin
            // Alirkan data secara vertikal dari bawah ke atas
            line_buf_0[col_ptr] <= pixel_in;
            line_buf_1[col_ptr] <= line_buf_0[col_ptr];
            
            // Reset pointer jika sudah mencapai ujung baris
            if (col_ptr == IMAGE_WIDTH - 1)
                col_ptr <= '0;
            else
                col_ptr <= col_ptr + 1;
        end
    end
    
    // --------------------------------------------------------
    // TODO 2: Shift Registers untuk Window 3x3
    // --------------------------------------------------------
    logic signed [31:0] window_3x3 [0:2][0:2];
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i=0; i<3; i++) begin
                for (int j=0; j<3; j++) begin
                    window_3x3[i][j] <= '0;
                end
            end
        end else if (valid_in) begin
            // Geser horizontal: Baris 0 (Paling Atas) - dari line_buf_1
            window_3x3[0][2] <= window_3x3[0][1];
            window_3x3[0][1] <= window_3x3[0][0];
            window_3x3[0][0] <= line_buf_1[col_ptr];
            
            // Geser horizontal: Baris 1 (Tengah) - dari line_buf_0
            window_3x3[1][2] <= window_3x3[1][1];
            window_3x3[1][1] <= window_3x3[1][0];
            window_3x3[1][0] <= line_buf_0[col_ptr];
            
            // Geser horizontal: Baris 2 (Bawah/Baru) - langsung dari pixel_in
            window_3x3[2][2] <= window_3x3[2][1];
            window_3x3[2][1] <= window_3x3[2][0];
            window_3x3[2][0] <= pixel_in;
        end
    end
    
// --------------------------------------------------------
    // TODO 3: Array Bobot (Weights) - Memuat dari .hex
    // --------------------------------------------------------
    // Mendeklarasikan 9 bobot untuk satu filter 3x3.
    // Dalam FPGA fisik, sintesis akan mengubah ini menjadi ROM internal.
    logic signed [31:0] weights [0:8];
    
    initial begin
        // Membaca file hex yang dihasilkan oleh skrip Python Anda
        $readmemh("tb_weight_3x3.hex", weights);
    end
    
    // --------------------------------------------------------
    // TODO 4: Logika Multiply-Accumulate (MAC) & Stride 2 FSM
    // --------------------------------------------------------
    logic signed [63:0] mac_temp;
    logic [9:0] pixel_x, pixel_y; // Pelacak koordinat piksel
    
    // Operasi MAC Kombinasional (Sirkuit DSP)
    // Mengalikan 9 piksel di window_3x3 dengan 9 bobot
    always_comb begin
        mac_temp = (window_3x3[0][0] * weights[0]) +
                   (window_3x3[0][1] * weights[1]) +
                   (window_3x3[0][2] * weights[2]) +
                   (window_3x3[1][0] * weights[3]) +
                   (window_3x3[1][1] * weights[4]) +
                   (window_3x3[1][2] * weights[5]) +
                   (window_3x3[2][0] * weights[6]) +
                   (window_3x3[2][1] * weights[7]) +
                   (window_3x3[2][2] * weights[8]);
    end

    // FSM Kontroler Stride 2 & Register Keluaran
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pixel_x   <= '0;
            pixel_y   <= '0;
            valid_out <= 1'b0;
            pixel_out <= '0;
        end else if (valid_in) begin
            // 1. Melacak Koordinat X dan Y
            if (pixel_x == IMAGE_WIDTH - 1) begin
                pixel_x <= '0;
                pixel_y <= pixel_y + 1;
            end else begin
                pixel_x <= pixel_x + 1;
            end

            // 2. Evaluasi Stride 2 & Eksekusi Keluaran
            // Stride 2 berarti kita hanya mengambil hasil saat X genap dan Y genap.
            // Kita juga harus memastikan window sudah terisi penuh (Y >= 2 dan X >= 2).
            if (pixel_y >= 2 && pixel_x >= 2 && 
               (pixel_x[0] == 1'b0) && (pixel_y[0] == 1'b0)) begin
                
                // Pemotongan Q16.16: Mengambil bit [47:16] dari hasil 64-bit
                pixel_out <= mac_temp[47:16]; 
                valid_out <= 1'b1;
            end else begin
                // Matikan sinyal valid_out untuk piksel ganjil (dilewati oleh stride)
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule