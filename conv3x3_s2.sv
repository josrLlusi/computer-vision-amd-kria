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
    logic [9:0] pixel_x, pixel_y;
    
    // --- PERBAIKAN PIPELINE: Register Penunda (Delay) 1 Siklus ---
    // Karena window_3x3 baru terisi penuh 1 siklus SETELAH pixel masuk,
    // kita harus menunda koordinat X dan Y agar sejajar dengan isi matriks.
    logic [9:0] win_x, win_y;
    logic win_valid;
    
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            win_x <= '0;
            win_y <= '0;
            win_valid <= 1'b0;
        end else begin
            // Menyalin koordinat saat ini untuk digunakan di siklus berikutnya
            win_x <= pixel_x;
            win_y <= pixel_y;
            win_valid <= valid_in;
        end
    end

    // Alokasi memori perantara untuk hasil perkalian 64-bit
    logic signed [63:0] p [0:8];
    
    always_comb begin
        // Perkalian Kombinasional MAC
        p[0] = window_3x3[0][2] * weights[0];
        p[1] = window_3x3[0][1] * weights[1];
        p[2] = window_3x3[0][0] * weights[2];
        
        p[3] = window_3x3[1][2] * weights[3];
        p[4] = window_3x3[1][1] * weights[4];
        p[5] = window_3x3[1][0] * weights[5];
        
        p[6] = window_3x3[2][2] * weights[6];
        p[7] = window_3x3[2][1] * weights[7];
        p[8] = window_3x3[2][0] * weights[8];
        
        mac_temp = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7] + p[8];
    end

    // FSM Pelacak Koordinat Input (pixel_x, pixel_y) - Bergerak lebih dulu
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            pixel_x <= '0;
            pixel_y <= '0;
        end else if (valid_in) begin
            if (pixel_x == IMAGE_WIDTH - 1) begin
                pixel_x <= '0;
                pixel_y <= pixel_y + 1;
            end else begin
                pixel_x <= pixel_x + 1;
            end
        end
    end

    // FSM Evaluasi Stride 2 & Keluaran - Dieksekusi berdasarkan waktu yang ditunda (win_x, win_y)
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            pixel_out <= '0;
        end else if (win_valid) begin
            // Menggunakan win_y dan win_x yang sudah selaras dengan data matriks
            if (win_y >= 2 && win_x >= 2 && 
               (win_x[0] == 1'b0) && (win_y[0] == 1'b0)) begin
                
                pixel_out <= mac_temp[47:16]; 
                valid_out <= 1'b1;
            end else begin
                valid_out <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule