`timescale 1ns / 1ps

module tb_conv3x3();

    // Sinyal Testbench
    logic clk;
    logic rst_n;
    logic signed [31:0] pixel_in;
    logic valid_in;
    logic ready_out;
    logic signed [31:0] pixel_out;
    logic valid_out;

    // Instansiasi Modul Utama (Design Under Test / DUT)
    conv3x3_s2 dut (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_in(pixel_in),
        .valid_in(valid_in),
        .ready_out(ready_out),
        .pixel_out(pixel_out),
        .valid_out(valid_out)
    );

    // Generator Clock (100 MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Variabel File I/O
    int file_in;
    int file_out;
    logic [31:0] read_data;
    int pixel_count;

    initial begin
        // Inisialisasi Sinyal
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;
        pixel_count = 0;
        
        // Membuka file Golden Data dan file Output Simulasi
        file_in = $fopen("tb_input_640x640.hex", "r");
        file_out = $fopen("sim_output.hex", "w");
        
        if (file_in == 0) begin
            $display("ERROR: File tb_input_640x640.hex tidak ditemukan!");
            $finish;
        end

        // Urutan Reset FSM
        #20 rst_n = 1;
        #10;

        $display("==================================================");
        $display("Memulai Injeksi Data Gambar ke Modul Konvolusi...");
        
        // Memompa 1 saluran (channel) pertama: 640 x 640 = 409,600 piksel
        while (!$feof(file_in) && pixel_count < 409600) begin
            @(posedge clk);
            if ($fscanf(file_in, "%h\n", read_data) == 1) begin
                pixel_in <= read_data;
                valid_in <= 1'b1;
                pixel_count++;
            end
        end

        // Menutup Aliran Data
        @(posedge clk);
        valid_in <= 1'b0; 

        // Menunggu sisa data di dalam pipa (Line Buffer) mengalir keluar
        #20000; 
        
        $display("Simulasi Selesai. Hasil disimpan di sim_output.hex");
        $display("==================================================");
        $fclose(file_in);
        $fclose(file_out);
        $finish;
    end

    // Menangkap Output dari Modul dan Menulisnya ke File
    always_ff @(posedge clk) begin
        if (valid_out) begin
            $fwrite(file_out, "%08x\n", pixel_out);
        end
    end

endmodule