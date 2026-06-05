#pragma once

#include <fstream>
#include <string>
#include <stdexcept>
#include <filesystem>

#include <Kokkos_Core.hpp>

#include "./Config.hpp"
#include "./ConfigEnums.hpp"
#include "./IOBlueprints.hpp"
#include "./../core/MCMem.hpp"
#include "./../core/Typedefs.hpp"


namespace KOps::IO::Binary {
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;
    namespace KIO = KOps::IO;

    // ========================================================================
    // SINGLE SOURCE OF TRUTH: BINARY FORMAT LAYOUT
    // ========================================================================
    struct BinHeader {
        const char file_key[4] = {'K', 'O', 'P', 'T'};
        int version = 1;
        int byte_precision = sizeof(KT::Real);
        int total_n_sims;
        int n_time_steps;
        KT::Real dt; // Allows to reconstruct the grid (works only for static time grid)
    };

    // ========================================================================
    // THE BINARY WRITER RESPONSIBLE
    // ========================================================================
    class BinWriter : public KIO::WriterBlueprint {
    public:
        explicit BinWriter(const Config::UInputs &conf) : config(conf) {
            // Create paths buffer if needed
            if (!config.output.filename_paths_out.empty()) {
                full_paths_out_path = std::filesystem::path(config.output.out_dir) / config.output.filename_paths_out;

                out_paths_stream.open(full_paths_out_path.string(), std::ios::out | std::ios::binary);
                if (!out_paths_stream.is_open()) {
                    throw std::runtime_error(
                        "[BinWriter Error] Failed to open binary output file: " + full_paths_out_path.string());
                }
                write_global_header();
            }
        }

        ~BinWriter() override {
            if (out_paths_stream.is_open()) out_paths_stream.close();
        }

        void save_paths_batch_if_needed(int current_batch_size, const Engine::MCBatchMem &BatchMem) override {
            // Return without writing if no file is specified
            if (config.output.filename_paths_out.empty()) {
                return;
            }

            //Calculate exactly how many bytes this specific batch occupies (N_sims x n_t_steps x sizeReal)
            // This is flexible and allows to consider cases where the batch is not complete
            const size_t bytes_to_write = static_cast<size_t>(current_batch_size) *
                                          static_cast<size_t>(config.time.N_time_steps) * sizeof(KT::Real);

            // Fetch the raw C-style pointer from the Kokkos Host View
            const KT::Real *raw_data_ptr = BatchMem.h_batch_view.data();

            // PATH A: The Fast Track (Only runs if the layout is physically contiguous)
            if (BatchMem.is_host_contiguous() && BatchMem.is_host_row_major()) {
                // Dump it to the SSD in one hardware instruction
                out_paths_stream.write(reinterpret_cast<const char *>(raw_data_ptr), bytes_to_write);
            }

            // PATH B: The Universal Buffered Fallback
            else {
                // 1. Allocate a contiguous memory buffer for exactly one row
                std::vector<KT::Real> row_buffer(config.time.N_time_steps);
                const size_t row_bytes = config.time.N_time_steps * sizeof(KT::Real);

                // Loop through row-by-row
                for (int i = 0; i < current_batch_size; ++i) {
                    // 2. Read the scattered RAM data (whatever layout it is) into our clean buffer
                    for (size_t j = 0; j < config.time.N_time_steps; ++j) {
                        row_buffer[j] = BatchMem.h_batch_view(i, j);
                    }

                    // 3. Dump the ENTIRE ROW to the SSD in exactly one hardware instruction
                    out_paths_stream.write(reinterpret_cast<const char *>(row_buffer.data()), row_bytes);
                }
            }
        }


        void print_planned_outputs_summary() const override {
            constexpr std::string_view indent = "  ";
            std::cout << "\n" << indent << "========================================================\n";
            std::cout << indent << "               BINARY I/O WRITER PERFORMANCE            \n";
            std::cout << indent << "========================================================\n";

            if (!config.output.filename_paths_out.empty()) {
                std::cout << indent << "   Export Format        :  High-Performance Binary\n";
                std::cout << indent << "--------------------------------------------------------\n";
                std::cout << indent << " [Binary File Layout Structure]\n";
                std::cout << indent << "   |-- GLOBAL HEADER (" << sizeof(BinHeader) << " Bytes)\n";
                std::cout << indent << "   |   |-- Magic Key    : 'KOPT' (4 bytes)\n";
                std::cout << indent << "   |   |-- Version      : 1 (int32)\n";
                std::cout << indent << "   |   |-- Precision    : " << sizeof(KT::Real) <<
                        " bytes per value (int32)\n";
                std::cout << indent << "   |   |-- Total Paths  : " << config.mc.N_Paths << " (int32)\n";
                std::cout << indent << "   |   |-- Time Steps   : " << config.time.N_time_steps << " (int32)\n";
                std::cout << indent << "   |   |-- Time dt      : " << config.time.dt << " (float64)\n";
                std::cout << indent << "   |\n";
                std::cout << indent << "   |-- MATRIX PAYLOAD\n";
                std::cout << indent << "       |-- Dimensions   : " << config.mc.N_Paths << " rows x " << config.
                        time.N_time_steps << " cols\n";
                std::cout << indent << "       |-- Ordering     : Row-Major (C-Style Sequential)\n";
                std::cout << indent << "       |-- Contents     : Price time series\n";
            } else {
                std::cout << indent << "  No output files will be written (filename empty).\n";
            }
            std::cout << indent << "========================================================\n" << std::endl;
        }

    private
    :
        const KC::UInputs config;
        std::ofstream out_paths_stream;
        std::filesystem::path full_paths_out_path;

        void write_global_header() {
            // 1. Setup Header Data
            BinHeader header;
            header.total_n_sims = config.mc.N_Paths;
            header.n_time_steps = config.time.N_time_steps;
            header.dt = config.time.dt;

            // 2. Dump Header Struct directly to disk
            // reinterpret_cast forces the compiler to treat the memory address as an array of raw, unsigned bytes (const char*).
            out_paths_stream.write(reinterpret_cast<const char *>(&header), sizeof(BinHeader));
        }
    };


    // ========================================================================
    // THE BINARY READER RESPONSIBLE
    // ========================================================================
    class BinReader : public KIO::ReaderBlueprint {
    public:
        explicit BinReader(const Config::UInputs &conf) : config(conf) {
            if (!config.output.filename_paths_out.empty()) {
                load_header_paths_file();
            }
        }

        std::vector<KT::Real> read_prices_at_target_time(double current_time) override {
            if (!inp_stream.is_open()) {
                throw std::runtime_error("[BinReader Error] Cannot read prices, file stream is closed.");
            }

            // Map continuous time to discrete index (a row major layour is assumed)
            int target_col_idx = static_cast<int>(std::round(current_time / bin_header.dt)) - 1;
            // Check That required time is in time domain
            if (target_col_idx < 0 || target_col_idx >= bin_header.n_time_steps) {
                throw std::runtime_error("[BinReader Error] Requested time " + std::to_string(current_time) +
                                         " is outside the simulated time grid bounds.");
            }
            // Check that the required time is in the time grid and do not required interpolation
            double actual_grid_time = (target_col_idx + 1) * bin_header.dt;
            if (std::abs(current_time - actual_grid_time) > 1e-10) {
                throw std::runtime_error("Requested time  " + std::to_string(current_time) +
                                         " does not align perfectly with the discrete simulation grid (dt = " +
                                         std::to_string(bin_header.dt) + ")."
                );
            }

            // Allocate the vector containing the prices at target times
            std::vector<KT::Real> target_prices(bin_header.total_n_sims);

            // Calculate the memory strides
            std::streampos start_offset_bytes = sizeof(BinHeader) + (target_col_idx * sizeof(KT::Real)); //Absolute location in a file from beginning
            std::streamoff row_stride_bytes = bin_header.n_time_steps * sizeof(KT::Real); // jumping of a simulation (relative distance)

            // Read Prices
            for (int i = 0; i < bin_header.total_n_sims; ++i) {
                // Calculate absolute byte position for this specific element by adding a relative distance (streamoff) to the starting point
                std::streampos element_pos = start_offset_bytes + (static_cast<std::streamoff>(i) * row_stride_bytes);
                // Jump to the exact byte location on the disk (Start measuring the element position from beginning of the file)
                inp_stream.seekg(element_pos, std::ios::beg);
                // Read exactly one floating point number into our vector
                inp_stream.read(reinterpret_cast<char*>(&target_prices[i]), sizeof(KT::Real));
            }

            // Clear any EOF flags that might have triggered, allowing future reads
            inp_stream.clear();

            return target_prices;

        }

        ~BinReader() override {
            if (inp_stream.is_open()) inp_stream.close();
        }

    private:
        KC::UInputs config;
        BinHeader bin_header;
        std::ifstream inp_stream;

        void load_header_paths_file() {
            // Open File Paths
            auto file_out_paths = std::filesystem::path(config.output.out_dir) / config.output.filename_paths_out;
            inp_stream.open(file_out_paths.string(), std::ios::in | std::ios::binary);
            if (!inp_stream.is_open()) {
                throw std::runtime_error("[BinReader Error] Failed to open file: " + file_out_paths.string());
            }
            // Read header
            inp_stream.read(reinterpret_cast<char *>(&bin_header), sizeof(BinHeader));
            // Check if heaser is ok
            if (bin_header.file_key[0] != 'K' || bin_header.file_key[1] != 'O' ||
                bin_header.file_key[2] != 'P' || bin_header.file_key[3] != 'T') {
                throw std::runtime_error(
                    "[BinReader Error] Magic key mismatch. File is corrupted or not a KOptions.paths file.");
            }
            if (bin_header.byte_precision != sizeof(KT::Real)) {
                throw std::runtime_error(
                    "[BinReader Error] Build precision mismatch. Data cannot safely map into target system precision.");
            }
        }
    };
}
