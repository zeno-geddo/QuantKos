#pragma once

#include <fstream>
#include <string>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "./Config.hpp"
#include "./ConfigEnums.hpp"
#include "./../core/MCMem.hpp"
#include "./../core/Typedefs.hpp"

namespace KOps::Out {
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;
    namespace KE = KOps::Engine;


    struct BinHeader {
        const char file_key[4] = {'K', 'O', 'P', 'T'};
        int version = 1;
        int byte_precision = sizeof(KT::Real);
        int total_n_sims;
        int n_time_steps;
        KT::Real dt; // Allows to reconstruct the grid (works only for static time grid)
    };


    class OutputManager {
    public:
        explicit OutputManager(const KC::UInputs &conf) : config(conf) {
            open_out_stream();
        }

        // Destructor automatically flushes and closes the file safely!
        ~OutputManager() {
            if (out_stream.is_open()) {
                out_stream.close();
            }
        }

        // High-level API exposed to the runner loop
        void save_batch(const int current_batch_size, const KE::MCBatchMem &BatchMem) {
            // Check the runtime configuration enum directly
            switch (config.output.format) {
                case KI::IOFormat::BIN:
                    write_binary_chunk(current_batch_size, BatchMem);
                    break;
                case KI::IOFormat::TXT:
                    write_text_chunk(current_batch_size, BatchMem);
                    break;
            }
        }

    private:
        const KC::UInputs config;
        std::ofstream out_stream; // The persistent hardware file pipe


        void open_out_stream() {
            // Open Stream for lifetime of the application
            if (config.output.format == KI::IOFormat::BIN) {
                out_stream.open(config.output.filename_out, std::ios::out | std::ios::binary);
                if (!out_stream.is_open()) {
                    throw std::runtime_error("Failed to open binary output file: " + config.output.filename_out);
                }
                write_global_bin_header();
            } else {
                throw std::runtime_error("TXT OUT NOT YET IMPLEMENTED");
            }
        }


        void write_global_text_header() {
            // Simple text-based formatting for 1D/2D debugging
        }


        void write_text_chunk(const int current_batch_size, const KE::MCBatchMem &BatchMem) {
            // Simple text-based formatting for 1D/2D debugging
        }


        void write_global_bin_header() {
            // 1. Setup Header Data
            BinHeader header;
            header.total_n_sims = config.mc.N_Paths;
            header.n_time_steps = config.time.N_time_steps;
            header.dt = config.time.dt;

            // 2. Dump Header Struct directly to disk
            // reinterpret_cast forces the compiler to treat the memory address as an array of raw, unsigned bytes (const char*).
            out_stream.write(reinterpret_cast<const char *>(&header), sizeof(BinHeader));
        }


        void write_binary_chunk(const int current_batch_size, const KE::MCBatchMem &BatchMem) {
            // Calculate exactly how many bytes this specific batch occupies (N_sims x n_t_steps x sizeReal)
            // This is flexible and allows to consider cases where the batch is not complete
            const size_t bytes_to_write = static_cast<size_t>(current_batch_size) *
                                          static_cast<size_t>(config.time.N_time_steps) * sizeof(KT::Real);

            // Fetch the raw C-style pointer from the Kokkos Host View
            const KT::Real *raw_data_ptr = BatchMem.h_batch_view.data();

            // PATH A: The Fast Track (Only runs if the layout is physically contiguous)
            if (BatchMem.is_host_contiguous() && BatchMem.is_host_row_major()) {
                // Dump it to the SSD in one hardware instruction
                out_stream.write(reinterpret_cast<const char *>(raw_data_ptr), bytes_to_write);
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
                    out_stream.write(reinterpret_cast<const char *>(row_buffer.data()), row_bytes);
                }
            }
        }
    };
}
