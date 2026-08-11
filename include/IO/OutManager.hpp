// Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <memory>
#include <stdexcept>

#include "./../core/config/Config.hpp"
#include "./../core/config/ConfigFileEnums.hpp"
#include "./IOBinary.hpp"
#include "../core/memory/PathsMCBatchMem.hpp"
#include "./../core/Typedefs.hpp"

namespace quantkos::IO {
    namespace KC = quantkos::Config;
    namespace KI = quantkos::Implemented;
    namespace KT = quantkos::Types;
    namespace KE = quantkos::Engine;
    namespace KB = quantkos::IO::Binary;

    /**
     * @brief High-level coordinator that manages how and where simulation results are written to disk.
     * * This class acts as a single, unified gateway for output tasks. Rather than forcing the main
     * simulation loops to know about specific file formats (like Binary or Text), this manager checks
     * the user configurations at startup, selects the appropriate file writer behind the scenes, and
     * delegates writing tasks automatically.
     * @note If no output filename for the paths is provided, the manager safely runs as a silent "no-op" (doing nothing),
     * saving computational overhead during the simulation (does move data around and write on disk, which are the slowest operations).
     */
    class OutputManager {
    public:
        /**
         * @brief Constructs the OutputManager and binds the appropriate file writer.
         * @note Looks at the output parameters inside the configuration. If output is enabled,
         * it instantiates the correct writer backend (such as the Binary writer) using a polymorphic
         * unique pointer.
         * * @param conf The master simulation user configuration tree.
         * @throw std::runtime_error If the requested file format is recognized but not yet supported,
         * or if the format is completely unknown.
         */
        explicit OutputManager(const Config::UInputs &conf) {
            // If the user didn't specify an output file, just leave active_writer as a nullptr.
            if (conf.output.filename_paths_out.empty()) {
                return;
            }

            /// Route to the correct specific writer based on the enum
            switch (conf.output.format) {
                case Implemented::IOFormat::BIN:
                    active_writer = std::make_unique<KB::BinWriter>(conf);
                    break;
                case Implemented::IOFormat::TXT:
                    // active_writer = std::make_unique<Text::TxtWriter>(conf);
                    // break;
                    throw std::runtime_error("TXT writer not yet implemented.");
                default:
                    throw std::runtime_error("Unknown output format requested.");
            }
        }

        /// @name Lifecycle & Resource Ownership Rules
        ///@{
        /* @brief Copy constructor is deleted.
        * * Duplicating the manager would create two objects trying to write to the exact same file
        * on your hard drive at the same time, leading to corrupted data and locked file handles.
        */
        OutputManager(const OutputManager &) = delete;

        OutputManager &operator=(const OutputManager &) = delete; ///< Copy assignment operator is deleted.

        /**
         * @brief Default move constructor.
         * * Safely transfers ownership of the unique file writer and its active disk handles from
         * one program scope to another (e.g., when returning from a factory function).
         */
        OutputManager(OutputManager &&) = default; // Default moves (Safely transfers ownership of the unique_ptr)

        OutputManager &operator=(OutputManager &&) = default; ///< Default move assignment operator.

        /**
         * @brief Default destructor.
         * @note Automatically flushes remaining buffers and safely closes any active file streams.
         */
        ~OutputManager() = default;

        ///@}

        /**
         * @brief Prints a quick, clear summary of where and how files will be saved.
         * * Displays the target path, file naming conventions, and selected format to the
         * standard console before the heavy simulations begin. If saving is disabled,
         * it notifies the user.
         */
        void print_paths_info_planned_outputs() const {
            if (active_writer) {
                active_writer->print_planned_outputs_summary();
            } else {
                std::cout << "   <<< [OutManager] No outputs files will be saved since filenames were not specified\n\n";
            }
        }

        /**
         * @brief Automatically saves a completed batch of price paths to disk if required by the user.
         * * Called directly inside the forward Monte Carlo execution loop. It hands the in-memory
         * path data over to the active file writer to write the segment to the hard drive.
         * * @param current_batch_size The active number of paths processed in the current batch.
         * @param BatchMem The active batch memory structure holding device/host path views.
         */
        void save_paths_batch_if_needed(const int current_batch_size, const Engine::PathsMCBatchMem &BatchMem) {
            if (active_writer) {
                active_writer->save_paths_batch_if_needed(current_batch_size, BatchMem);
            }
        }

    private:
        // Polymorphic pointer that can hold ANY writer (BIN, TXT, etc.)
        /**
         * @brief Polymorphic pointer holding the active underlying writer (Binary, Text, etc.).
         * @note Remains empty (nullptr) if the user did not specify output file paths.
         */
        std::unique_ptr<WriterBlueprint> active_writer;
    };
}
