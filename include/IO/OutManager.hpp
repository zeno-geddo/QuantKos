#pragma once

#include <memory>
#include <stdexcept>

#include "./../core/config/Config.hpp"
#include "./../core/config/ConfigEnums.hpp"
#include "./IOBinary.hpp"
#include "../core/memory/PathsMCBatchMem.hpp"
#include "./../core/Typedefs.hpp"

namespace KOps::IO {
    namespace KC = KOps::Config;
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;
    namespace KE = KOps::Engine;
    namespace KB = KOps::IO::Binary;


    class OutputManager {
    public:
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

        // Delete copies (Cannot duplicate a unique pointer / file writer)
        OutputManager(const OutputManager&) = delete;
        OutputManager& operator=(const OutputManager&) = delete;

        // Default moves (Safely transfers ownership of the unique_ptr)
        OutputManager(OutputManager&&) = default;
        OutputManager& operator=(OutputManager&&) = default;

        // Default destructor
        ~OutputManager() = default;

        void print_paths_info_planned_outputs() const {
            if (active_writer) {
                active_writer->print_planned_outputs_summary();
            } else {
                std::cout << "  No outputs files will be saved since filenames were not specified.\n";
            }
        }

        void save_paths_batch_if_needed(const int current_batch_size, const Engine::PathsMCBatchMem &BatchMem) {
            if (active_writer) {
                active_writer->save_paths_batch_if_needed(current_batch_size, BatchMem);
            }
        }



    private:
        // Polymorphic pointer that can hold ANY writer (BIN, TXT, etc.)
        std::unique_ptr<WriterBlueprint> active_writer;
    };

}
