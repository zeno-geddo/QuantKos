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
#include "../core/memory/PathsMCBatchMem.hpp"
#include "./../core/Typedefs.hpp"

/**
 * @namespace KOps::IO
 * @brief Handles all Input/Output operations, serialization, and disk-persistence interfaces for the Monte Carlo engine.
 *
 * @details The `KOps::IO` namespace serves as the primary abstraction layer for data movement between
 * system memory and external storage. Its architecture is built around three core principles:
 * * - **Polymorphism-First Design**: Uses interface blueprints (`WriterBlueprint`, `ReaderBlueprint`) to decouple
 * high-performance compute kernels from specific I/O backends (e.g., CSV, Binary, HDF5).
 * - **Memory Safety**: Enforces strict lifecycle management by deleting copy/move constructors, preventing
 * accidental duplication of file handles or heavy memory-mapped structures.
 * - **Batch-Oriented Throughput**: Designed to handle high-dimensional Monte Carlo paths in blocks.
 *
 * @note This namespace relies heavily on `KOps::Types` for precision control, ensuring that
 * data written to disk matches the numerical precision requirements of the solver engines.
 *
 * @todo Implement `CSVWriter` class to handle streaming storage to CSV files for rapid inspections.
 * @todo Implement `DatabaseWriter` class to handle streaming storage directly into SQL-based repositories.
 * @todo Add compression support (Zlib/LZ4) for binary path serialization to minimize disk footprint.
 */
namespace KOps::IO {
    namespace KT = KOps::Types;

    /**
    * @brief Abstract base class (interface) defining the contract for path and payoff output writers.
    * * This blueprint establishes the mandatory interface for writing simulation results to disk.
    * Implementations of this interface (such as binary, CSV, or custom database writers) handle
    * writing simulated asset paths and terminal payoffs during batch execution loops.
    * * * ### Lifecycle & Slicing Safeguards
     * Since this is an interface intended for polymorphic usage, all copy and move operations are
     * explicitly deleted.
    */
    class WriterBlueprint {
    public:
        /// @name Lifecycle Safeguards
        ///@{
        WriterBlueprint() = default; ///< Default constructor
        virtual ~WriterBlueprint() = default; ///< Virtual destructor guarantees derived classes close their files!
        WriterBlueprint(const WriterBlueprint &) = delete; ///< Copy constructor is deleted.
        WriterBlueprint &operator=(const WriterBlueprint &) = delete; ///< Copy assignment is deleted.
        WriterBlueprint(WriterBlueprint &&) = delete; ///< Move constructor is deleted.
        WriterBlueprint &operator=(WriterBlueprint &&) = delete; ///< Move assignment is deleted.
        ///@}

        /**
         * @brief Saves the current batch of simulated price paths to disk storage if configured.
         * @note If required, This function is invoked sequentially at the end of each Monte Carlo batch loop, writing
         * only the active simulation subset to prevent memory-mapped disk overflow.
         * * @param current_batch_size The active number of simulated pathways in the give batch iteration.
         * @param BatchMem The active batch memory structure holding device/host path views.
         */
        virtual void save_paths_batch_if_needed(int current_batch_size, const Engine::PathsMCBatchMem &BatchMem) = 0;

        /**
         * @brief Outputs a detailed summary report of the planned files, paths, and formats to the console.
         * @note Typically executed prior to launching the SDE solvers to verify I/O directory permissions
         * and configure target output structures.
         */
        virtual void print_planned_outputs_summary() const = 0;
    };

    /**
     * @brief Abstract base class (interface) defining the contract for reading path data back into memory.
     * * This blueprint establishes the interface for streaming or reading simulated price paths from disk.
     * * ### Lifecycle & Slicing Safeguards
     * Since this is an interface intended for polymorphic usage, all copy and move operations are
     * explicitly deleted.
     */
    class ReaderBlueprint {
    public:
        /// @name Lifecycle Safeguards
        ///@{
        ReaderBlueprint() = default; ///< Default constructor.
        virtual ~ReaderBlueprint() = default; ///< Virtual destructor guarantees derived classes close their files!
        ReaderBlueprint(const ReaderBlueprint&) = delete;            ///< Copy constructor is deleted.
        ReaderBlueprint& operator=(const ReaderBlueprint&) = delete; ///< Copy assignment is deleted.
        ReaderBlueprint(ReaderBlueprint&&) = delete;                 ///< Move constructor is deleted.
        ReaderBlueprint& operator=(ReaderBlueprint&&) = delete;      ///< Move assignment is deleted.
        ///@}

        /**
         * @brief Reads and extracts the cross-sectional asset prices across all paths at a specific target time.
         * * Maps historical file records from disk into a contiguous host vector representing the spatial
         * price distribution at time step $t$.
         * * @param target_time The continuous timeline coordinate (in years) to extract.
         * @return A standard vector containing the reconstructed spot prices across all simulation lines.
         */
        virtual std::vector<KT::Real> read_prices_at_target_time(KT::Real target_time) = 0;
    };
}
