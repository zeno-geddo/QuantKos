#pragma once
#include "../core/memory/PathsMCBatchMem.hpp"
#include "./../core/Typedefs.hpp"

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
