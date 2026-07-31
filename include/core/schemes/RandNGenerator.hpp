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

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "../config/Config.hpp"

namespace quantkos::Engine {
    namespace KC = quantkos::Config;

    // ========================================================================
    // 1. HOST-SIDE MANAGER: Handles allocation and lifecycle of the global pool
    // ========================================================================
    /**
     * @brief Host-side manager for the massive GPU random number generator pool.
     * * Generating random numbers in parallel on a GPU is tricky because thousands of threads
     * cannot share the same random seed without generating the exact same numbers.
     * * This class initializes a `Kokkos::Random_XorShift64_Pool`, which allocates thousands of
     * independent random number sequences directly inside the GPU's memory. Each sequence is
     * seeded with a slightly shifted variant of your master initial seed.
     */
    class RNGManager {
    public:
        // Expose the underlying Kokkos Pool type for the kernels
        /// @brief Expose the underlying Kokkos Pool type so other classes know what to expect.
        using GlobalRNGPool = Kokkos::Random_XorShift64_Pool<>;
        /// @brief Expose the individual thread-level generator type.
        using RNGeneratorState = GlobalRNGPool::generator_type;

        /**
         * @brief Constructs the master random pool based on the user's initial seed.
         * * Kokkos looks at the physical hardware (e.g., your GPU architecture) and automatically
         * creates enough independent random states to supply every possible simultaneous thread.
         * @param config reference to the master configuration tree containing the Monte Carlo initial seed.
         */
        explicit RNGManager(const KC::UInputs &config) {
            // Kokkos allocates a massive, flat array of independent mathematical state registers directly inside the device memory space (VRAM or system RAM).
            // The number of states allocated is chosen by Kokkos to match the maximum physical hardware concurrency of your chip (often tens of thousands of individual slots).
            // Each slot is initialized with a slightly scrambled version of your seed.
            uint64_t initial_seed = config.mc.rng_seed;
            global_rng_pool = GlobalRNGPool(initial_seed);
            std::cout << "  [RNGenerator] Global Random Number Pool initialized correctly." << std::endl;
        }

        /// @name Lifecycle Safeguards (Single Source of Truth)
        ///@{
        /** * @brief Copying is deleted to enforce a Single Source of Truth.
        * * We absolutely do not want two managers trying to reset or manage the exact same GPU memory pool.
        */
        RNGManager(const RNGManager &) = delete;

        RNGManager &operator=(const RNGManager &) = delete;

        /** @brief Moving is allowed to pass the manager between setup functions or store it in containers. */
        RNGManager(RNGManager &&) = default;

        RNGManager &operator=(RNGManager &&) = default;

        ~RNGManager() = default; ///<  Default destructor
        ///@}

        /**
         * @brief Passes access to the GPU random memory pool to the parallel execution batches.
         * @note This does *not* duplicate the massive array of random states. It only
         * creates a lightweight, shallow copy (like passing a pointer), allowing the new batch of
         * threads to continue drawing from the exact same random sequences where the last batch left off.
         * @return A lightweight wrapper pointing to the global pool.
         */
        [[nodiscard]] GlobalRNGPool get_global_rng_pool() const {
            // Kokkos only creates a shallow copy of the pool wrapper.
            // It does not reallocate or re-seed the massive array of random states in the GPU VRAM.
            // It simply gives the new batch a pointer to the existing pool in memory.
            return global_rng_pool;
        }

        // Placeholder for future variance reduction math modifications
        /**
         * @brief Resets the entire GPU pool with a brand new master seed.
         * @note Reserved for variance reduction techniques or running completely independent simulation loops.
         * @param new_seed The new 64-bit integer seed.
         */
        void reset_global_rn_pool(uint64_t new_seed) {
            global_rng_pool = GlobalRNGPool(new_seed);
        }

    private:
        GlobalRNGPool global_rng_pool; ///< The underlying Kokkos pool structure.
    };


    // ========================================================================
    // 2. DEVICE-SIDE RAII GUARD: Automates get_state() and free_state()
    // ========================================================================
    /**
     * @brief Device-side tool for managing individual thread random states.
     * * **The Problem:** When a GPU thread wants to generate a random number, it needs to "check out"
     * a unique random state from the global pool, use it, and then "return" it when it finishes. If a thread
     * forgets to return it (or crashes), that state is locked forever (a state leak).
     * * **The Solution:** This class uses RAII (Resource Acquisition Is Initialization). It automatically
     * checks out a state the moment it is created, and its destructor guarantees the state is returned
     * back to the pool the exact moment the thread finishes its work, no matter what happens.
     */
    class ScopedRNG {
    public:
        // Constructor, acquire state on creation
        // KOKKOS_INLINE_FUNCTION allows instantiation directly on the GPU
        /**
         * @brief Constructs the guard and checks out a unique random sequence for the calling thread.
         * @param pool A reference to the massive global pool of random states.
         */
        KOKKOS_INLINE_FUNCTION
        explicit ScopedRNG(const RNGManager::GlobalRNGPool &pool) : global_rng_pool(pool),
                                                                    unique_rng_state(global_rng_pool.get_state()) {
        }

        /// @name Lifecycle Safeguards
        ///@{
        /** * @brief Copying is deleted (Prevent copying to ensure a state isn't double-freed by accident).
         * * If a thread copied this object, the destructor would run twice, trying to "return" the
         * same random state to the pool twice, crashing the GPU.
         */
        ScopedRNG(const ScopedRNG &) = delete; // delete the copy constructor
        ScopedRNG &operator=(const ScopedRNG &) = delete; // delete copy assignment operator.

        /** @brief Moving is deleted because the state is strictly tied to the executing thread lane. */
        ScopedRNG(ScopedRNG &&) = delete;

        ScopedRNG &operator=(ScopedRNG &&) = delete;

        // Destructor
        /**
         * @brief Destroys the guard and safely returns the state back to the pool.
         */
        KOKKOS_INLINE_FUNCTION
        ~ScopedRNG() {
            global_rng_pool.free_state(unique_rng_state); // Release state automatically on destruction
        }

        ///@}

        /**
         * @brief Accessor to expose the raw random number generator state to the mathematical SDE schemes.
         * @return A reference to this thread's uniquely checked-out generator.
         */
        KOKKOS_INLINE_FUNCTION
        RNGManager::RNGeneratorState &return_unique_rng_state() {
            return unique_rng_state;
        }

    private:
        RNGManager::GlobalRNGPool global_rng_pool; ///< Local reference to the master pool.
        RNGManager::RNGeneratorState unique_rng_state; ///< The exact sequence "checked out" by the calling thread.
    };
}
