#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "../config/Config.hpp"

namespace KOps::Engine {
    namespace KC = KOps::Config;

    // ========================================================================
    // 1. HOST-SIDE MANAGER: Handles allocation and lifecycle of the global pool
    // ========================================================================
    class RNGManager {
    public:
        // Expose the underlying Kokkos Pool type for the kernels
        using GlobalRNGPool = Kokkos::Random_XorShift64_Pool<>;
        using RNGeneratorState = GlobalRNGPool::generator_type;

        // Explicit constructor requiring configuration input
        explicit RNGManager(const KC::UInputs &config) {
            // Kokkos allocates a massive, flat array of independent mathematical state registers directly inside the device memory space (VRAM or system RAM).
            // The number of states allocated is chosen by Kokkos to match the maximum physical hardware concurrency of your chip (often tens of thousands of individual slots).
            // Each slot is initialized with a slightly scrambled version of your seed.
            uint64_t initial_seed = config.mc.rng_seed;
            global_rng_pool = GlobalRNGPool(initial_seed);
            std::cout << "  [RNGenerator] Global Random Number Pool initialized correctly." << std::endl;
        }

        [[nodiscard]] GlobalRNGPool get_global_rng_pool() const {
            // Kokkos only creates a shallow copy of the pool wrapper.
            // It does not reallocate or re-seed the massive array of random states in the GPU VRAM.
            // It simply gives the new batch a pointer to the existing pool in memory.
            return global_rng_pool;
        }

        // Placeholder for future variance reduction math modifications
        void reset_global_rn_pool(uint64_t new_seed) {
            global_rng_pool = GlobalRNGPool(new_seed);
        }

    private:
        GlobalRNGPool global_rng_pool;
    };


    // ========================================================================
    // 2. DEVICE-SIDE RAII GUARD: Automates get_state() and free_state()
    // ========================================================================
    class ScopedRNG {
    public:
        // KOKKOS_INLINE_FUNCTION allows instantiation directly on the GPU
        KOKKOS_INLINE_FUNCTION
        explicit ScopedRNG(const RNGManager::GlobalRNGPool &pool) : global_rng_pool(pool),
                                                                    unique_rng_state(global_rng_pool.get_state()) {
        } // Acquire state on creation

        KOKKOS_INLINE_FUNCTION
        ~ScopedRNG() {
            global_rng_pool.free_state(unique_rng_state); // Release state automatically on destruction
        }

        // Prevent copying to ensure a state isn't double-freed by accident
        ScopedRNG(const ScopedRNG &) = delete; // delete the copy constructor

        ScopedRNG &operator=(const ScopedRNG &) = delete; // delete copy assignment operator.

        // Accessor to pass to your mathematical schemes
        KOKKOS_INLINE_FUNCTION
        RNGManager::RNGeneratorState &return_unique_rng_state() {
            return unique_rng_state;
        }

    private:
        RNGManager::GlobalRNGPool global_rng_pool;
        RNGManager::RNGeneratorState unique_rng_state;
    };
}
