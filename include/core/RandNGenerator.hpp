#pragma once

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "./../IO/Config.hpp"

namespace KOps::Engine {
    namespace KC = KOps::Config;

    class RNGenerator {
    public:
        // Expose the underlying Kokkos Pool type for the kernels
        using PoolType = Kokkos::Random_XorShift64_Pool<>;

        // Explicit constructor requiring configuration input
        explicit RNGenerator(const KC::UInputs& config) {
            //Kokkos allocates a massive, flat array of independent mathematical state registers directly inside the device memory space (VRAM or system RAM). The number of states allocated is chosen by Kokkos to match the maximum physical hardware concurrency of your chip (often tens of thousands of individual slots). Each slot is initialized with a slightly scrambled version of your 54321ULL seed.
            uint64_t initial_seed = 54321ULL;
            m_pool = PoolType(initial_seed);
            std::cout << "  [RNGenerator] Random Number Pool initialized correctly." << std::endl;
        }

        // Kokkos RNG pools are shallow-copy safe because they wrap internal Views.
        // Returning it by value allows the GPU parallel_for lambda to capture it perfectly.
        [[nodiscard]] PoolType get_pool() const {
            return m_pool;
        }

        // Placeholder for future variance reduction math modifications
        void reset(uint64_t new_seed) {
            m_pool = PoolType(new_seed);
        }

    private:
        PoolType m_pool;
    };
}