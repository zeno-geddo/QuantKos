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

#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../../core/schemes/RandNGenerator.hpp"
#include "../../core/config/Config.hpp"
#include "../../core/Typedefs.hpp"

/**
 * @brief Integration tests for validating parallel random number generation (RNG) distributions.
*/
namespace quantkos::Tests::RNG {
    namespace KT = quantkos::Types;
    namespace KE = quantkos::Engine;
    namespace KC = quantkos::Config;

    /**
    * @brief Generate a dummy configuration needed to initialize the random number generator.
    */
    inline KC::UInputs get_dummy_config() {
        auto dummy_config = KC::UInputs();
        dummy_config.mc.rng_seed = 123456789;
        dummy_config.mc.batch_size = 1'000'000;
        dummy_config.time.N_time_steps = 365;
        dummy_config.output.out_dir = "";
        dummy_config.output.filename_paths_out = "";
        dummy_config.output.filename_log = "";
        return dummy_config;
    }

    /**
    * @brief Initialize the random number generator and return it.
    */
    inline auto initialize_rng_manager() {
        const KC::UInputs dummy_config = get_dummy_config();
        return KE::RNGManager(dummy_config);
    }


    // ==============================================================================
    // Default Normal Variable Kokkos Test
    // ==============================================================================

    /**
     * @brief A simplified stochastic differential equation (SDE) scheme used for RNG validation.
     * @details This helper scheme isolates the random number generator from actual model dynamics,
     * directly returning standard normal (Gaussian) draws to verify statistical properties.
     * @note Use kokkos default box-muller normal rn generator
     */
    struct DummyKokkosDefGaussianSDEScheme {
        template<typename RNGeneratorType>
        /**
         * @brief Draws a standard normal random variable from the generator state.
         * @tparam RNGeneratorType The type of individual thread-level generator.
         * @param local_rn_generator The thread-local state instance of the random number generator.
         * @return A single standard normal realization cast to the engine's active real precision.
         */

        KOKKOS_INLINE_FUNCTION
        KT::Real evolve_step(RNGeneratorType &local_rn_generator) const {
            return static_cast<KT::Real>(local_rn_generator.normal());
        }
    };

    /**
     * @brief Parallel execution functor testing standard normal random number generation.
     * @details Schedules independent GPU/CPU threads to draw sequences of random numbers
     * and store them in a shared view for statistical analysis.
     * @note Use kokkos default box-muller normal rn generator
     */
    struct KokkosDefGaussianRNGTestKernel {
        int n_t_steps; ///< Number of times steps to use.
        Kokkos::View<KT::Real **> dummy_path_view; ///< Buffer to store generated normal values.
        KE::RNGManager::GlobalRNGPool rng_pool; ///< Global hardware random number state pool.
        DummyKokkosDefGaussianSDEScheme Scheme; ///< Simplified SDE step provider

        /**
         * @brief Parallel thread execution operator generating random paths.
         * @details Checks out a thread-unique random state, loops through the designated
         * time steps, draws Gaussian values, and securely returns the state to the pool via RAII.
         * @param n_p The globally indexed unique parallel lane (path) ID.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            quantkos::Engine::ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state();
            for (int i = 0; i < n_t_steps; ++i) {
                dummy_path_view(n_p, i) = Scheme.evolve_step(rn_generator);
            }
        }
    };

    /**
   * @brief Executes the Gaussian RNG test kernel independently.
   * @details This function is separated to ensure it only captures POD (Plain Old Data)
   * types, preventing the GPU compiler from trying to inspect or capture Host-only
   * configuration objects containing std::string.
   * @note Use kokkos default box-muller normal rn generator
   */
    inline void execute_rng_kernel_kookkos_def(
        const int batch_size,
        const int N_time_steps,
        Kokkos::View<KT::Real **> d_z_values,
        KE::RNGManager::GlobalRNGPool rng_pool) {
        // The kernel is instantiated here. Because we are not passing the UInputs
        // object (which contains std::string), there is no Host-only dependency
        // for the compiler to complain about.
        KokkosDefGaussianRNGTestKernel test_kernel{
            N_time_steps,
            d_z_values,
            rng_pool,
            DummyKokkosDefGaussianSDEScheme{}
        };

        Kokkos::parallel_for("Test_RNG_Pipeline", batch_size, test_kernel);
        Kokkos::fence();
    }


    /**
    * @brief Validates that the parallel RNG pool produces a correct standard normal distribution.
    * @note Use kokkos default box-muller normal rn generator
    */
    inline bool test_default_kokkos_gaussian() {
        const std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          TEST 1a : Standard Normal Distribution Check  (kokkos box-muller) \n"
                << indent << "====================================================================\n";

        std::cout << indent << "[   RUN   ] Standard Normal Distribution Check\n";


        bool test_passed = true;

        // Define size test
        const int batch_size = 1'000'000;
        const int N_time_steps = 365;

        // Initialize Random number generation manager
        const auto rng_man = initialize_rng_manager();

        // Allocate Memory for the generated random number
        auto d_z_values = Kokkos::View<KT::Real **>("device_Z_values",
                                                    batch_size,
                                                    N_time_steps);
        auto h_z_values = Kokkos::create_mirror_view(d_z_values);

        // Launch Dummy Kernel to see if we are really generating guassina values
        const auto start_time = std::chrono::steady_clock::now();
        execute_rng_kernel_kookkos_def(batch_size,
                                       N_time_steps,
                                       d_z_values,
                                       rng_man.get_global_rng_pool());
        const auto end_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << indent << "[   INFO   ] Kernel exe time : "<< elapsed.count()<< "s \n";

        // Copy results on host if not already there
        Kokkos::deep_copy(h_z_values, d_z_values);

        // Calculate Statistics
        double sum = 0.0;
        double sq_sum = 0.0;
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < N_time_steps; ++j) {
                const auto z = static_cast<double>(h_z_values(i, j));
                sum += z;
                sq_sum += (z * z);
            }
        }

        const auto N_draws = static_cast<double>(batch_size * N_time_steps);
        const double mean = sum / N_draws;
        const double variance = (sq_sum / N_draws) - (mean * mean);

        std::cout << indent << "[   INFO   ] Number Samples     : " << N_draws << "\n";
        std::cout << indent << "[   INFO   ] Measured Mean     : " << mean << "\n";
        std::cout << indent << "[   INFO   ] Measured Variance : " << variance << "\n";

        if (std::abs(mean - 0.0) > 0.005) {
            std::cerr << indent << "[  FAILED  ] RNG Mean drifted unacceptably far from 0.0!\n";
            test_passed = false;
        }

        if (std::abs(variance - 1.0) > 0.01) {
            std::cerr << indent << "[  FAILED  ] RNG Variance drifted unacceptably far from 1.0!\n";
            test_passed = false;
        }

        if (test_passed) {
            std::cout << indent << "[  PASSED  ] Standard Normal Distribution Check \n";
        }


        std::cout << indent << "====================================================================\n" << std::endl;

        return test_passed;
    }



    // ==============================================================================
    // Custom NormalPair Functor Test
    // ==============================================================================

    /**
     * @brief A stochastic scheme that utilizes the new paired custom Normal RNG generator.
     */
    struct DummyCustomGaussianSDEScheme {
        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION
        KE::RVPair evolve_step(RNGeneratorType &local_rn_generator) const {
            // Instantiate and call the optimized functor we created
            return KE::NormalPair<RNGeneratorType>{}(local_rn_generator);
        }
    };

    /**
     * @brief Parallel execution functor testing the Custom NormalPair generation.
     */
    struct CustomGaussianRNGTestKernel {
        int n_t_steps;
        Kokkos::View<KT::Real **> dummy_path_view;
        KE::RNGManager::GlobalRNGPool rng_pool;
        DummyCustomGaussianSDEScheme Scheme;

        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            quantkos::Engine::ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state();

            // Advance by 2 because our functor generates TWO numbers at once
            for (int i = 0; i < n_t_steps; i += 2) {
                const KE::RVPair z_pair = Scheme.evolve_step(rn_generator);

                // Always write Z1
                dummy_path_view(n_p, i) = z_pair.Z1;

                // Only write Z2 if we haven't exceeded the time steps (e.g., if n_t_steps is odd like 365)
                if (i + 1 < n_t_steps) {
                    dummy_path_view(n_p, i + 1) = z_pair.Z2;
                }
            }
        }
    };

    inline void execute_custom_rng_kernel(
        const int batch_size,
        const int N_time_steps,
        Kokkos::View<KT::Real **> d_z_values,
        KE::RNGManager::GlobalRNGPool rng_pool) {

        CustomGaussianRNGTestKernel test_kernel{
            N_time_steps,
            d_z_values,
            rng_pool,
            DummyCustomGaussianSDEScheme{}
        };

        Kokkos::parallel_for("Test_Custom_RNG_Pipeline", batch_size, test_kernel);
        Kokkos::fence();
    }


    /**
     * @brief Validates the Custom NormalPair functor (Box-Muller/Marsaglia).
     */
    inline bool run_test_custom_gaussian() {
        const std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                  << indent << "          TEST 1b : Custom Normal Distribution Check  (GPU->Box-Muller ; CPU->Marsaglia)      \n"
                  << indent << "====================================================================\n";

        std::cout << indent << "[   RUN   ] Custom Paired Normal Distribution Check\n";

        bool test_passed = true;

        int batch_size = 1'000'000;
        int N_time_steps = 365; // The kernel safely handles this odd number

        const auto rng_man = initialize_rng_manager();

        auto d_z_values = Kokkos::View<KT::Real **>("device_Z_values_custom",
                                                    batch_size,
                                                    N_time_steps);
        auto h_z_values = Kokkos::create_mirror_view(d_z_values);

        // Execute the new custom paired kernel
        const auto start_time = std::chrono::steady_clock::now();
        execute_custom_rng_kernel(batch_size,
                                  N_time_steps,
                                  d_z_values,
                                  rng_man.get_global_rng_pool());
        const auto end_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << indent << "[   INFO   ] Kernel exe time : "<< elapsed.count()<< "s \n";

        Kokkos::deep_copy(h_z_values, d_z_values);

        double sum = 0.0;
        double sq_sum = 0.0;

        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < N_time_steps; ++j) {
                const double z = static_cast<double>(h_z_values(i, j));
                sum += z;
                sq_sum += (z * z);
            }
        }

        const double N_draws = static_cast<double>(batch_size * N_time_steps);
        const double mean = sum / N_draws;
        const double variance = (sq_sum / N_draws) - (mean * mean);

        std::cout << indent << "[   INFO   ] Number Samples    : " << N_draws << "\n";
        std::cout << indent << "[   INFO   ] Measured Mean     : " << mean << "\n";
        std::cout << indent << "[   INFO   ] Measured Variance : " << variance << "\n";

        if (std::abs(mean - 0.0) > 0.005) {
            std::cerr << indent << "[  FAILED  ] Custom RNG Mean drifted unacceptably far from 0.0!\n";
            test_passed = false;
        }

        if (std::abs(variance - 1.0) > 0.01) {
            std::cerr << indent << "[  FAILED  ] Custom RNG Variance drifted unacceptably far from 1.0!\n";
            test_passed = false;
        }

        if (test_passed) {
            std::cout << indent << "[  PASSED  ] Custom Paired Normal Distribution Check \n";
        }

        std::cout << indent << "====================================================================\n" << std::endl;

        return test_passed;
    }

    /**
 * @brief Validates that the parallel RNG pool produces a correct standard normal distribution.
 * @details Generates a massive sample size (1,000,000 paths over 365 steps) of normal draws on the
 * active execution space, copies them back to host memory, and verifies that the measured
 * mean and variance fall within highly strict statistical thresholds:
 * - **Expected Mean**: 0.0 (Tolerance: +/- 0.005)
 * - **Expected Variance**: 1.0 (Tolerance: +/- 0.01)
 * @return true if both statistics satisfy their tolerance boundaries; false otherwise.
 @todo Fix the warning generated when using GPUs.
*/
    inline bool run_test_gaussian() {
        bool test_passed = test_default_kokkos_gaussian();
        test_passed =run_test_custom_gaussian();
        return test_passed;
    }
}
