#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "../core/RandNGenerator.hpp"
#include "../IO/Config.hpp"
#include "../core/Typedefs.hpp"

namespace KOps::Tests::RNG {
    namespace KT = KOps::Types;
    namespace KE = KOps::Engine;
    namespace KC = KOps::Config;

    struct DummySDEScheme {
        template<typename RNGeneratorType>



        KOKKOS_INLINE_FUNCTION
        KT::Real evolve_step(RNGeneratorType &local_rn_generator) const {
            return static_cast<KT::Real>(local_rn_generator.normal());
        }
    };

    struct RNGTestKernel {
        KC::UInputs conf;
        Kokkos::View<KT::Real **> dummy_path_view;
        KE::RNGManager::GlobalRNGPool rng_pool;
        DummySDEScheme Scheme;

        KOKKOS_INLINE_FUNCTION
        void operator()(const int n_p) const {
            KOps::Engine::ScopedRNG scoped_rng(rng_pool);
            auto &rn_generator = scoped_rng.return_unique_rng_state();
            for (int i = 0; i < conf.time.N_time_steps; ++i) {
                dummy_path_view(n_p, i) = Scheme.evolve_step(rn_generator);
            }
        }
    };

    bool run_test() {
        std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          TEST 1 : Standard Normal Distribution Check             \n"
                << indent << "====================================================================\n";

        std::cout << indent << "[   RUN   ] Standard Normal Distribution Check\n";


        bool test_passed = true;


        // Set Data
        KC::UInputs dummy_config;
        dummy_config.mc.rng_seed = 123456789;
        dummy_config.mc.batch_size = 1'000'000;
        dummy_config.time.N_time_steps = 365;

        // Initialize Random number generation manager
        auto rng_man = KE::RNGManager(dummy_config);
        auto global_pool = rng_man.get_global_rng_pool();

        // Allocate Memory for the generated random number
        auto d_z_values = Kokkos::View<KT::Real **>("device_Z_values",
                                                    dummy_config.mc.batch_size,
                                                    dummy_config.time.N_time_steps);
        auto h_z_values = Kokkos::create_mirror_view(d_z_values);

        // Initialize Kernel test
        RNGTestKernel test_kernel{
            dummy_config,
            d_z_values,
            global_pool,
            DummySDEScheme{}
        };

        // Launch Dummy Kernel to see if we are really generating guassina values
        Kokkos::parallel_for("Test_RNG_Pipeline", dummy_config.mc.batch_size, test_kernel);
        Kokkos::fence();
        Kokkos::deep_copy(h_z_values, d_z_values);

        // Calculate Statistics
        double sum = 0.0;
        double sq_sum = 0.0;

        for (int i = 0; i < dummy_config.mc.batch_size; ++i) {
            for (int j = 0; j < dummy_config.time.N_time_steps; ++j) {
                double z = static_cast<double>(h_z_values(i, j));
                sum += z;
                sq_sum += (z * z);
            }
        }

        double N_draws = static_cast<double>(dummy_config.mc.batch_size * dummy_config.time.N_time_steps);
        double mean = sum / N_draws;
        double variance = (sq_sum / N_draws) - (mean * mean);

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

        return test_passed;

        std::cout << "====================================================================\n" << std::endl;
    }
}

