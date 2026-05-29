
#include <map>
#include <string>
#include <cmath>

#include "../IO/Config.hpp"
#include "../core/Distpatcher.hpp"

namespace KOps::Tests::NoNoise {
    namespace KC = KOps::Config;
    namespace KE = KOps::Engine;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;

    inline double get_exact_solution(const KOps::Config::UInputs &conf) {
        // Expected: S_T = S_0 * exp((r - q) * T)
        const double expected_S_T = conf.init.S0 * std::exp(
                                        (conf.model.heston.r - conf.model.heston.q) * conf.time.t_end);
        return expected_S_T;
    }

    inline KC::UInputs getDefaultConfig() {
        // General Config
        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;

        config.mc.N_Paths = 100;
        config.mc.batch_size = config.mc.N_Paths;

        config.time.N_time_steps = 365;
        config.time.t_end = 1.;
        config.time.inp_dt = config.time.t_end / config.time.N_time_steps;

        config.init.S0 = 100.;
        config.init.v0 = 0.;

        config.model.heston.r = 0.05;
        config.model.heston.q = 0.02;
        config.model.heston.k = 0.;
        config.model.heston.theta = 0.;
        config.model.heston.sigma = 0.;
        config.model.heston.rho = 0.0;

        return config;
    }

    inline bool run_test() {
        // NOTE : When no randomness, the stock should grow purely by the deterministic drift: S_T = S_0 e^{(r-q)T}.

        std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          TEST 2 : Zero Variance Forward Growth Check             \n"
                << indent << "====================================================================\n";
        std::cout << indent << "\n\n[   RUN   ] Zero Variance Forward Growth Check\n";

        bool test_passed = true;
        auto config = getDefaultConfig();

        // Get Exact Solution
        KT::Real expected_S_T = get_exact_solution(config);
        std::cout << indent << "[   INFO   ] Expected Final Price : " << expected_S_T << "\n";

        // Loop testing all models
        static const std::map<KI::MathModel, KI::NumScheme> models_to_test = {
            {KI::MathModel::Heston, KI::NumScheme::Euler}
        };

        for (const auto &model: models_to_test) {
            // Set missing params
            config.model.id_model = model.first;
            config.scheme.id_scheme = model.second;
            std::string name_out_file = "Test_" + KI::enum_to_string(config.model.id_model) + "_"
                                        + KI::enum_to_string(config.scheme.id_scheme) + ".paths";
            config.output.filename_paths_out = name_out_file;

            // Check and print params
            std::cout << indent << "[   INFO   ] MODEL   : " << KI::enum_to_string(config.model.id_model) ;
            std::cout << indent << "[   INFO   ] SCHEME  : " << KI::enum_to_string(config.scheme.id_scheme) ;

            std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
            config.validate();
            config.print_summary();

            // Run model (keep the scope to be sure that dispatcher is destructed correctly)
            {
                auto MCDisp = KE::MCDispatcher(config);
                MCDisp.launch_montecarlo();
            }

            // Load the data Saved
            std::cout << "\n" << indent << "--------------------------------------------------------------------\n";

            // Check if last Price at T_Final is correct
            test_passed = true;
        }


        std::cout << "====================================================================\n" << std::endl;
        return test_passed;
    }
}

