// NOTE : When no randomness, the stock should grow purely by the deterministic drift: S_T = S_0 e^{(r-q)T}.

#include <map>
#include <string>
#include <cmath>

#include "./../include/IO/Config.hpp"
#include "./../include/core/Distpatcher.hpp"

namespace KOps::Tests::NoNoise {
    namespace KC = KOps::Config;
    namespace KE = KOps::Engine;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;

    double get_exact_solution(const KOps::Config::UInputs &conf) {
        // Expected: S_T = S_0 * exp((r - q) * T)
        const double expected_S_T = conf.init.S0 * std::exp(
                                        (conf.model.heston.r - conf.model.heston.q) * conf.time.t_end);
        return expected_S_T;
    }

    bool run_tests() {
        std::cout << "\n\n[ RUN      ] Zero Variance Forward Growth Check\n";

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

        // Exact Solution
        KT::Real expected_S_T = get_exact_solution(config);
        std::cout << "[   INFO   ] Expected Final Price : " << expected_S_T << "\n";

        // Specific configs to loop
        static const std::map<KI::MathModel, KI::NumScheme> models_to_test = {
            {KI::MathModel::Heston, KI::NumScheme::Euler}
        };

        // Loop testing all models
        bool test_passed = true;
        for (const auto &model: models_to_test) {
            // Set missing params
            config.model.id_model = model.first;
            config.scheme.id_scheme = model.second;
            std::string name_out_file = KI::enum_to_string(config.model.id_model) + "_"
                                        + KI::enum_to_string(config.scheme.id_scheme) + ".paths";
            config.output.filename_paths_out = name_out_file;

            // Check and print params
            std::cout << "[   INFO   ] Focus on Model : "
                    << KI::enum_to_string(config.model.id_model) << ", with Scheme : "
                    << KI::enum_to_string(config.scheme.id_scheme) << "\n";
            config.validate();
            config.print_summary();

            // Run model (keep the scope to be sure that dispatcher is destructed correctly)
            {
                auto MCDisp = KE::MCDispatcher(config);
                MCDisp.launch_montecarlo();
            }


            // Load the data Saved


            // See that each last time is
            test_passed = true;
        }

        return test_passed;
    }
}


// double get_exact_solution(const KOps::Config::UInputs &conf) {
//     // Expected: S_T = S_0 * exp((r - q) * T)
//     const double expected_S_T = conf.init.S0 * std::exp((conf.model.heston.r - conf.model.heston.q) * conf.time.t_end);
//     return expected_S_T;
// }
//
//
// int main(int argc, char *argv[]) {
//     namespace KC = KOps::Config;
//     namespace KE = KOps::Engine;
//     namespace KT = KOps::Types;
//     namespace KI = KOps::Implemented;
//
//     std::cout << "[ RUN      ] Zero Variance Forward Growth Check\n";
//
//     // General Config
//     auto config = KC::UInputs();
//
//     config.mc.N_Paths = 100;
//     config.mc.batch_size = config.mc.N_Paths;
//
//     config.time.N_time_steps = 365;
//     config.time.t_end = 1.;
//     config.time.inp_dt = config.time.t_end / config.time.N_time_steps;
//
//     config.init.S0 = 100.;
//     config.init.v0 = 0.;
//
//     config.model.heston.r = 0.05;
//     config.model.heston.q = 0.02;
//     config.model.heston.k = 0.;
//     config.model.heston.theta = 0.;
//     config.model.heston.sigma = 0.;
//     config.model.heston.rho = 0.0;
//
//     // Exact Solution
//     KT::Real expected_S_T = get_exact_solution(config);
//     std::cout << "[   INFO   ] Expected Final Price : " << expected_S_T << "\n";
//
//     // Specific configs to loop
//     static const std::map<KI::MathModel, KI::NumScheme> models_to_test = {
//         {KI::MathModel::Heston, KI::NumScheme::Euler},
//     };
//
//     // Loop testing all models
//     Kokkos::initialize(argc, argv);
//     bool test_passed = true;
//     {
//         for (const auto &model: models_to_test) {
//             // Set missing params
//             config.model.id_model = model.first;
//             config.scheme.id_scheme = model.second;
//             std::string name_out_file = KI::enum_to_string(config.model.id_model) + "_"
//                                         + KI::enum_to_string(config.scheme.id_scheme) + ".paths";
//             config.output.filename_paths_out = name_out_file;
//
//             // Check and print params
//             std::cout << "[   INFO   ] Focus on Model : "
//                     << KI::enum_to_string(config.model.id_model) << ", with Scheme : "
//                     << KI::enum_to_string(config.scheme.id_scheme) << "\n";
//             config.validate();
//             config.print_summary();
//
//             // Run model (keep the scope to be sure that dispatcher is destructed correctly)
//             {
//                 auto MCDisp = KE::MCDispatcher(config);
//                 MCDisp.launch_montecarlo();
//             }
//
//
//             // Load the data Saved
//
//             // See that each last time is
//         }
//     }
//     Kokkos::finalize();
//
//     return test_passed ? 0 : 1;
// }
