#include "../IO/Config.hpp"
#include "../core/Distpatcher.hpp"

namespace KOps::Tests::BlackScholes {
    namespace KC = KOps::Config;
    namespace KE = KOps::Engine;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KB = KOps::IO::Binary;

    inline KC::UInputs getDefaultConfig() {
        // General Config
        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 1.;

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.K = 100.0;

        config.init.S0 = 100.;
        config.init.v0 = 0.04; // This implies the volatility is 0.20, similat to that of the S&P500

        config.model.heston.r = 0.04;
        config.model.heston.q = 0.;
        config.model.heston.k = 0.;
        config.model.heston.theta = 0.;
        config.model.heston.sigma = 0.;
        config.model.heston.rho = 0.;

        return config;
    }

    inline KT::Real normalCDF(KT::Real x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    inline KT::Real get_black_scholes_exact_solution(const KC::UInputs &conf) {
        KT::Real S = conf.init.S0;
        KT::Real K = conf.options.K;
        KT::Real T = conf.time.t_end;
        KT::Real r = conf.model.heston.r;
        KT::Real q = conf.model.heston.q;
        KT::Real vol = std::sqrt(conf.init.v0); // Extract Volatility from Variance

        KT::Real d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        KT::Real d2 = d1 - vol * std::sqrt(T);

        // Assume considering European Call
        bool condition = (conf.options.opt_right == KI::OptRight::Call) and (
                             conf.options.opt_type == KI::OptType::European);
        if (condition) {
            return S * std::exp(-q * T) * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
        } else {
            throw std::runtime_error("Must consider a European Call option for the Black Scholes weak convergence !");
        }
    }

    std::map<KI::MathModel, KI::NumScheme> get_models_to_test() {
        static const std::map<KI::MathModel, KI::NumScheme> models_to_test = {
            {KI::MathModel::Heston, KI::NumScheme::Euler}
        };
        return models_to_test;
    }

    std::vector<int> get_N_time_steps(const int lower_exponent = 3, const int upper_exponent = 9) {
        // Calculates 2^i
        if (lower_exponent > upper_exponent) return {};

        std::vector<int> steps;
        steps.reserve(upper_exponent - lower_exponent + 1);

        for (int i = lower_exponent; i <= upper_exponent; ++i) {
            steps.push_back(1 << i);
        }

        return steps;
    }

    struct ConvData {
        int N_dt;
        double dt;
        double error;
        double stat_error;
    };

    bool run_test() {
        std::string_view indent{"   "};
        std::cout << "\n\n" << indent << "====================================================================\n"
                << indent << "          TEST 4 : Weak Convergence to Black Scholes              \n"
                << indent << "====================================================================\n";
        std::cout << indent << "[   RUN   ] Weak Convergence to Black Scholes \n";

        bool test_passed = true;
        auto config = getDefaultConfig();

        // Get Exact Solution
        KT::Real expected_O_P = get_black_scholes_exact_solution(config);
        std::cout << indent << "[   INFO   ] Exact BS Option Price : " << expected_O_P << "\n";
        std::cout << indent << "--------------------------------------------------------------------\n";

        // Loop over models
        for (auto &model: get_models_to_test()) {
            config.model.id_model = model.first;
            config.scheme.id_scheme = model.second;

            std::vector<ConvData> convergence_results;

            // Loop over time steps
            for (int N_Tsteps: get_N_time_steps()) {
                config.time.N_time_steps = N_Tsteps;
                config.time.inp_dt = config.time.t_end / N_Tsteps;

                // Print info
                std::cout << indent << "[   INFO   ] MODEL   : " << KI::enum_to_string(config.model.id_model) << "\n";
                std::cout << indent << "[   INFO   ] SCHEME  : " << KI::enum_to_string(config.scheme.id_scheme) << "\n";
                std::cout << indent << "[   INFO   ] dt: " << std::fixed << std::setprecision(6) << config.time.inp_dt
                        << "\n";

                // Run simulation
                std::cout << indent << ">>> Calling the solver ...\n";
                std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
                {
                    config.validate();
                    config.print_summary();
                    auto MCDisp = KE::MCDispatcher(config);
                    const auto MCRes = MCDisp.launch_montecarlo();
                    const KT::Real abs_error = std::abs(MCRes.OptionPrice.option_price - expected_O_P);
                    const KT::Real stat_error = MCRes.OptionPrice.standard_error;
                    std::cout << " Price: " << MCRes.OptionPrice.option_price << " | Error: " << abs_error <<  " | Stat Error: " << stat_error <<"\n";
                    convergence_results.push_back({N_Tsteps, config.time.dt, abs_error, stat_error});
                }

                std::cout << "\n" << indent << "--------------------------------------------------------------------\n";
                std::cout << indent << ">>> Go back to the tester ...\n";
            }

            // Compute Convergence Order
            std::cout << indent <<
                    "\n::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n";
            std::cout << indent << "\n\nAnalyzing convergence order for give model and scheme...\n";
            std::cout << indent <<
                    "Total Error ^2= Discretization Error (O(dt))  ^2+ Statistical Noise (O(std/sqrt(N_paths)) ^2)\n";
            for (size_t i = 1; i < convergence_results.size(); ++i) {
                const int N_dt = convergence_results[i-1].N_dt;
                const double dt = convergence_results[i-1].dt;
                const double error = convergence_results[i-1].error;
                const double stat_error = convergence_results[i-1].stat_error;
                double z_score = error / stat_error;
                const double error_next = convergence_results[i].error;
                const double dt_next = convergence_results[i].dt;
                const double log_dt_diff = std::log(dt_next) - std::log(dt);
                const double log_err_diff = std::log(error_next) - std::log(error);
                const double order = log_err_diff / log_dt_diff;
                //double disc_err_squared = (error * error) - (stat_error * stat_error);
                //double estimated_disc_err = (disc_err_squared > 0.0) ? std::sqrt(disc_err_squared) : 0.0;
                std::cout << indent << "[   INFO   ] Step " << i
                        << ": Ndt= " << N_dt
                        << ", ERR= " << error
                        << ", STAT ERR = " << stat_error << "\n"
                        << ", Zscore = " << z_score << "\n";
                       // << ", DISC. ERR= " << estimated_disc_err << "\n";
                // NOTE at some point, the stat err likely dominate and the convergence rate brakes!

                std::cout << indent << "[   INFO   ] Step " << i << " -> " << i + 1 << " Convergence Order: " << order
                        << "\n";

                // Calculate how many "Standard Errors" away we are from exact value
                if (z_score > 3.) {
                    test_passed = false;
                }
            }
            std::cout << indent <<
                    "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n";
        }

        std::cout << "\n" << indent << "====================================================================\n";
        if (test_passed) {
            std::cout << indent << "[  PASSED  ] Discretization correctly converges to analytical Black-Scholes.\n";
        } else {
            std::cerr << indent << "[  FAILED  ] Convergence rate dropped below expected mathematical bounds.\n";
        }
        std::cout << indent << "====================================================================\n\n";

        return test_passed;
    }


    // NOTE : The Heston model must perfectly collapse into the Black-Scholes model if you turn off the stochastic volatility.
}
