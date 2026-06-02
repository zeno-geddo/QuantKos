#pragma once

#include "../IO/Config.hpp"
#include "TestsUtils.hpp"
#include "TestsExactSolutions.hpp"


namespace KOps::Tests::BlackScholes {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KES = KOps::Tests::ExactSolutions;


    inline KC::UInputs getDefaultConfig() {
        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 5.;

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.K = 100.0;

        config.init.S0 = 100.;
        config.init.v0 = 0.04; // Starting variance, implying volatility is 0.20, similat to that of the S&P500

        config.model.heston.r = 0.04;
        config.model.heston.q = 0.;
        config.model.heston.k = 1.5;
        config.model.heston.theta = 0.04;
        config.model.heston.sigma = 0.3;
        config.model.heston.rho = -0.5;
        // Feller Condition : 2*k*theta >= sigma*sigma
        // The feller condition is satisfied in this test

        return config;
    }


    bool run_test() {
        std::string id_test {"TEST 5 : Weak Convergence to Heston SDE exact option price"};
        auto config = getDefaultConfig();
        KT::Real exact_price = KES::get_exact_option_price_BlackScholes(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              KTU::get_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}
