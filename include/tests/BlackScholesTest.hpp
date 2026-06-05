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

        config.time.t_end = 1.;

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.StrikePrice = 100.0;

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


    bool run_test() {
        std::string id_test {"TEST 4 : Weak Convergence to Black-Scholes SDE exact option price"};
        auto config = getDefaultConfig();
        KT::Real exact_price = KES::BlackScholes::get_exact_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              KTU::get_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}
