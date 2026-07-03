#pragma once

#include "../../core/config/Config.hpp"
#include "../../core/analytical/BatesExact.hpp"
#include "TestsUtils.hpp"


namespace KOps::Tests::Bates {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KB = KOps::Engine::Analytical::Bates;


    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // Heston params from pag 28, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        // Bates params


        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 0.5; // 6 months

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.StrikePrice = 100.0;

        config.init.S0 = 100.;
        config.init.v0 = 0.05; // Starting variance

        config.model.bates.r = 0.03;
        config.model.bates.q = 0.02;
        config.model.bates.k = 5.;
        config.model.bates.theta = 0.05;
        config.model.bates.sigma = 0.5;
        config.model.bates.rho = -0.8;

        config.model.bates.lambda_J = 0.11;  // Roughly 1 jump every ~9 years
        config.model.bates.mu_J = -0.15;     // When a jump happens, it averages a -15% drop (Crash)
        config.model.bates.sigma_J = 0.11;   // The volatility/uncertainty (std) of that jump size

        return config;
    }

    inline std::vector<std::pair<KI::MathModel, KI::NumScheme> > get_bates_models_to_test() {
        static const std::vector<std::pair<KI::MathModel, KI::NumScheme> > models_to_test = {
            {KI::MathModel::Bates, KI::NumScheme::Euler},
            {KI::MathModel::Bates, KI::NumScheme::Milstein},
            {KI::MathModel::Bates, KI::NumScheme::AndersonQE},
        };
        return models_to_test;
    }


    bool run_test() {
        std::string id_test {"TEST 6 : Bates Weak Convergence to Bates SDE exact option price"};
        auto config = getDefaultConfigGoodIntegrand();
        const KT::Real exact_price = KB::get_exact_eu_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              get_bates_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}


