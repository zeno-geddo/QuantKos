#pragma once

#include "../../core/config/Config.hpp"
#include "../../core/analytical/HestonExact.hpp"
#include "TestsUtils.hpp"


namespace KOps::Tests::Heston {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KH = KOps::Engine::Analytical::Heston;


    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // See pag 28, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        // Expected European call price : 6.2528

        // Feller Condition : 2*k*theta >= sigma*sigma
        // The feller condition is satisfied in this test


        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 0.5; // 6 months

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.StrikePrice = 100.0;

        config.market.S0 = 100.;
        config.market.v0 = 0.05; // Starting variance
        config.market.r = 0.03;
        config.market.q = 0.02;

        // Heston
        config.model.heston.k = 5.;
        config.model.heston.theta = 0.05;
        config.model.heston.sigma = 0.5;
        config.model.heston.rho = -0.8;

        // Bates
        config.model.bates.k = 5.;
        config.model.bates.theta = 0.05;
        config.model.bates.sigma = 0.5;
        config.model.bates.rho = -0.8;
        config.model.bates.lambda_J = 0.;
        config.model.bates.mu_J = 0.;
        config.model.bates.sigma_J = 0.;

        return config;
    }

    
    bool run_test() {
        std::string id_test {"TEST 5 : Heston Weak Convergence to Heston SDE exact option price"};
        auto config = getDefaultConfigGoodIntegrand();
        const KT::Real exact_price = KH::get_exact_eu_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              KTU::get_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}


