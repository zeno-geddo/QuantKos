#pragma once

#include "../IO/Config.hpp"
#include "TestsUtils.hpp"
#include "TestsExactSolutions.hpp"


namespace KOps::Tests::Heston {
    namespace KC = KOps::Config;
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KTU = KOps::Tests::Utils;
    namespace KES = KOps::Tests::ExactSolutions;


    inline KC::UInputs getDefaultConfigGoodIntegrand() {
        // See pag 28, F.Rouah, The Heston Model and its Extensions in Matlab and C#
        // Expected European call price : 6.2528
        auto config = KC::UInputs();

        config.output.format = KI::IOFormat::BIN;
        config.output.filename_paths_out = "";

        config.mc.N_Paths = 1'000'000;
        config.mc.batch_size = 0;

        config.time.t_end = 0.5; // 6 months

        config.options.opt_type = KI::OptType::European;
        config.options.opt_right = KI::OptRight::Call;
        config.options.K = 100.0;

        config.init.S0 = 100.;
        config.init.v0 = 0.05; // Starting variance

        config.model.heston.r = 0.03;
        config.model.heston.q = 0.02;
        config.model.heston.k = 5.;
        config.model.heston.theta = 0.05;
        config.model.heston.sigma = 0.5;
        config.model.heston.rho = -0.8;
        // Feller Condition : 2*k*theta >= sigma*sigma
        // The feller condition is satisfied in this test

        return config;
    }

    
    bool run_test() {
        std::string id_test {"TEST 5 : Weak Convergence to Heston SDE exact option price"};
        auto config = getDefaultConfigGoodIntegrand();
        KT::Real exact_price = KES::Heston::get_exact_call_option_price(config);
        return KTU::run_weak_convergence_test(id_test,
                                              config,
                                              exact_price,
                                              KTU::get_models_to_test(),
                                              KTU::get_time_grid_resolutions());
    }
}


