#pragma once

#include <stdexcept>
#include <iostream>

#include "FMCEngine.hpp"
#include "../config/Config.hpp"


namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;


    class MCDispatcher {
    public:
        explicit MCDispatcher(const KC::UInputs &conf) : config(conf) {
        }

        MCResults launch_montecarlo() {
            std::cout << "\n>>> Starting Monte Carlo Simulation...\n" << std::endl;
            const MCResults results = dispatch_model();
            std::cout << "\n>>> Simulation completed successfully!\n" << std::endl;
            return results;
        }

    private:
        const KC::UInputs config;

        // ====================================================================
        // LEVEL 1: Dispatch the SDE Model (The Entry Point)
        // ====================================================================
        MCResults dispatch_model() {
            switch (config.model.id_model) {
                case KI::MathModel::Heston:
                    return dispatch_scheme<KI::MathModel::Heston>();
                case KI::MathModel::Bates:
                    return dispatch_scheme<KI::MathModel::Bates>();
                default:
                    throw std::runtime_error("Unknown SDE Model");
            }
        }

        // ====================================================================
        // LEVEL 2: Dispatch the Discretization Scheme
        // ====================================================================
        template<KI::MathModel ModelPolicy>
        MCResults dispatch_scheme() {
            switch (config.scheme.id_scheme) {
                case KI::NumScheme::Euler:
                    return dispatch_opt_style<ModelPolicy, KI::NumScheme::Euler>();
                case KI::NumScheme::Milstein:
                    return dispatch_opt_style<ModelPolicy, KI::NumScheme::Milstein>();
                case KI::NumScheme::AndersonQE:
                    return dispatch_opt_style<ModelPolicy, KI::NumScheme::AndersonQE>();
                default:
                    throw std::runtime_error("Unknown Discretization Scheme");
            }
        }

        // ====================================================================
        // LEVEL 3: Dispatch the Option Style (European, Asian, etc.)
        // ====================================================================
        template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
        MCResults dispatch_opt_style() {
            switch (config.options.opt_type) {
                // BASIC OPTIONS
                case KI::OptType::European:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::European>();
                case KI::OptType::Asian:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::Asian>();

                // BARRIER OPTIONS
                case KI::OptType::BarrierUpAndOut:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::BarrierUpAndOut>();
                case KI::OptType::BarrierDownAndOut:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::BarrierDownAndOut>();
                case KI::OptType::BarrierUpAndIn:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::BarrierUpAndIn>();
                case KI::OptType::BarrierDownAndIn:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::BarrierDownAndIn>();

                // LOOKBACK OPTIONS
                case KI::OptType::LookbackFloatingStrike:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::LookbackFloatingStrike>();
                case KI::OptType::LookbackFixedStrike:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::LookbackFixedStrike>();

                // BINARY OPTIONS
                case KI::OptType::BinaryCashOrNothing:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::BinaryCashOrNothing>();
                case KI::OptType::BinaryAssetOrNothing:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::BinaryAssetOrNothing>();

                // BACKWARD OPTIONS
                case KI::OptType::American:
                    return dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::American>();

                default:
                    throw std::runtime_error("Unknown Option Style");
            }
        }


        // ====================================================================
        // LEVEL 4: Dispatch the Option Right
        // ====================================================================
        template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType>
        MCResults dispatch_opt_right() {
            switch (config.options.opt_right) {
                case KI::OptRight::Call:
                    return execute_selected_runner<ModelPolicy, SchemePolicy, OptType, KI::OptRight::Call>();
                case KI::OptRight::Put:
                    return execute_selected_runner<ModelPolicy, SchemePolicy, OptType, KI::OptRight::Put>();
                default:
                    throw std::runtime_error("Unknown Option Right");
            }
        }

        // ====================================================================
        // LEVEL 5: Compile-Time Router
        // All templates are now resolved. Instantiate the actual MCRunner here.
        // ====================================================================
        template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
        MCResults execute_selected_runner() {
            if constexpr (OptType == KI::OptType::American) {
                // If it is American, the compiler ONLY consider this branch.
                // Notice the new architectural name: ForwardBackwardMCRunner
                return ForwardMCRunner<ModelPolicy, SchemePolicy, OptType, OptRight>(config).get_option_prices();
                // return ForwardBackwardMCRunner<ModelPolicy, SchemePolicy, OptRight>(config).run_mc_simulation();
            } else {
                // For all other options, the compiler ONLY consider this branch.
                return ForwardMCRunner<ModelPolicy, SchemePolicy, OptType, OptRight>(config).get_option_prices();
            }
        }
    };
}
