#pragma once

#include <stdexcept>
#include <iostream>

#include "MCEngine.hpp"
#include "./../IO/Config.hpp"


namespace KOps::Engine {
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;


    class MCDispatcher {
    public:
        explicit MCDispatcher(const KC::UInputs &conf) : config(conf) {
        }

        void launch_montecarlo() {
            std::cout << "\n>>> Starting Monte Carlo Simulation...\n" << std::endl;
            dispatch_model();
            std::cout << "\n>>> Simulation completed successfully!\n" << std::endl;
        }

    private:
        const KC::UInputs config;

        // ====================================================================
        // LEVEL 1: Dispatch the SDE Model (The Entry Point)
        // ====================================================================
        void dispatch_model() {
            switch (config.model.id_model) {
                case KI::MathModel::Heston:
                    dispatch_scheme<KI::MathModel::Heston>();
                    break;
                case KI::MathModel::Bates:
                    dispatch_scheme<KI::MathModel::Bates>();
                    break;
                default:
                    throw std::runtime_error("Unknown SDE Model");
            }
        }

        // ====================================================================
        // LEVEL 2: Dispatch the Discretization Scheme
        // ====================================================================
        template<KI::MathModel ModelPolicy>
        void dispatch_scheme() {
            switch (config.scheme.id_scheme) {
                case KI::NumScheme::Euler:
                    dispatch_opt_style<ModelPolicy, KI::NumScheme::Euler>();
                    break;
                case KI::NumScheme::Milstein:
                    dispatch_opt_style<ModelPolicy, KI::NumScheme::Milstein>();
                    break;
                case KI::NumScheme::AndersonQE:
                    dispatch_opt_style<ModelPolicy, KI::NumScheme::AndersonQE>();
                    break;
                default:
                    throw std::runtime_error("Unknown Discretization Scheme");
            }
        }

        // ====================================================================
        // LEVEL 3: Dispatch the Option Style (European, Asian, etc.)
        // ====================================================================
        template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
        void dispatch_opt_style() {
            switch (config.options.opt_type) {
                case KI::OptType::European:
                    dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::European>();
                    break;
                case KI::OptType::Asian:
                    dispatch_opt_right<ModelPolicy, SchemePolicy, KI::OptType::Asian>();
                    break;
                default:
                    throw std::runtime_error("Unknown Option Style");
            }
        }


        // ====================================================================
        // LEVEL 4: Dispatch the Option Right (The Leaf Node)
        // All templates are now resolved. Instantiate the actual MCRunner here.
        // ====================================================================
        template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType>
        void dispatch_opt_right() {
            switch (config.options.opt_right) {
                case KI::OptRight::Call:
                    MCRunner<ModelPolicy, SchemePolicy, OptType, KI::OptRight::Call>(config).run_mc_simulation();
                    break;
                case KI::OptRight::Put:
                    MCRunner<ModelPolicy, SchemePolicy, OptType, KI::OptRight::Put>(config).run_mc_simulation();
                    break;
                default:
                    throw std::runtime_error("Unknown Option Right");
            }
        }
    };
}
