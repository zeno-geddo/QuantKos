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
        explicit MCDispatcher(const KC::UInputs& conf) : config(conf) {}

        void launch_montecarlo() {
            std::cout << "Starting Chunked Monte Carlo Simulation..." << std::endl;
            dispatch_model();
            std::cout << "Simulation completed successfully!" << std::endl;
        }

    private:
        const KC::UInputs config;

        // ====================================================================
        // LEVEL 1: Dispatch the SDE Model
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
                    // Instantiate the correct engine and run it!
                    MCRunner<ModelPolicy, KI::NumScheme::Euler>(config).run_mc_simulation();
                    break;
                case KI::NumScheme::Milstein:
                    MCRunner<ModelPolicy, KI::NumScheme::Milstein>(config).run_mc_simulation();
                    break;
                case KI::NumScheme::AndersonQE:
                    MCRunner<ModelPolicy, KI::NumScheme::AndersonQE>(config).run_mc_simulation();
                    break;
                default:
                    throw std::runtime_error("Unknown Discretization Scheme");
            }
        }
    };
}