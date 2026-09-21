// Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <stdexcept>
#include <iostream>

#include "ForwardMC.hpp"
#include "ForwardBackwardMC.hpp"
#include "../config/Config.hpp"

/**
 * @brief Namespace aggregating all tools to numerically compute options prices.
 * @todo Optimize Console String Formatting and Printing, adding level of verbosity and redirection to log file.
  *@todo Should add other namespaces into this namespace to make the structure of the code clearer (e.g. memory, sde, ...)
*/
namespace quantkos::Engine {
    namespace KI = quantkos::Implemented;
    namespace KC = quantkos::Config;

    // Transfer service class
    /**
     * @brief Orchestrates the runtime-to-compile-time dispatching loop for Monte Carlo execution.
     * * This class acts as a centralized **Template Router (Service Pattern)**. It reads the dynamic
     * configuration provided at runtime, unrolls the parameters through a cascade of switch-case statements,
     * and maps them onto compile-time template arguments.
     * * ### Execution Routing Pipeline:
     * 1. **Model Policy Evaluation** (e.g, Heston vs. Bates, etc.)
     * 2. **Numerical Scheme Evaluation** (e.g, Euler vs. Milstein vs. AndersonQE, etc.)
     * 3. **Option Payoff Structure Evaluation** (e.g, European, Asian, Barrier variants, etc.)
     * 4. **Option Exercise Right Evaluation** (Call vs. Put)
     * 5. **Compile-Time Branch Allocation** (`if constexpr` split between standard Forward vs. Forward-Backward LSM solvers)
     * * This approach guarantees that inside the heavy simulation loops, there are zero virtual function overheads,
     * dynamic allocations, or branch-prediction penalties.
     * * @note This is a stateless execution utility. Copy and move operations are strictly deleted to enforce a
     * unique lifecycle context.
     * @todo Should improve the dispatching procedure to reduce the compilation time, especially if many other models will be added.
     */
    class MCDispatcher {
    public:
        /**
        * @brief Constructs the dispatcher with the user input.
        * @param conf Master inputs configuration tree.
        */
        explicit MCDispatcher(const KC::UInputs &conf) : config(conf) {
        }

        /// @name Lifecycle Safeguards
        ///@{
        // Force unique lifecycle: Delete copy operations
        MCDispatcher(const MCDispatcher&) = delete; ///< Deleted copy constructor.
        MCDispatcher& operator=(const MCDispatcher&) = delete; ///< Deleted copy assignment.

        // Delete move operations as well, this is just an execution tool
        MCDispatcher(MCDispatcher&&) = delete; ///< Deleted move constructor.
        MCDispatcher& operator=(MCDispatcher&&) = delete; ///< Deleted move constructor.

        ~MCDispatcher() = default; ///< Default destructor.
        ///@}

        /**
         * @brief High-level entry point to initialize, and run the Monte Carlo engine.
         * @return An aggregated MCResults structure containing option prices, uncertainties, and execution metrics.
         */
        MCResults launch_montecarlo() {
            std::cout << "\n>>> Starting Monte Carlo Simulation...\n\n";
            const MCResults results = dispatch_model();
            std::cout << "\n>>> Simulation completed successfully!\n\n";
            return results;
        }

    private:
        const KC::UInputs config; ///< Local copy of the input configuration.

        // ====================================================================
        // LEVEL 1: Dispatch the SDE Model (The Entry Point)
        // ====================================================================
        /**
         * @brief LEVEL 1: Resolves the MathModel identifier.
         * @return MonteCarlo results structure after cascading evaluations.
         * @throw std::runtime_error If an unhandled or unregistered stochastic model is provided.
         */
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
        /**
         * @brief LEVEL 2: Resolves the discretization scheme template policy.
         * @tparam ModelPolicy Resolved compile-time stochastic model policy.
         * @return MonteCarlo results structure.
         * @throw std::runtime_error If an invalid discretization scheme index is processed.
         */
        template<KI::MathModel ModelPolicy>
        MCResults dispatch_scheme() {
            switch (config.scheme.id_scheme) {
                case KI::NumScheme::Euler:
                    return dispatch_opt_style<ModelPolicy, KI::NumScheme::Euler>();
                case KI::NumScheme::ImplicitMilstein:
                    return dispatch_opt_style<ModelPolicy, KI::NumScheme::ImplicitMilstein>();
                case KI::NumScheme::AndersonQE:
                    return dispatch_opt_style<ModelPolicy, KI::NumScheme::AndersonQE>();
                default:
                    throw std::runtime_error("Unknown Discretization Scheme");
            }
        }

        // ====================================================================
        // LEVEL 3: Dispatch the Option Style (European, Asian, etc.)
        // ====================================================================
        /**
         * @brief LEVEL 3: Resolves the option template policy.
         * @tparam ModelPolicy Resolved compile-time stochastic model policy.
         * @tparam SchemePolicy Resolved compile-time SDE discretization technique.
         * @return MonteCarlo results structure.
         * @throw std::runtime_error If the option style payoff pattern is unknown.
         */
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
        /**
         * @brief LEVEL 4: Resolves the exercise right (Call/Put) policy.
         * @tparam ModelPolicy Resolved compile-time stochastic model policy.
         * @tparam SchemePolicy Resolved compile-time SDE discretization technique.
         * @tparam OptType Resolved compile-time option policy.
         * @return MonteCarlo results structure.
         * @throw std::runtime_error If the exercise right string identifier is invalid.
         */
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
        /**
         * @brief LEVEL 5: Terminal Compile-Time Router.
         * * At this point, all structural parameter configurations are fully mapped onto
         * template compile-time parameters. This method utilizes static polymorphism via `if constexpr`
         * to instantiate the appropriate runner engine, preventing code bloat by isolating the backward
         * Longstaff-Schwartz components to American execution traces only.
         * * @tparam ModelPolicy Completed stochastic process model context.
         * @tparam SchemePolicy Completed step integration method context.
         * @tparam OptType Completed payoff valuation contract layout.
         * @tparam OptRight Completed contract exercise execution right.
         * @return MonteCarlo results structure.
         */
        template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy, KI::OptType OptType, KI::OptRight OptRight>
        MCResults execute_selected_runner() {
            if constexpr (OptType == KI::OptType::American) {
                // If it is American, the compiler ONLY consider this branch.
                return ForwardBackwardMCRunner<ModelPolicy, SchemePolicy, OptRight>(config).run();
            } else {
                // For all other options, the compiler ONLY consider this branch.
                return ForwardMCRunner<ModelPolicy, SchemePolicy, OptType, OptRight>(config).run();
            }
        }
    };
}
