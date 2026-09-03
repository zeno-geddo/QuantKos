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

// File (YAML) → Parser → Validator → Config → Solver
#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <filesystem>

#include "../Typedefs.hpp"
#include "ConfigFileEnums.hpp"
#include "ConfigFileKeys.hpp"
#include "ConfigFileKeysEnumMaps.hpp"

/**
 * @brief Namespace aggregating all structures containing parameters required to initialize and run a simulation.
 */
namespace quantkos::Config {
    /** @brief Internal helper to make configuration error messages cleaner */
    inline std::string config_err_msg(std::string_view block, std::string_view key, const std::string &msg) {
        return "[" + std::string(block) + "." + std::string(key) + "] " + msg;
    }

    // Alias for easier access to the keys
    namespace KI = quantkos::Implemented;
    namespace KK = quantkos::Keys;
    using Real = quantkos::Types::Real;

    /**
     * @brief Market environment configuration.
     *
     * Defines the initial state of the underlying asset and the surrounding
     * financial parameters (interest rates and dividends).
     */
    struct MarketConfig {
        Real S0 = 100.0; ///< Initial asset spot price.
        Real v0 = 1.0; ///< Initial stochastic variance.
        Real r = 0.05; ///< Annualized risk-free interest rate.
        Real q = 0.0; ///< Annualized continuous dividend yield.

        /**
         * @brief Validates that the market parameters lie within physically
         *        meaningful and numerically stable ranges.
         * @throw std::invalid_argument If any parameter violates stability constraints.
         */
        void validate() const {
            if (S0 <= 0.0)
                throw std::invalid_argument(
                    config_err_msg(KK::Market, KK::MarketParams::Price, "must be positive."));
            if (v0 < 0.0)
                throw std::invalid_argument(
                    config_err_msg(KK::Market, KK::MarketParams::Variance, "must be positive."));

            if (q < 0.0)
                throw std::invalid_argument(
                    config_err_msg(KK::Market, KK::MarketParams::q, "Dividend yield (q) cannot be negative."));

            if (std::abs(r) > 0.5)
                throw std::invalid_argument(
                    config_err_msg(KK::Market, KK::MarketParams::r,
                                   "Risk-free rate (r) is unphysical (keep between -50% and +50%)."));

            if (q > 0.8)
                throw std::invalid_argument(
                    config_err_msg(KK::Market, KK::MarketParams::q,
                                   "Dividend yield (q) is unphysical (keep under 80%)."));
        }

        /**
         * @brief Prints the market configuration to standard output.
         * @param indent String prefix for printing alignment.
         */
        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Market << "]\n"
                    << indent << "    Initial Price (S0"
                    ")      :    " << S0 << "\n"
                    << indent << "    Initial Variance (v0"
                    ")      :    " << v0 << "\n"
                    << indent << "    Risk-free interest rate (" << KK::MarketParams::r <<
                    ")      :   " << r << "\n"
                    << indent << "      Continuous dividend yield (" << KK::MarketParams::q <<
                    ")      :   " << q << "\n";
        }
    };

    /**
    * @brief Options configuration.
    *
    * Defines the option type and right, as well as striking and barrier price.
    */
    struct OptionsConfig {
        KI::OptType opt_type = KI::OptType::European; ///< Option Type
        KI::OptRight opt_right = KI::OptRight::Call; ///< Option Right
        Real StrikePrice = 100.; ///< Strike Price
        Real BarrierPrice = 100.; ///< Barrier Price

        /**
        * @brief Validates that the options parameters lie within physically
        *        meaningful ranges.
        * @throw std::invalid_argument If any parameter violates stability constraints.
        */
        void validate(const Real Spot_price) const {
            // Check if Strike price make sense (compared to the spot price)
            if (StrikePrice <= 0.0)
                throw std::invalid_argument(
                    config_err_msg(KK::Options, KK::OptionsParams::StrikePrice, "must be positive."));

            const Real min_meaningful_strike = Spot_price * static_cast<Real>(0.0001);
            if (StrikePrice < min_meaningful_strike) {
                throw std::invalid_argument("Config Error: Strike price (K) is too close to zero. "
                    "Must be at least 1000 times smaller than the initial asset price (S0) to prevent numerical problems.");
            }


            const Real max_meaningful_strike = Spot_price * static_cast<Real>(10000);
            if (StrikePrice > max_meaningful_strike) {
                throw std::invalid_argument("Config Error: Strike price (K) is absurdly high (exceeds 10000x of S0). "
                    "This will likely result in zero-variance path generation and statistical breakdown.");
            }
        }

        /**
         * @brief Prints the options configuration to standard output.
         * @param indent String prefix for printing alignment.
         */
        void print(std::string_view indent = "") const {
            const bool is_barrier_option = (opt_type == KI::OptType::BarrierDownAndIn ||
                                            opt_type == KI::OptType::BarrierDownAndOut ||
                                            opt_type == KI::OptType::BarrierUpAndIn ||
                                            opt_type == KI::OptType::BarrierUpAndOut);

            std::cout << indent << "  [" << KK::Options << "]\n";
            std::cout << indent << "      Option Type        :     " << enum_to_string(opt_type) << "\n";
            std::cout << indent << "      Option Right       :     " << enum_to_string(opt_right) << "\n";
            std::cout << indent << "      Strike Price (K)   :     " << StrikePrice << "\n";
            if (is_barrier_option) {
                std::cout << indent << "      Barrier Price      :     " << BarrierPrice << "\n";
            }
        }
    };


    /**
     * @brief Manages SDE model and parameters.
     *
     * Defines the mathematical dynamics for asset pricing (e.g., Heston or Bates).
     * Includes validation logic to enforce financial constraints (e.g., Feller condition)
     * and provides formatted SDE equation output for audit logs.
     */
    struct MathModelConfig {
        KI::MathModel id_model = KI::MathModel::Heston; ///< ID Active SDE model.

        /**
         * @brief Core Heston stochastic volatility parameters.
         */
        struct Heston {
            Real k = 2.0; ///< Mean reversion speed.
            Real theta = 0.04; ///< Long-term mean variance.
            Real sigma = 0.3; ///< Volatility of variance (Vol-of-Vol).
            Real rho = -0.7; ///< Correlation between asset and variance shocks.
        } heston;

        /**
         * @brief Bates jump-diffusion extension.
         *
         * Extends Heston dynamics by adding log-normal jumps (Merton process).
         * @note Inherits from Heston to share common stochastic volatility parameters.
         */
        struct Bates : public Heston {
            Real lambda_J = 0.1; ///< Average jump intensity per unit time. [1/years]
            Real mu_J = -0.1;
            ///< Mean of the jump size distribution in log-space (Expected log-return of a jump). It is dimensionless, the jumps act multiplicatively on the stock price. Example: $\mu_J = -0.1 \implies \text{median price multiplier } e^{-0.1} \approx 0.9048$. When a jump hits, the asset price drops on average by $\approx 9.52\%$.
            Real sigma_J = 0.15;
            ///< Standard deviation of the jump size in log-space (Standard deviation of jump log-return). Dimensionless. Example: $\sigma_J = 0.15$ means a $15\%$ volatility spread around the mean jump size in log-space.
        } bates;

        /**
         * @brief Performs physical constraints validation on the selected model.
        * @throw std::invalid_argument If model-specific physical bounds are violated.
        */
        void validate() const {
            if (id_model == KI::MathModel::Heston) {
                validate_heston_params(heston);
            } else if (id_model == KI::MathModel::Bates) {
                validate_bates_params(bates);
            }
        }

        /**
     * @brief Prints a mathematical representation of the model and its current parameters.
     *
     * @param market The market context (r, q) required for accurate SDE display.
     * @param indent String indentation for log formatting.
     * @throw std::invalid_argument if model selected is not valid
     */
        void print(const MarketConfig &market, std::string_view indent = "") const {
            // Assemble the equations
            std::string str_math_model = "";
            std::string ind(indent);

            if (id_model == KI::MathModel::Heston) {
                str_math_model = ind + "\t\t\t dSt = (r - q)·St·dt + √(vt)·St·dW1,t\n" +
                                 ind + "\t\t\t dvt = κ·(θ - vt)·dt + σ·√(vt)·dW2,t\n" +
                                 ind + "\t\t\t where : E[dW1,t·dW2,t] = ρ·dt\n";
            } else if (id_model == KI::MathModel::Bates) {
                // dNt = Merton jump process
                str_math_model = ind + "\t\t dSt = (r - q - λ·κ_J)·St·dt + √(vt)·St·dW1,t + St⁻·(J - 1)·dNt\n" +
                                 ind + "\t\t dvt = κ·(θ - vt)·dt + σ·√(vt)·dW2,t\n" +
                                 ind + "\t\t where :\n" + // λ·κJ = Martingale compensator
                                 ind + "\t\t   ln(J) ~ N(μ_J, σ_J²),\n" +
                                 // log normal jump size (J = Realized Random jump size)
                                 ind + "\t\t   κ_J = exp(μ_J + 0.5·σ_J²) - 1,\n" +
                                 // expected percentage change in the asset price caused by a single jump
                                 ind + "\t\t   E[dW1,t·dW2,t] = ρ·dt\n";
            }

            if (id_model == KI::MathModel::Heston || id_model == KI::MathModel::Bates) {
                // Handle parameter reference mapping cleanly based on active execution mode (upcast if needed)
                const Heston &base_params = (id_model == KI::MathModel::Bates)
                                                ? static_cast<const Heston &>(bates)
                                                : heston;

                std::cout << indent << "  [" << KK::Model << "]\n"
                        << indent << "    ID Model       :   " << id_model << "\n"
                        << indent << "    --------------------------------------------------------\n"
                        << str_math_model
                        << indent << "    --------------------------------------------------------\n"
                        << indent << "    Core Parameters :\n"
                        << indent << "      Risk-free interest rate (" << KK::MarketParams::r <<
                        ")           :   " << market.r << "\n"
                        << indent << "      Continuous dividend yield (" << KK::MarketParams::q <<
                        ")           :   " << market.q << "\n"
                        << indent << "      Mean reversion speed of the variance (" << KK::MathModelParams::k <<
                        ")           :   " << base_params.k << "\n"
                        << indent << "      Mean reversion level of the variance (" << KK::MathModelParams::theta <<
                        ") :   " << base_params.theta << "\n"
                        << indent << "      Volatility of the variance (" << KK::MathModelParams::sigma <<
                        ")           :   " << base_params.sigma << "\n"
                        << indent << "      Price-Variance correlation (" << KK::MathModelParams::rho <<
                        ")           :   " << base_params.rho << "\n";

                if (2.0 * base_params.k * base_params.theta <= base_params.sigma * base_params.sigma) {
                    std::cout << indent <<
                            "      Feller condition (2κθ > σ²) is violated. Variance paths may touch zero.\n";
                } else {
                    std::cout << indent <<
                            "      Feller condition (2κθ > σ²) respected.\n";

                }

                if (id_model == KI::MathModel::Bates) {
                    std::cout << indent << "    Jump Parameters:\n"
                            << indent << "      Jump Intensity (λ)            :   " << bates.lambda_J << "\n"
                            << indent << "      Mean Jump Size (μJ)           :   " << bates.mu_J << "\n"
                            << indent << "      Jump Volatility (σJ)          :   " << bates.sigma_J << "\n";
                }
            } else {
                //const std::string msg = "[Config Error] Model selected ('" + id_model + "') is not a valid model !";
                throw std::invalid_argument(config_err_msg(KK::Model, KK::Model,
                                                           "Model selected is not a valid model !"));
            }
        }

    private:
        /** @brief Validates Heston-specific stochastic constraints. */
        void validate_heston_params(const Heston &h) const {
            if (h.k < 0.0)
                throw std::invalid_argument("[Model Config] κ must be positive.");
            if (h.theta < 0.0)
                throw std::invalid_argument("[Model Config] θ must be positive.");
            if (h.sigma < 0.0)
                throw std::invalid_argument("[Model Config] σ must be positive.");
            if (h.rho < -1.0 || h.rho > 1.0)
                throw std::invalid_argument("[Model Config] ρ must fall within [-1.0, 1.0].");
        }

        /** @brief Validates jump-diffusion parameters, including Heston base. */
        void validate_bates_params(const Bates &b) const {
            // Leverage inheritance! Validate the stoch-vol layer first.
            validate_heston_params(b);

            // Validate Jump Bounds
            if (b.lambda_J < 0.0) {
                throw std::invalid_argument("[Model Config] Bates: Jump intensity (λ) cannot be negative.");
            }
            if (b.sigma_J < 0.0) {
                throw std::invalid_argument("[Model Config] Bates: Jump volatility (σJ) must be positive.");
            }
        }
    };


    /**
    * @brief Numerical Scheme configuration.
    *
    * Defines the Numerical Scheme to solve the SDE.
    */
    struct NumSchemeConfig {
        KI::NumScheme id_scheme = KI::NumScheme::Euler; ///< ID Numerical Scheme

        void validate() const {
        }

        /**
         * @brief Prints the Numerical Scheme configuration to standard output.
         * @param indent String prefix for printing alignment.
         */
        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Numerics << "]\n"
                    << indent << "    Numerical Scheme    :  " << enum_to_string(id_scheme) << "\n";
        }
    };

    // Substructure: Time Stepping
    /**
    * @brief Time stepping configuration.
    *
    * Defines the final time as well as the time step and the number of times steps for the discretization.
    */
    struct TimeConfig {
        Real t_end = 1.0; ///< Final Time
        Real inp_dt = 0.2; ///< Adjusted time step
        Real dt; ///< Input time step
        int N_time_steps; ///< Number of times steps

        /**
        * @brief Validates that the time stepping parameters lie within physically
        *        meaningful ranges. If the input time step is not valid, it adjusts it giving a warning.
        * @throw std::invalid_argument If any parameter violates stability constraints.
        */
        void validate() {
            // Check inputs
            if (t_end <= 0.0)
                throw std::invalid_argument(
                    config_err_msg(KK::Time, KK::TimeParams::T_End, "must be positive."));
            if (inp_dt <= 0.0)
                throw std::invalid_argument(
                    config_err_msg(KK::Time, KK::TimeParams::DT, "must be positive."));
            if (inp_dt > t_end)
                throw std::invalid_argument(
                    config_err_msg(KK::Time, KK::TimeParams::DT, "must be smaller that total time."));

            // 1. Calculate the number of steps by rounding up (ceil)
            N_time_steps = static_cast<int>(std::ceil(t_end / inp_dt));

            // 2. Uniformly distribute the time steps to fit t_end exactly
            dt = t_end / N_time_steps;

            // 3. Inform/Warn the user if their input was altered
            // (Using a small epsilon for floating-point comparison safety)
            if (std::abs(dt - inp_dt) > 1e-12) {
                std::cout << "[Warning] Time step dt (" << dt
                        << "s) does not evenly divide End Time (" << t_end << ").\n"
                        << "          Automatically adjusted dt to " << dt
                        << "across " << N_time_steps << " uniform steps.\n" << std::endl;
            }
        }

        /**
         * @brief Prints the time stepping configuration to standard output.
         * @param indent String prefix for printing alignment.
         */
        void print(const std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Time << "]\n"
                    << indent << "    End Time (" << KK::TimeParams::T_End << ")                     :   " << t_end <<
                    "\n"
                    << indent << "    User Input Time Step (" << KK::TimeParams::Inp_DT << ")        :   " << inp_dt <<
                    "\n"
                    << indent << "    Actual Time Step (" << KK::TimeParams::DT << ")                :   " << dt << "\n"
                    << indent << "    Number Actual Time Steps (" << KK::TimeParams::N_TSteps << ")  :   " <<
                    N_time_steps << "\n";
        }
    };

    /**
     * @brief Configuration parameters for the Monte Carlo engine.
     *
     * Controls the MonteCarlo size, parallelization batching, hardware memory limits,
     * and the stochastic seed for path generation.
     */
    struct MCConfig {
        bool normalize_prices = true;
        ///< Automatically scales prices (monetary parameters) by a factor N=S0 (so that S0=1) for numerical stability if set to true. If false it does nothing.
        int N_Paths = 1000; ///< Total number of SDE realizations.
        int batch_size = 0;
        ///< Number of paths per kernel launch. (Set to 0 for automatic tuning, -1 for tot numb simulations.)
        uint64_t rng_seed = 184467440737095ULL; ///< Initial seed for the independent random number generation.
        long long Max_VRAM_MB = 256; ///< Limit on GPU VRAM allocation.
        long long Max_CPU_RAM_MB = 4000; ///< Limit on CPU host memory allocation.
        bool analyze_risk_neutral_payoff_distribution = false;
        ///< Analyze the distribution of the risk-neutral discounted payoffs (The payoffs are sorted and the percentiles are computed and discounting is applied)
        // Greeks
        bool compute_delta_et_gamma = false;
        ///< If true, compute delta, the sensitivity to underlying spot price and gamma, Rate of change of Delta. Computed using Central Difference on Spot price.
        bool compute_vega = false;
        ///< If true, compute vega, i.e. the sensitivity to volatility. Computed using Central Difference on Initial Volatility.
        bool compute_rho = false;
        ///< If true, compute rho, i.e. the sensitivity to interest rates. Computed using Central Difference on Risk-Free Rate.
        bool compute_theta = false; ///< If true, compute theta, i.e. the Sensitivity to final time
        double spot_price_relative_bump_size = 0.001;
        ///< Relative bump size for Delta and Gamma (Spot Price). Default is 0.001 (0.1%). Must be strictly positive and <= 0.05.
        double volatility_absolute_bump_size = 0.01;
        ///< Absolute bump size for Vega (Initial Volatility, bump on sqrt(v0)).  Default is 0.01. Must be strictly positive and <= 0.10.
        double risk_free_rate_absolute_bump_size = 0.0001;
        ///< Absolute bump size for Rho (Risk-Free Interest Rate). Default is 0.0001. Must be strictly positive and <= 0.05.
        /**
       * @brief Absolute bump size for Theta (Time to Maturity).
       * @details Applied as a fixed reduction to the continuous time to maturity ($T$):
       * $h = \text{time\_absolute\_bump\_size}$. The number of simulation steps ($N$)
       * is kept strictly constant, meaning the solver smoothly shrinks the grid step
       * $dt_{new} = (T - h) / N$. This preserves Common Random Numbers (CRN) and
       * avoids memory reallocation.
       * @note Default is 1/365.0 (one day). Must be strictly positive.
       */
        double time_absolute_bump_size = 1.0 / 365.0;


        /**
       * @brief Validates hardware limits and simulation parameters.
       * @throw std::invalid_argument If inputs are out of range, or violate memory bounds.
       */
        void validate() const {
            if (N_Paths < 1)
                throw std::invalid_argument(
                    config_err_msg(KK::MC, KK::MCParams::N_Realizations, "Must be a positive integer."));
            if (batch_size < -1)
                throw std::invalid_argument(config_err_msg(KK::MC, KK::MCParams::Batch_Size,
                                                           "Must be a positive integer if you want to impose it, -1 if you want it equal to the max number of paths, "
                                                           "and 0 if you want to let the machine handle this."));
            if (batch_size > N_Paths)
                throw std::invalid_argument(config_err_msg(
                    KK::MC, KK::MCParams::Batch_Size,
                    "The batch size must be smaller that the notal number of realizations."));
            if (rng_seed == 0) {
                throw std::invalid_argument(config_err_msg(KK::MC, KK::MCParams::RNG_Seed,
                                                           "The random number seed cannot be 0."));
            }
            if (rng_seed >= 18446744073709551615ULL) {
                config_err_msg(KK::MC, KK::MCParams::RNG_Seed,
                               "[Warning] Seed is at maximum uint64 limit. Did you pass a negative number?");
            }
            if (Max_VRAM_MB <= 0)
                throw std::invalid_argument(
                    config_err_msg(KK::MC, KK::MCParams::Max_VRAM_MB, "must be a positive integer."));
            if (Max_CPU_RAM_MB <= 0)
                throw std::invalid_argument(
                    config_err_msg(KK::MC, KK::MCParams::Max_CPU_RAM_MB, "must be a positive integer."));

            validate_bump_sizes_for_greeks_computations();
        }

        /**
     * @brief Validates that Greek bump sizes are mathematically safe.
     * @details Prevents division-by-zero (if h=0) and catastrophic truncation
     * errors caused by excessively large finite difference steps.
     * @throws std::invalid_argument If any bump size falls outside the mathematically stable bounds.
     */
        void validate_bump_sizes_for_greeks_computations() const {
            // 1. Spot Bump Check (Relative)
            if (spot_price_relative_bump_size <= 0.0 || spot_price_relative_bump_size > 0.05) {
                throw std::invalid_argument(
                    config_err_msg(KK::MC,
                                   KK::MCParams::spot_price_relative_bump_size,
                                   "must be strictly positive and <= 5% (0.05)."));
            }

            // 2. Volatility Bump Check (Absolute)
            if (volatility_absolute_bump_size <= 0.0 || volatility_absolute_bump_size > 0.10) {
                throw std::invalid_argument(
                    config_err_msg(KK::MC,
                                   KK::MCParams::volatility_absolute_bump_size,
                                   "must be strictly positive and <= 0.10."));
            }

            // 3. Interest Rate Bump Check (Absolute)
            if (risk_free_rate_absolute_bump_size <= 0.0 || risk_free_rate_absolute_bump_size > 0.05) {
                throw std::invalid_argument(
                    config_err_msg(KK::MC,
                                   KK::MCParams::volatility_absolute_bump_size,
                                   "must be strictly positive and <= 0.05."));
            }

            // 4. Time Bump Check (Absolute)
            if (time_absolute_bump_size <= 0.0 || time_absolute_bump_size > 1.0) {
                throw std::invalid_argument(
                config_err_msg(KK::MC,
                               KK::MCParams::time_absolute_bump_size,
                               "must be strictly positive and <= 1.0 year."));
            }
        }

        /** @brief Outputs MC configuration summary to standard console. */
        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::MC << "]\n"
                    << indent << "    Normalize Prices               :    " << normalize_prices << "\n"
                    << indent << "    Number of Realizations         :    " << N_Paths << "\n"
                    << indent << "    Batch Size required            :    " << batch_size << "\n"
                    << indent << "    Seed Random Number Generator   :    " << rng_seed << "\n"
                    << indent << "    User CPU RAM Limit (MB)        :     " << Max_CPU_RAM_MB << "\n"
                    << indent << "    User GPU VRAM Limit (MB)       :    " << Max_VRAM_MB << "\n"
                    << indent << "    Analyze risk-neutral discounted payoffs distribution   :    " <<
                    analyze_risk_neutral_payoff_distribution << "\n";

            // Print greeks info
            // 2. Delta & Gamma
            std::cout << indent << "    Evaluate Delta and Gamma       :    " << compute_delta_et_gamma << "\n";
            if (compute_delta_et_gamma) {
                std::cout << indent << "      -> S0 Bump Size (Rel)      :    " << spot_price_relative_bump_size << "\n";
            }
            // 3. Vega
            std::cout << indent << "    Evaluate Vega                  :    " << compute_vega << "\n";
            if (compute_vega) {
                std::cout << indent << "      -> Vol0 Bump Size (Abs)       :    " << volatility_absolute_bump_size <<
                        "\n";
            }
            // 4. Rho
            std::cout << indent << "    Evaluate Rho                   :    " << compute_rho << "\n";
            if (compute_rho) {
                std::cout << indent << "      -> r Bump Size (Abs)      :    " << risk_free_rate_absolute_bump_size <<
                        "\n";
            }
            // 5. Theta
            std::cout << indent << "    Evaluate Theta                 :    " << compute_theta << "\n";
            if (compute_theta) {
                std::cout << indent << "      -> T Bump Size (Abs)      :    " << time_absolute_bump_size<<"\n";
            }
        }
    };


    /**
     * @brief Configuration for file system I/O and logging.
     *
     * Manages output paths, naming conventions, and file formats (binary/text).
     * Automatically ensures the target directory tree exists on disk upon validation.
     */
    struct OutputConfig {
        std::string out_dir = "outputs"; ///< Target directory for generated logs and paths.
        std::string filename_paths_out = "QuantKos.paths"; ///< Filename for the simulated path storage.
        std::string filename_log = "QuantKos.log"; ///< Filename for the runtime execution log.
        KI::IOFormat format = KI::IOFormat::TXT; ///< Output Data format (e.g., BIN or TXT).

        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Output << "]\n"
                    << indent << "    Output Directory         :     " << out_dir << "\n"
                    << indent << "    Format Output Files      :     " << enum_to_string(format) << "\n"
                    << indent << "    Name Output File Paths   :     " << filename_paths_out << "\n"
                    << indent << "    Name Log File            :     " << filename_log << "\n";
        }

        /**
     * @brief Creates the output directory structure if it does not exist.
     * @throw std::invalid_argument If filesystem permissions prevent directory creation.
     */
        void validate() const {
            // Check if a file path is empty
            if (filename_paths_out.empty()) {
                // ...
            }

            // Create Output directories if given
            if (!out_dir.empty()) {
                try {
                    // Generates full tree tree structure (e.g., "build/outputs/bin") safely.
                    // Does absolutely nothing if the directory already exists.
                    std::filesystem::create_directories(out_dir);
                } catch (const std::filesystem::filesystem_error &e) {
                    // If OS denies creation (No permissions, invalid disk path, etc.), crash instantly!
                    throw std::invalid_argument(
                        "[Configuration Error] The assigned output directory '" + out_dir +
                        "' could not be prepared or accessed.\nDetails: " + e.what()
                    );
                }
            }
        }
    };

    /**
     * @brief Master configuration container for the simulation engine.
     *
     * Aggregates all market, model, Numerical, and hardware settings. This structure serves
     * as the single source of truth for the SDE solvers and Monte Carlo kernels.
     */
    struct UInputs {
        MarketConfig market;
        OptionsConfig options;
        MathModelConfig model;
        NumSchemeConfig scheme;
        TimeConfig time;
        MCConfig mc;
        OutputConfig output;

        /**
         * @brief Performs a full validation of all sub-configurations.
         *
         * Should be called immediately after parsing input files (e.g., JSON/YAML)
         * and before initializing the simulation engines.
         *
         * @throw std::runtime_error If any component parameter fails validation.
         */
        void validate() {
            market.validate();
            options.validate(market.S0);
            model.validate();
            scheme.validate();
            time.validate();
            mc.validate();
            output.validate();

            std::cout << "  [ Config ] Input configuration validated successfully\n\n";
        }

        /**
          * @brief Normalizes monetary parameters relative to S0.
          * Help protecting algorithms from floating-point overflow/underflows and ill-conditioned matrices.
          */
        void apply_price_scaling() {
            if (mc.normalize_prices) {
                // 1. Save the original S0 to scale the final option price back later
                price_scale_factor = market.S0;

                // 2. Scale the Market and Option inputs
                market.S0 = 1.0;
                options.StrikePrice /= price_scale_factor;
                options.BarrierPrice /= price_scale_factor;

                std::cout << "  [ Config ] Monetary parameters scaled. "
                        << "Scale Factor = N = " << price_scale_factor << "  (S0_n = S0/N = 1)\n";
            }
        }

        /** @brief Prints a formatted summary of the entire simulation configuration. */
        void print_summary() const {
            constexpr std::string_view indent = "    ";

            std::cout << "\n" << indent << "========================================================\n";
            std::cout << indent << "                      MC CONFIGURATION                    \n";
            std::cout << indent << "========================================================\n";
            market.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            options.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            model.print(market, indent);
            std::cout << indent << "--------------------------------------------------------\n";
            scheme.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            time.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            mc.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            output.print(indent);
            std::cout << indent << "========================================================\n" << std::endl;
        }

        /** @brief Tells if you should sync the paths at the end of each batch from device to host using kokkos
         */
        [[nodiscard]] bool requires_paths_allocated() const {
            // Check if the user requested path outputs (assuming std::string)
            const bool wants_file_output = !output.filename_paths_out.empty();

            // Check if the Longstaff-Schwartz backward induction needs the paths
            const bool is_american = (options.opt_type == KI::OptType::American);

            return wants_file_output || is_american;
        }

        /** @brief Safely get scaling factor
        */
        [[nodiscard]] double get_price_scaling_factor() const {
            return price_scale_factor;
        }

        /** @brief Tells if at least one of the greeks must be computed
        */
        [[nodiscard]] bool must_compute_greeks() const {
            if (mc.compute_delta_et_gamma || mc.compute_vega || mc.compute_rho || mc.compute_theta) {
                return true;
            }

            return false;
        }

    private:
        double price_scale_factor = 1.0;
        ///< Stores the scaling factor used by the engine to normalize the monetary parameters (S0, K, Barrier, etc). Default is 1.0 (no scaling).
    };
}
