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

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "../Typedefs.hpp"
#include "../config/Config.hpp"
#include "../config/ConfigFileEnums.hpp"
#include "../memory/MemoryTypes.hpp"

namespace quantkos::Engine {
    namespace KT = quantkos::Types;
    namespace KI = quantkos::Implemented;
    namespace KC = quantkos::Config;

    /**
     * @brief Evaluates and reports final statistical metrics from Monte Carlo simulation payoffs.
     * * This class accumulates raw path payouts from device memory buffers, calculates unbiased
     * statistical metrics (mean option price and standard error), applies financial discounting,
     * and performs a percentile-based distribution analysis on the host CPU.
     */
    class OptionPricer {
    public:

        /**
         * @brief Storage container for final simulation metrics and distribution percentiles.
         */
        struct MCOpPrices {
            KT::Real option_price = -999.; ///< Discounted expected value of the option.
            KT::Real standard_error = -999.; ///< Statistical standard error of the estimate.
            KT::Real prob_itm = -999.; ///< Empirical probability of finishing In-The-Money (payoff > 0).
            KT::Real p0 = -999.; ///< Minimum discounted payout observed.
            KT::Real p1 = -999.; ///< 1st percentile.
            KT::Real p5 = -999.; ///< 5th percentile.
            KT::Real p10 = -999.; ///< 10th percentile.
            KT::Real p20 = -999.; ///< 20th percentile.
            KT::Real p30 = -999.; ///< 30th percentile.
            KT::Real p40 = -999.; ///< 40th percentile.
            KT::Real median = -999.; ///< 50th percentile (Median payout).
            KT::Real p60 = -999.; ///< 60th percentile.
            KT::Real p70 = -999.; ///< 70th percentile.
            KT::Real p80 = -999.; ///< 80th percentile.
            KT::Real p90 = -999.; ///< 90th percentile.
            KT::Real p95 = -999.; ///< 95th percentile.
            KT::Real p99 = -999.; ///< 99th percentile.
            KT::Real p100 = -999.; ///< Maximum discounted payout observed.
            bool is_computed = false; ///< Safety flag protecting against premature reads.
        };

        /**
         * @brief Constructs the pricer and reserves heap memory for path payloads on Host CPU.
         * @param conf Reference to simulation user parameters.
         */
        explicit OptionPricer(const KC::UInputs &conf) : config(conf) {
            all_payoffs.reserve(conf.mc.N_Paths);
        }

        /// @name Lifecycle Safeguards
        ///@{
        // Prevent accidental deep-copying of the payoffs
        OptionPricer(const OptionPricer&) = delete; ///< Copying is prohibited
        OptionPricer& operator=(const OptionPricer&) = delete; ///< Copy assignment is prohibited.

        // Allow payoffs to be moved safely if handled by orchestrators
        OptionPricer(OptionPricer&&) = default; ///< Move constructor is supported
        OptionPricer& operator=(OptionPricer&&) = delete; ///< Move assignment is prohibited.

        // Default destructor
        ~OptionPricer() = default; ///< Default destructor.
        ///@}

        // To be called inside the batch loop
        /**
         * @brief Collects batch-specific path results from the host memory mirror views into a single heap vector for the payoffs of the entire MonteCarlo.
         * * @param h_payoffs Host mirror view holding the current batch payoffs.
         * @param curr_batch_size The active number of paths processed in this batch loop.
         * @note To be executed inside your main Monte Carlo batch iterations loop.
         */
        void accumulate_batch_payoffs(const HostBatchPayoffView &h_payoffs, const int curr_batch_size) {
            for (int i = 0; i < curr_batch_size; ++i) {
                const KT::Real curr_payoff = h_payoffs(i); // Get payoff value from device
                total_payoff_sum += curr_payoff;
                total_squared_payoff_sum += (curr_payoff * curr_payoff);
                all_payoffs.push_back(curr_payoff); // Store payoff to host std::vector
            }
        }

        /**
         * @brief Computes final option statistics and performs empirical quantiles.
         * * Calculates unbiased sample statistics, determines the probability of ending
         * in-the-money, and applies the continuous risk-free discount factor:
         * * $$D = e^{-r \cdot T}$$
         * * @note For American options processed via backward induction (LSM), discounting is bypassed
         * here because the temporal discounting operation is handled step-by-step during the backward induction loop.
         * * @throw std::runtime_error If called before any path payoffs have been accumulated.
         */
        void evaluate_option_price() {
            const KT::Real N = config.mc.N_Paths;
            if (all_payoffs.empty() || N <= 0) {
                throw std::runtime_error("Cannot compute metrics: No paths accumulated.");
            }

            const auto start_time = std::chrono::steady_clock::now();

            // Compute discount factor (do not apply to backwards options since already applied)
            KT::Real discount_factor = KT::real_one;
            if (config.options.opt_type != KI::OptType::American) {
                discount_factor = std::exp(-config.market.r * config.time.t_end);
            }

            // Compute stats
            const KT::Real sample_mean = total_payoff_sum / N;

            size_t itm_count = 0;
            KT::Real sum_sq_diff = 0.0;
            for(const auto& p : all_payoffs) {
                sum_sq_diff += (p - sample_mean) * (p - sample_mean);
                if (p > 1e-12) itm_count++; // Count paths that survived/won
            }

            const KT::Real safe_variance = sum_sq_diff / (N - 1.0); // Unbiased sample variance
            metrics.prob_itm = static_cast<KT::Real>(itm_count) / N;

            // Apply Discounting to get the Option Price etc.
            metrics.option_price = sample_mean * discount_factor;
            metrics.standard_error = std::sqrt(safe_variance / N) * discount_factor;

            // Analyze shape of option price distribution
            analyze_option_price_distribution(discount_factor, N);

            // Mark as successfully computed
            metrics.is_computed = true;

            // Store time
            const auto end_time = std::chrono::steady_clock::now();
            const std::chrono::duration<double> elapsed = end_time - start_time;
            computation_time = elapsed.count();
        }

        // 3. Pure Reporting Phase (Uses the stored metrics)
        /**
         * @brief Outputs a comprehensive simulation summary report (containing option prices etc.) to the standard console.
         * @throw std::runtime_error If executed before evaluate_option_price() completes successfully.
         */
        void print_info_option_price() const {
            if (!metrics.is_computed) {
                throw std::runtime_error(
                    "Attempted to print report before computing metrics. Call compute_metrics() first.");
            }

            std::cout << "  ========================================================\n";
            std::cout << "                     SIMULATION RESULTS\n";
            std::cout << "  ========================================================\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "   Estimated Option Price   :  " << metrics.option_price << "\n";
            std::cout << "   Statistical Std Error    :  " << metrics.standard_error << "\n";
            std::cout << "   Probability of ITM       :  " << (metrics.prob_itm * 100.0) << " %\n";
            std::cout << "  --------------------------------------------------------\n";
            std::cout << "   [Discounted Payout Distribution Tail Metrics]\n";
            std::cout << "   Min  (Lower)             :  " << metrics.p0 << "\n";
            std::cout << "   1st  Percentile          :  " << metrics.p1 << "\n";
            std::cout << "   5th  Percentile          :  " << metrics.p5 << "\n";
            std::cout << "   10th Percentile          :  " << metrics.p10 << "\n";
            std::cout << "   20th Percentile          :  " << metrics.p20 << "\n";
            std::cout << "   30th Percentile          :  " << metrics.p30 << "\n";
            std::cout << "   40th Percentile          :  " << metrics.p40 << "\n";
            std::cout << "   50th Percentile (Median) :  " << metrics.median << "\n";
            std::cout << "   60th Percentile          :  " << metrics.p60 << "\n";
            std::cout << "   70th Percentile          :  " << metrics.p70 << "\n";
            std::cout << "   80th Percentile          :  " << metrics.p80 << "\n";
            std::cout << "   90th Percentile          :  " << metrics.p90 << "\n";
            std::cout << "   95th Percentile          :  " << metrics.p95 << "\n";
            std::cout << "   99th Percentile          :  " << metrics.p99 << "\n";
            std::cout << "   Max  (Upper)             :  " << metrics.p100 << "\n";
            std::cout << "  --------------------------------------------------------\n";
            std::cout << "   Time Spent to Compute Metrics   :  " << computation_time << "s \n";
            std::cout << "  ========================================================\n" << std::endl;
        }

        // 4. Getter so other parts of the program can use the raw numbers
        /**
         * @brief Exposes read-only access to the final computed metrics data structure.
         * @return An immutable reference to the completed option data structure.
         * @throw std::runtime_error If metrics have not been computed yet.
         */
        [[nodiscard]] const MCOpPrices &get_option_price_data() const {
            if (!metrics.is_computed) {
                throw std::runtime_error("Metrics have not been computed yet.");
            }
            return metrics;
        }

    private:
        const KC::UInputs &config; ///< Reference to the user input configuration.
        std::vector<KT::Real> all_payoffs; ///< Aggregated flat array containing all individual path payoff results.
        KT::Real total_payoff_sum = 0.; ///< Cumulative summary of all processed payoffs values.
        KT::Real total_squared_payoff_sum = 0.; ///< Cumulative summary of squared payoffs values (for analytical tracking).
        double computation_time = 0.; ///< Internal profiling metric tracking processing overhead.
        MCOpPrices metrics; ///< Local state storage wrapper instance.

        /**
         * @brief Sorts vector results in-place to compute distribution empirical quantiles.
         * @param discount_factor Constant asset continuous discount modifier asset multiplier.
         * @param N_paths Total active configurations paths dimension scalar value.
         */
        void analyze_option_price_distribution(const KT::Real discount_factor, const KT::Real N_paths) {
            // Compute the percentiles

            std::sort(all_payoffs.begin(), all_payoffs.end());

            metrics.p0 = all_payoffs[0] * discount_factor;
            metrics.p1 = all_payoffs[static_cast<size_t>(N_paths * 0.01)] * discount_factor;
            metrics.p5 = all_payoffs[static_cast<size_t>(N_paths * 0.05)] * discount_factor;
            metrics.p10 = all_payoffs[static_cast<size_t>(N_paths * 0.10)] * discount_factor;
            metrics.p20 = all_payoffs[static_cast<size_t>(N_paths * 0.20)] * discount_factor;
            metrics.p30 = all_payoffs[static_cast<size_t>(N_paths * 0.30)] * discount_factor;
            metrics.p40 = all_payoffs[static_cast<size_t>(N_paths * 0.40)] * discount_factor;
            metrics.median = all_payoffs[static_cast<size_t>(N_paths * 0.50)] * discount_factor;
            metrics.p60 = all_payoffs[static_cast<size_t>(N_paths * 0.60)] * discount_factor;
            metrics.p70 = all_payoffs[static_cast<size_t>(N_paths * 0.70)] * discount_factor;
            metrics.p80 = all_payoffs[static_cast<size_t>(N_paths * 0.80)] * discount_factor;
            metrics.p90 = all_payoffs[static_cast<size_t>(N_paths * 0.90)] * discount_factor;
            metrics.p95 = all_payoffs[static_cast<size_t>(N_paths * 0.95)] * discount_factor;
            metrics.p99 = all_payoffs[static_cast<size_t>(N_paths * 0.99)] * discount_factor;
            metrics.p100 = all_payoffs[all_payoffs.size() - 1] * discount_factor;
        }
    };
} // namespace KOps::Engine
