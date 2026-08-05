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
#include <stdexcept>
#include <Kokkos_Sort.hpp>

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
         * @brief Aggregated flat array containing all individual path payoff results.
         */
        Kokkos::View<KT::Real *, Kokkos::HostSpace> all_payoffs;

        /**
         * @brief Storage container for final simulation metrics and distribution percentiles.
         */
        struct MCOpPrices {
            double option_price = -999.; ///< Discounted expected value of the option.
            double standard_error = -999.; ///< Statistical standard error of the estimate.
            double prob_itm = -999.; ///< Empirical probability of finishing In-The-Money (payoff > 0).
            double p0 = -999.; ///< Minimum discounted payout observed.
            double p1 = -999.; ///< 1st percentile.
            double p5 = -999.; ///< 5th percentile.
            double p10 = -999.; ///< 10th percentile.
            double p20 = -999.; ///< 20th percentile.
            double p30 = -999.; ///< 30th percentile.
            double p40 = -999.; ///< 40th percentile.
            double median = -999.; ///< 50th percentile (Median payout).
            double p60 = -999.; ///< 60th percentile.
            double p70 = -999.; ///< 70th percentile.
            double p80 = -999.; ///< 80th percentile.
            double p90 = -999.; ///< 90th percentile.
            double p95 = -999.; ///< 95th percentile.
            double p99 = -999.; ///< 99th percentile.
            double p100 = -999.; ///< Maximum discounted payout observed.
            bool is_computed = false; ///< Safety flag protecting against premature reads.
        };

        /**
         * @brief Constructs the pricer and reserves heap memory for path payloads on Host CPU.
         * @param conf Reference to simulation user parameters.
         */
        explicit OptionPricer(const KC::UInputs &conf) : config(conf) {
            //all_payoffs.reserve(conf.mc.N_Paths);
            all_payoffs = Kokkos::View<KT::Real *, Kokkos::HostSpace>("Host_All_Payoffs_For_Sorting",
                                                                      conf.mc.N_Paths);
        }

        /// @name Lifecycle Safeguards
        ///@{
        // Prevent accidental deep-copying of the payoffs
        OptionPricer(const OptionPricer &) = delete; ///< Copying is prohibited
        OptionPricer &operator=(const OptionPricer &) = delete; ///< Copy assignment is prohibited.

        // Allow payoffs to be moved safely if handled by orchestrators
        OptionPricer(OptionPricer &&) = default; ///< Move constructor is supported
        OptionPricer &operator=(OptionPricer &&) = delete; ///< Move assignment is prohibited.

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
                //all_payoffs.push_back(curr_payoff); // Store payoff to host std::vector
                all_payoffs(current_path_idx++) = curr_payoff; // Store payoff to host kokkos view
                const auto payoff = static_cast<double>(curr_payoff); // Cast to double befor multipying
                total_payoff_sum += payoff;
                total_squared_payoff_sum += (payoff * payoff); // Safely multiply payoff in double precision
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
            const auto N = static_cast<double>(config.mc.N_Paths);
            if (all_payoffs.empty() || N <= 0) {
                throw std::runtime_error("Cannot compute metrics: No paths accumulated.");
            }

            const auto start_time = std::chrono::steady_clock::now();

            // Compute discount factor (do not apply to backwards options since already applied)
            double discount_factor = KT::real_one;
            if (config.options.opt_type != KI::OptType::American) {
                const auto r = static_cast<double>(config.market.r);
                const auto t = static_cast<double>(config.time.t_end);
                discount_factor = std::exp(-r * t); // Safely compute in double precision
            }

            // Compute stats
            const double sample_mean = total_payoff_sum / N;

            size_t itm_count = 0;
            double sum_sq_diff = 0.0;
            //for(const auto& p : all_payoffs) {
            for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
                //const auto payoff = static_cast<double>(p);
                const KT::Real curr_payoff = all_payoffs(i);
                const double diff = static_cast<double>(curr_payoff) - sample_mean;
                sum_sq_diff += diff * diff;
                if (curr_payoff > eps_payoff) itm_count++; // Count paths that survived/won
            }

            const double safe_variance = sum_sq_diff / (N - 1.0); // Unbiased sample variance
            metrics.prob_itm = static_cast<double>(itm_count) / N;

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
        //std::vector<KT::Real> all_payoffs; ///< Aggregated flat array containing all individual path payoff results.
        size_t current_path_idx = 0; ///< Track the payoff is being added to all payoffs
        double total_payoff_sum = 0.; ///< Cumulative summary of all processed payoffs values.
        double total_squared_payoff_sum = 0.; ///< Cumulative summary of squared payoffs values (for stats).
        double computation_time = 0.; ///< Internal profiling metric tracking processing overhead.
        const KT::Real eps_payoff = KT::is_real_using_single_precision() ? 1e-6f : 1e-12;
        MCOpPrices metrics; ///< Local state storage wrapper instance.

        /**
         * @brief Sorts vector results in-place to compute distribution empirical quantiles.
         * @param discount_factor Constant asset continuous discount modifier asset multiplier.
         * @param N_paths Total active configurations paths dimension scalar value.
         */
        void analyze_option_price_distribution(const double discount_factor, const double N_paths) {
            // Compute the percentiles

            //std::sort(std::execution::par, all_payoffs.begin(), all_payoffs.end());
            Kokkos::sort(all_payoffs);

            metrics.p0 = static_cast<double>(all_payoffs(0)) * discount_factor;
            metrics.p1 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.01))) * discount_factor;
            metrics.p5 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.05))) * discount_factor;
            metrics.p10 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.10))) * discount_factor;
            metrics.p20 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.20))) * discount_factor;
            metrics.p30 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.30))) * discount_factor;
            metrics.p40 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.40))) * discount_factor;
            metrics.median = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.50))) * discount_factor;
            metrics.p60 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.60))) * discount_factor;
            metrics.p70 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.70))) * discount_factor;
            metrics.p80 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.80))) * discount_factor;
            metrics.p90 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.90))) * discount_factor;
            metrics.p95 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.95))) * discount_factor;
            metrics.p99 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.99))) * discount_factor;
            metrics.p100 = static_cast<double>(all_payoffs(all_payoffs.size() - 1)) * discount_factor;
        }
    };
} // namespace KOps::Engine
