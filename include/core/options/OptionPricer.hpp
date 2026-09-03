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
         * @brief Storage container for first reduction loop in batch accumulation.
         */
        struct PayoffsBatchStats {
            double sum = 0.0;
            size_t itm = 0;

            // Tell Kokkos how to combine the results of two different threads
            KOKKOS_INLINE_FUNCTION void operator+=(const PayoffsBatchStats &src) {
                sum += src.sum;
                itm += src.itm;
            }
        };

        /**
         * @brief Constructs the pricer and reserves heap memory for path payloads on Host CPU.
         * @param conf Reference to simulation user parameters.
         */
        explicit OptionPricer(const KC::UInputs &conf) : config(conf) {
            //all_payoffs.reserve(conf.mc.N_Paths);
            // Allocate memory for all payoffs to the host only if they has to be sorted to analyzed their distribution
            if (config.mc.analyze_risk_neutral_payoff_distribution) {
                all_payoffs = Kokkos::View<KT::Real *, Kokkos::HostSpace>("Host_All_Payoffs_For_Sorting",
                                                                          conf.mc.N_Paths);
            }
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


        /**
         * @brief Accumulate mean and variance (to be then renormalized since using chang's algorithm) of the payoffs.
         * Also accumulates the payoffs to host if they need to be sorted later.
         * * @param d_payoffs Device view holding the current batch payoffs.
         * @param curr_batch_size The active number of paths processed in this batch loop.
         * @note To be executed at the end of each batch of the montecarlo.
         * @note The variance is computed using Chang's algorithm, while the mean as a weighted average to prevent the case where you have 2 or 3 batches only with nearly equal means
         * (Safe from cancellation, and 64-bit double prevents swamping for typical N).  More on Chang's algorithm on : https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        */
        void accumulate_batch_payoffs(const DevBatchPayoffView &d_payoffs,
                                      const int curr_batch_size) {
            // STEP 1 : Get Batch Sum and ITM count on device
            PayoffsBatchStats batch_stats;
            const auto eps = eps_payoff;
            Kokkos::parallel_reduce("Batch_Sum_And_ITM_Pass", curr_batch_size,
                                    KOKKOS_LAMBDA(const int i, PayoffsBatchStats &l_stats) {
                                        const double payoff = static_cast<double>(d_payoffs(i));
                                        l_stats.sum += payoff; // Accumulate sum
                                        if (payoff > eps) l_stats.itm += 1; // Accumulate ITM count
                                    }, batch_stats);
            const double payoff_batch_sum = batch_stats.sum;
            const size_t batch_itm_count = batch_stats.itm;
            const double batch_mean = payoff_batch_sum / curr_batch_size;


            // STEP 2: Get Batch M2 (Sum of squared differences) on device---
            double payoff_batch_sum_diff2 = 0.0;
            Kokkos::parallel_reduce("Batch_M2_Pass", curr_batch_size,
                                    KOKKOS_LAMBDA(const int i, double &sum_diff2) {
                                        const double diff = static_cast<double>(d_payoffs(i)) - batch_mean;
                                        sum_diff2 += (diff * diff);
                                    }, payoff_batch_sum_diff2);


            // ====================================================================
            // --- CPU MERGE: Mean + Chan's Exact Variance ---
            // ====================================================================
            const size_t new_total_paths = n_payoffs_processed + curr_batch_size;

            // 1. Calculate delta using the OLD global_mean (Required for payoff_sum_diff2)
            const double delta = batch_mean - payoff_mean;

            // 2. Weighted Average  to update the mean
            payoff_mean = (payoff_mean * static_cast<double>(n_payoffs_processed) +
                           batch_mean * static_cast<double>(curr_batch_size)
                          ) / static_cast<double>(new_total_paths);

            // 3. Chan's Formula to update sum of diff squared (needed for variance merging)
            // Note : M2 Chang's parallel algo : https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            payoff_M2_Chang += payoff_batch_sum_diff2 +
                    (delta * delta) *
                    (static_cast<double>(n_payoffs_processed * curr_batch_size)
                     / static_cast<double>(new_total_paths));

            // Update trackers
            n_payoffs_processed = new_total_paths;
            itm_count += batch_itm_count;


            // STEP 4 : OPTIONAL: Store arrays ONLY if distribution analysis is active ---
            if (config.mc.analyze_risk_neutral_payoff_distribution) {
                // 1. Destination subview (Where to put it in the global array)
                const auto dest_slice = std::make_pair(current_path_idx, current_path_idx + curr_batch_size);
                const auto destination_subview = Kokkos::subview(all_payoffs, dest_slice);

                // 2. Source subview (Only copy the valid computed paths from this batch buffer!)
                const auto src_slice = std::make_pair(0, curr_batch_size);
                const auto source_subview = Kokkos::subview(d_payoffs, src_slice);

                // 3. Perform device-to-device copy with matching extents
                Kokkos::deep_copy(destination_subview, source_subview);
            }
        }

        /**
        * @brief Computes final option and performs empirical quantiles risk neutral discounted payoff distribution if needed.
        * * Calculates unbiased sample statistics, determines the probability of ending
        * in-the-money, and applies the continuous risk-free discount factor:
        * * $$D = e^{-r \cdot T}$$
        * * @note For American options processed via backward induction (LSM), discounting is bypassed
        * here because the temporal discounting operation is handled step-by-step during the backward induction loop.
        * * @throw std::runtime_error If called with an invalid number of paths.
        */
        void evaluate_option_price() {
            const auto N = static_cast<double>(config.mc.N_Paths);
            if (N <= 0) {
                throw std::runtime_error("Cannot compute metrics: Invalid number of paths.");
            }

            // 1. Compute discount factor (do not apply to backwards options since already applied)
            double discount_factor = KT::real_one;
            if (config.options.opt_type != KI::OptType::American) {
                const auto r = static_cast<double>(config.market.r);
                const auto t = static_cast<double>(config.time.t_end);
                discount_factor = std::exp(-r * t); // Safely compute in double precision
            }

            // 2. Compute Unbiased sample variance: M2 / (N - 1)
            // (We add a small ternary check to prevent division by zero if N == 1)
            const double payoff_variance = (N > 1.0) ? (payoff_M2_Chang / (N - 1.0)) : 0.0;

            // 3. Apply Discounting and scaling to get the Option Price etc.
            // Note: standard_error : how confident we are that our calculated average is the true option price
            const double effective_scale = discount_factor * get_scaling_factor();
            metrics.option_price = payoff_mean * effective_scale;
            metrics.standard_error = std::sqrt(payoff_variance / N) * effective_scale;
            metrics.prob_itm = static_cast<double>(itm_count) / N;


            // 4. Analyze shape of option price distribution ONLY if requested
            if (config.mc.analyze_risk_neutral_payoff_distribution) {
                if (all_payoffs.empty()) {
                    // Safety check just in case memory allocation failed earlier
                    throw std::runtime_error("Distribution analysis requested, but payoffs array is empty.");
                }

                const auto start_time = std::chrono::steady_clock::now();

                analyze_risk_neutral_discounted_payoff_distribution(effective_scale, N);

                // Store time
                const auto end_time = std::chrono::steady_clock::now();
                const std::chrono::duration<double> elapsed = end_time - start_time;
                sorting_time = elapsed.count();
            }

            // Mark as successfully computed
            metrics.is_computed = true;
        }

        /**
         * @brief Outputs a comprehensive simulation summary report (containing option prices etc.) to the standard console.
         * @throw std::runtime_error If executed before evaluate_option_price() completes successfully.
         */
        void print_info_option_price() const {
            if (!metrics.is_computed) {
                throw std::runtime_error(
                    "Attempted to print report before computing metrics. Call the method that computes the payoffs first.");
            }

            std::cout << "  ========================================================\n";
            std::cout << "                     SIMULATION RESULTS\n";
            std::cout << "  ========================================================\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "   Estimated Option Price   :  " << metrics.option_price << "\n";
            std::cout << "   Statistical Std Error    :  " << metrics.standard_error << "\n";
            std::cout << "   Probability of ITM       :  " << (metrics.prob_itm * 100.0) << " %\n";

            if (config.mc.analyze_risk_neutral_payoff_distribution) {
                std::cout << "  --------------------------------------------------------\n";
                std::cout << "   [Risk-Neutral Discounted Payout Distribution]\n";
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
                std::cout << "   Time Spent to Sorting the Payoffs   :  " << sorting_time << "s \n";
            }

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
        size_t current_path_idx = 0; ///< Track the payoff is being added to all payoffs
        size_t n_payoffs_processed = 0; ///< Track the number of payoffs processed.
        size_t itm_count = 0; ///< Count of the ITM paths.
        double payoff_mean = 0.; ///< Global payoffs mean.
        double payoff_M2_Chang = 0.;
        ///< Global M2 updated using Chang's approach.
        double sorting_time = 0.; ///< Internal profiling variable to tracking time spent sorting the payoffs.
        const KT::Real eps_payoff = KT::is_real_using_single_precision() ? 1e-6f : 1e-12; ///< eps to see if a path has itm
        MCOpPrices metrics; ///< Local state storage wrapper instance.

        /**
        * @brief helper function getting the scaling factor needed
         * @return  The scaling factor to be used to get the correct options prices
         */
        [[nodiscard]] double get_scaling_factor() const {
            if (config.options.opt_type == KI::OptType::BinaryCashOrNothing) {
                // For Binary Cash, the GPU just outputs 1.0 if ITM.
                // It is directly scaled by the requested cash payout, ignoring S0.
                return 1.;
            }

            // For all other options, the GPU outputs normalized asset prices. We scale back by S0.
            return config.get_price_scaling_factor();
        }


        /**
         * @brief Sorts payoffs in-place to compute distribution empirical quantiles and apply discount to them.
         * @param effective_scale Discount factor multiplied by scaling factor.
         * @param N_paths Total active configurations paths dimension scalar value.
         */
        void analyze_risk_neutral_discounted_payoff_distribution(const double effective_scale, const double N_paths) {
            // Sort the payoffs
            Kokkos::sort(all_payoffs);

            // Compute the percentiles
            metrics.p0 = static_cast<double>(all_payoffs(0)) * effective_scale;
            metrics.p1 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.01))) * effective_scale;
            metrics.p5 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.05))) * effective_scale;
            metrics.p10 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.10))) * effective_scale;
            metrics.p20 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.20))) * effective_scale;
            metrics.p30 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.30))) * effective_scale;
            metrics.p40 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.40))) * effective_scale;
            metrics.median = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.50))) * effective_scale;
            metrics.p60 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.60))) * effective_scale;
            metrics.p70 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.70))) * effective_scale;
            metrics.p80 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.80))) * effective_scale;
            metrics.p90 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.90))) * effective_scale;
            metrics.p95 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.95))) * effective_scale;
            metrics.p99 = static_cast<double>(all_payoffs(static_cast<size_t>(N_paths * 0.99))) * effective_scale;
            metrics.p100 = static_cast<double>(all_payoffs(all_payoffs.size() - 1)) * effective_scale;
        }
    };
} // namespace KOps::Engine
