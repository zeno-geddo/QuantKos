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

namespace KOps::Engine {
    namespace KT = KOps::Types;
    namespace KI = KOps::Implemented;
    namespace KC = KOps::Config;

    class OptionPricer {
    public:
        struct MCEngine {
            KT::Real option_price = -999.;
            KT::Real standard_error = -999.;
            KT::Real prob_itm = -999.; // Probability of finishing ITM
            KT::Real p0 = -999.;
            KT::Real p1 = -999.;
            KT::Real p5 = -999.;
            KT::Real p10 = -999.;
            KT::Real p20 = -999.;
            KT::Real p30 = -999.;
            KT::Real p40 = -999.;
            KT::Real median = -999.;
            KT::Real p60 = -999.;
            KT::Real p70 = -999.;
            KT::Real p80 = -999.;
            KT::Real p90 = -999.;
            KT::Real p95 = -999.;
            KT::Real p99 = -999.;
            KT::Real p100 = -999.;
            bool is_computed = false; // Safety flag
        };

        explicit OptionPricer(const KC::UInputs &conf) : config(conf) {
            all_payoffs.reserve(conf.mc.N_Paths);
        }

        // Prevent accidental deep-copying of the payoffs
        OptionPricer(const OptionPricer&) = delete;
        OptionPricer& operator=(const OptionPricer&) = delete;

        // Allow payoffs to be moved safely if handled by orchestrators
        OptionPricer(OptionPricer&&) = default;
        OptionPricer& operator=(OptionPricer&&) = delete;

        // Default destructor
        ~OptionPricer() = default;

        // To be called inside the batch loop
        void accumulate_batch_payoffs(const HostPayoffView &h_payoffs, const int curr_batch_size) {
            for (int i = 0; i < curr_batch_size; ++i) {
                const KT::Real curr_payoff = h_payoffs(i); // Get payoff value from device
                total_payoff_sum += curr_payoff;
                total_squared_payoff_sum += (curr_payoff * curr_payoff);
                all_payoffs.push_back(curr_payoff); // Store payoff to host std::vector
            }
        }

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
        [[nodiscard]] const MCEngine &get_option_price_data() const {
            if (!metrics.is_computed) {
                throw std::runtime_error("Metrics have not been computed yet.");
            }
            return metrics;
        }

    private:
        const KC::UInputs &config;
        std::vector<KT::Real> all_payoffs;
        KT::Real total_payoff_sum = 0.;
        KT::Real total_squared_payoff_sum = 0.;
        double computation_time = 0.;
        MCEngine metrics;


        void analyze_option_price_distribution(KT::Real discount_factor, KT::Real N_paths) {
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
