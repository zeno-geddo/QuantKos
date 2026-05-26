#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace KOps::Engine {

    class RiskQuantifier {
    public:
        // 1. Define a clean struct to hold the final computed metrics
        struct ResultMetrics {
            double option_price = 0.0;
            double standard_error = 0.0;
            double p1 = 0.0;
            double median = 0.0;
            double p99 = 0.0;
            bool is_computed = false; // Safety flag
        };

        explicit RiskQuantifier(const int total_paths) {
            N = static_cast<double>(total_paths);
            all_payoffs.reserve(total_paths);
        }

        // Called inside the batch loop
        void accumulate_batch_payoffs(const HostPayoffView &h_payoffs, const int batch_size) {
            for (int i = 0; i < batch_size; ++i) {
                const KT::Real p = static_cast<KT::Real>(h_payoffs(i));
                total_payoff_sum += p;
                total_squared_payoff_sum += (p * p);
                all_payoffs.push_back(p);
            }
        }

        // 2. Pure Computation Phase (No I/O)
        void compute_metrics(const double r, const double T_End) {
            if (all_payoffs.empty() || N <= 0) {
                throw std::runtime_error("Cannot compute metrics: No paths accumulated.");
            }

            // Core Statistics
            const double sample_mean = total_payoff_sum / N;
            const double discount_factor = std::exp(-r * T_End);

            metrics.option_price = sample_mean * discount_factor;

            const double sample_variance = (total_squared_payoff_sum / N) - (sample_mean * sample_mean);
            // Ensure variance doesn't go slightly negative due to floating point inaccuracies
            const double safe_variance = std::max(0.0, sample_variance);
            metrics.standard_error = std::sqrt(safe_variance / N) * discount_factor;

            // Percentile Sorting
            std::sort(all_payoffs.begin(), all_payoffs.end());

            metrics.p1     = all_payoffs[static_cast<size_t>(N * 0.01)] * discount_factor;
            metrics.median = all_payoffs[static_cast<size_t>(N * 0.50)] * discount_factor;
            metrics.p99    = all_payoffs[static_cast<size_t>(N * 0.99)] * discount_factor;

            // Mark as successfully computed
            metrics.is_computed = true;
        }

        // 3. Pure Reporting Phase (Uses the stored metrics)
        void print_report() const {
            if (!metrics.is_computed) {
                throw std::runtime_error("Attempted to print report before computing metrics. Call compute_metrics() first.");
            }

            std::cout << "\n\n  ========================================================\n";
            std::cout << "                     SIMULATION RESULTS\n";
            std::cout << "  ========================================================\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "   Estimated Option Price   :  " << metrics.option_price << "\n";
            std::cout << "   Statistical Std Error    :  " << metrics.standard_error << "\n";
            std::cout << "  --------------------------------------------------------\n";
            std::cout << "   [Discounted Payout Distribution Tail Metrics]\n";
            std::cout << "   1st Percentile (Lower)   :  " << metrics.p1 << "\n";
            std::cout << "   50th Percentile (Median) :  " << metrics.median << "\n";
            std::cout << "   99th Percentile (Upper)  :  " << metrics.p99 << "\n";
            std::cout << "  ========================================================\n" << std::endl;
        }

        // 4. Getter so other parts of the program can use the raw numbers
        [[nodiscard]] const ResultMetrics& get_metrics() const {
            if (!metrics.is_computed) {
                throw std::runtime_error("Metrics have not been computed yet.");
            }
            return metrics;
        }

    private:
        double N;
        double total_payoff_sum = 0.0;
        double total_squared_payoff_sum = 0.0;
        std::vector<KT::Real> all_payoffs;

        // The attribute holding the results
        ResultMetrics metrics;
    };

} // namespace KOps::Engine