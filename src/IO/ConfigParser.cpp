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

#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include "../../include/IO/InputParser.hpp"
#include "../../include/core/config/ConfigFileKeys.hpp"
#include "../../include/core/config/ConfigFileKeysEnumMaps.hpp"


namespace quantkos::Config {
    // Short alias for cleaner code within this file
    namespace K = quantkos::Keys;
    namespace KI = quantkos::Implemented;

    // Helper to cast string_view to string for yaml-cpp (older versions compatibility)
    inline std::string str(std::string_view sv) {
        return std::string(sv);
    }

    UInputs Parser::parse(const std::string &file) {
        const auto start_time = std::chrono::steady_clock::now();

        std::cout << "\n>>> [ Parser ] Start Parsing User Inputs...\n\n";

        // ---------------------------------------------------------
        // 1. Sanity Check
        // ---------------------------------------------------------
        if (!std::filesystem::exists(file)) {
            throw std::runtime_error("Config Error: File not found -> " + file);
        }

        // ---------------------------------------------------------
        // 2. Load YAML
        // ---------------------------------------------------------
        YAML::Node root;
        try {
            root = YAML::LoadFile(file);
        } catch (const YAML::ParserException &e) {
            throw std::runtime_error("Config Error: Invalid YAML syntax -> " + std::string(e.what()));
        }

        UInputs conf;

        // ---------------------------------------------------------
        // 3. Market Section
        // ---------------------------------------------------------
        if (root[str(KK::Market)]) {
            const auto &node = root[str(KK::Market)];

            if (node[str(KK::MarketParams::Price)])
                conf.market.S0 = node[str(KK::MarketParams::Price)].as<Real>();
            if (node[str(KK::MarketParams::Variance)])
                conf.market.v0 = node[str(KK::MarketParams::Variance)].as<Real>();
            if (node[str(KK::MarketParams::r)])
                conf.market.r = node[str(KK::MarketParams::r)].as<Real>();
            if (node[str(KK::MarketParams::q)])
                conf.market.q = node[str(KK::MarketParams::q)].as<Real>();
        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::Market) + "' missing.");
        }


        // ---------------------------------------------------------
        // 4. Option Section
        // ---------------------------------------------------------
        if (root[str(KK::Options)]) {
            const auto &node = root[str(KK::Options)];

            if (node[str(KK::OptionsParams::OptionType)]) {
                conf.options.opt_type = KI::string_to_enum<KI::OptType>(
                    node[str(KK::OptionsParams::OptionType)].as<std::string>(),
                    str(KK::Options) + "." + str(KK::OptionsParams::OptionType));
            }

            if (node[str(KK::OptionsParams::OptionRight)]) {
                conf.options.opt_right = KI::string_to_enum<KI::OptRight>(
                    node[str(KK::OptionsParams::OptionRight)].as<std::string>(),
                    str(KK::Options) + "." + str(KK::OptionsParams::OptionRight));
            }

            if (node[str(KK::OptionsParams::StrikePrice)]) {
                conf.options.StrikePrice = node[str(KK::OptionsParams::StrikePrice)].as<Real>();
            }
            if (node[str(KK::OptionsParams::BarrierPrice)]) {
                conf.options.BarrierPrice = node[str(KK::OptionsParams::BarrierPrice)].as<Real>();
            }
        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::Options) + "' missing.");
        }


        // ---------------------------------------------------------
        // 5. Model Section (Heston / Bates)
        // ---------------------------------------------------------
        if (root[str(KK::Model)]) {
            const auto &node = root[str(KK::Model)];

            // Check for Heston Sub-block
            if (node[str(KK::MathModelParams::IDHestonBlock)]) {
                const auto &h = node[str(KK::MathModelParams::IDHestonBlock)];
                if (h[str(KK::MathModelParams::k)])
                    conf.model.heston.k = h[str(KK::MathModelParams::k)].as<Real>();
                if (h[str(KK::MathModelParams::theta)])
                    conf.model.heston.theta = h[str(KK::MathModelParams::theta)].as<Real>();
                if (h[str(KK::MathModelParams::sigma)])
                    conf.model.heston.sigma = h[str(KK::MathModelParams::sigma)].as<Real>();
                if (h[str(KK::MathModelParams::rho)])
                    conf.model.heston.rho = h[str(KK::MathModelParams::rho)].as<Real>();
            }

            // Check for Bates Sub-block (Placeholder for future)
            if (node[str(KK::MathModelParams::IDBatesBlock)]) {
                // Implementation for Bates parameters would go here
                const auto &b = node[str(KK::MathModelParams::IDBatesBlock)];

                if (b[str(KK::MathModelParams::k)])
                    conf.model.heston.k = b[str(KK::MathModelParams::k)].as<Real>();
                if (b[str(KK::MathModelParams::theta)])
                    conf.model.heston.theta = b[str(KK::MathModelParams::theta)].as<Real>();
                if (b[str(KK::MathModelParams::sigma)])
                    conf.model.heston.sigma = b[str(KK::MathModelParams::sigma)].as<Real>();
                if (b[str(KK::MathModelParams::rho)])
                    conf.model.heston.rho = b[str(KK::MathModelParams::rho)].as<Real>();
            }
        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::Model) + "' missing.");
        }


        // ---------------------------------------------------------
        // 6. Numerics Section
        // ---------------------------------------------------------
        if (root[str(KK::Numerics)]) {
            const auto &node = root[str(KK::Numerics)];

            if (node[str(KK::NumSchemeParams::Scheme)]) {
                conf.scheme.id_scheme = KI::string_to_enum<KI::NumScheme>(
                    node[str(KK::NumSchemeParams::Scheme)].as<std::string>(),
                    str(KK::Numerics) + "." + str(KK::NumSchemeParams::Scheme)
                );
            }
        }

        // ---------------------------------------------------------
        // 7. Time Section
        // ---------------------------------------------------------
        if (root[str(KK::Time)]) {
            const auto &node = root[str(KK::Time)];

            if (node[str(KK::TimeParams::T_End)])
                conf.time.t_end = node[str(KK::TimeParams::T_End)].as<Real>();

            if (node[str(KK::TimeParams::Inp_DT)])
                conf.time.inp_dt = node[str(KK::TimeParams::Inp_DT)].as<Real>();
        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::Time) + "' missing.");
        }

        // ---------------------------------------------------------
        // 8. Monte Carlo Section
        // ---------------------------------------------------------
        if (root[str(KK::MC)]) {
            const auto &node = root[str(KK::MC)];
            // Normalization
            if (node[str(KK::MCParams::normalize_prices)])
                try {
                    conf.mc.normalize_prices = node[str(KK::MCParams::normalize_prices)].as<bool>();
                } catch (...) {
                    throw std::runtime_error(config_err_msg(
                        KK::MC,
                        KK::MCParams::normalize_prices,
                        "must be a boolean value (true/false, yes/no, 1/0)."
                    ));
                }
            // MC details
            if (node[str(KK::MCParams::N_Realizations)])
                conf.mc.N_Paths = node[str(KK::MCParams::N_Realizations)].as<int>();
            if (node[str(KK::MCParams::Batch_Size)])
                conf.mc.batch_size = node[str(KK::MCParams::Batch_Size)].as<int>();
            if (node[str(KK::MCParams::RNG_Seed)])
                conf.mc.rng_seed = node[str(KK::MCParams::RNG_Seed)].as<u_int64_t>();
            // Memory constraints
            if (node[str(KK::MCParams::Max_VRAM_MB)])
                conf.mc.Max_VRAM_MB = node[str(KK::MCParams::Max_VRAM_MB)].as<long long>();
            if (node[str(KK::MCParams::Max_CPU_RAM_MB)])
                conf.mc.Max_CPU_RAM_MB = node[str(KK::MCParams::Max_CPU_RAM_MB)].as<long long>();
            // Payoffs
            if (node[str(KK::MCParams::analyze_risk_neutral_payoff_distribution)])
                conf.mc.analyze_risk_neutral_payoff_distribution = node[str(
                    KK::MCParams::analyze_risk_neutral_payoff_distribution)].as<bool>();
            // Greeks
            if (node[str(KK::MCParams::compute_delta_et_gamma)])
                conf.mc.compute_delta_et_gamma = node[str(KK::MCParams::compute_delta_et_gamma)].as<bool>();
            if (node[str(KK::MCParams::spot_price_relative_bump_size)])
                conf.mc.spot_price_relative_bump_size = node[str(KK::MCParams::spot_price_relative_bump_size)].as<double>();
            if (node[str(KK::MCParams::compute_vega_et_vomma)])
                conf.mc.compute_vega_et_vomma = node[str(KK::MCParams::compute_vega_et_vomma)].as<bool>();
            if (node[str(KK::MCParams::compute_vanna)])
                conf.mc.compute_vanna = node[str(KK::MCParams::compute_vanna)].as<bool>();
            if (node[str(KK::MCParams::volatility_absolute_bump_size)])
                conf.mc.volatility_absolute_bump_size = node[str(KK::MCParams::volatility_absolute_bump_size)].as<double>();
            if (node[str(KK::MCParams::compute_rho)])
                conf.mc.compute_rho = node[str(KK::MCParams::compute_rho)].as<bool>();
            if (node[str(KK::MCParams::risk_free_rate_absolute_bump_size)])
                conf.mc.risk_free_rate_absolute_bump_size = node[str(KK::MCParams::risk_free_rate_absolute_bump_size)].as<double>();
            if (node[str(KK::MCParams::compute_theta)])
                conf.mc.compute_theta = node[str(KK::MCParams::compute_theta)].as<bool>();
            if (node[str(KK::MCParams::time_absolute_bump_size)])
                conf.mc.time_absolute_bump_size = node[str(KK::MCParams::time_absolute_bump_size)].as<double>();

        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::MC) + "' missing.");
        }

        // ---------------------------------------------------------
        // 9. Output Section
        // ---------------------------------------------------------
        if (root[str(KK::Output)]) {
            const auto &node = root[str(KK::Output)];

            if (node[str(KK::OutParams::Verbosity)]) {
                conf.output.verbosity = KI::string_to_enum<KI::VerbosityLevel>(
                    node[str(KK::OutParams::Verbosity)].as<std::string>(),
                    str(KK::OutParams::Verbosity)
                    );
            }

            if (node[str(KK::OutParams::out_dir)])
                conf.output.out_dir = node[str(KK::OutParams::out_dir)].as<std::string>();

            if (node[str(KK::OutParams::Name_Paths_Out_File)])
                conf.output.filename_paths_out = node[str(KK::OutParams::Name_Paths_Out_File)].as<std::string>();

            if (node[str(KK::OutParams::Name_Log_File)])
                conf.output.filename_log = node[str(KK::OutParams::Name_Log_File)].as<std::string>();

            if (node[str(KK::OutParams::Format)]) {
                conf.output.format = KI::string_to_enum<KI::IOFormat>(
                    node[str(KK::OutParams::Format)].as<std::string>(),
                    str(KK::Output) + "." + str(KK::OutParams::Format)
                );
            }
        }

        // ---------------------------------------------------------
        // 10. Final Validation
        // ---------------------------------------------------------
        conf.validate();
        conf.print_summary();

        // ---------------------------------------------------------
        // report timing
        // ---------------------------------------------------------
        const auto end_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "  [ Parser ] Time taken to parse the input  : "<< elapsed.count() <<"s \n";


        return conf;
    }
} // namespace Labes
