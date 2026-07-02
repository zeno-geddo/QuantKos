#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include "../../include/IO/InputParser.hpp"
#include "../../include/core/config/ConfigFileKeys.hpp"
#include "../../include/core/config/ConfigFileKeysEnumMaps.hpp"


namespace KOps::Config {
    // Short alias for cleaner code within this file
    namespace K = KOps::Keys;
    namespace KI = KOps::Implemented;

    // Helper to cast string_view to string for yaml-cpp (older versions compatibility)
    inline std::string str(std::string_view sv) {
        return std::string(sv);
    }

    UInputs Parser::parse(const std::string &file) {
        std::cout << "\n>>> Start Parsing User Inputs...\n" << std::endl;

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
        // 3. Option Section (Heston / Bates)
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
        // 3. Model Section (Heston / Bates)
        // ---------------------------------------------------------
        if (root[str(KK::Model)]) {
            const auto &node = root[str(KK::Model)];

            // Check for Heston Sub-block
            if (node[str(KK::MathModelParams::IDHestonBlock)]) {
                const auto &h = node[str(KK::MathModelParams::IDHestonBlock)];

                if (h[str(KK::MathModelParams::r)])
                    conf.model.heston.r = h[str(KK::MathModelParams::r)].as<Real>();
                if (h[str(KK::MathModelParams::q)])
                    conf.model.heston.q = h[str(KK::MathModelParams::q)].as<Real>();
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

                if (b[str(KK::MathModelParams::r)])
                    conf.model.heston.r = b[str(KK::MathModelParams::r)].as<Real>();
                if (b[str(KK::MathModelParams::q)])
                    conf.model.heston.q = b[str(KK::MathModelParams::q)].as<Real>();
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
        // 4. Initialization Section
        // ---------------------------------------------------------
        if (root[str(KK::Init)]) {
            const auto &node = root[str(KK::Init)];

            if (node[str(KK::InitParams::Price)])
                conf.init.S0 = node[str(KK::InitParams::Price)].as<Real>();

            if (node[str(KK::InitParams::Variance)])
                conf.init.v0 = node[str(KK::InitParams::Variance)].as<Real>();
        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::Init) + "' missing.");
        }

        // ---------------------------------------------------------
        // 5. Numerics Section
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
        // 6. Time Section
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
        // 7. Monte Carlo Section
        // ---------------------------------------------------------
        if (root[str(KK::MC)]) {
            const auto &node = root[str(KK::MC)];
            if (node[str(KK::MCParams::N_Realizations)])
                conf.mc.N_Paths = node[str(KK::MCParams::N_Realizations)].as<int>();
            if (node[str(KK::MCParams::Batch_Size)])
                conf.mc.batch_size = node[str(KK::MCParams::Batch_Size)].as<int>();
            if (node[str(KK::MCParams::RNG_Seed)])
                conf.mc.rng_seed = node[str(KK::MCParams::RNG_Seed)].as<u_int64_t>();
            if (node[str(KK::MCParams::Max_VRAM_MB)])
                conf.mc.Max_VRAM_MB = node[str(KK::MCParams::Max_VRAM_MB)].as<long long>();
            if (node[str(KK::MCParams::Max_CPU_RAM_MB)])
                conf.mc.Max_CPU_RAM_MB = node[str(KK::MCParams::Max_CPU_RAM_MB)].as<long long>();
        } else {
            throw std::runtime_error("Config Error: Mandatory block '" + str(KK::MC) + "' missing.");
        }

        // ---------------------------------------------------------
        // 9. Output Section
        // ---------------------------------------------------------
        if (root[str(KK::Output)]) {
            const auto &node = root[str(KK::Output)];

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

        return conf;
    }
} // namespace Labes
