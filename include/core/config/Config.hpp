// File (YAML) → Parser → Validator → Config → Solver
#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <filesystem>

#include "../Typedefs.hpp"
#include "ConfigEnums.hpp"
#include "ConfigKeys.hpp"
#include "ConfigKeysEnumMaps.hpp"


namespace KOps::Config {
    // Internal helper to make error messages clean
    inline std::string config_err_msg(std::string_view block, std::string_view key, const std::string &msg) {
        return "[" + std::string(block) + "." + std::string(key) + "] " + msg;
    }

    // Alias for easier access to the keys
    namespace KI = KOps::Implemented;
    namespace KK = KOps::Keys;
    using Real = KOps::Types::Real;

    // Substructure: Options types and tigh
    struct OptionsConfig {
        KI::OptType opt_type = KI::OptType::European;
        KI::OptRight opt_right = KI::OptRight::Call;
        Real StrikePrice = 100.; //
        Real BarrierPrice = 100.; //

        void validate(Real Spot_price) const {
            // Check if Strike price make sense (compared to the spot price)
            if (StrikePrice <= 0.0)
                throw std::runtime_error(
                    config_err_msg(KK::Options, KK::OptionsParams::StrikePrice, "must be positive."));

            Real min_meaningful_strike = Spot_price * static_cast<Real>(0.0001);
            if (StrikePrice < min_meaningful_strike) {
                throw std::runtime_error("Config Error: Strike price (K) is too close to zero. "
                    "Must be at least 1000 times smaller than the initial asset price (S0) to prevent numerical problems.");
            }


            Real max_meaningful_strike = Spot_price * static_cast<Real>(10000);
            if (StrikePrice > max_meaningful_strike) {
                throw std::runtime_error("Config Error: Strike price (K) is absurdly high (exceeds 10000x of S0). "
                    "This will likely result in zero-variance path generation and statistical breakdown.");
            }
        }

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


    // Substructure: Physics & Model
    struct MathModelConfig {
        KI::MathModel id_model = KI::MathModel::Heston;

        struct Heston {
            Real r = 1.;
            Real q = 1.;
            Real k = 1.;
            Real theta = 1.;
            Real sigma = 1.;
            Real rho = 1.;
        } heston;

        struct Bates {
            // To be done in the future ...
        } bates;

        void validate() const {
            // To be done in the future ...
        }

        void print(std::string_view indent = "") const {
            // Assemble the equations
            std::string str_math_model = "";
            if (id_model == KI::MathModel::Heston) {
                // Use string concatenation (+) and convert string_view to string smoothly
                std::string ind(indent);
                str_math_model =
                        ind + "\t\t\t dSt = (r - q)·St·dt + √(vt)·St·dW1,t\n" +
                        ind + "\t\t\t dvt = κ·(θ - vt)·dt + σ·√(vt)·dW2,t\n" +
                        ind + "\t\t\t where E[dW1,t·dW2,t] = ρ·dt\n";
            } else if (id_model == KI::MathModel::Bates) {
                str_math_model = "... ";
            }

            std::cout << indent << "  [" << KK::Model << "]\n"
                    << indent << "    ID Model       :   " << id_model << "\n"
                    << indent << "    --------------------------------------------------------\n"
                    << str_math_model
                    << indent << "    --------------------------------------------------------\n"
                    << indent << "    Parameters :\n"
                    << indent << "      Risk-free interest rate (" << KK::MathModelParams::r <<
                    ")                  :   " << heston.r << "\n"
                    << indent << "      Continuous dividend yield (" << KK::MathModelParams::q <<
                    ")                :   " << heston.q << "\n"
                    << indent << "      Mean reversion speed of the variance (" << KK::MathModelParams::k <<
                    ")     :   " << heston.k << "\n"
                    << indent << "      Mean reversion level of the variance (" << KK::MathModelParams::theta <<
                    ") :   " << heston.theta << "\n"
                    << indent << "      Volatility of the variance (" << KK::MathModelParams::sigma <<
                    ")           :   " << heston.sigma << "\n"
                    << indent << "      Correlation between price and variance Brownian motions (" <<
                    KK::MathModelParams::rho << ")   :   " << heston.rho << "\n";
        }
    };


    // Substructure: Initial Conditions
    struct InitConfig {
        Real S0 = 100.0;
        Real v0 = 1.0;

        void validate() const {
            if (S0 <= 0.0)
                throw std::runtime_error(
                    config_err_msg(KK::Init, KK::InitParams::Price, "must be positive."));
            if (v0 < 0.0)
                throw std::runtime_error(
                    config_err_msg(KK::Init, KK::InitParams::Variance, "must be positive."));
        }

        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Init << "]\n"
                    << indent << "    Initial Price (S0)     :    " << S0 << "\n"
                    << indent << "    Initial Variance (v0)  :    " << v0 << "\n";
        }
    };


    // Substructure: Numerical Scheme
    struct NumSchemeConfig {
        KI::NumScheme id_scheme = KI::NumScheme::Euler;

        void validate() const {
        }

        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Numerics << "]\n"
                    << indent << "    Numerical Scheme    :  " << enum_to_string(id_scheme) << "\n";
        }
    };

    // Substructure: Time Stepping
    struct TimeConfig {
        Real t_end = 1.0;
        Real inp_dt = 0.2;
        Real dt;
        int N_time_steps;

        void validate() {
            // Check inputs
            if (t_end <= 0.0)
                throw std::runtime_error(
                    config_err_msg(KK::Time, KK::TimeParams::T_End, "must be positive."));
            if (inp_dt <= 0.0)
                throw std::runtime_error(
                    config_err_msg(KK::Time, KK::TimeParams::DT, "must be positive."));
            if (inp_dt > t_end)
                throw std::runtime_error(
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

        void print(std::string_view indent = "") const {
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

    // Substructure: MonteCarlo
    struct MCConfig {
        int N_Paths = 1000;
        int batch_size = 0; // 0 means autotune
        uint64_t rng_seed = 184467440737095ULL;
        long long Max_VRAM_MB = 256;
        long long Max_CPU_RAM_MB = 4000;


        void validate() const {
            if (N_Paths < 1)
                throw std::runtime_error(
                    config_err_msg(KK::MC, KK::MCParams::N_Realizations, "Must be a positive integer."));
            if (batch_size < -1)
                throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::Batch_Size,
                                                        "Must be a positive integer if you want to impose it, -1 if you want it equal to the max number of paths, "
                                                        "and 0 if you want to let the machine handle this."));
            if (batch_size > N_Paths)
                throw std::runtime_error(config_err_msg(
                    KK::MC, KK::MCParams::Batch_Size,
                    "The batch size must be smaller that the notal number of realizations."));
            if (rng_seed == 0) {
                throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::RNG_Seed,
                                                        "The random number seed cannot be 0."));
            }
            if (rng_seed >= 18446744073709551615ULL) {
                config_err_msg(KK::MC, KK::MCParams::RNG_Seed,
                               "[Warning] Seed is at maximum uint64 limit. Did you pass a negative number?");
            }
            if (Max_VRAM_MB <= 0)
                throw std::runtime_error(
                    config_err_msg(KK::MC, KK::MCParams::Max_VRAM_MB, "must be a positive integer."));
            if (Max_CPU_RAM_MB <= 0)
                throw std::runtime_error(
                    config_err_msg(KK::MC, KK::MCParams::Max_CPU_RAM_MB, "must be a positive integer."));
        }

        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::MC << "]\n"
                    << indent << "    Number of Realizations         :    " << N_Paths << "\n"
                    << indent << "    Batch Size required            :    " << batch_size << "\n"
                    << indent << "    Seed Random Number Generator   :    " << rng_seed << "\n"
                    << indent << "    User VRAM Limit (MB)           :    " << Max_VRAM_MB << "\n"
                    << indent << "    User CPU RAM Limit (MB)        :    " << Max_CPU_RAM_MB << "\n";
        }
    };


    // Substructure: IO & Diagnostics
    struct OutputConfig {
        std::string out_dir = "outputs";
        std::string filename_paths_out = "KOptions.paths";
        std::string filename_log = "KOptions.log";
        KI::IOFormat format = KI::IOFormat::TXT;

        void print(std::string_view indent = "") const {
            std::cout << indent << "  [" << KK::Output << "]\n"
                    << indent << "    Output Directory         :     " << out_dir << "\n"
                    << indent << "    Format Output Files      :     " << enum_to_string(format) << "\n"
                    << indent << "    Name Output File Paths   :     " << filename_paths_out << "\n"
                    << indent << "    Name Log File            :     " << filename_log << "\n";
        }

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
                    throw std::runtime_error(
                        "[Configuration Error] The assigned output directory '" + out_dir +
                        "' could not be prepared or accessed.\nDetails: " + e.what()
                    );
                }
            }
        }
    };

    // Master Configuration (information container)
    struct UInputs {
        OptionsConfig options;
        MathModelConfig model;
        InitConfig init;
        NumSchemeConfig scheme;
        TimeConfig time;
        MCConfig mc;
        OutputConfig output;

        void validate() {
            options.validate(init.S0);
            model.validate();
            init.validate();
            scheme.validate();
            time.validate();
            mc.validate();
            output.validate();
            std::cout << ">>> Input configuration validated successfully.\n" << std::endl;
        }

        void print_summary() const {
            constexpr std::string_view indent = "    ";

            std::cout << "\n" << indent << "========================================================\n";
            std::cout << indent << "                      MC CONFIGURATION                    \n";
            std::cout << indent << "========================================================\n";
            options.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            model.print(indent);
            std::cout << indent << "--------------------------------------------------------\n";
            init.print(indent);
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
    };
}
