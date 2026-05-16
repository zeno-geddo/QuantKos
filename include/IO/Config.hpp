// File (YAML) → Parser → Validator → Config → Solver
#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "../core/Typedefs.hpp"
#include "../IO/ConfigEnums.hpp"
#include "../IO/ConfigKeys.hpp"
#include "../IO/ConfigKeysEnumMaps.hpp"


namespace KOps::Config {

    // Internal helper to make error messages clean
    inline std::string config_err_msg(std::string_view block, std::string_view key, const std::string &msg) {
        return "[" + std::string(block) + "." + std::string(key) + "] " + msg;
    }

    // Alias for easier access to the keys
    namespace KI = KOps::Implemented;
    namespace KK = KOps::Keys;
    using Real = KOps::Types::Real;

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

        void print() const {
            std::cout << "  [" << KK::Model << "]\n"
                    << "    ID Model                                                                    :       " << id_model << "\n"
                    << "    Parameters Heston Model :"
                    << "    .... (" << KK::MathModelParams::r << ")                                     :       " << heston.r << "\n"
                    << "    .... (" << KK::MathModelParams::r << ")                                     :       " << heston.r << "\n"
                    << "    .... (" << KK::MathModelParams::q << ")                                     :       " << heston.q << "\n"
                    << "    Mean reversion speed of the variance (" << KK::MathModelParams::k << ")     :       " << heston.k << "\n"
                    << "    Mean reversion level of the variance (" << KK::MathModelParams::theta << ") :       " << heston.theta << "\n"
                    << "    Volatility of the variance (" << KK::MathModelParams::sigma << ")           :       " << heston.sigma << "\n"
                    << "    Correlation between Brownian motions (" << KK::MathModelParams::rho << ")   :       " << heston.rho << "\n";
        }
    };


    // Substructure: Initial Conditions
    struct InitConfig {
        Real S0 = 100.0;
        Real v0 = 1.0;

        void validate() const {
            if (S0 <= 0.0) throw std::runtime_error(config_err_msg(KK::Init, KK::InitParams::Price, "must be positive."));
        }

        void print() const {
            std::cout << "  [" << KK::Init << "]\n"
                    << "    Initial Price (S0)     :     " << S0 << "\n"
                    << "    Initial Variance (v0)  :     " << v0 << "\n";
        }
    };


    // Substructure: Numerical Scheme
    struct NumSchemeConfig {
        KI::NumScheme id_scheme = KI::NumScheme::Euler;

        void validate() const {
        }

        void print() const {
            std::cout << "  [" << KK::Numerics << "]\n"
                    << "    Numerical Scheme:  " <<  enum_to_string(id_scheme) << "\n";
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
            if (t_end <= 0.0) throw std::runtime_error(config_err_msg(KK::Time, KK::TimeParams::T_End, "must be positive."));
            if (inp_dt <= 0.0) throw std::runtime_error(config_err_msg(KK::Time, KK::TimeParams::DT, "must be positive."));
            if (inp_dt > t_end) throw std::runtime_error(config_err_msg(KK::Time, KK::TimeParams::DT, "must be smaller that total time."));

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
                          << "across " <<N_time_steps << " uniform steps.\n" << std::endl;
            }

        }

        void print() const {
            std::cout << "  [" << KK::Time << "]\n"
                    << "    End Time (" << KK::TimeParams::T_End << ")\t:\t" << t_end << "\n"
                    << "    User Input Time Step (" << KK::TimeParams::Inp_DT << ")\t:\t" << inp_dt << "\n"
                    << "    Actual Time Step (" << KK::TimeParams::DT << ")\t:\t" << dt << "\n"
                    << "    Number Actual Time Steps (" << KK::TimeParams::N_TSteps << ")\t:\t" << N_time_steps << "\n";
        }
    };

    // Substructure: MonteCarlo
    struct MCConfig {
        int N_Paths = 1000;
        int batch_size = 0; // 0 means autotune
        long long Max_VRAM_MB = 256;
        long long Max_CPU_RAM_MB = 4000;


        void validate() const {
            if (N_Paths < 1) throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::N_Realizations, "Must be a positive integer."));
            if (batch_size < - 1) throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::Batch_Size,
                "Must be a positive integer if you want to impose it, -1 if you want it equal to the max number of paths, "
                "and 0 if you want to let the machine handle this."));
            if (batch_size > N_Paths) throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::Batch_Size, "The batch size must be smaller that the notal number of realizations."));
            if (Max_VRAM_MB <= 0) throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::Max_VRAM_MB, "must be a positive integer."));
            if (Max_CPU_RAM_MB <= 0) throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::Max_CPU_RAM_MB, "must be a positive integer."));
        }

        void print() const {
            std::cout << "  [" << KK::MC << "]\n"
                    << "    Number of Realizations :      " << N_Paths << "\n"
                    << "    Batch Size required :         " << N_Paths << "\n"
                    << "    User VRAM Limit :             " << Max_VRAM_MB << "\n"
                    << "    User CPU RAM Limit :          " << Max_CPU_RAM_MB << "\n";
        }
    };


    // Substructure: IO & Diagnostics
    struct OutputConfig {
        std::string filename_out = "KOptions.out";
        std::string filename_log = "KOptions.log";
        KI::IOFormat format = KI::IOFormat::TXT;

        void print() const {
            std::cout << "  [" << KK::Output << "]\n"
                    << "    Name Output File:     " << filename_out << "\n"
                    << "    Format Output File:   " << enum_to_string(format) << "\n"
                    << "    Name Log File:        " << filename_log << "\n";
        }
    };

    // Master Configuration
    struct UInputs {
        MathModelConfig model;
        InitConfig init;
        NumSchemeConfig scheme;
        TimeConfig time;
        MCConfig mc;
        OutputConfig output;

        void validate() const {
            model.validate();
            init.validate();
            scheme.validate();
            mc.validate();
            std::cout << ">>> Input configuration validated successfully. \n" << std::endl;
        }

        void print_summary() const {
            std::cout << "========================================================\n";
            std::cout << "          KOptions SIMULATION CONFIGURATION             \n";
            std::cout << "========================================================\n";
            model.print();
            std::cout << "--------------------------------------------------------\n";
            init.print();
            std::cout << "--------------------------------------------------------\n";
            scheme.print();
            std::cout << "--------------------------------------------------------\n";
            time.print();
            std::cout << "--------------------------------------------------------\n";
            mc.print();
            std::cout << "--------------------------------------------------------\n";
            output.print();
            std::cout << "========================================================\n" << std::endl;
        }
    };
}
