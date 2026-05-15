// File (YAML) → Parser → Validator → Config → Solver
#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
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
                    << "    .... (" << KK::MathModelParams::r << ") :           " << heston.r << "\n"
                    << "    .... (" << KK::MathModelParams::q << ") :           " << heston.q << "\n"
                    << "    Mean reversion speed of the variance (" << KK::MathModelParams::k << ")     :       " << heston.k << "\n"
                    << "    Mean reversion level of the variance (" << KK::MathModelParams::theta << ") :       " << heston.theta << "\n"
                    << "    Volatility of the variance (" << KK::MathModelParams::sigma << ")           :       " << heston.sigma << "\n"
                    << "    Correlation between Brownian motions (" << KK::MathModelParams::rho << ")   :         " << heston.rho << "\n";
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
        KI::NumScheme num_scheme = KI::NumScheme::Euler;

        void validate() const {
        }

        void print() const {
            std::cout << "  [" << KK::Numerics << "]\n"
                    << "    Numerical Scheme:  " <<  enum_to_string(num_scheme) << "\n";
        }


    };

    // Substructure: Time Stepping
    struct TimeConfig {
        Real t_end = 1.0;
        Real dt = 0.4;

        void validate() const {
            if (t_end <= 0.0) throw std::runtime_error(config_err_msg(KK::Time, KK::TimeParams::T_End, "must be positive."));
            if (dt <= 0.0) throw std::runtime_error(config_err_msg(KK::Time, KK::TimeParams::DT, "must be positive."));
            if (dt > t_end) throw std::runtime_error(config_err_msg(KK::Time, KK::TimeParams::DT, "must be smallet that total time."));
        }

        void print() const {
            std::cout << "  [" << KK::Time << "]\n"
                    << "    End Time (" << KK::TimeParams::T_End << ") :      " << t_end << " s\n"
                    << "    Time Step (" << KK::TimeParams::DT << "):         " << dt << "s\n";
        }
    };

    // Substructure: MonteCarlo
    struct MCConfig {
        int N_Paths = 1000;

        void validate() const {
            if (N_Paths < 1) throw std::runtime_error(config_err_msg(KK::MC, KK::MCParams::N_Realizations, "must be a positive integer."));
        }

        void print() const {
            std::cout << "  [" << KK::MC << "]\n"
                    << "    Number of Realizations :      " << N_Paths << " s\n";
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
