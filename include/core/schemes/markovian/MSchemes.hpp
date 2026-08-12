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

#include <Kokkos_Core.hpp>
#include "../../config/Config.hpp"
#include "../../config/ConfigFileEnums.hpp"
#include "core/schemes/RandNGenerator.hpp"

namespace quantkos::Engine {
    namespace KI = quantkos::Implemented;
    namespace KC = quantkos::Config;
    namespace KT = quantkos::Types;
    namespace KE = quantkos::Engine;

    /**
     * @brief Output data structure returned by all SDE integration step functions.
     * * Encapsulates the current state of the SDE (Asset Price and Variance).
     * @note Designed to be trivially copyable, good for GPU registers.
     */
    struct SDEState {
        KT::Real S; ///< The underlying asset spot price ($S_t$).
        KT::Real v; ///< The stochastic variance ($v_t$).
    };


    // Master template blueprint (fall back if no specialization is available), it is trivially copiable struct!
    /**
     * @brief Master template blueprint for Stochastic Differential Equation (SDE) solvers.
     * * This unspecialized base struct serves as a compile-time fallback and safety guard.
     * @throw std::runtime_error If a user requests a MathModel/NumScheme combination that has not been explicitly implemented
     * (e.g., Bates + UnknownScheme).
     * * @tparam ModelPolicy The mathematical model to consider (e.g., Heston, Bates, etc.).
     * @tparam SchemePolicy The numerical discretization scheme to use (e.g., Euler, Milstein, etc.).
     */
    template<KI::MathModel ModelPolicy, KI::NumScheme SchemePolicy>
    struct SDEScheme {
        explicit SDEScheme(const KC::UInputs &config) {
            // This protects you at runtime if you accidentally try to run an unimplemented scheme
            throw std::runtime_error("SDESolver math kernel not yet implemented for this scheme combination!");
        }

        /**
         * @brief Dummy fallback function to allow un-specialized paths to compile successfully
         * @return Just return the input price and variance $(S_{n}, v_{n})$.
         */
        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION SDEState evolve_step(const KT::Real S_n,
                                                    const KT::Real v_n,
                                                    RNGeneratorType &rn_generator) const {
            return SDEState{.S = S_n, .v = v_n};
        }
    };

    // ========================================================================
    // SPECIALIZATION: Heston + Euler Maruyama
    // BOOK : The Heston model and its extensions in matlab and C#
    // Eq 7.8 -> v_n+1 = v_n + k(theta - v_n)dt + sigma * sqrt(v_n) * sqrt(dt) * Z_v
    // Eq 7.12 -> S_n+1 = S_n * exp( (r - q - v_n+1 * 1/2) * dt + sqrt(v_n) sqrt(dt) * Z_S)
    // ========================================================================
    /**
     * @brief Euler-Maruyama discretization scheme for the Heston Stochastic Volatility Model.
     * * **Source Literature**: Fabrice D. Rouah, *The Heston Model and its Extensions in Matlab and C#*.
     * * Implements the standard explicit Euler-Maruyama temporal integration.
     * - **Strong Convergence Order**: 0.5
     * - **Weak Convergence Order**: 1.0
     * * ### Mathematical Formulation:
     * - **Variance (Eq 7.8)**: $v_{n+1} = \max\left(0, v_n + \kappa(\theta - v_n)\Delta t + \sigma \sqrt{v_n \Delta t} Z_v\right)$
     * - **Asset Price (Eq 7.12)**: $S_{n+1} = S_n \exp\left( (r - q - \frac{1}{2}v_n)\Delta t + \sqrt{v_n \Delta t} Z_S \right)$
     * * @note Precomputes invariant scalar factors on the CPU Host during initialization to
     * prevent redundant multiplications inside the GPU execution threads
     * (This is ok since the registers pressure is not too high).
     */
    template<>
    struct SDEScheme<KI::MathModel::Heston, KI::NumScheme::Euler> {
        // Weak Convergence 1., Strong convergence 0.5
        // Precomputed constant scalar invariants (Computed once on CPU Host)
        KT::Real one_minus_k_dt; ///< $1 - \kappa \Delta t$
        KT::Real k_theta_dt; ///< $\kappa \theta \Delta t$
        KT::Real sigma_sqrt_dt; ///< $\sigma \sqrt{\Delta t}$
        KT::Real r_minus_q_dt; ///< $(r - q) \Delta t$
        KT::Real half_dt; ///< $0.5 \Delta t$
        KT::Real sqrt_dt; ///< $\sqrt{\Delta t}$
        KT::Real rho; ///< Correlation $\rho$
        KT::Real rho_complement; ///< Cholesky factor $\sqrt{1 - \rho^2}$

        // Constructor
        /** @brief Precomputes invariant time-step terms on the CPU Host. */
        explicit SDEScheme(const KC::UInputs &config) {
            const KT::Real r = config.market.r;
            const KT::Real q = config.market.q;
            const KT::Real k = config.model.heston.k;
            const KT::Real theta = config.model.heston.theta;
            const KT::Real sigma = config.model.heston.sigma;
            rho = config.model.heston.rho;
            const KT::Real dt = config.time.dt;

            // Compute roots once on the host so the GPU doesn't have to
            sqrt_dt = Kokkos::sqrt(dt);
            rho_complement = Kokkos::sqrt(KT::real_one - rho * rho);

            // PreCompute variance factors
            one_minus_k_dt = KT::real_one - k * dt;
            k_theta_dt = k * theta * dt;
            sigma_sqrt_dt = sigma * sqrt_dt;

            // PreCompute price factors
            r_minus_q_dt = (r - q) * dt;
            half_dt = KT::real_05 * dt; // Avoid division on GPU since it is a slow operation
        }

        /**
         * @brief Evolves the Heston process forward by one discrete time step $\Delta t$.
         * @param S_n Asset price at time $t_n$.
         * @param v_n Variance at time $t_n$.
         * @param local_rn_generator Kokkos Thread-local normal random number generator.
         * @return The updated state $(S_{n+1}, v_{n+1})$.
         */
        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION SDEState evolve_step(const KT::Real S_n,
                                                    const KT::Real v_n,
                                                    RNGeneratorType &local_rn_generator) const {
            // Get Random Normal Variables
            //const KT::Real Z_1 = static_cast<KT::Real>(local_rn_generator.normal());
            //const KT::Real Z_2 = static_cast<KT::Real>(local_rn_generator.normal());
            const auto [Z_1, Z_2] = NormalPair<RNGeneratorType>{}(local_rn_generator);

            // Cholesky Decomposition for correlated Brownian Motion
            const KT::Real Z_v = Z_1;
            const KT::Real Z_S = rho * Z_1 + rho_complement * Z_2;

            // EVOLVE ASSET (NOTE: MUST USE OLD VARIANCE TO RESPECT ITO INTEGRAL, REQUIRING WITH THE VAL AT BEGINNIG OF TIME SETP)
            // Mathematically: (r - q)dt - (v_n * 0.5 * dt) + (sqrt(v_n) * sqrt(dt)) * Z_S
            const KT::Real sqrt_v_n = Kokkos::sqrt(v_n);
            const KT::Real exponent = r_minus_q_dt - (v_n * half_dt) + (sqrt_v_n * sqrt_dt) * Z_S;
            const KT::Real S_np1 = S_n * Kokkos::exp(exponent); // Special Function Unit (SFU) Call (slow)

            // EVOLVE & TRUNCATE VARIANCE (Fused Multiply-Add (FMA) optimized variance step)
            KT::Real v_np1 = v_n * one_minus_k_dt + k_theta_dt + (sigma_sqrt_dt * sqrt_v_n) * Z_v;
            v_np1 = Kokkos::fmax(v_np1, KT::real_zero);

            return SDEState{.S = S_np1, .v = v_np1};
        }
    };


    // ========================================================================
    // SPECIALIZATION: Heston + Milstein Implicit
    // BOOK : The Heston model and its extensions in matlab and C#
    // Eq 7.8 -> To update price
    // Eq 7.26 -> To update variance
    // ========================================================================
    /**
     * @brief Implicit Milstein discretization scheme for the Heston Model.
     * * **Source Literature**: Fabrice D. Rouah, *The Heston Model and its Extensions in Matlab and C#*.
     * * Enhances the Euler approach by adding a second-order Itô-Taylor expansion term (the Milstein correction)
     * to the variance process, improving strong convergence. Uses an *implicit* algebraic formulation
     * to naturally prevent the variance from dropping below zero as frequently.
     * - **Strong Convergence Order**: 1.0
     * - **Weak Convergence Order**: 1.0
     * * ### Mathematical Formulation:
     * - **Variance (Eq 7.26)**: $v_{n+1} = \frac{v_n + \kappa\theta\Delta t + \sigma\sqrt{v_n\Delta t}Z_v + \frac{1}{4}\sigma^2\Delta t(Z_v^2 - 1)}{1 + \kappa\Delta t}$
     * - **Asset Price (Eq 7.8)**: Uses the exact same exponential Euler formulation as above.
     */
    template<>
    struct SDEScheme<KI::MathModel::Heston, KI::NumScheme::ImplicitMilstein> {
        // Weak Convergence 1., Strong convergence 1.
        // Precomputed constant scalar invariants
        KT::Real r_minus_q_dt;
        KT::Real half_dt;
        KT::Real sqrt_dt;
        KT::Real rho;
        KT::Real rho_complement;

        // Milstein-specific invariants
        KT::Real k_theta_dt;
        KT::Real sigma_sqrt_dt;
        KT::Real quarter_sigma_sq_dt; ///< Milstein correction multiplier: $\frac{1}{4}\sigma^2\Delta t$
        KT::Real implicit_denominator_v; ///< Implicit denominator: $\frac{1}{1 + \kappa\Delta t}$

        /** @brief Precomputes Milstein invariants on the Host. */
        explicit SDEScheme(const KC::UInputs &config) {
            const KT::Real r = config.market.r;
            const KT::Real q = config.market.q;
            const KT::Real k = config.model.heston.k;
            const KT::Real theta = config.model.heston.theta;
            const KT::Real sigma = config.model.heston.sigma;
            rho = config.model.heston.rho;
            const KT::Real dt = config.time.dt;

            sqrt_dt = Kokkos::sqrt(dt);
            rho_complement = Kokkos::sqrt(KT::real_one - rho * rho);

            r_minus_q_dt = (r - q) * dt;
            half_dt = KT::real_05 * dt;

            // Precompute Milstein invariants
            k_theta_dt = k * theta * dt;
            sigma_sqrt_dt = sigma * sqrt_dt;
            quarter_sigma_sq_dt = KT::real_025 * sigma * sigma * dt;

            // Host precomputes denominator to eliminate GPU division
            implicit_denominator_v = KT::real_one / (KT::real_one + k * dt);
        }

        /**
         * @brief Evolves the Heston process forward by one discrete time step $\Delta t$.
         * @param S_n Asset price at time $t_n$.
         * @param v_n Variance at time $t_n$.
         * @param local_rn_generator Kokkos Thread-local normal random number generator.
         * @return The updated state  $(S_{n+1}, v_{n+1})$.
         */
        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION SDEState evolve_step(const KT::Real S_n,
                                                    const KT::Real v_n,
                                                    RNGeneratorType &local_rn_generator) const {
            // Get Random Normal Variables
            //const KT::Real Z_1 = static_cast<KT::Real>(local_rn_generator.normal());
            //const KT::Real Z_2 = static_cast<KT::Real>(local_rn_generator.normal());
            const auto [Z_1, Z_2] = NormalPair<RNGeneratorType>{}(local_rn_generator);

            // Cholesky Decomposition for correlated Brownian Motion
            const KT::Real Z_v = Z_1;
            const KT::Real Z_S = rho * Z_1 + rho_complement * Z_2;


            // EVOLVE EVOLUTION (Evaluated at t_n to respect Ito calculus)
            const KT::Real sqrt_v_n = Kokkos::sqrt(v_n);
            const KT::Real exponent = r_minus_q_dt - (v_n * half_dt) + (sqrt_v_n * sqrt_dt) * Z_S;
            const KT::Real S_np1 = S_n * Kokkos::exp(exponent);

            // EVOLVE & TRUNCATE VARIANCE VARIANCE EVOLVE: Equation 7.26 (Algebraic Implicit Milstein)
            const KT::Real milstein_correction = quarter_sigma_sq_dt * (Z_v * Z_v - KT::real_one);
            KT::Real v_np1 = implicit_denominator_v *
                             (v_n + k_theta_dt + (sigma_sqrt_dt * sqrt_v_n) * Z_v + milstein_correction);
            v_np1 = Kokkos::fmax(v_np1, KT::real_zero);
            return SDEState{.S = S_np1, .v = v_np1};
        }
    };

    // ========================================================================
    // SPECIALIZATION: Heston + AndersonQE
    // BOOK : The Heston model and its extensions in matlab and C#
    // Eq 7.48, 7.49, 7.50, 7.51 -> To update price
    // Eq 7.37 -> To update variance
    // ========================================================================
    /**
     * @brief Andersen's Quadratic-Exponential (QE) scheme for the Heston Model.
     * * **Source Literature**: Leif B. G. Andersen (2008), detailed in F. Rouah, *The Heston Model*.
     * * A state-of-the-art, nearly bias-free integration scheme. It solves the Feller condition
     * negativity problem by matching the moments of the exact non-central chi-squared distribution
     * of the variance using two distinct asymptotic regimes.
     * * ### The Two Regimes:
     * - **Regime 1 (Quadratic / High Variance)**: If $\psi \le 1.5$, variance is drawn from an approximated
     * non-central chi-squared distribution (Eq 7.37).
     * - **Regime 2 (Exponential / Low Variance)**: If $\psi > 1.5$, variance is drawn from an exponential
     * distribution with a probability mass at zero (Eq 7.37).
     * * Integrates the asset price using a Martingale-corrected method (Eq 7.48, 7.49, 7.50, 7.51).
    @note This is not optimal on gpu because it can trigger several wrap divergence and requires calling heavy mathematical function such erf
    */
    template<>
    struct SDEScheme<KI::MathModel::Heston, KI::NumScheme::AndersonQE> {
        // Precomputed constant scalar invariants
        KT::Real r_minus_q_dt;
        KT::Real half_dt;
        KT::Real sqrt_dt;
        KT::Real rho;
        KT::Real rho_complement;

        // QE-Specific eps security values
        KT::Real eps_param = KT::is_real_using_single_precision() ? 1e-6f : 1e-12;
        KT::Real eps_psi = KT::is_real_using_single_precision() ? 1e-6f : 1e-12;

        // QE-Specific Structural Constants
        KT::Real psi_c = KT::real_1p5; // 1.5 as in Anderson
        KT::Real gamma1 = KT::real_05; // Central Predictor-Corrector
        KT::Real gamma2 = KT::real_05;
        KT::Real inv_sqrt_2; // For the Error Function (Normal CDF)

        // QE-Specific invariants
        KT::Real exp_minus_k_dt;
        KT::Real theta_m_factor; // theta * (1 - e^{-k dt})
        KT::Real s2_factor_1; // (sigma^2 * e^{-k dt} / k) * (1 - e^{-k dt})
        KT::Real s2_factor_2; // (theta * sigma^2 / (2k)) * (1 - e^{-k dt})^2
        KT::Real K1, K2, K3, K4, A;

        /** @brief Precomputes analytical QE constants and Martingale corrections. */
        explicit SDEScheme(const KC::UInputs &config) {
            const KT::Real r = config.market.r;
            const KT::Real q = config.market.q;
            const KT::Real k = config.model.heston.k;
            const KT::Real theta = config.model.heston.theta;
            const KT::Real sigma = config.model.heston.sigma;
            rho = config.model.heston.rho;
            const KT::Real dt = config.time.dt;

            // Base invariants
            sqrt_dt = std::sqrt(dt);
            r_minus_q_dt = (r - q) * dt;
            half_dt = KT::real_05 * dt;

            // Anderson QE invariants
            inv_sqrt_2 = static_cast<KT::Real>(1.0 / std::sqrt(2.0));
            exp_minus_k_dt = std::exp(-k * dt);

            // Handle case k = 0
            if (k < eps_param) {
                // Catch case when there is not drift
                // Compute them with theorem de l'hopital
                theta_m_factor = KT::real_zero;
                s2_factor_1 = sigma * sigma * dt;
                s2_factor_2 = KT::real_zero;
            } else {
                const KT::Real one_minus_exp = KT::real_one - exp_minus_k_dt;
                theta_m_factor = theta * one_minus_exp; // in Eq 7.41, F. Rouah
                const KT::Real sig_sq = sigma * sigma; // in Eq 7.41, F. Rouah
                s2_factor_1 = (sig_sq * exp_minus_k_dt / k) * one_minus_exp; // in Eq, 7.41 F. Rouah
                s2_factor_2 = (theta * sig_sq / (KT::real_two * k)) * (one_minus_exp * one_minus_exp);
            }

            // Handle case sigma=0
            KT::Real rho_div_simga;
            if (sigma < eps_param) {
                // To handle Black Scholes degeneration
                rho_div_simga = KT::real_zero;
                rho = KT::real_zero; // If sigma is near zero, correlation is mathematically meaningless
            } else {
                rho_div_simga = rho / sigma; // after Eq 7.47, F. Rouah
            }
            const KT::Real one_m_rho2 = KT::real_one - rho * rho; // after Eq 7.47, F. Rouah
            const KT::Real K12_fact = (k * rho_div_simga - KT::real_05); // after Eq 7.47, F. Rouah
            K1 = dt * gamma1 * K12_fact - rho_div_simga; // after Eq 7.47, F. Rouah
            K2 = dt * gamma2 * K12_fact + rho_div_simga; // after Eq 7.47, F. Rouah
            K3 = dt * gamma1 * one_m_rho2; // after Eq 7.47, F. Rouah
            K4 = dt * gamma2 * one_m_rho2; // after Eq 7.47, F. Rouah
            // Martingale Constants precomputed for the GPU
            A = K2 + K4 * KT::real_05; // Terms for matingale correction K0
        }

        /**
         * @brief Evolves the Heston process forward by one discrete time step $\Delta t$.
         * @param S_n Asset price at time $t_n$.
         * @param v_n Variance at time $t_n$.
         * @param local_rn_generator Kokkos Thread-local normal random number generator.
         * @return The updated state tuple $(S_{n+1}, v_{n+1})$.
         */
        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION SDEState evolve_step(const KT::Real S_n,
                                                    const KT::Real v_n,
                                                    RNGeneratorType &local_rn_generator) const {
            // Generate noise
            //const KT::Real Z_V = static_cast<KT::Real>(local_rn_generator.normal());
            //const KT::Real Z_indep = static_cast<KT::Real>(local_rn_generator.normal());
            const auto [Z_V, Z_indep] = NormalPair<RNGeneratorType>{}(local_rn_generator);

            // Determine Distribution Shape (Psi)
            // (Compute Conditional Mean (m) and Variance (s^2) of V(t+dt))
            const KT::Real m = v_n * exp_minus_k_dt + theta_m_factor; // (Eq. 7.41, F. Rouah)
            const KT::Real s2 = v_n * s2_factor_1 + s2_factor_2; // (Eq. 7.41, F. Rouah)
            // const KT::Real psi = s2 / (m * m); // (After Eq. 7.42, F. Rouah)

            // ====================================================================
            // DIVERGENCE CONTROL BLOCK (for variance computation)
            // ====================================================================
            KT::Real v_np1;
            KT::Real M; // Terms for martingale correction used to compute K0
            if (m <= eps_param) {
                // REGIME 0a: Case when m=0 since vn=0 and theta=0
                v_np1 = KT::real_zero;
                M = KT::real_one;
            } else {
                const KT::Real psi = s2 / (m * m); // (After Eq. 7.42, F. Rouah)
                if (psi <= eps_psi) {
                    // REGIME 0b: Deterministic Limit (Black-Scholes Degeneration)
                    // NOte: this first if will not cause divergence because all threads will go here if using BS
                    // If psi is zero, variance is deterministic. Bypass the division-by-zero.
                    v_np1 = m;
                    M = KT::real_one; // The MGF of a deterministic constant is exactly 1.0
                } else if (psi <= psi_c) {
                    // REGIME 1: Quadratic (High Variance)
                    const KT::Real inv_psi = KT::real_one / psi;
                    const KT::Real b2 = KT::real_two * inv_psi - KT::real_one +
                                        Kokkos::sqrt(KT::real_two * inv_psi *
                                                     (KT::real_two * inv_psi - KT::real_one)
                                        ); // Eq. 7.42, F. Rouah
                    const KT::Real a = m / (KT::real_one + b2); // Eq. 7.42, F. Rouah
                    const KT::Real b_plus_Zv = Kokkos::sqrt(b2) + Z_V; // Eq. 7.37, F. Rouah
                    v_np1 = a * b_plus_Zv * b_plus_Zv; // Eq. 7.37, F. Rouah
                    // Term for Martingale correction
                    const KT::Real safe_A = Kokkos::fmin(A, (KT::real_05 / a) - eps_param);
                    const KT::Real term_Aa = KT::real_one - KT::real_two * safe_A * a;
                    M = Kokkos::exp((safe_A * b2 * a) / term_Aa) / Kokkos::sqrt(term_Aa); // Eq. 7.50, F. Rouah
                } else {
                    // REGIME 2: Exponential (Low Variance)
                    const KT::Real p = (psi - KT::real_one) / (psi + KT::real_one); // Eq. 7.43, F. Rouah
                    const KT::Real beta = (KT::real_one - p) / m; // Eq. 7.43, F. Rouah

                    // Recover the Uniform variable U_V from Z_V : U_V = CDF(Z_V).
                    const KT::Real U_V = KT::real_05 * (KT::real_one + Kokkos::erf(Z_V * inv_sqrt_2));
                    // Eq. 7.40, F. Rouah
                    // Update v
                    if (U_V <= p) {
                        v_np1 = KT::real_zero; // Eq. 7.40, F. Rouah
                    } else {
                        // Protect against log singularity at U_V == 1.0
                        const KT::Real safe_U = Kokkos::fmin(U_V, KT::real_one - eps_param);
                        v_np1 = (Kokkos::log((KT::real_one - p) / (KT::real_one - safe_U))) / beta;
                        // Eq. 7.40, F. Rouah
                    }
                    // Term for Martingale correction
                    const KT::Real safe_A = Kokkos::fmin(A, beta - eps_param);
                    M = p + (beta * (KT::real_one - p)) / (beta - safe_A); // Eq. 7.51, F. Rouah
                }
            }

            // ====================================================================
            // ASSET EVOLUTION (Predictor-Corrector Integration)
            // ====================================================================
            const KT::Real K0 = -Kokkos::log(M) - (K1 + K3 * KT::real_05) * v_n; // Eq. 7.49, F. Rouah (4Martingale)
            const KT::Real integrated_sigma = (K3 * v_n + K4 * v_np1); // Eq. 7.48, F. Rouah
            const KT::Real exponent = r_minus_q_dt + K0 +
                                      (K1 * v_n + K2 * v_np1) +
                                      (Kokkos::sqrt(integrated_sigma) * Z_indep); // Eq. 7.48, F. Rouah
            const KT::Real S_np1 = S_n * Kokkos::exp(exponent); // Eq. 7.48, F. Rouah

            return SDEState{.S = S_np1, .v = v_np1};
        }
    };


    // ========================================================================
    // SPECIALIZATION: Bates Model (Universal Wrapper via Composition)
    // Works automatically for any heston scheme
    // ========================================================================
    /**
     * @brief Universal Jump-Diffusion wrapper for the Bates model.
     * @note **Mathematical Design via Composition**: The Bates model is essentially the continuous Heston
     * volatility model combined with Merton-style discrete log-normal jumps.
     * Rather than rewriting the complex Euler/Milstein/QE integrations, this struct *wraps* a standard
     * Heston core object.
     * * ### The Martingale Compensator
     * Adding random price jumps creates an artificial arbitrage opportunity unless the baseline drift
     * is corrected. We calculate the expected percentage change caused by a single jump ($\kappa_J$)
     * and modify the continuous dividend yield:
     * $$ q_{\text{new}} = q_{\text{old}} + \lambda \cdot \kappa_J $$
     * By "spoofing" the underlying Heston configuration with this new $q_{\text{new}}$, the continuous
     * process drifts downward just enough to perfectly offset the average upward jump risk.
     * * ### Poisson Jump Draw
     * Uses Donald E. Knuth's algorithm to draw a random integer $N \sim \text{Poisson}(\lambda \Delta t)$
     * representing the number of jumps occurring inside the discrete step.
     * * @tparam SchemePolicy The underlying continuous integration technique (e.g., Eurle, AndersonQE, etc.).
     */
    template<KI::NumScheme SchemePolicy>
    struct SDEScheme<KI::MathModel::Bates, SchemePolicy> {
        // 1. Composition: The underlying Heston Engine
        SDEScheme<KI::MathModel::Heston, SchemePolicy> heston_core; ///< Encapsulated heston driver.

        // 2. Bates-Specific Invariants
        KT::Real lambda_dt; ///< $\lambda \Delta t$
        KT::Real exp_minus_lambda_dt; ///< $e^{-\lambda \Delta t}$
        KT::Real mu_J; ///< Mean log-jump magnitude
        KT::Real sigma_J; ///< std of the log-jump

        /** @brief Precomputes jump parameters and initializes the spoofed Heston core. */
        explicit SDEScheme(const KC::UInputs &config)
            : heston_core(create_spoofed_config_from_original(config)) {
            // Precompute jump parameters for the GPU
            lambda_dt = config.model.bates.lambda_J * config.time.dt;
            exp_minus_lambda_dt = std::exp(-lambda_dt);
            mu_J = config.model.bates.mu_J;
            sigma_J = config.model.bates.sigma_J;
        }

        /**
        * @brief Evolves the Heston model and then applies discrete Poisson jumps.
        */
        template<typename RNGeneratorType>
        KOKKOS_INLINE_FUNCTION SDEState evolve_step(const KT::Real S_n,
                                                    const KT::Real v_n,
                                                    RNGeneratorType &local_rn_generator) const {
            // 1. Evolve the continuous part using the spoofed Heston core
            auto [S_temp, v_next] = heston_core.evolve_step(S_n, v_n, local_rn_generator);

            // 2. Poisson Draw (Draw a random integer N from a Poisson distribution with mean λΔt)
            // Note : This represents how many times the stock jumps during this specific time step.
            // Note : The KRUNT algorithm is used (https://math.stackexchange.com/questions/3628801/proving-knuth-s-algorithm-for-generating-a-poisson-distribution)
            int N = -1; // Start at -1 since Krunt's loop overshoots by 1. It draws the poisson jump count
            KT::Real p = KT::real_one;
            do {
                // Multiply random fractions util accumulated value drops below the target threshold
                // The number of multiplications done maps to the exact number of jumps occurred inside the discrete interval
                N++;
                p *= get_uniform(local_rn_generator); //static_cast<KT::Real>(local_rn_generator.drand());
            } while (p >= exp_minus_lambda_dt);

            // 3. Draw the jump size and update price
            KT::Real S_next = S_temp;
            if (N > 0) {
                // Compute current jump size
                // Note : The total log-jump size is \sum_{i=1}^{N}(mu_J + sigma_J*Z_{J,I}) = N*mu_J + sqrt(N)*sigma_J*Z_{J}
                const auto real_N = static_cast<KT::Real>(N);
                const KT::Real Z = NormalPair<decltype(local_rn_generator), true>{}(local_rn_generator).Z1;
                const KT::Real aggregate_mu = real_N * mu_J;
                const KT::Real aggregate_sigma = Kokkos::sqrt(real_N) * sigma_J;
                const KT::Real jump_magnitude = Kokkos::exp(aggregate_mu + aggregate_sigma * Z);

                // Update Price : multiply the continuous stock price by the exponentiated jump sum.
                S_next *= jump_magnitude;
            }
            return SDEState{.S = S_next, .v = v_next};
        }

    private:
        // Helper function that runs exclusively on the Host during construction
        /**
         * @brief Host-side helper that add the Bates jump compensator into the Heston continuous drift.
         * @param orig_config The master user configuration.
         * @return A modified configuration object designed for continuous Heston core where the mertingale correction has been applied.
         */
        KOKKOS_INLINE_FUNCTION static KC::UInputs create_spoofed_config_from_original(const KC::UInputs &orig_config) {
            KC::UInputs spoofed_config = orig_config; // copy the input config

            // 1. C++ Object Slicing: Copies ALL base Heston fields (k, theta, sigma, rho) in one go
            spoofed_config.model.heston = static_cast<Config::MathModelConfig::Heston>(orig_config.model.bates);

            // 2. Calculate the Bates Martingale Compensator
            const KT::Real mu = orig_config.model.bates.mu_J;
            const KT::Real sig2 = orig_config.model.bates.sigma_J * orig_config.model.bates.sigma_J;
            const KT::Real kappa_J = std::exp(mu + KT::real_05 * sig2) - KT::real_one; // Theoretical expected jump mean

            // 3. Spoof the continuous dividend yield: q_new = q_old + (lambda * kappa_J)
            spoofed_config.market.q += (orig_config.model.bates.lambda_J * kappa_J);

            return spoofed_config;
        }
    };
}
