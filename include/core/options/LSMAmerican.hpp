#pragma once

#include <Kokkos_Core.hpp>

#include "../Typedefs.hpp"
#include "../options/Payoff.hpp"

/**
 * @brief Namespace containing the core components for the Longstaff-Schwartz American pricing engine.
 * * Provides structural containers for regression matrices, custom polynomial basis sets,
 * and parallel reduction kernels to evaluate optimal early exercise boundaries.
 * @todo Should implement the control variate technique to improve the precision, see Pag. 208, eq. 7, the the book : Fabrice D. Rouah's *The Heston Model and its Extensions in Matlab and C#*.
 */
namespace KOps::Engine::LSM {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;

    /**
     * @brief Fixed-size container for the calculated ordinary least-squares regression coefficients.
     * * Represents the solution vector $C$ of the system:
     * $$M \cdot C = B$$
     * where $M = X^T X$ is the design covariance matrix, and $B = X^T Y$ is the projected cash flow vector.
     * * @tparam NBasis The number of basis functions (polynomial order + 1).
     */
    template<size_t NBasis = 4>
    struct Coeffs {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found
        KT::Real C[NBasis] = {0.0}; ///< Flat stack allocation storing the polynomial coefficients.

        /** @brief Provides mutable reference access to a coefficient by index. */
        KOKKOS_INLINE_FUNCTION
        KT::Real &operator[](size_t i) { return C[i]; }

        /** @brief Provides read-only constant reference access to a coefficient by index. */
        KOKKOS_INLINE_FUNCTION
        const KT::Real &operator[](size_t i) const { return C[i]; } // Allows reading: auto val = my_coeffs[0];
    };

    /**
     * @brief Represents the covariance matrix $M = X^T X$.
     * * Allocated entirely on the stack within execution threads to track cross-product combinations
     * of polynomial functions evaluated along asset price paths.
     * * @tparam NBasis Dimension of the square regression system (The number of basis functions (polynomial order + 1)).
     */
    template<size_t NBasis = 4>
    struct Matrix {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found
        // Components Matrix $M = (X^T X)$ (symmetric matrix)
        KT::Real M[NBasis][NBasis] = {}; ///< Fixed stack array representing the $N \times N$ matrix.

        /** @brief Elements access operator for mutable configurations. */
        KOKKOS_INLINE_FUNCTION
        KT::Real &operator()(size_t i, size_t j) { return M[i][j]; }

        /** @brief Elements access operator for read-only constant matrices. */
        KOKKOS_INLINE_FUNCTION
        const KT::Real &operator()(size_t i, size_t j) const { return M[i][j]; }

        /**
         * @brief In-place component-wise accumulation operator.
         * @note Uses loop unrolling directives to eliminate branch evaluation counters
         * inside parallel reductions.
         * * @param src Source matrix to add.
         * @return Reference to this updated matrix instance.
         */
        KOKKOS_INLINE_FUNCTION
        Matrix &operator+=(const Matrix &src) {
#pragma unroll
            for (size_t i = 0; i < NBasis; ++i) {
#pragma unroll
                for (size_t j = 0; j < NBasis; ++j) {
                    M[i][j] += src.M[i][j];
                }
            }
            return *this;
        }
    };

    /**
     * @brief Represents the target projection vector $B = X^T Y$.
     * * @tparam NBasis The number of basis functions (polynomial order + 1).
     */
    template<size_t NBasis = 4>
    struct Vector {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found
        //  Components Vector $B = (X^T Y)$
        KT::Real B[NBasis] = {}; ///< 1D Stack buffer tracking cross-product results.

        /** @brief Access operator for localized mutable parameters. */
        KOKKOS_INLINE_FUNCTION
        KT::Real &operator()(size_t i) { return B[i]; }

        /** @brief Access operator for read-only constant vectors. */
        KOKKOS_INLINE_FUNCTION
        const KT::Real &operator()(size_t i) const { return B[i]; }

        /**
         * @brief Component-wise accumulation operator.
         * @param src Input vector segment.
         * @return Reference to this updated instance.
         */
        KOKKOS_INLINE_FUNCTION
        Vector &operator+=(const Vector &src) {
#pragma unroll
            for (size_t i = 0; i < NBasis; ++i) {
                B[i] += src.B[i];
            }
            return *this;
        }
    };

    /**
     * @brief Aggregator structure used during Kokkos parallel reductions.
     * * Combines the matrix $M$, vector $B$, and an In-The-Money (ITM) pathway sample counter
     * into a single structure. This allows a single reduction pass over the path matrix.
     * * @tparam NBasis The number of basis functions (polynomial order + 1).
     */
    template<size_t NBasis = 4>
    struct RegressionSums {
        KT::Real count{0.0}; ///< Number of paths that successfully qualified as In-The-Money.
        Matrix<NBasis> M; ///< Aggregated equation matrix.
        Vector<NBasis> B; ///< Aggregated projected cash flow vector.

        /**
         * @brief Accumulation operator for reduction.
         * * Invoked automatically by Kokkos backend threads across warp boundaries
         * during parallel reduction cycles.
         */
        KOKKOS_INLINE_FUNCTION
        RegressionSums &operator+=(const RegressionSums &src) {
            count += src.count;
            M += src.M;
            B += src.B;
            return *this;
        }
    };

    /**
     * @brief Compile-time evaluator for standardized orthogonal Laguerre polynomials.
     * @note Generates a localized basis array from a normalized asset price coordinate $X = S_t / K$.
     * @note Uses `if constexpr` branch to dynamically unroll statements according to the requested
     * system size, preventing branch divergence inside hardware execution wraps.
     * * @tparam NBasis The number of basis functions (polynomial order + 1).
     */
    template<size_t NBasis>
    struct LaguerreBasis {
        /**
         * @brief Evaluates orthogonal Laguerre functions up to degree 4.
         * * Values are filled directly into a fixed stack array passed by reference:
         * - $L_0(x) = 1$
         * - $L_1(x) = 1 - x$
         * - $L_2(x) = 1 - 2x + \frac{1}{2}x^2$
         * - $L_3(x) = 1 - 3x + \frac{3}{2}x^2 - \frac{1}{6}x^3$
         * - $L_4(x) = 1 - 4x + 3x^2 - \frac{2}{3}x^3 + \frac{1}{24}x^4$
         * * @param X The normalized price coordinate ($S_t / K$).
         * @param L Stack reference array to populate with evaluated polynomial levels.
         */
        KOKKOS_INLINE_FUNCTION
        static void evaluate(const KT::Real X, KT::Real (&L)[NBasis]) {
            // L Passed by reference as a fixed array size
            L[0] = 1.0;
            if constexpr (NBasis > 1) L[1] = 1.0 - X;
            if constexpr (NBasis > 2) L[2] = 1.0 - 2.0 * X + 0.5 * X * X;
            if constexpr (NBasis > 3) L[3] = 1.0 - 3.0 * X + 1.5 * X * X - (X * X * X) / 6.0;
            if constexpr (NBasis > 4)
                L[4] = 1.0 - 4.0 * X + 3.0 * X * X - (2.0 / 3.0) * X * X * X + (1.0 / 24.0) * X *
                       X * X * X;
        }
    };


    /**
     * @brief Core template processing structural steps of the Longstaff-Schwartz American valuation induction.
     *  @note Operations are split between high-throughput data reduction on the Device (GPU) and matrix linear algebra solutions on the Host (CPU) to
     * maximize processing performance.
     * * ### Mathematical Overview & Execution Split:
     * 1. **Device Phase 1 (`Kokkos::parallel_reduce`)**: Screens all simulation lines to isolate components that are
     * currently In-The-Money. Evaluates polynomial basis configurations and aggregates entries into a structural matrix $M$ and vector $B$.
     * 2. **Host Pass (CPU Linear Algebra)**: Performs Tikhonov regularization on the tiny $NBasis \times NBasis$ matrix
     * and solves for the regression weights $C = M^{-1}B$ via explicit Cholesky decomposition.
     * 3. **Device Phase 2 (`Kokkos::parallel_for`)**: Evaluates expected continuation values using the completed regression weights,
     * compares values against immediate exercise payoffs, and updates the path cash flow vectors.
     * * @tparam OptRight Exercise right contract configuration policy (Call vs Put).
     * @tparam Basis Target polynomial basis configuration selection.
     * @todo Should implement the control variate technique to improve the precision, see Pag. 208, eq. 7, the the book : Fabrice D. Rouah's *The Heston Model and its Extensions in Matlab and C#*.
     */
    template<KI::OptRight OptRight, KI::LSRegressionBasis Basis = KI::LSRegressionBasis::LaguerreP03> //MonomialO2
    class LSMEngine {
    public:
        using DevView1D = Kokkos::View<KT::Real *, Kokkos::LayoutLeft>;

        // =================================================================
        // STEP 0 : Chose the basis for the regression
        // =================================================================
        /** @brief Translates structural configuration enums to compile-time capacity constraints. */
        static constexpr size_t get_n_basis() {
            if constexpr (Basis == KI::LSRegressionBasis::LaguerreP02) {
                return 3;
            } else if constexpr (Basis == KI::LSRegressionBasis::LaguerreP03) {
                return 4;
            } else if constexpr (Basis == KI::LSRegressionBasis::LaguerreP04) {
                return 5;
            } else {
                static_assert(Basis != Basis, "Unsupported LSM Regression Basis!");
                return 0;
            }
        }

        static constexpr size_t NBasis = get_n_basis(); ///< System rank constant resolved at compile-time.

        // =================================================================
        // STEP 1 : Initialize Terminal Payoffs
        // =================================================================
        /**
         * @brief STEP 1: Initializes terminal payoff vectors at maturity ($t = T$).
         * * Maps terminal asset spot prices across all paths into the base cash flow tracking array
         * using parallel execution kernels.
         * * @param d_slice_prices_at_target_time 1D View containing path spot values at maturity.
         * @param d_best_future_outcomes 1D Output View containing the initialized cash flows.
         * @param strike_price Contract exercise strike price ($K$).
         */
        static void initialize_cashflows(const DevView1D &d_slice_prices_at_target_time,
                                         const DevView1D &d_best_future_outcomes,
                                         const KT::Real strike_price) {
            // Note: It assumes that d_slice_prices_at_target_time is given at last time (not passing mem since the labda would try capture and load it in kokkos)
            const int N = d_slice_prices_at_target_time.extent(0);
            Kokkos::parallel_for("LSM_Init_Cashflows", N, KOKKOS_LAMBDA(const int i)
            {
                KT::Real target_price = d_slice_prices_at_target_time(i);
                d_best_future_outcomes(i) = Payoff<OptRight>::evaluate_payoff(target_price, strike_price);
            }
            )
            ;
            Kokkos::fence();
        }

        // =================================================================
        // STEP 2 : The Cross-Paths Regression
        // =================================================================
        /**
         * @brief STEP 2: Main coordinator for the cross-sectional path regression.
         * * Combines parallel device reduction kernels and host linear system solvers to
         * determine the optimal regression coefficients.
         * * @param d_slice_prices_at_target_time 1D View of asset prices at the current time step.
         * @param d_best_future_outcomes 1D View tracking historical future optimal cash flows.
         * @param discount_factor Continuous discount factor spanning a single step ($\Delta t$).
         * @param strike_price Contract exercise strike price.
         * @return The solved Coeffs structure containing the regression weights.
         */
        static Coeffs<NBasis> perform_cross_paths_regression(const DevView1D &d_slice_prices_at_target_time,
                                                             const DevView1D &d_best_future_outcomes,
                                                             const KT::Real discount_factor,
                                                             const KT::Real strike_price) {
            // 1. Compute terms M and B of linear system MC = B
            // NOTE : Run the parallel reduction on the GPU to get terms needed in the matrix multiplication
            const RegressionSums<NBasis> sums = compute_regression_sums_on_device(d_slice_prices_at_target_time,
                d_best_future_outcomes,
                discount_factor,
                strike_price);

            // 2. Get C = M^-1 B
            // Solve the system containing the (NBasisxNBasis) matrix on the CPU Host
            return solve_system_on_host_cholesky(sums);
        }


        // =================================================================
        // STEP 3: The Early Exercise Decision
        // =================================================================
        /**
         * @brief STEP 3: Evaluates the early exercise boundary and updates the path cash flows.
         * * Compares the immediate intrinsic payoff value against the conditional expected continuation
         * value (derived from the regression coefficients). If the intrinsic value is higher, early exercise
         * is optimal; the future cash flow is overwritten, and future payoffs along that path are canceled.
         * * @param d_slice_prices_at_target_time 1D View of asset prices at the current time step.
         * @param d_best_future_outcomes 1D View tracking optimal future cash flows.
         * @param ls_coeffs Completed regression coefficients struct (passed by value to device memory).
         * @param discount_factor Step continuous discount calculation scalar factor.
         * @param strike_price Contract execution strike price constraint.
         */
        static void update_cashflows(const DevView1D &d_slice_prices_at_target_time,
                                     const DevView1D &d_best_future_outcomes,
                                     const Coeffs<NBasis> ls_coeffs,
                                     // Pass by value since going to GPU and small struct
                                     const KT::Real discount_factor,
                                     const KT::Real strike_price) {
            const int N = d_slice_prices_at_target_time.extent(0);

            Kokkos::parallel_for("LSM_Update_Cashflows", N, KOKKOS_LAMBDA(const int i)
            {
                const KT::Real S = d_slice_prices_at_target_time(i);
                const KT::Real intrinsic_val = Payoff<OptRight>::evaluate_payoff(S, strike_price);
                const KT::Real discounted_future_cf = d_best_future_outcomes(i) * discount_factor;

                // Early exercise decision (ITM branch)
                if (intrinsic_val > KT::real_zero) {
                    KT::Real expected_val_of_holding = evaluate_expected_val_of_holding(ls_coeffs, S, strike_price);
                    // Continuation value
                    d_best_future_outcomes(i) = (intrinsic_val > expected_val_of_holding)
                                                    ? intrinsic_val
                                                    : discounted_future_cf;
                    // if : intrinsic_val > expected_val_of_holding the option is exercised (holding will be statistically worst). The future cash flow is overwritten with the intrinsic_val of today
                    // else : we hold the option, based on statistical average of all paths, holding will likely yield a bigg payout in the future
                } else {
                    d_best_future_outcomes(i) = discounted_future_cf; // Hold since exiting is impossible
                }
            }
            )
            ;
            Kokkos::fence();
        }

        // =================================================================
        // STEP 4: Final t=1 to t=0 Discounting
        // =================================================================
        /**
         * @brief STEP 4: Applies the final discount factor from $t = 1$ to $t = 0$.
         * * Multiplies the remaining optimal cash flow trajectories by the final discount factor
         * to bring all values back to time zero.
         * * @param d_best_future_outcomes 1D cash flow target view matrix vector.
         * @param discount_factor Final continuous discount factor fraction step value.
         */
        static void apply_final_discount(const DevView1D &d_best_future_outcomes,
                                         const KT::Real discount_factor) {
            const int N = d_best_future_outcomes.extent(0);
            Kokkos::parallel_for("LSM_Final_Discount", N, KOKKOS_LAMBDA(const int i)
            {
                d_best_future_outcomes(i) *= discount_factor;
            }
            )
            ;
            Kokkos::fence();
        }

        // =================================================================
        // HELPER FUNCTIONS TO PERFORM THE LEAST SQUARES REGRESSIONS
        // =================================================================
        /**
         * @brief Compiles and reduces the cross-product regression sums on the device.
         * * Filters the path cross-section to isolate lines that are In-The-Money ($Payoff(S_t) > 0$).
         * For those paths, it evaluates the Laguerre polynomials and updates the normal equations
         * matrix components using a single `Kokkos::parallel_reduce` pass.
         * * @note This method must remain public to grant CUDA/HIP device kernels visibility
         * and permission to execute the reduction code.
         * @return The regression sums composing the M matrix and the B Vector
         */
        static RegressionSums<NBasis> compute_regression_sums_on_device(const DevView1D &d_slice_prices,
                                                                        const DevView1D &d_best_future_outcomes,
                                                                        const KT::Real discount_factor,
                                                                        const KT::Real strike_price) {
            // Compute terms M and B of linear system MC = B
            // NOTE : Method kept public since if private cuda does not have access to it

            RegressionSums<NBasis> total_sums;
            const int N_prices = d_slice_prices.extent(0);

            Kokkos::parallel_reduce("LSM_Compute_Regression_Sums", N_prices,
                                    KOKKOS_LAMBDA(const int i, RegressionSums<NBasis> & local)
            {
                const KT::Real S = d_slice_prices(i);
                const KT::Real intrinsic_val = Payoff<OptRight>::evaluate_payoff(S, strike_price);

                if (intrinsic_val > KT::real_zero) {
                    const KT::Real Y = d_best_future_outcomes(i) * discount_factor;
                    const KT::Real X = S / strike_price;

                    // Evaluate the basis polynomials into a local stack buffer
                    KT::Real L[NBasis];
                    LaguerreBasis<NBasis>::evaluate(X, L);

                    // Update M and B using the unified templated logic
                    // The compiler should unroll these loops, making it as fast as manual coding
#pragma unroll
                    for (size_t row = 0; row < NBasis; ++row) {
                        local.B(row) += L[row] * Y;
#pragma unroll
                        for (size_t col = 0; col < NBasis; ++col) {
                            local.M(row, col) += L[row] * L[col];
                        }
                    }
                    local.count += 1.0;
                }
            }
            ,
            total_sums
            )
            ;
            return total_sums;
        }

        /**
         * @brief Computes the expected continuation value (value of holding the contract) at a given asset price.
         * * Calculates the dot product of the solved regression weights and the evaluated polynomial basis functions:
         * $$ \hat{V}_{\text{continue}} = \sum_{i=0}^{N-1} C_i \cdot L_i(X) $$
         * * @param coeffs Solved regression coefficients container reference.
         * @param S Current asset price level.
         * @param strike_price Contract exercise strike price.
         * @return The computed expected continuation value.
         */
        KOKKOS_INLINE_FUNCTION
        static KT::Real evaluate_expected_val_of_holding(const Coeffs<NBasis> &coeffs,
                                                         const KT::Real S,
                                                         const KT::Real strike_price) {
            const KT::Real X = S / strike_price;
            KT::Real L[NBasis];

            // Use the unified basis evaluator
            LaguerreBasis<NBasis>::evaluate(X, L);

            KT::Real continuation_value = 0.0;

            // Accumulate the weighted sum (coeffs * basis)
#pragma unroll
            for (size_t i = 0; i < NBasis; ++i) {
                continuation_value += coeffs.C[i] * L[i];
            }

            return continuation_value;
        }

        /**
         * @brief Solves the system of normal equations ($M \cdot C = B$) on the CPU host via Cholesky decomposition.
         * * Applies Tikhonov regularization ($\text{reg} \cdot I$) to the matrix diagonal to ensure positive-definiteness
         * and numerical stability, then resolves the coefficient vector $C$ using standard forward and backward substitution:
         * 1. Decomposition: $M = L \cdot L^T$
         * 2. Forward Substitution: $L \cdot y = B$
         * 3. Backward Substitution: $L^T \cdot C = y$
         * * @param sums Aggregated regression matrix and vector entries computed on the device.
         * @return A populated Coeffs structure containing the solved weights. If the matrix is
         * singular or there are fewer ITM paths than basis functions, it returns a zeroed structure.
         */
        static Coeffs<NBasis> solve_system_on_host_cholesky(const RegressionSums<NBasis> &sums) {
            // If not enough paths to form a basis, return zero coefficients
            if (sums.count < static_cast<KT::Real>(NBasis)) return Coeffs<NBasis>{};

            // Tikhonov Regularization
            // Add a tiny amount of noise (1 part per million) relative to the signal present in the data (average diagonal element)
            //             KT::Real trace = 0.0;
            // #pragma unroll
            //             for (size_t i = 0; i < NBasis; ++i) {
            //                 trace += sums.M(i, i);
            //             }
            //             const KT::Real avg_diag = trace / static_cast<KT::Real>(NBasis);
            //             constexpr KT::Real eps_factor = std::is_same_v<KT::Real, float> ? 1e-4 : 1e-6;
            //             const KT::Real reg = std::max(eps_factor * avg_diag, eps_factor);

            constexpr KT::Real reg = std::is_same_v<KT::Real, float> ? 1e-4 : 1e-6;

            // 1. Prepare local M and B
            KT::Real M[NBasis][NBasis];
            KT::Real B[NBasis];

#pragma unroll
            for (size_t i = 0; i < NBasis; ++i) {
                B[i] = sums.B(i);
#pragma unroll
                for (size_t j = 0; j < NBasis; ++j) {
                    M[i][j] = sums.M(i, j) + ((i == j) ? reg : 0.0);
                }
            }

            // 2. Cholesky Decomposition: M = L * L^T
            KT::Real L[NBasis][NBasis] = {0.0};

            const KT::Real singularity_limit = std::is_same_v<KT::Real, float> ? 1e-6 : 1e-14;

            for (size_t i = 0; i < NBasis; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    KT::Real s = 0.0;
#pragma unroll
                    for (size_t k = 0; k < j; ++k) s += L[i][k] * L[j][k];

                    if (i == j) {
                        KT::Real val = M[i][i] - s;
                        if (val < singularity_limit) return Coeffs<NBasis>{}; // Not pos-def
                        L[i][i] = std::sqrt(val);
                    } else {
                        L[i][j] = (M[i][j] - s) / L[j][j];
                    }
                }
            }

            // 3. Forward substitution: L * y = B
            KT::Real y[NBasis];
            for (size_t i = 0; i < NBasis; ++i) {
                KT::Real s = 0.0;
#pragma unroll
                for (size_t k = 0; k < i; ++k) s += L[i][k] * y[k];
                y[i] = (B[i] - s) / L[i][i];
            }

            // 4. Backward substitution: L^T * C = y
            Coeffs<NBasis> C;
            for (int i = static_cast<int>(NBasis) - 1; i >= 0; --i) {
                KT::Real s = 0.0;
#pragma unroll
                for (size_t k = i + 1; k < NBasis; ++k) s += L[k][i] * C[k];
                C[i] = (y[i] - s) / L[i][i];
            }

            return C;
        }
    };
}
