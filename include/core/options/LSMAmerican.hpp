#pragma once

#include <Kokkos_Core.hpp>

#include "../Typedefs.hpp"
#include "../options/Payoff.hpp"

namespace KOps::Engine::LSM {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;

    template<size_t NBasis = 4>
    struct Coeffs {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found
        KT::Real C[NBasis] = {0.0};

        KOKKOS_INLINE_FUNCTION
        KT::Real &operator[](size_t i) {return C[i];}

        // Allows reading: auto val = my_coeffs[0];
        KOKKOS_INLINE_FUNCTION
        const KT::Real &operator[](size_t i) const { return C[i];}
    };

    template<size_t NBasis = 4>
    struct Matrix {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found
        // Components Matrix $M = (X^T X)$ (symmetric matrix)
        KT::Real M[NBasis][NBasis] = {};

        KOKKOS_INLINE_FUNCTION
        KT::Real &operator()(size_t i, size_t j) { return M[i][j]; }

        KOKKOS_INLINE_FUNCTION
        const KT::Real &operator()(size_t i, size_t j) const { return M[i][j]; }

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

    template<size_t NBasis = 4>
    struct Vector {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found
        //  Components Vector $B = (X^T Y)$
        KT::Real B[NBasis] = {};

        KOKKOS_INLINE_FUNCTION
        KT::Real &operator()(size_t i) { return B[i]; }

        KOKKOS_INLINE_FUNCTION
        const KT::Real &operator()(size_t i) const { return B[i]; }


        KOKKOS_INLINE_FUNCTION
        Vector &operator+=(const Vector &src) {
#pragma unroll
            for (size_t i = 0; i < NBasis; ++i) {
                B[i] += src.B[i];
            }
            return *this;
        }
    };

    template<size_t NBasis = 4>
    struct RegressionSums {
        KT::Real count{0.0};
        Matrix<NBasis> M;
        Vector<NBasis> B;

        KOKKOS_INLINE_FUNCTION
        RegressionSums &operator+=(const RegressionSums &src) {
            count += src.count;
            M += src.M;
            B += src.B;
            return *this;
        }
    };

    template<size_t NBasis>
    struct LaguerreBasis {
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


    template<KI::OptRight OptRight, KI::LSRegressionBasis Basis = KI::LSRegressionBasis::LaguerreP03> //MonomialO2
    class LSMEngine {
    public:
        using DevView1D = Kokkos::View<KT::Real *, Kokkos::LayoutLeft>;

        // =================================================================
        // STEP 0 : Chose the basis for the regression
        // =================================================================
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

        static constexpr size_t NBasis = get_n_basis();

        // =================================================================
        // STEP 1 : Initialize Terminal Payoffs
        // =================================================================
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
