#pragma once

#include <Kokkos_Core.hpp>

#include "../Typedefs.hpp"
#include "../options/Payoff.hpp"

namespace KOps::Engine::LSM {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;

    struct LSCoeffs {
        KT::Real c0 = 0.0;
        KT::Real c1 = 0.0;
        KT::Real c2 = 0.0;
        KT::Real c3 = 0.0;
    };

    struct RegressionSums_MonomialO2 {
        // Case in which the regression is done with a second order polynomial
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found later

        // Components Matrix $M = (X^T X)$ (symmetric matrix)
        KT::Real m11{0.0}; // N (number of ITM paths)
        KT::Real m12{0.0}; // Sum(X)
        KT::Real m13{0.0}; // Sum(X^2)
        KT::Real m23{0.0}; // Sum(X^3)
        KT::Real m33{0.0}; // Sum(X^4)

        //  Components Vector $B = (X^T Y)$
        KT::Real b1{0.0}; // Sum(Y)
        KT::Real b2{0.0}; // Sum(X * Y)
        KT::Real b3{0.0}; // Sum(X^2 * Y)

        KOKKOS_INLINE_FUNCTION
        RegressionSums_MonomialO2 &operator+=(const RegressionSums_MonomialO2 &src) {
            // Matrix $M = (X^T X)$
            m11 += src.m11;
            m12 += src.m12;
            m13 += src.m13;
            m23 += src.m23;
            m33 += src.m33;

            //  Vector $B = (X^T Y)$
            b1 += src.b1;
            b2 += src.b2;
            b3 += src.b3;
            return *this;
        }
    };

    struct RegressionSums_Laguerre03 {
        KT::Real count{0.0};

        // The 10 unique elements of the 4x4 Symmetric Matrix (L_i * L_j)
        KT::Real m00{0.0}, m01{0.0}, m02{0.0}, m03{0.0};
        KT::Real m11{0.0}, m12{0.0}, m13{0.0};
        KT::Real m22{0.0}, m23{0.0};
        KT::Real m33{0.0};

        // The 4 elements of the Vector (L_i * Y)
        KT::Real b0{0.0}, b1{0.0}, b2{0.0}, b3{0.0};

        KOKKOS_INLINE_FUNCTION
        RegressionSums_Laguerre03 &operator+=(const RegressionSums_Laguerre03 &src) {
            count += src.count;

            m00 += src.m00;
            m01 += src.m01;
            m02 += src.m02;
            m03 += src.m03;
            m11 += src.m11;
            m12 += src.m12;
            m13 += src.m13;
            m22 += src.m22;
            m23 += src.m23;
            m33 += src.m33;

            b0 += src.b0;
            b1 += src.b1;
            b2 += src.b2;
            b3 += src.b3;
            return *this;
        }
    };


    template<KI::OptRight OptRight, KI::LSRegressionBasis Basis = KI::LSRegressionBasis::LaguerreO3> //MonomialO2
    class LSMEngine {
    public:
        using DevView1D = Kokkos::View<KT::Real *, Kokkos::LayoutLeft>;

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
        static LSCoeffs perform_cross_paths_regression(const DevView1D &d_slice_prices_at_target_time,
                                                       const DevView1D &d_best_future_outcomes,
                                                       const KT::Real discount_factor,
                                                       const KT::Real strike_price) {
            if constexpr (Basis == KI::LSRegressionBasis::MonomialO2) {
                // 1. Compute terms M and B of linear system MC = B
                // NOTE : Run the parallel reduction on the GPU to get terms needed in the matrix multiplication
                const RegressionSums_MonomialO2 sums = compute_regression_sums_P2_on_device(
                    d_slice_prices_at_target_time,
                    d_best_future_outcomes,
                    discount_factor,
                    strike_price);
                // 2. Get C = M^-1 B
                // Solve the 3x3 matrix on the CPU Host, returns the coeffs needed to compute the best future outcomes
                return solve_3x3_system_on_host_cramer(sums);
            } else {
                // 1. Compute terms M and B of linear system MC = B
                // NOTE : Run the parallel reduction on the GPU to get terms needed in the matrix multiplication
                const RegressionSums_Laguerre03 sums = compute_regression_sums_P3_on_device(
                    d_slice_prices_at_target_time,
                    d_best_future_outcomes,
                    discount_factor,
                    strike_price);
                // 2. Get C = M^-1 B
                // Solve the 4x4 matrix on the CPU Host, returns the coeffs needed to compute the best future outcomes
                return solve_4x4_system_on_host_cholesky(sums);
            }
        }

        // =================================================================
        // STEP 3: The Early Exercise Decision
        // =================================================================
        static void update_cashflows(const DevView1D &d_slice_prices_at_target_time,
                                     const DevView1D &d_best_future_outcomes,
                                     const LSCoeffs &ls_coeffs,
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
                    KT::Real expected_val_of_holding = evaluate_continuation_value(ls_coeffs, S, strike_price);
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

        static RegressionSums_MonomialO2 compute_regression_sums_P2_on_device(const DevView1D &d_slice_prices,
                                                                              const DevView1D &d_best_future_outcomes,
                                                                              const KT::Real discount_factor,
                                                                              const KT::Real strike_price) {
            // Compute terms M and B of linear system MC = B
            // NOTE : Method kept public since if private cuda does not have access to it
            const int N = d_slice_prices.extent(0);
            RegressionSums_MonomialO2 total_sums;

            Kokkos::parallel_reduce("LSM_Compute_Regression_Sums_Monomial02",
                                    N,
                                    KOKKOS_LAMBDA(const int i, RegressionSums_MonomialO2 & local_sum)
            {
                const KT::Real S = d_slice_prices(i);
                const KT::Real intrinsic_val = Payoff<OptRight>::evaluate_payoff(S, strike_price);

                // We ONLY include paths that are In-The-Money in the regression
                if (intrinsic_val > KT::real_zero) {
                    const KT::Real Z = S / strike_price; // normalize for numerical stability

                    const KT::Real Z2 = Z * Z;
                    const KT::Real Z3 = Z2 * Z;
                    const KT::Real Z4 = Z3 * Z;

                    // Y is the future cash flow discounted to today
                    const KT::Real Y = d_best_future_outcomes(i) * discount_factor;

                    local_sum.m11 += 1.0;
                    local_sum.m12 += Z;
                    local_sum.m13 += Z2;
                    local_sum.m23 += Z3;
                    local_sum.m33 += Z4;

                    local_sum.b1 += Y;
                    local_sum.b2 += Z * Y;
                    local_sum.b3 += Z2 * Y;
                }
            }
            ,
            total_sums
            )
            ; // total_sums is automatically copied back to the Host CPU here
            Kokkos::fence();
            return total_sums;
        }


        static RegressionSums_Laguerre03 compute_regression_sums_P3_on_device(const DevView1D &d_slice_prices,
                                                                              const DevView1D &d_best_future_outcomes,
                                                                              const KT::Real discount_factor,
                                                                              const KT::Real strike_price) {
            RegressionSums_Laguerre03 total_sums;
            const int N = d_slice_prices.extent(0);

            Kokkos::parallel_reduce("LSM_Compute_Laguerre_Sums", N,
                                    KOKKOS_LAMBDA(const int i, RegressionSums_Laguerre03 & local)
            {
                const KT::Real S = d_slice_prices(i);
                const KT::Real intrinsic_val = Payoff<OptRight>::evaluate_payoff(S, strike_price);

                if (intrinsic_val > KT::real_zero) {
                    const KT::Real Y = d_best_future_outcomes(i) * discount_factor;
                    const KT::Real X = S / strike_price;

                    // Laguerre Polynomials (Order 3)
                    const KT::Real L0 = 1.0;
                    const KT::Real L1 = 1.0 - X;
                    const KT::Real L2 = 1.0 - 2.0 * X + 0.5 * X * X;
                    const KT::Real L3 = 1.0 - 3.0 * X + 1.5 * X * X - (X * X * X) / 6.0;

                    // Update Vector B (L_i * Y)
                    local.b0 += L0 * Y;
                    local.b1 += L1 * Y;
                    local.b2 += L2 * Y;
                    local.b3 += L3 * Y;

                    // Update Symmetric Matrix M (L_i * L_j)
                    // Note: Only need to compute the upper triangle (10 elements)
                    local.m00 += L0 * L0;
                    local.m01 += L0 * L1;
                    local.m02 += L0 * L2;
                    local.m03 += L0 * L3;
                    local.m11 += L1 * L1;
                    local.m12 += L1 * L2;
                    local.m13 += L1 * L3;
                    local.m22 += L2 * L2;
                    local.m23 += L2 * L3;
                    local.m33 += L3 * L3;

                    local.count += 1.0;
                }
            }
            ,
            total_sums
            )
            ;

            Kokkos::fence();
            return total_sums;
        }


        KOKKOS_INLINE_FUNCTION
        static KT::Real evaluate_continuation_value(const LSCoeffs &coeffs, const KT::Real S,
                                                    const KT::Real strike_price) {
            const KT::Real Z = S / strike_price;
            if constexpr (Basis == KI::LSRegressionBasis::MonomialO2) {
                return coeffs.c0 + (coeffs.c1 * Z) + (coeffs.c2 * Z * Z);
            } else {
                const KT::Real L0 = 1.0;
                const KT::Real L1 = 1.0 - Z;
                const KT::Real L2 = 1.0 - 2.0 * Z + 0.5 * Z * Z;
                const KT::Real L3 = 1.0 - 3.0 * Z + 1.5 * Z * Z - (Z * Z * Z) / 6.0;
                return coeffs.c0 * L0 + coeffs.c1 * L1 + coeffs.c2 * L2 + coeffs.c3 * L3;
            }
        }

        static LSCoeffs solve_3x3_system_on_host_cramer(const RegressionSums_MonomialO2 &sums) {
            // Solving the sistem C = M-1 * B using cramer rule
            // NOTE : in a system of linear equations MC=B, you can find any individual coefficient C_i
            // simply by calculating the ratio of two determinants det(M_i)/ det(M), where M_i swap colum i with B
            // NOTE : Method kept public since if private cuda does not have access to it

            // If there are fewer than 3 ITM paths, we cannot fit a quadratic curve.
            if (sums.m11 < 3.0) {
                return LSCoeffs{0.0, 0.0, 0.0, 0.0};
            }

            // The 3x3 Matrix M = X^T X
            const KT::Real m11 = sums.m11;
            const KT::Real m12 = sums.m12;
            const KT::Real m13 = sums.m13;
            const KT::Real m21 = m12;
            const KT::Real m22 = sums.m13;
            const KT::Real m23 = sums.m23;
            const KT::Real m31 = sums.m13;
            const KT::Real m32 = m23;
            const KT::Real m33 = sums.m33;

            // The 3x1 Vector B = X^T Y
            const KT::Real B0 = sums.b1;
            const KT::Real B1 = sums.b2;
            const KT::Real B2 = sums.b3;

            // Calculate Determinant of M
            const KT::Real detM = m11 * (m22 * m33 - m23 * m32) - m12 * (m21 * m33 - m23 * m31) + m13 * (
                                      m21 * m32 - m22 * m31);

            // Safety check for singular or nearly-singular matrix (e.g., all ITM paths have the exact same price)
            if (std::abs(detM) < 1e-12) {
                return LSCoeffs{0.0, 0.0, 0.0, 0.0};
            }

            // Cramer's Rule determinants for C0, C1, C2
            const KT::Real det0 = B0 * (m22 * m33 - m23 * m32) - m12 * (B1 * m33 - m23 * B2) + m13 * (
                                      B1 * m32 - m22 * B2);
            const KT::Real det1 = m11 * (B1 * m33 - m23 * B2) - B0 * (m21 * m33 - m23 * m31) + m13 * (
                                      m21 * B2 - B1 * m31);
            const KT::Real det2 = m11 * (m22 * B2 - B1 * m32) - m12 * (m21 * B2 - B1 * m31) + B0 * (
                                      m21 * m32 - m22 * m31);

            return LSCoeffs{
                det0 / detM, // c0
                det1 / detM, // c1
                det2 / detM, // c2
                0.
            };
        }

        static LSCoeffs solve_4x4_system_on_host_cholesky(const RegressionSums_Laguerre03 &sums) {
            // If fewer than 4 ITM paths, regression is ill-conditioned
            // Note: allocating matices and vector direcly on the stack
            if (sums.count < 4.0) return LSCoeffs{0.0, 0.0, 0.0, 0.0};

            // 1. Fill the symmetric matrix M (4x4)
            constexpr KT::Real reg = 1e-6; // Add Tikhonov Regularization
            const KT::Real M[4][4] = {
                {sums.m00 + reg, sums.m01, sums.m02, sums.m03},
                {sums.m01, sums.m11 + reg, sums.m12, sums.m13},
                {sums.m02, sums.m12, sums.m22 + reg, sums.m23},
                {sums.m03, sums.m13, sums.m23, sums.m33 + reg}
            };
            const KT::Real B[4] = {sums.b0, sums.b1, sums.b2, sums.b3};

            // 2. Cholesky Decomposition: M = L * L^T
            // L is lower triangular
            KT::Real L[4][4] = {0.0};

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j <= i; j++) {
                    KT::Real s = 0.0;
                    for (int k = 0; k < j; k++) s += L[i][k] * L[j][k];

                    if (i == j) {
                        KT::Real val = M[i][i] - s;
                        if (val < 1e-14) return LSCoeffs{0.0, 0.0, 0.0, 0.0}; // Not pos-def
                        L[i][i] = std::sqrt(val);
                    } else {
                        L[i][j] = (M[i][j] - s) / L[j][j];
                    }
                }
            }

            // 3. Forward substitution: L * y = B
            KT::Real y[4];
            y[0] = B[0] / L[0][0];
            y[1] = (B[1] - L[1][0] * y[0]) / L[1][1];
            y[2] = (B[2] - L[2][0] * y[0] - L[2][1] * y[1]) / L[2][2];
            y[3] = (B[3] - L[3][0] * y[0] - L[3][1] * y[1] - L[3][2] * y[2]) / L[3][3];

            // 4. Backward substitution: L^T * C = y
            LSCoeffs C;
            C.c3 = y[3] / L[3][3];
            C.c2 = (y[2] - L[3][2] * C.c3) / L[2][2];
            C.c1 = (y[1] - L[2][1] * C.c2 - L[3][1] * C.c3) / L[1][1];
            C.c0 = (y[0] - L[1][0] * C.c1 - L[2][0] * C.c2 - L[3][0] * C.c3) / L[0][0];

            return C;
        }
    };
}
