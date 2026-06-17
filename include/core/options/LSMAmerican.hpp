#pragma once

#include <Kokkos_Core.hpp>

#include "../Typedefs.hpp"
#include "../options/Payoff.hpp"

namespace KOps::Engine::LSM {
    namespace KI = KOps::Implemented;
    namespace KT = KOps::Types;

    struct LSCoeffs {
        KT::Real b0 = 0.0;
        KT::Real b1 = 0.0;
        KT::Real b2 = 0.0;
    };

    struct RegressionSums {
        //  C = M^{-1} B, where M = (X^T X), B = (X^T Y), C regression coeffs to be found later

        // Components Matrix $M = (X^T X)$
        KT::Real count{0.0}; // N (number of ITM paths)
        KT::Real x1{0.0}; // Sum(X)
        KT::Real x2{0.0}; // Sum(X^2)
        KT::Real x3{0.0}; // Sum(X^3)
        KT::Real x4{0.0}; // Sum(X^4)

        //  Components Vector $B = (X^T Y)$
        KT::Real y{0.0}; // Sum(Y)
        KT::Real xy{0.0}; // Sum(X * Y)
        KT::Real x2y{0.0}; // Sum(X^2 * Y)

        KOKKOS_INLINE_FUNCTION
        RegressionSums &operator+=(const RegressionSums &src) {
            // Matrix $M = (X^T X)$
            count += src.count;
            x1 += src.x1;
            x2 += src.x2;
            x3 += src.x3;
            x4 += src.x4;

            //  Vector $B = (X^T Y)$
            y += src.y;
            xy += src.xy;
            x2y += src.x2y;
            return *this;
        }
    };

    template<KI::OptRight OptRight>
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
            // 1. Compute terms M and B of linear system MC = B
            // NOTE : Run the parallel reduction on the GPU to get terms needed in the matrix multiplication
            const RegressionSums sums = compute_regression_sums_on_device(
                d_slice_prices_at_target_time,
                d_best_future_outcomes,
                discount_factor,
                strike_price
            );

            // 2. Get C = M^-1 B
            // Solve the 3x3 matrix on the CPU Host, returns the coeffs needed to compute the best future outcomes
            return solve_linear_system_on_host(sums);
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
                    const KT::Real X = S / strike_price; // normalize for numerical stability
                    const KT::Real expected_val_of_holding = ls_coeffs.b0 + (ls_coeffs.b1 * X) + (ls_coeffs.b2 * X * X);
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

        static RegressionSums compute_regression_sums_on_device(const DevView1D &d_slice_prices,
                                                                const DevView1D &d_best_future_outcomes,
                                                                const KT::Real discount_factor,
                                                                const KT::Real strike_price) {
            // Compute terms M and B of linear system MC = B
            // NOTE : Method kept public since if private cuda does not have access to it
            const int N = d_slice_prices.extent(0);
            RegressionSums total_sums;

            Kokkos::parallel_reduce("LSM_Compute_Regression_Sums",
                                    N,
                                    KOKKOS_LAMBDA(const int i, RegressionSums & local_sum)
            {
                const KT::Real S = d_slice_prices(i);
                const KT::Real intrinsic_val = Payoff<OptRight>::evaluate_payoff(S, strike_price);

                // We ONLY include paths that are In-The-Money in the regression
                if (intrinsic_val > KT::real_zero) {
                    const KT::Real X = S / strike_price; // normalize for numerical stability
                    const KT::Real X2 = X * X;
                    const KT::Real X3 = X2 * X;
                    const KT::Real X4 = X3 * X;

                    // Y is the future cash flow discounted to today
                    const KT::Real Y = d_best_future_outcomes(i) * discount_factor;

                    local_sum.count += 1.0;
                    local_sum.x1 += X;
                    local_sum.x2 += X2;
                    local_sum.x3 += X3;
                    local_sum.x4 += X4;

                    local_sum.y += Y;
                    local_sum.xy += X * Y;
                    local_sum.x2y += X2 * Y;
                }
            }
            ,
            total_sums
            )
            ; // total_sums is automatically copied back to the Host CPU here

            Kokkos::fence();
            return total_sums;
        }


        static LSCoeffs solve_linear_system_on_host(const RegressionSums &sums) {
            // Solving the sistem C = M-1 * B using cramer rule
            // NOTE : in a system of linear equations MC=B, you can find any individual coefficient C_i
            // simply by calculating the ratio of two determinants det(M_i)/ det(M), where M_i swap colum i with B
            // NOTE : Method kept public since if private cuda does not have access to it

            // If there are fewer than 3 ITM paths, we cannot fit a quadratic curve.
            if (sums.count < 3.0) {
                return LSCoeffs{0.0, 0.0, 0.0};
            }

            // The 3x3 Matrix M = X^T X
            const KT::Real a11 = sums.count;
            const KT::Real a12 = sums.x1;
            const KT::Real a13 = sums.x2;
            const KT::Real a21 = a12;
            const KT::Real a22 = sums.x2;
            const KT::Real a23 = sums.x3;
            const KT::Real a31 = sums.x2;
            const KT::Real a32 = a23;
            const KT::Real a33 = sums.x4;

            // The 3x1 Vector B = X^T Y
            const KT::Real Y0 = sums.y;
            const KT::Real Y1 = sums.xy;
            const KT::Real Y2 = sums.x2y;

            // Calculate Determinant of M
            const KT::Real detA = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (
                                      a21 * a32 - a22 * a31);

            // Safety check for singular or nearly-singular matrix (e.g., all ITM paths have the exact same price)
            if (std::abs(detA) < 1e-12) {
                return LSCoeffs{0.0, 0.0, 0.0};
            }

            // Cramer's Rule determinants for C0, C1, C2
            const KT::Real det0 = Y0 * (a22 * a33 - a23 * a32) - a12 * (Y1 * a33 - a23 * Y2) + a13 * (
                                      Y1 * a32 - a22 * Y2);
            const KT::Real det1 = a11 * (Y1 * a33 - a23 * Y2) - Y0 * (a21 * a33 - a23 * a31) + a13 * (
                                      a21 * Y2 - Y1 * a31);
            const KT::Real det2 = a11 * (a22 * Y2 - Y1 * a32) - a12 * (a21 * Y2 - Y1 * a31) + Y0 * (
                                      a21 * a32 - a22 * a31);

            return LSCoeffs{
                det0 / detA, // c0
                det1 / detA, // c1
                det2 / detA // c2
            };
        }
    };
}
