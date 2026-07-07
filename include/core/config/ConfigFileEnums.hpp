// File (YAML) → Parser → Validator → Config → Solver
#pragma once


namespace KOps::Implemented {
    /**
    * @brief Defines the supported option exercise styles and payoff structures.
    */
    enum class OptType {
        // ------------------------------------------------
        // Standard Options (Terminal & Averages)
        // ------------------------------------------------
        /** @name Standard Vanilla Options */
        ///@{
        European, ///< Standard vanilla option exercisable only at maturity.
        Asian, ///< Payoff depends on the average price of the underlying.
        ///@}

        /** @name Barrier Options (Conditional Survival/Activation)*/
        ///@{
        BarrierUpAndOut, ///< Option becomes worthless if the asset rises above the barrier.
        BarrierDownAndOut, ///< Option becomes worthless if the asset falls below the barrier.
        BarrierUpAndIn, ///< Option activates only if the asset rises to touch the barrier.
        BarrierDownAndIn, ///< Option activates only if the asset falls to touch the barrier.
        ///@}

        /** @name Lookback Options */
        ///@{
        LookbackFloatingStrike, ///< Strike price is set to the historical minimum/maximum of the asset.
        LookbackFixedStrike, ///< Strike is fixed, but payoff uses the absolute max (Call) or min (Put)
        ///@}

        /** @name Binary Options */
        ///@{
        BinaryCashOrNothing, ///< Pays a fixed cash amount if the option finishes in-the-money.
        BinaryAssetOrNothing, ///< Pays the value of the terminal asset price if in-the-money.
        ///@}

        /** @name Early Exercise Options */
        ///@{
        American,
        ///@}
    };

    /**
    * @brief Represents the exercise right of the option contract.
    */
    enum class OptRight {
        /** @brief The right to purchase the underlying asset. */
        Call,
        /** @brief The right to sell the underlying asset. */
        Put
    };

    /** @brief Supported stochastic math models for asset dynamics.
     */
    enum class MathModel {
        Heston,
        Bates,
    };

    /** @brief Numerical discretization schemes for SDE paths. */
    enum class NumScheme {
        Euler, ///< Euler-Maruyama discretization.
        Milstein, ///< First-order Implicit Milstein scheme.
        AndersonQE, ///< Quadratic Exponential (QE) scheme for Heston.
    };

    /** @brief Supported output data formats. */
    enum class IOFormat {
        BIN, ///< High-performance custom binary protocol.
        TXT, ///< Human-readable text format for debugging/plotting.
    };

    /** @brief Supported basis for Longstaff-Schwarz Algorithm (for American Options). */
    enum class LSRegressionBasis {
        LaguerreP02, ///< Order 2 Laguerre polynomial.
        LaguerreP03, ///< Order 3 Laguerre polynomial.
        LaguerreP04, ///< Order 4 Laguerre polynomial.
    };
}
