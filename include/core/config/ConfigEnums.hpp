// File (YAML) → Parser → Validator → Config → Solver
#pragma once

namespace KOps::Implemented {

    enum class OptType {
        // ------------------------------------------------
        // Standard Options (Terminal & Averages)
        // ------------------------------------------------
        European, // The standard vanilla option.
        Asian, // The average rate option.

        // ------------------------------------------------
        // Barrier Option (Conditional Survival/Activation)
        // ------------------------------------------------
        BarrierUpAndOut, // Starts active. If the asset price rises above the barrier, the option instantly becomes worthless (dies).
        BarrierDownAndOut, // Starts active. If the asset price drops below the barrier, the option dies.
        BarrierUpAndIn, // Starts dead. It only becomes a valid option if the asset price rises and touches the upper barrier.
        BarrierDownAndIn, // Starts dead. It only becomes a valid option if the asset price drops and touches the lower barrier.

        // ------------------------------------------------
        // Lookback Options (Extrema Tracking)
        // ------------------------------------------------
        LookbackFloatingStrike, // Strike floats to the absolute min (Call) or max (Put)
        LookbackFixedStrike,    // Strike is fixed, but payoff uses the absolute max (Call) or min (Put)


        // ------------------------------------------------
        // Binary Options ( options
        // ------------------------------------------------
        BinaryCashOrNothing, // Pays a fixed cash amount if In-The-Money (ITM) // ? should add such amount as input ?
        BinaryAssetOrNothing, // Pays the terminal asset price if ITM

        // ------------------------------------------------
        // Backwards path dependent options
        // ------------------------------------------------
        American,
    };

    enum class OptRight {
        Call,
        Put
    };

    enum class MathModel {
        Heston,
        Bates,
    };

    enum class NumScheme {
        // Should add exact?
        Euler,
        Milstein,
        AndersonQE,
    };

    enum class IOFormat {
        BIN, // Specific binary protocol
        TXT, // For simple 1D debugging
    };

    enum class LSRegressionBasis {
        MonomialO2,
        LaguerreO3
    };
}
