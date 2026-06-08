// File (YAML) → Parser → Validator → Config → Solver
#pragma once

namespace KOps::Implemented {

    enum class OptType {
        // Standard Options
        European, // The standard vanilla option.
        Asian, // The average rate option.

        // Barrier Option
        BarrierUpAndOut, // Starts active. If the asset price rises above the barrier, the option instantly becomes worthless (dies).
        BarrierDownAndOut, // Starts active. If the asset price drops below the barrier, the option dies.
        BarrierUpAndIn, // Starts dead. It only becomes a valid option if the asset price rises and touches the upper barrier.
        BarrierDownAndIn // Starts dead. It only becomes a valid option if the asset price drops and touches the lower barrier.
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
}
