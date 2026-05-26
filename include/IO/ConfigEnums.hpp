// File (YAML) → Parser → Validator → Config → Solver
#pragma once

namespace KOps::Implemented {

    enum class OptType {
        European,
        Asian
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
