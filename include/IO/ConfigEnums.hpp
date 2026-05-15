// File (YAML) → Parser → Validator → Config → Solver
#pragma once

namespace KOps::Implemented {

    enum class MathModel {
        Heston,
        Bates,
    };

    enum class NumScheme {
        Euler,
        Milstein,
        AndersonQE,
    };

    enum class IOFormat {
        BIN,           // Specific binary protocol
        TXT,           // For simple 1D debugging
    };

}
