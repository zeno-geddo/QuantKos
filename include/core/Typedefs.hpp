// #pragma once
#ifndef KOPTIONS_TYPEDEFS_HPP
#define KOPTIONS_TYPEDEFS_HPP


namespace KOps::Types {
    // Check if CMake defined USE_SINGLE_PRECISION
#ifdef KOPS_ENABLE_SINGLE_PRECISION
    using Real = float;
#else
    using Real = double; // Default to double
#endif

    // Globally accessible compile-time type-safe literals
    constexpr Real real_zero = static_cast<Real>(0.0);
    constexpr Real real_one = static_cast<Real>(1.0);
    constexpr Real real_05 = static_cast<Real>(0.5);
}


#endif //KOPTIONS_TYPEDEFS_HPP
