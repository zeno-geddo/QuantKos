// #pragma once
#ifndef KOPTIONS_TYPEDEFS_HPP
#define KOPTIONS_TYPEDEFS_HPP

/**
 * @brief Namespace containing core mathematical type definitions and compile-time constants.
 * * Provides a unified way to manage floating-point precision across the entire KOps engine
 * via CMake-defined preprocessor macros.
 */
namespace KOps::Types {

    /**
     * @brief The primary floating-point type used throughout the numerical simulation engine.
     * * Defaults to @c double precision for maximum accuracy. If the CMake option @c KOPS_ENABLE_SINGLE_PRECISION
     * is enabled during the build process, this alias automatically resolves to @c float to improve
     * memory bandwidth and throughput on GPU architectures.
     */
#ifdef KOPS_ENABLE_SINGLE_PRECISION
    using Real = float;
#else
    using Real = double; // Default to double
#endif

    /** @name Type-Safe Literal Constants
     * Compile-time constants cast to the engine's current @c Real precision.
     * Using these literals ensures consistency and prevents implicit precision
     * loss during mixed-type arithmetic.
     */
    ///@{
    constexpr Real real_zero = static_cast<Real>(0.0); ///< Zero literal constant.
    constexpr Real real_one = static_cast<Real>(1.0); ///< One literal constant.
    constexpr Real real_two = static_cast<Real>(2.0); ///< Two literal constant.
    constexpr Real real_05 = static_cast<Real>(0.5); ///< 0.5 literal constant.
    constexpr Real real_025 = static_cast<Real>(0.25); ///< 0.25 literal constant.
    constexpr Real real_1p5 = static_cast<Real>(1.5); ///< 1.5 literal constant.
    ///@}
}


#endif //KOPTIONS_TYPEDEFS_HPP
