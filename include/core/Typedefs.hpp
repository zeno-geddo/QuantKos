// Copyright (C) 14/07/2026 Zeno GEDDO <zeno.geddo@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

/**
 * @brief Namespace containing core mathematical type definitions and compile-time constants.
 * * Provides a unified way to manage floating-point precision across the entire QuantKos engine
 * via CMake-defined preprocessor macros.
 */
namespace quantkos::Types {
    /**
     * @brief The primary floating-point type used throughout the numerical simulation engine.
     * * Defaults to @c double precision for maximum accuracy. If the CMake option @c QKOS_ENABLE_SINGLE_PRECISION
     * is enabled during the build process, this alias automatically resolves to @c float to improve
     * memory bandwidth and throughput on GPU architectures.
     * @todo Improve the way single precision is handled
     */
#ifdef QKOS_ENABLE_SINGLE_PRECISION
    using Real = float;
#else
    using Real = double; // Default to double
#endif

    /**
* @brief Helper function to know is using single precision at run time
*/
    inline constexpr bool is_real_using_single_precision() {
#ifdef QKOS_ENABLE_SINGLE_PRECISION
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief Helper function to get a string showing the precision used.
    */
    inline constexpr const char * get_precision_string() {
        return is_real_using_single_precision() ? "float (single precision)": "double (double precision)";
    }


    /** @name Type-Safe Literal Constants
     * Compile-time constants cast to the engine's current @c Real precision.
     * Using these literals ensures consistency and prevents implicit precision
     * loss during mixed-type arithmetic.
     */
    ///@{
    constexpr Real real_zero = static_cast<Real>(0.0); ///< Zero literal constant.
    constexpr Real real_one = static_cast<Real>(1.0); ///< One literal constant.
    constexpr Real real_two = static_cast<Real>(2.0); ///< Two literal constant.
    constexpr Real real_3 = static_cast<Real>(3.0); ///< 3. literal constant.
    constexpr Real real_4 = static_cast<Real>(4.0); ///< 4. literal constant.
    constexpr Real real_05 = static_cast<Real>(0.5); ///< 0.5 literal constant.
    constexpr Real real_025 = static_cast<Real>(0.25); ///< 0.25 literal constant.
    constexpr Real real_1p5 = static_cast<Real>(1.5); ///< 1.5 literal constant.
    ///@}




    /**
     * @brief Checks if Fast Math optimizations are active at compile time.
     */
    inline constexpr bool is_fast_math_enabled() {
#if defined(QKOS_USE_FAST_MATH) || defined(__FAST_MATH__) || defined(_M_FP_FAST)
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief Returns a human-readable string for Fast Math state.
     */
    inline constexpr const char *get_fast_math_string() {
        return is_fast_math_enabled() ? "True (Non-IEEE)" : "False (Strict IEEE)";
    }


        /**
         * @brief Returns the active build type ("Release", "Debug", "RelWithDebInfo").
         */
        inline constexpr const char* get_build_mode_string() {
#if defined(QKOS_BUILD_TYPE)
            return QKOS_BUILD_TYPE;
#elif defined(NDEBUG)
            return "Release";
#else
            return "Debug";
#endif
        }

}
