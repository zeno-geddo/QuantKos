// #pragma once
#ifndef KOPTIONS_TYPEDEFS_HPP
#define KOPTIONS_TYPEDEFS_HPP


namespace KOps::Types {

    // Check if CMake defined USE_SINGLE_PRECISION
#ifdef LABES_USE_SINGLE_PRECISION
    using Real = float;
#else
    using Real = double; // Default to double
#endif

    // ... rest of typedefs
}





#endif //KOPTIONS_TYPEDEFS_HPP