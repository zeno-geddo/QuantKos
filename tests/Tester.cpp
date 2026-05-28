#include <iostream>
#include <Kokkos_Core.hpp>

#include "./RNGenTest.hpp"
#include "./PutCallParityTest.hpp"
#include "./NoNoiseTest.hpp"

int main(int argc, char *argv[]) {
    // 1. Turn on the hardware ONCE
    Kokkos::initialize(argc, argv);

    int failed_tests = 0;

    // 2. Run the math (The GPU is active and ready for all of them)
    if (!KOps::Tests::RNG::run_test()) {
        failed_tests++;
    }

    if (!KOps::Tests::NoNoise::run_tests()) {
        failed_tests++;
    }

    // 3. Turn off the hardware ONCE
    Kokkos::finalize();

    return failed_tests == 0 ? 0 : 1;
}