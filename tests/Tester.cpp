#include <iostream>
#include <Kokkos_Core.hpp>

#include "./../include/tests/RNGenTest.hpp"
#include "./../include/tests/PutCallParityTest.hpp"
#include "./../include/tests/NoNoiseTest.hpp"

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    int failed_tests = 0;
    {
        namespace KTE = KOps::Tests;

        if (!KTE::RNG::run_test()) {
            failed_tests++;
        }

        if (!KTE::NoNoise::run_test()) {
            failed_tests++;
        }

    }
    Kokkos::finalize();

    return failed_tests == 0 ? 0 : 1;
}
