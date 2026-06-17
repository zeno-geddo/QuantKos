#include <iostream>
#include <Kokkos_Core.hpp>

#include "./../include/tests/IO/BinaryFormatTest.hpp"
#include "./../include/tests/RNG/RNGenTest.hpp"
#include "./../include/tests/Options/PutCallParityTest.hpp"
#include "./../include/tests/Options/NoNoiseTest.hpp"
#include "./../include/tests/Options/BlackScholesTest.hpp"
#include "./../include/tests/Options/HestonTest.hpp"
#include "./../include/tests/Options/AmericanOption.hpp"



struct TestSuite {
    std::string name;
    std::function<bool()> execute; // Can hold any function matching: bool fn()
};

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    int failed_tests = 0;
    {
        constexpr std::string_view indent{"   "};

        std::cout << "\n\n" << indent << "**************************************************\n";
        std::cout << indent << "          LAUNCHING KOPTIONS TEST ENGINE          \n";
        std::cout << indent << "**************************************************\n\n";

        // Tests to RUN
        namespace KTE = KOps::Tests;
        const std::vector<TestSuite> tests_to_run = {
            //{"Random Number Generator", KTE::RNG::run_test},
            //{"Binary File Format I/O", KTE::IOBIN::run_test},
            //{"Zero-Variance SDE Drift", KTE::NoNoise::run_test},
            //{"Weak Convergence to Black Scholes", KTE::BlackScholes::run_test},
            //{"Weak Convergence to Heston", KTE::Heston::run_test},
            {"American Option LSM", KTE::LSM::run_test}
        };

        // Run Tests
        for (const auto &test: tests_to_run) {
            bool success = test.execute();
            if (!success) {
                failed_tests++;
                std::cerr << indent << "[ FAILURE ] " << test.name << " Suite detected an error!\n\n";
            } else {
                std::cout << indent << "[ SUCCESS ] " << test.name << " Suite passed successfully.\n\n";
            }
        }

        std::cout << indent << "**************************************************\n";
        std::cout << indent << "             EXECUTION RUN COMPLETE               \n";
        if (failed_tests == 0) {
            std::cout << indent << " STATUS   : ALL PASSED OK \n";
        } else {
            std::cout << indent << " STATUS   : FAILED (" << failed_tests << " suite(s) broke constraints)\n";
        }
        std::cout << indent << "**************************************************\n\n" << std::endl;
    }

    Kokkos::finalize();

    return failed_tests == 0 ? 0 : 1;
}
