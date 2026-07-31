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
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <yaml-cpp/yaml.h>

/**
 * @brief Provides a lightweight testing framework for registering and executing test suites.
 */
namespace quantkos::Tests {
    /**
     * @struct TestSuite
     * @brief Represents an individual test case, containing its short description and the logic (i.e. funciton) to execute it.
     */
    struct TestSuite {
        const std::string description; ///< A human-readable description of the test.
        const std::function<bool()> test_function; ///< The function to execute. Must return true on success.

        // Explicit constructor helps the compiler handle the assignment
        /**
         * @brief Constructs a new TestSuite by passing the required arguments.
         * @param desc A short description of what the test does.
         * @param func A function returning a boolean (true if passed).
         */
        TestSuite(std::string desc, std::function<bool()> func)
            : description(std::move(desc)), test_function(std::move(func)) {
        }

        // Default constructor needed for map[] operator
        /**
         * @brief Default constructor. Allows to generate the struct without passing arguments. Required for internal map operations, since the suit must be generated internally.
         */
        TestSuite() = default;
    };

    /**
     * @class TestEngine
     * @brief Manages the registration and execution of multiple TestSuite objects.
     */
    class TestEngine {
    public:
        std::map<std::string, TestSuite> tests_registry; ///< Map of unique test IDs to their respective suites.
        static constexpr std::string_view indent{"   "}; ///< Formatting constant for console output.

        /**
         * @brief Adds a new test to the engine's registry.
         * @param id_test A unique string identifier for the test.
         * @param test_description A short description of the test's purpose.
         * @param test_func The logic (i.e. the function returning a bool) to run for this test.
         */
        void register_test(const std::string &id_test,
                           const std::string &test_description,
                           const std::function<bool()> test_func) {
            // Note : The emplace function is used to construct elements directly in a container
            tests_registry.emplace(id_test, TestSuite(test_description, test_func));
        }

        /**
         * @brief Executes registered tests based on provided configuration in input yalm file or runs all if none provided.
         * @param argc The number of command-line arguments.
         * @param argv Array of command-line arguments. If argv[1] is a YAML file, only tests listed there are run.
         * @return int The total number of failed tests. Returns 0 if all tests passed.
         */
        int run_tests(const int argc, char *argv[]) {
            // Check for help flag
            if (handle_help(argc, argv)) {
                return 0;
            }

            // Determine tests to run
            std::vector<std::string> id_tests_to_run;
            if (argc > 1) {
                try {
                    // Try to see in the yalm configuration file which are the test to run
                    std::cout << "Opening YAML file at: " << argv[1] << std::endl;
                    const YAML::Node tests_config = YAML::LoadFile(argv[1]);
                    for (const auto &node: tests_config["run_tests"]) {
                        id_tests_to_run.push_back(node.as<std::string>());
                    }
                } catch (...) {
                    std::cerr << "Error: Could not parse YAML file or file not found.\n";
                    std::cerr << get_help_message() << std::endl;
                    return 1;
                }
            } else {
                // Run all tests if not yalm file is passed
                for (const auto &[id_test, test_suite]: tests_registry) {
                    id_tests_to_run.push_back(id_test);
                }
            }

            // Execute tests
            int n_tests_failed = 0;
            for (const auto &id_test: id_tests_to_run) {
                const bool test_required = static_cast<bool>(tests_registry.count(id_test));
                // 1 if there is the key, 0 else
                if (test_required) {
                    const auto &target_test_suite = tests_registry.at(id_test);
                    try {
                        const bool test_passed = target_test_suite.test_function();
                        if (test_passed) {
                            std::cout << indent << "[ SUCCESS ] " << target_test_suite.description <<
                                    " Suite passed successfully.\n\n";
                        } else {
                            std::cerr << indent << "[ FAILURE ] " << target_test_suite.description <<
                                    " Suite detected an error!\n\n";
                            n_tests_failed++;
                        }
                    } catch (const std::runtime_error &e) {
                        std::cerr << indent << "[ FATAL RUNTIME ERROR ] " << target_test_suite.description
                                << " Suite made the software crash failed with: \n\n " << e.what() << "\n\n";
                        n_tests_failed++;
                    } catch (...) {
                        std::cerr << indent << "[ FATAL ERROR ] " << target_test_suite.description <<
                                " Suite made the software crash!\n\n";
                        n_tests_failed++;
                    }
                }
            }
            return n_tests_failed;
        }

    private:
        std::string get_help_message() {
            // Start with a short explanation
            std::ostringstream oss;
            oss << "Usage: ./Test [YAML_CONFIG_FILE] [--help]\n\n"
                                    "Options:\n" <<
                                    indent << "--help    Display this help message.\n\n"
                                    "YAML Configuration Format:\n"
                                    "To run specific tests, create a YAML file with a 'run_tests' sequence:\n\n"
                                    "run_tests:\n"
                                    "  - \"test_id_1\"\n"
                                    "  - \"test_id_2\"\n\n"
                                    "If no YAML file is provided, the engine will execute all registered tests.\n"
                                    "List Available tests :\n";
            // Show the id of the implemented tests
            for (const auto& [id, suite] : tests_registry) {
                oss << indent << " - " << id << "  #  " << suite.description + "\n";
            }

            return oss.str();
        }

        /**
         * @brief Checks for --help flag and prints usage documentation.
         * @return true if help was displayed, false otherwise.
        */
        bool handle_help(int argc, char *argv[]) {
            for (int i = 1; i < argc; ++i) {
                if (std::string(argv[i]) == "--help") {
                    std::cout << get_help_message();
                    return true;
                }
            }
            return false;
        }
    };
}
