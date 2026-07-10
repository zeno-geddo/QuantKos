# pragma once

#include <string>
#include "./../core/config/Config.hpp"

namespace KOps::Config {

    /**
     * @brief Static utility class responsible for loading and parsing file-based configurations.
     * * Provides a centralized static entry point to load YAML/JSON options, validate parameter constraints
     * (such as positive volatility values and non-zero path counts), and map them into the engine's
     * structural configuration representation.
     * * @note This class acts as a pure stateless utility. Instantiation is disabled to prevent
     * redundant object allocations.
     */
    class Parser {        /**
         * @brief Parses a YAML configuration file.
         * @param file Path to the .yaml file.
         * @return A fully populated Config object.
         * @throws std::runtime_error if file is missing or invalid.
         */
    public:

        /// @name Lifecycle Safeguards
        ///@{
        /**
         * @brief Deleted constructor to enforce static-only usage.
         * * Prohibits creating instances of this class, keeping the configuration load sequence clean and stateless.
         */
        Parser() = delete; // Prevent anyone from ever creating a 'Parser' object
        ///@}

        /**
         * @brief Parses a YAML configuration file and populates the master input parameters block.
         * * Opens, reads, and parses the specified file, returning a validated, ready-to-use
         * configuration tree for the Monte Carlo orchestrators.
         * * @param file System path to the configuration file (typically a `.yaml` specification).
         * @return A fully populated and validated UInputs config object.
         * @throws std::runtime_error If the configuration file is missing, structurally invalid, or fails semantic checks.
         */
        static UInputs parse(const std::string& file);
    };

} // namespace Labes