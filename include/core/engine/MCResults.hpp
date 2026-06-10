
#include "../config/Config.hpp"
#include "../options/OptionPricer.hpp"

namespace KOps::Engine {
    struct MCResults {
        const Config::UInputs &MCConfig; // Just give the address, it will be valid since it lives in the main
        const OptionPricer::MCResults OptionPrice; // Copy the structure so that it is ok when simulation scope end
    };
}
