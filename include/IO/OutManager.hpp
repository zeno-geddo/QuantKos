#pragma once

#include <fstream>
#include <string>
#include <stdexcept>
#include <Kokkos_Core.hpp>

#include "./../IO/Config.hpp"

namespace KOps::Out {
    namespace KC = KOps::Config;
    class OutputManager {

    public:
        explicit OutputManager(const KC::OutputConfig& out_conf) : config(out_conf) {}

    private:
        const KC::OutputConfig config;

    };

}