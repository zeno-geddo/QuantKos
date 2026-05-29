#pragma once
#include "./../core/MCMem.hpp"
#include "./../core/Typedefs.hpp"

namespace KOps::IO {

    namespace KT = KOps::Types;

    class WriterBlueprint {
    public:
        // Virtual destructor guarantees derived classes close their files!
        virtual ~WriterBlueprint() = default;

        // The unified methods that all specific writers must implement
        virtual void save_paths_batch_if_needed(int current_batch_size, const Engine::MCBatchMem &BatchMem) = 0;
        virtual void print_planned_outputs_summary() const = 0;
    };

    class ReaderBlueprint {
    public:
        // Virtual destructor guarantees derived classes close their files!
        virtual ~ReaderBlueprint() = default;

        // The unified methods that all specific writers must implement
        virtual std::vector<KT::Real> read_prices_at_target_time(KT::Real target_time) = 0;
    };


}