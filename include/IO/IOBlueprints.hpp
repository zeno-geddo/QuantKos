#pragma once
#include "./../core/MCMem.hpp"

namespace KOps::IO {

    class WriterBlueprint {
    public:
        // Virtual destructor guarantees derived classes close their files!
        virtual ~WriterBlueprint() = default;

        // The unified methods that all specific writers must implement
        virtual void save_paths_batch_if_needed(int current_batch_size, const Engine::MCBatchMem &BatchMem) = 0;
        virtual void print_planned_outputs_summary() const = 0;
    };

}