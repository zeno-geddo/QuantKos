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
#include <string>
#include <unistd.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__) || defined(__MACH__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#else
#include <fstream>
#endif

namespace quantkos::Engine {


    /**
     * @brief Retrieves the amount of physical RAM currently available on the system.
     * * This function queries the operating system to determine how much free
     * physical memory is currently available for use.
     * * @return The available memory measured in Kilobytes (KB).
     * Returns 0 if the system information could not be retrieved.
     */
    inline size_t get_ram_available_memory_from_os_kb() {
#if defined(_WIN32)
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            return (size_t) (memInfo.ullAvailPhys / 1024);
        }
        return 0;

#elif defined(__APPLE__) || defined(__MACH__)
        vm_size_t page_size;
        mach_port_t mach_port = mach_host_self();
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

        if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
            KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t) & vm_stats, &count)) {
            // Free memory is the count of free pages multiplied by page size
            return (size_t) ((uint64_t) vm_stats.free_count * (uint64_t) page_size / 1024);
        }
        return 0;

#else
        // Linux implementation
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 12, "MemAvailable") == 0) {
                // Extracts the number from strings like "MemAvailable:  123456 kB"
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    return std::stoul(line.substr(pos));
                }
            }
        }
        return 0;
#endif
    }

    /**
     * @brief Retrieves the amount of physical RAM currently available on the system.
     * * A convenience wrapper for @ref get_ram_available_memory_from_os_kb() that
     * converts the result into a more readable Megabyte unit.
     * * @return The available memory measured in Megabytes (MB).
     */
    inline size_t get_available_memory_from_os_mb() {
        return get_ram_available_memory_from_os_kb() / 1024;
    }
}
