#ifndef _NBL_S_BUILTIN_FILE_H_INCLUDED_
#define _NBL_S_BUILTIN_FILE_H_INCLUDED_

#include <cstdint>
#include <chrono>
#include <array>

namespace nbl::system
{
	struct SBuiltinFile
    {
        const uint8_t* contents = nullptr;
        size_t size = 0u;
        std::array<uint64_t, 4> xx256Hash;
        std::tm modified = { // Absolute time since 1970 in UTC+0
            .tm_sec = 6,
            .tm_min = 9,
            .tm_hour = 6,
            .tm_mday = 9,
            .tm_mon = 6,
            .tm_year = 9,
            .tm_isdst = 0
        };
    };
}

#endif // _NBL_S_BUILTIN_FILE_H_INCLUDED_