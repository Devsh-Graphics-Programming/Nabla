// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_OS_H_INCLUDED__
#define __NBL_OS_H_INCLUDED__

#include "nbl/core/core.h"
#include "nbl/core/math/floatutil.h"

#include "irrString.h"
#include "path.h"
#include "ILogger.h"
#include "ITimer.h"

namespace nbl
{
namespace os
{
class Byteswap
{
    Byteswap() = delete;

public:
#if defined(_NBL_COMPILE_WITH_SDL_DEVICE_)
#include <SDL/SDL_endian.h>
#define bswap_16(X) SDL_Swap16(X)
#define bswap_32(X) SDL_Swap32(X)
#elif defined(_NBL_WINDOWS_API_) && defined(_MSC_VER) && (_MSC_VER > 1298)
#include <stdlib.h>
#define bswap_16(X) _byteswap_ushort(X)
#define bswap_32(X) _byteswap_ulong(X)
#elif defined(_NBL_OSX_PLATFORM_)
#include <libkern/OSByteOrder.h>
#define bswap_16(X) OSReadSwapInt16(&X, 0)
#define bswap_32(X) OSReadSwapInt32(&X, 0)
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/endian.h>
#define bswap_16(X) bswap16(X)
#define bswap_32(X) bswap32(X)
#elif !defined(_NBL_SOLARIS_PLATFORM_) && !defined(__PPC__) && !defined(_NBL_WINDOWS_API_)
#include <byteswap.h>
#else
#define bswap_16(X) ((((X)&0xFF) << 8) | (((X)&0xFF00) >> 8))
#define bswap_32(X) ((((X)&0x000000FF) << 24) | (((X)&0xFF000000) >> 24) | (((X)&0x0000FF00) << 8) | (((X)&0x00FF0000) >> 8))
#endif

    static inline uint16_t byteswap(uint16_t num)
    {
        return bswap_16(num);
    }
    static inline int16_t byteswap(int16_t num) { return bswap_16(num); }
    static inline uint32_t byteswap(uint32_t num) { return bswap_32(num); }
    static inline int32_t byteswap(int32_t num) { return bswap_32(num); }
    static inline float byteswap(float num)
    {
        uint32_t tmp = core::IR(num);
        tmp = bswap_32(tmp);
        return core::FR(tmp);
    }
    // prevent accidental byte swapping of chars
    static inline uint8_t byteswap(uint8_t num) { return num; }
    static inline int8_t byteswap(int8_t num) { return num; }
};

class Printer
{
    Printer() = delete;

public:
    // prints out a string to the console out stdout or debug log or whatever
    static void print(const std::string& message);
    static void log(const std::string& message, ELOG_LEVEL ll = ELL_INFORMATION);
    static void log(const std::wstring& message, ELOG_LEVEL ll = ELL_INFORMATION);
    static void log(const std::string& message, const std::string& hint, ELOG_LEVEL ll = ELL_INFORMATION);

    static ILogger* Logger;
};

}  // end namespace os
}  // end namespace nbl

#endif
