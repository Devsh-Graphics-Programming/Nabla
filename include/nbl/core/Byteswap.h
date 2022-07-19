// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_BYTESWAP_H_INCLUDED__
#define __NBL_CORE_BYTESWAP_H_INCLUDED__

#include <stdint.h>
#include <type_traits>

#if defined(_NBL_WINDOWS_API_) && defined(_MSC_VER) && (_MSC_VER > 1298)
#include <stdlib.h>
#define bswap_16(X) _byteswap_ushort(X)
#define bswap_32(X) _byteswap_ulong(X)
#elif defined(_NBL_OSX_PLATFORM_)
#include <libkern/OSByteOrder.h>
#define bswap_16(X) OSReadSwapInt16(&X,0)
#define bswap_32(X) OSReadSwapInt32(&X,0)
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/endian.h>
#define bswap_16(X) bswap16(X)
#define bswap_32(X) bswap32(X)
#else
#define bswap_16(X) ((((X)&0xFF) << 8) | (((X)&0xFF00) >> 8))
#define bswap_32(X) ( (((X)&0x000000FF)<<24) | (((X)&0xFF000000) >> 24) | (((X)&0x0000FF00) << 8) | (((X) &0x00FF0000) >> 8))
#endif

namespace nbl::core
{
	class NBL_API Byteswap
	{
		Byteswap() = delete;
	public:

		static inline int8_t byteswap(const int8_t number)
		{
			return number;
		}

		static inline uint8_t byteswap(const uint8_t number)
		{
			return number;
		}

		static inline uint16_t byteswap(const uint16_t number)
		{
			return bswap_16(number);
		}

		static inline int16_t byteswap(const int16_t number)
		{
			return bswap_16(number);
		}

		static inline uint32_t byteswap(const uint32_t number)
		{
			return bswap_32(number);
		}

		static inline int32_t byteswap(const int32_t number)
		{
			return bswap_32(number);
		}

		static inline float byteswap(const float number)
		{
			uint32_t value = core::IR(number);
			value = bswap_32(value);
			return core::FR(value);
		}
	};
} // end namespace nbl::core

#endif //__NBL_CORE_BYTESWAP_H_INCLUDED__
