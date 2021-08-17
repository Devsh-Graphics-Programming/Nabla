// Copyright (C) 2018-2021 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_BYTESWAP_H_INCLUDED__
#define __NBL_CORE_BYTESWAP_H_INCLUDED__

#include <stdint.h>
#include <type_traits>

namespace nbl::core
{
	class Byteswap
	{
		Byteswap() = delete;
	public:

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

		template<typename NUM_TYPE>
		static inline NUM_TYPE byteswap(NUM_TYPE number)
		{
			if constexpr (std::is_same<NUM_TYPE, int8_t>::value)
				return number;
			else if (std::is_same<NUM_TYPE, uint8_t>::value)
				return number;
			else if (std::is_same<NUM_TYPE, uint16_t>::value)
				return bswap_16(number);
			else if (std::is_same<NUM_TYPE, int16_t>::value)
				return bswap_16(number);
			else if (std::is_same<NUM_TYPE, uint32_t>::value)
				return bswap_32(number);
			else if (std::is_same<NUM_TYPE, int32_t>::value)
				return bswap_32(number);
			else if (std::is_same<NUM_TYPE, float>::value)
			{
				uint32_t value = core::IR(number);
				value = bswap_32(value); 
				return core::FR(value);
			}
			else
				static_assert(false, "there are not another supported types!");
		}
	};
} // end namespace nbl::core

#endif //__NBL_CORE_BYTESWAP_H_INCLUDED__
