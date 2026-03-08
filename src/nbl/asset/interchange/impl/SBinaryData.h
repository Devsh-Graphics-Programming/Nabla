// Internal src-only header. Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_BINARY_DATA_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_BINARY_DATA_H_INCLUDED_
#include <algorithm>
#include <cstdint>
#include <cstring>
namespace nbl::asset::impl
{
//! Binary helpers for endian conversion and unaligned loads/stores.
struct BinaryData
{
	//! Returns `value` with byte order reversed.
	template<typename T>
	static inline T byteswap(const T value) { auto retval = value; const auto* it = reinterpret_cast<const char*>(&value); std::reverse_copy(it, it + sizeof(retval), reinterpret_cast<char*>(&retval)); return retval; }

	//! Loads one trivially copyable value from unaligned memory and optionally byte-swaps it.
	template<typename T>
	static inline T loadUnaligned(const void* src, const bool swapEndian = false)
	{
		T value = {};
		if (!src)
			return value;
		std::memcpy(&value, src, sizeof(value));
		return swapEndian ? byteswap(value) : value;
	}

	//! Stores one trivially copyable value into unaligned memory.
	template<typename T>
	static inline void storeUnaligned(void* dst, const T& value) { std::memcpy(dst, &value, sizeof(value)); }

	//! Stores one value and advances the destination pointer by `sizeof(T)`.
	template<typename T>
	static inline void storeUnalignedAdvance(uint8_t*& dst, const T& value) { storeUnaligned(dst, value); dst += sizeof(value); }
};
}
#endif
