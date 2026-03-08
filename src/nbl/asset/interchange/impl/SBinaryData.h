// Internal src-only header.
// Do not include from public headers.
#ifndef _NBL_ASSET_IMPL_S_BINARY_DATA_H_INCLUDED_
#define _NBL_ASSET_IMPL_S_BINARY_DATA_H_INCLUDED_
#include <algorithm>
#include <cstdint>
#include <cstring>
namespace nbl::asset::impl
{
struct BinaryData
{
	template<typename T>
	static inline T byteswap(const T value)
	{
		auto retval = value;
		const auto* it = reinterpret_cast<const char*>(&value);
		std::reverse_copy(it, it + sizeof(retval), reinterpret_cast<char*>(&retval));
		return retval;
	}

	template<typename T>
	static inline T loadUnaligned(const void* src, const bool swapEndian = false)
	{
		T value = {};
		if (!src)
			return value;
		std::memcpy(&value, src, sizeof(value));
		return swapEndian ? byteswap(value) : value;
	}

	template<typename T>
	static inline void storeUnaligned(void* dst, const T& value)
	{
		std::memcpy(dst, &value, sizeof(value));
	}

	template<typename T>
	static inline void storeUnalignedAdvance(uint8_t*& dst, const T& value)
	{
		storeUnaligned(dst, value);
		dst += sizeof(value);
	}
};
}
#endif
