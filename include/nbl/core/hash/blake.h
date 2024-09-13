// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_HASH_BLAKE3_H_INCLUDED_
#define _NBL_CORE_HASH_BLAKE3_H_INCLUDED_


#include "blake3.h"

#include <span>


namespace nbl::core
{
struct blake3_hash_t final
{
	inline bool operator==(const blake3_hash_t&) const = default;

	// could initialize this to a hash of a zero-length array,
	// but that requires a .cpp file and a static
	uint8_t data[BLAKE3_OUT_LEN];
};

const blake3_hash_t INVALID_HASH = {};

class blake3_hasher final
{
		template<typename T, typename Dummy=void>
		struct update_impl
		{
			static inline void __call(blake3_hasher& hasher, const T& input)
			{
				// unfortunately there's no concept like StandardLayout or Aggregate for "just structs/classes of non-pointer types" so need to play it safe
				constexpr bool ForbiddenType = std::is_compound_v<T> || std::is_enum_v<T> || std::is_class_v<T>;
				// use __FUNCTION__ to print something with `T` to the error log
				static_assert(!ForbiddenType, __FUNCTION__ "Hashing Specialization for this Type is not implemented!");
				hasher.update(&input,sizeof(input));
			}
		};

		::blake3_hasher m_state;

	public:
		inline blake3_hasher()
		{
			::blake3_hasher_init(&m_state);
		}

		inline blake3_hasher& update(const void* data, const size_t bytes)
		{
			::blake3_hasher_update(&m_state,data,bytes);
			return *this;
		}

		template<typename T>
		blake3_hasher& operator<<(const T& input)
		{
			update_impl<T>::__call(*this,input);
			return *this;
		}

		explicit inline operator blake3_hash_t() const
		{
			blake3_hash_t retval;
			// the blake3 docs say that the hasher can be finalized multiple times
			::blake3_hasher_finalize(&m_state,retval.data,sizeof(retval));
			return retval;
		}
};

// Useful specializations
template<typename Dummy>
struct blake3_hasher::update_impl<blake3_hash_t,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const blake3_hash_t& input)
	{
		hasher.update(&input,sizeof(input));
	}
};
template<typename T, typename Dummy> requires std::is_enum_v<T>
struct blake3_hasher::update_impl<T,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const T& input)
	{
		hasher.update(&input,sizeof(input));
	}
};
template<typename U, size_t N, typename Dummy>
struct blake3_hasher::update_impl<std::span<U,N>,Dummy>
{
	static inline void __call(blake3_hasher& hasher, const std::span<U,N>& input)
	{
		if constexpr (std::is_fundamental_v<U>)
			hasher.update(input.data(),input.size()*sizeof(U));
		else // Note ideally I'd have some check for a `trivially_serializable` trait or something
		for (const auto& item : input)
			hasher << item;

	}
};
template<typename U, size_t N, typename Dummy>
struct blake3_hasher::update_impl<U[N],Dummy>
{
	static inline void __call(blake3_hasher& hasher, const U input[N])
	{
		update_impl<std::span<U>>::__call(hasher,input);
	}
};
}


namespace std
{
template<>
struct hash<nbl::core::blake3_hash_t>
{
	inline size_t operator()(const nbl::core::blake3_hash_t& blake3) const
	{
		auto* as_p_uint64_t = reinterpret_cast<const size_t*>(blake3.data);
		size_t retval = as_p_uint64_t[0];
		for (auto i=1; i<BLAKE3_OUT_LEN/sizeof(size_t); i++)
			retval ^= as_p_uint64_t[i] + 0x9e3779b97f4a7c15ull + (retval << 6) + (retval >> 2);
		return retval;
	}
};
}

#endif // _NBL_CORE_HASH_BLAKE3_H_INCLUDED_