// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_C_BITFLAG_H_INCLUDED_
#define _NBL_CORE_C_BITFLAG_H_INCLUDED_

#include "BuildConfigOptions.h"

namespace nbl::core
{

template <typename ENUM_TYPE>
struct bitflag final
{
	using UNDERLYING_TYPE = std::underlying_type_t<ENUM_TYPE>;

	static_assert(std::is_enum<ENUM_TYPE>::value);

	ENUM_TYPE value = static_cast<ENUM_TYPE>(0);

	constexpr bitflag() = default;
	constexpr bitflag(const ENUM_TYPE value) : value(value) {}
	template<typename Integer, std::enable_if_t<std::is_integral<Integer>::value, bool> = true, std::enable_if_t<std::is_unsigned<Integer>::value, bool> = true>
	constexpr explicit bitflag(const Integer value) : value(static_cast<ENUM_TYPE>(value)) {}

	inline bitflag<ENUM_TYPE> operator~() { return static_cast<ENUM_TYPE>(~static_cast<UNDERLYING_TYPE>(value)); }
	inline bitflag<ENUM_TYPE> operator|(bitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(static_cast<UNDERLYING_TYPE>(value) | static_cast<UNDERLYING_TYPE>(rhs.value)); }
	inline bitflag<ENUM_TYPE> operator&(bitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(static_cast<UNDERLYING_TYPE>(value) & static_cast<UNDERLYING_TYPE>(rhs.value)); }
	inline bitflag<ENUM_TYPE> operator^(bitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(static_cast<UNDERLYING_TYPE>(value) ^ static_cast<UNDERLYING_TYPE>(rhs.value)); }
	inline bitflag<ENUM_TYPE>& operator|=(bitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(static_cast<UNDERLYING_TYPE>(value) | static_cast<UNDERLYING_TYPE>(rhs.value)); return *this; }
	inline bitflag<ENUM_TYPE>& operator&=(bitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(static_cast<UNDERLYING_TYPE>(value) & static_cast<UNDERLYING_TYPE>(rhs.value)); return *this; }
	inline bitflag<ENUM_TYPE>& operator^=(bitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(static_cast<UNDERLYING_TYPE>(value) ^ static_cast<UNDERLYING_TYPE>(rhs.value)); return *this; }

	inline bool hasFlags(bitflag<ENUM_TYPE> val) const { return (static_cast<UNDERLYING_TYPE>(value) & static_cast<UNDERLYING_TYPE>(val.value)) == static_cast<UNDERLYING_TYPE>(val.value); }
};

}

#endif