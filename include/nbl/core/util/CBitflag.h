// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_C_BITFLAG_H_INCLUDED__
#define __NBL_CORE_C_BITFLAG_H_INCLUDED__

namespace nbl
{
namespace core
{

template <typename ENUM_TYPE>
struct CBitflag final
{
	static_assert(std::is_enum<ENUM_TYPE>::value);

	ENUM_TYPE value;

	CBitflag() = default;
	CBitflag(ENUM_TYPE value) : value(value) {}

	inline CBitflag<ENUM_TYPE> operator~() { return static_cast<ENUM_TYPE>(~value); }
	inline CBitflag<ENUM_TYPE> operator|(CBitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(value | rhs.value); }
	inline CBitflag<ENUM_TYPE> operator&(CBitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(value & rhs.value); }
	inline CBitflag<ENUM_TYPE> operator^(CBitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(value ^ rhs.value); }
	inline CBitflag<ENUM_TYPE>& operator|=(CBitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(value | rhs.value); return *this; }
	inline CBitflag<ENUM_TYPE>& operator&=(CBitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(value | rhs.value); return *this; }
	inline CBitflag<ENUM_TYPE>& operator^=(CBitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(value | rhs.value); return *this; }
};

}
}

#endif