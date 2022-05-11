// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_C_BITFLAG_H_INCLUDED_
#define _NBL_CORE_C_BITFLAG_H_INCLUDED_

namespace nbl::core
{

template <typename ENUM_TYPE>
struct bitflag final
{
	static_assert(std::is_enum<ENUM_TYPE>::value);

	ENUM_TYPE value = static_cast<ENUM_TYPE>(0);

	bitflag() = default;
	bitflag(const ENUM_TYPE value) : value(value) {}
	template<typename Integer, std::enable_if_t<std::is_integral<Integer>::value, bool> = true, std::enable_if_t<std::is_unsigned<Integer>::value, bool> = true>
	explicit bitflag(const Integer value) : value(static_cast<ENUM_TYPE>(value)) {}

	inline bitflag<ENUM_TYPE> operator~() { return static_cast<ENUM_TYPE>(~value); }
	inline bitflag<ENUM_TYPE> operator|(bitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(value | rhs.value); }
	inline bitflag<ENUM_TYPE> operator&(bitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(value & rhs.value); }
	inline bitflag<ENUM_TYPE> operator^(bitflag<ENUM_TYPE> rhs) const { return static_cast<ENUM_TYPE>(value ^ rhs.value); }
	inline bitflag<ENUM_TYPE>& operator|=(bitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(value | rhs.value); return *this; }
	inline bitflag<ENUM_TYPE>& operator&=(bitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(value & rhs.value); return *this; }
	inline bitflag<ENUM_TYPE>& operator^=(bitflag<ENUM_TYPE> rhs) { value = static_cast<ENUM_TYPE>(value ^ rhs.value); return *this; }
    //TODO:Rename bitflag::hasValue to hasFlags
	inline bool hasValue(bitflag<ENUM_TYPE> val) const { return (value & val.value) == val.value; }
};

}

#endif