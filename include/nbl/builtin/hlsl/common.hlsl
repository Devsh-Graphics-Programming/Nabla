
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMMON_INCLUDED_

namespace nbl
{
namespace hlsl
{


bool3 greaterThanEqual(float3 x, float3 y)
{
	return select(x>=y, bool3(true, true, true), bool3(false, false, false));
}
bool3 lessThanEqual(float3 x, float3 y)
{
	return select(x<=y, bool3(true, true, true), bool3(false, false, false));
}
bool3 greaterThan(float3 x, float3 y)
{
	return select(x>y, bool3(true, true, true), bool3(false, false, false));
}
bool3 lessThan(float3 x, float3 y)
{
	return select(x<y, bool3(true, true, true), bool3(false, false, false));
}

uint bitfieldExtract(uint value, int offset, int bits)
{
	uint mask = uint((1 << bits) - 1);
	return uint(value >> offset) & mask;
}
uint bitfieldInsert(uint value, int insert, int offset, int bits)
{
	uint mask = ~(0xffffffff << bits) << offset;
	mask = ~mask;
	value &= mask;
	return value | (insert << offset);
}


}
}

#endif