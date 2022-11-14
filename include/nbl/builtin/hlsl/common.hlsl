
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



uint spvPackUnorm4x8(float4 value)
{
    uint4 Packed = uint4(round(saturate(value) * 255.0));
    return Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24);
}

float4 spvUnpackUnorm4x8(uint value)
{
    uint4 Packed = uint4(value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24);
    return float4(Packed) / 255.0;
}

uint spvPackSnorm4x8(float4 value)
{
    int4 Packed = int4(round(clamp(value, -1.0, 1.0) * 127.0)) & 0xff;
    return uint(Packed.x | (Packed.y << 8) | (Packed.z << 16) | (Packed.w << 24));
}

float4 spvUnpackSnorm4x8(uint value)
{
    int SignedValue = int(value);
    int4 Packed = int4(SignedValue << 24, SignedValue << 16, SignedValue << 8, SignedValue) >> 24;
    return clamp(float4(Packed) / 127.0, -1.0, 1.0);
}

uint spvPackUnorm2x16(float2 value)
{
    uint2 Packed = uint2(round(saturate(value) * 65535.0));
    return Packed.x | (Packed.y << 16);
}

float2 spvUnpackUnorm2x16(uint value)
{
    uint2 Packed = uint2(value & 0xffff, value >> 16);
    return float2(Packed) / 65535.0;
}

uint spvPackSnorm2x16(float2 value)
{
    int2 Packed = int2(round(clamp(value, -1.0, 1.0) * 32767.0)) & 0xffff;
    return uint(Packed.x | (Packed.y << 16));
}

float2 spvUnpackSnorm2x16(uint value)
{
    int SignedValue = int(value);
    int2 Packed = int2(SignedValue << 16, SignedValue) >> 16;
    return clamp(float2(Packed) / 32767.0, -1.0, 1.0);
}

}
}

#endif