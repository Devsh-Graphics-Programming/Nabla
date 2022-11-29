
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_UTILS_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_COMMON_INCLUDED_


namespace nbl
{
namespace hlsl
{


struct SBasicViewParameters
{
    float4x4 MVP;
    float4x3 MV;
    float4x3 NormalMatAndEyePos;

	float3x3 GetNormalMat(in float4x3 _NormalMatAndEyePos)
	{
	    return (float3x3)(_NormalMatAndEyePos);
	}
	float3 GetEyePos(in float4x3 _NormalMatAndEyePos)
	{
	    return _NormalMatAndEyePos[3];
	}
};


}
}



#endif