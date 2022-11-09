
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SCENE_NODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCENE_NODE_INCLUDED_

#include <nbl/builtin/hlsl/math/animations.hlsl>

namespace nbl
{
namespace hlsl
{
namespace scene
{
namespace node
{

void initializeLinearSkin(
	out float4 accVertexPos, out float3 accVertexNormal,
	const float3 inVertexPos, const float3 inVertexNormal,
	const float4x4 boneTransform, const float3x3 boneOrientationInvT, const float boneWeight)
{
	accVertexPos = mul(boneTransform, float4(inVertexPos * boneWeight, boneWeight));
	accVertexNormal = mul(boneOrientationInvT, inVertexNormal * boneWeight);
}



void accumulateLinearSkin(
	inout float4 accVertexPos, inout float3 accVertexNormal,
	const float3 inVertexPos, const float3 inVertexNormal,
	const float4x4 boneTransform, const float3x3 boneOrientationInvT, const float boneWeight)
{
	accVertexPos += mul(boneTransform, float4(inVertexPos * boneWeight, boneWeight));
	accVertexNormal += mul(boneOrientationInvT, inVertexNormal * boneWeight);
}

}
}
}
}

#endif