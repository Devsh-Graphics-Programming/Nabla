
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCENE_NODE_INCLUDED_
#define _NBL_HLSL_SCENE_NODE_INCLUDED_



#include <nbl/builtin/hlsl/math/animations.hlsl>

namespace nbl
{
	namespace hlsl
	{
		namespace scene
		{
			void Node_initializeLinearSkin(
				out float4 accVertexPos, out float3 accVertexNormal,
				in float3 inVertexPos, in float3 inVertexNormal,
				in float4x4 boneTransform, in float3x3 boneOrientationInvT, in float boneWeight)
			{
				accVertexPos = boneTransform * float4(inVertexPos * boneWeight, boneWeight);
				accVertexNormal = boneOrientationInvT * inVertexNormal * boneWeight;
			}



			void Node_accumulateLinearSkin(
				inout float4 accVertexPos, inout float3 accVertexNormal,
				in float3 inVertexPos, in float3 inVertexNormal,
				in float4x4 boneTransform, in float3X3 boneOrientationInvT, in float boneWeight)
			{
				accVertexPos += boneTransform * float4(inVertexPos * boneWeight, boneWeight);
				accVertexNormal += boneOrientationInvT * inVertexNormal * boneWeight;
			}
		}
	}
}



#endif
