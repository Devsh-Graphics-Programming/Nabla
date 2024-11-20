// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_

// #include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace spirv
{

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryInitializeKHR)]]
void rayQueryInitializeEXT(/* spv::OpTypeRayQueryKHR function_ptr query, spv::OpTypeAccelerationStructureKHR AS, uint32_t flags, uint32_t cull mask, float3 origin, float32_t tmin, float3 direction, float32_t tmax*/);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryProceedKHR)]]
bool rayQueryProceedEXT(/* spv::OpTypeRayQueryKHR function_ptr query */);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTypeKHR)]]
int rayQueryGetIntersectionTypeEXT(/* spv::OpTypeRayQueryKHR function_ptr query, uint32_t intersection (candidate 0, commited 1) */);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionInstanceIdKHR)]]
int rayQueryGetIntersectionInstanceIdEXT(/* spv::OpTypeRayQueryKHR function_ptr query, uint32_t intersection (candidate 0, commited 1) */);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionPrimitiveIndexKHR)]]
int rayQueryGetIntersectionPrimitiveIndexEXT(/* spv::OpTypeRayQueryKHR function_ptr query, uint32_t intersection (candidate 0, commited 1) */);

// position fetch for ray tracing uses gl_HitTriangleVertexPositionsEXT -> HitTriangleVertexPositionsKHR decorated OpVariable

[[vk::ext_capability(spv::CapabilityRayQueryPositionFetchKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)]]   // ray query version
// get intersection triangle vertex position - gl_HitTriangleVertexPositionsEXT
// returns 3-array float3 vertex positions
void rayQueryGetIntersectionTriangleVertexPositionsEXT(/* spv::OpTypeRayQueryKHR function_ptr query, uint32_t intersection (candidate 0, commited 1), out float3 positions[3] */);

}
}
}

#endif  // _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_