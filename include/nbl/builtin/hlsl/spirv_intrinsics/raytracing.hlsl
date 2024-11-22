// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace spirv
{

//[[vk::ext_capability(spv::CapabilityRayQueryKHR)]] https://github.com/microsoft/DirectXShaderCompiler/issues/6958
using RayQueryKHR = vk::SpirvOpaqueType<spv::OpTypeRayQueryKHR>;

//[[vk::ext_capability(spv::CapabilityAccelerationStructureKHR)]]
using AccelerationStructureKHR = vk::SpirvOpaqueType<spv::OpTypeAccelerationStructureKHR>;

// matching Ray Query Committed Intersection Type
static const uint32_t RayQueryCommittedIntersectionNoneKHR = 0;
static const uint32_t RayQueryCommittedIntersectionTriangleKHR = 1;
static const uint32_t RayQueryCommittedIntersectionGeneratedKHR = 2;

// matching Ray Query Candidate Intersection Type
static const uint32_t RayQueryCandidateIntersectionTriangleKHR = 0;
static const uint32_t RayQueryCandidateIntersectionAABBKHR = 1;

[[vk::ext_instruction(spv::OpConvertUToAccelerationStructureKHR)]]
AccelerationStructureKHR accelerationStructureKHR(uint64_t u);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryInitializeKHR)]]
void rayQueryInitializeKHR([[vk::ext_reference]] RayQueryKHR query, AccelerationStructureKHR AS, uint32_t flags, uint32_t cullMask, float32_t3 origin, float32_t tmin, float32_t3 direction, float32_t tmax);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryInitializeKHR)]]
void rayQueryInitializeKHR([[vk::ext_reference]] RayQueryKHR query, RaytracingAccelerationStructure AS, uint32_t flags, uint32_t cullMask, float32_t3 origin, float32_t tmin, float32_t3 direction, float32_t tmax);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryProceedKHR)]]
bool rayQueryProceedKHR([[vk::ext_reference]] RayQueryKHR query);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTypeKHR)]]
int rayQueryGetIntersectionTypeKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionInstanceIdKHR)]]
int rayQueryGetIntersectionInstanceIdKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionPrimitiveIndexKHR)]]
int rayQueryGetIntersectionPrimitiveIndexKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionBarycentricsKHR)]]
float2 rayQueryGetIntersectionBarycentricsKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

// position fetch for ray tracing uses gl_HitTriangleVertexPositionsEXT -> HitTriangleVertexPositionsKHR decorated OpVariable
[[vk::ext_builtin_input(spv::BuiltInHitTriangleVertexPositionsKHR)]]
static const float32_t3 HitTriangleVertexPositionsKHR[3];

using __arr3_float3 =  float32_t3[3];

[[vk::ext_capability(spv::CapabilityRayQueryPositionFetchKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing_position_fetch")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)]]   // ray query version
__arr3_float3 rayQueryGetIntersectionTriangleVertexPositionsKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

}
}
}

#endif  // _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_
