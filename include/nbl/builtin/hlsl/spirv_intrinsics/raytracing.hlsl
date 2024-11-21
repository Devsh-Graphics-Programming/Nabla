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

using RayQueryKHR = vk::SpirvOpaqueType<spv::OpTypeRayQueryKHR>;
using AccelerationStructureKHR = vk::SpirvOpaqueType<spv::OpTypeAccelerationStructureKHR>;

// matching Ray Query Committed Intersection Type
static const uint32_t RayQueryCommittedIntersectionNoneKHR = 0;
static const uint32_t RayQueryCommittedIntersectionTriangleKHR = 1;
static const uint32_t RayQueryCommittedIntersectionGeneratedKHR = 2;

// matching Ray Query Candidate Intersection Type
static const uint32_t RayQueryCandidateIntersectionTriangleKHR = 0;
static const uint32_t RayQueryCandidateIntersectionAABBKHR = 1;

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryInitializeKHR)]]
void rayQueryInitializeEXT([[vk::ext_reference]] RayQueryKHR query, [[vk::ext_reference]] AccelerationStructureKHR AS, uint32_t flags, uint32_t cull mask, float3 origin, float32_t tmin, float3 direction, float32_t tmax);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryInitializeKHR)]]
void rayQueryInitializeEXT([[vk::ext_reference]] RayQueryKHR query, [[vk::ext_reference]] RaytracingAccelerationStructure AS, uint32_t flags, uint32_t cull mask, float3 origin, float32_t tmin, float3 direction, float32_t tmax);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryProceedKHR)]]
bool rayQueryProceedEXT([[vk::ext_reference]] RayQueryKHR query);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTypeKHR)]]
int rayQueryGetIntersectionTypeEXT([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionInstanceIdKHR)]]
int rayQueryGetIntersectionInstanceIdEXT([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionPrimitiveIndexKHR)]]
int rayQueryGetIntersectionPrimitiveIndexEXT([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection);

// position fetch for ray tracing uses gl_HitTriangleVertexPositionsEXT -> HitTriangleVertexPositionsKHR decorated OpVariable

[[vk::ext_capability(spv::CapabilityRayQueryPositionFetchKHR)]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)]]   // ray query version
void rayQueryGetIntersectionTriangleVertexPositionsEXT([[vk::ext_reference]] RayQueryKHR query, uint32_t intersection, out float32_t3 pos[3]);

}
}
}

#endif  // _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_RAYTRACING_INCLUDED_