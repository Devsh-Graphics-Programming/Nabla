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


//[[vk::ext_capability(spv::CapabilityAccelerationStructureKHR)]]
using AccelerationStructureKHR = vk::SpirvOpaqueType<spv::OpTypeAccelerationStructureKHR>;

[[vk::ext_instruction(spv::OpConvertUToAccelerationStructureKHR)]]
AccelerationStructureKHR accelerationStructureKHR(uint64_t u);


//! Ray Query stuff

//[[vk::ext_capability(spv::CapabilityRayQueryKHR)]] https://github.com/microsoft/DirectXShaderCompiler/issues/6958
using RayQueryKHR = vk::SpirvOpaqueType<spv::OpTypeRayQueryKHR>;

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
int rayQueryGetIntersectionTypeKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionInstanceCustomIndexKHR)]]
int rayQueryGetIntersectionInstanceCustomIndexKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionInstanceIdKHR)]]
int rayQueryGetIntersectionInstanceIdKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionGeometryIndexKHR)]]
int rayQueryGetIntersectionGeometryIndexKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionPrimitiveIndexKHR)]]
int rayQueryGetIntersectionPrimitiveIndexKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionBarycentricsKHR)]]
float2 rayQueryGetIntersectionBarycentricsKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

[[vk::ext_capability(spv::CapabilityRayQueryKHR)]]
[[vk::ext_extension("SPV_KHR_ray_query")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionFrontFaceKHR)]]
float2 rayQueryGetIntersectionFrontFaceKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed);

// position fetch for ray tracing uses gl_HitTriangleVertexPositionsEXT -> HitTriangleVertexPositionsKHR decorated OpVariable
[[vk::ext_builtin_input(spv::BuiltInHitTriangleVertexPositionsKHR)]]
static const float32_t3 HitTriangleVertexPositionsKHR[3];

// ray query version
[[vk::ext_capability(spv::CapabilityRayQueryPositionFetchKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing_position_fetch")]]
[[vk::ext_instruction(spv::OpRayQueryGetIntersectionTriangleVertexPositionsKHR)]]
float3 rayQueryGetIntersectionTriangleVertexPositionsKHR([[vk::ext_reference]] RayQueryKHR query, uint32_t committed)[3];

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInLaunchIdKHR)]]
static const uint32_t3 LaunchIdKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInLaunchSizeKHR)]]
static const uint32_t3 LaunchSizeKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInInstanceCustomIndexKHR)]]
static const uint32_t InstanceCustomIndexKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInRayGeometryIndexKHR)]]
static const uint32_t RayGeometryIndexKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInWorldRayOriginKHR)]]
static const float32_t3 WorldRayOriginKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInWorldRayDirectionKHR)]]
static const float32_t3 WorldRayDirectionKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInObjectRayOriginKHR)]]
static const float32_t3 ObjectRayOriginKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInObjectRayDirectionKHR)]]
static const float32_t3 ObjectRayDirectionKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInRayTminKHR)]]
static const float32_t RayTminKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInRayTmaxKHR)]]
static const float32_t RayTmaxKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInObjectToWorldKHR)]]
static const float32_t4x3 ObjectToWorldKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInWorldToObjectKHR)]]
static const float32_t4x3 WorldToObjectKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInHitKindKHR)]]
static const uint32_t HitKindKHR;

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_builtin_input(spv::BuiltInIncomingRayFlagsKHR)]]
static const uint32_t IncomingRayFlagsKHR;

template <typename PayloadT>
[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_instruction(spv::OpTraceRayKHR)]]
void traceRayKHR(AccelerationStructureKHR AS, uint32_t rayFlags, uint32_t cullMask, uint32_t sbtOffset, uint32_t sbtStride, uint32_t missIndex, float32_t3 rayOrigin, float32_t rayTmin, float32_t3 rayDirection, float32_t rayTmax, [[vk::ext_reference]] PayloadT payload);

template <typename PayloadT>
[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_instruction(spv::OpTraceRayKHR)]]
void traceRayKHR(RaytracingAccelerationStructure AS, uint32_t rayFlags, uint32_t cullMask, uint32_t sbtOffset, uint32_t sbtStride, uint32_t missIndex, float32_t3 rayOrigin, float32_t rayTmin, float32_t3 rayDirection, float32_t rayTmax, [[vk::ext_reference]] PayloadT payload);

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_instruction(spv::OpReportIntersectionKHR)]]
bool reportIntersectionKHR(float32_t hit, uint32_t hitKind);

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_instruction(spv::OpIgnoreIntersectionKHR)]]
void ignoreIntersectionKHR();

[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_instruction(spv::OpTerminateRayKHR)]]
void terminateRayKHR();

template <typename T>
[[vk::ext_capability(spv::CapabilityRayTracingKHR)]]
[[vk::ext_extension("SPV_KHR_ray_tracing")]]
[[vk::ext_instruction(spv::OpExecuteCallableKHR)]]
void executeCallable(uint32_t sbtIndex, [[vk::ext_reference]] T payload);


}
}
}

#endif
