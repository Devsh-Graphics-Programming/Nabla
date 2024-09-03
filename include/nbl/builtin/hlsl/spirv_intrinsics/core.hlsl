// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_

#ifdef __HLSL_VERSION
#include "spirv/unified1/spirv.hpp"
#include "spirv/unified1/GLSL.std.450.h"
#endif

#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl 
{
namespace hlsl
{
#ifdef __HLSL_VERSION
namespace spirv
{

//! General Decls
template<uint32_t StorageClass, typename T>
using pointer_t = vk::SpirvOpaqueType<spv::OpTypePointer, vk::Literal< vk::integral_constant<uint32_t, StorageClass> >, T>;

// The holy operation that makes addrof possible
template<uint32_t StorageClass, typename T>
[[vk::ext_instruction(spv::OpCopyObject)]]
pointer_t<StorageClass, T> copyObject([[vk::ext_reference]] T value);

//! Std 450 Extended set operations
template<typename SquareMatrix>
[[vk::ext_instruction(GLSLstd450MatrixInverse)]]
SquareMatrix matrixInverse(NBL_CONST_REF_ARG(SquareMatrix) mat);

// Add specializations if you need to emit a `ext_capability` (this means that the instruction needs to forward through an `impl::` struct and so on)
template<typename T, typename U>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
enable_if_t<is_spirv_type_v<T> && is_spirv_type_v<U>, T> bitcast(U);

template<typename T>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
uint64_t bitcast(pointer_t<spv::StorageClassPhysicalStorageBuffer,T>);

template<typename T>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
pointer_t<spv::StorageClassPhysicalStorageBuffer,T> bitcast(uint64_t);

template<class T, class U>
[[vk::ext_instruction(spv::OpBitcast)]]
T bitcast(U);

//! Builtins
namespace builtin
{[[vk::ext_builtin_output(spv::BuiltInPosition)]]
static float32_t4 Position;
[[vk::ext_builtin_input(spv::BuiltInHelperInvocation)]]
static const bool HelperInvocation;
[[vk::ext_builtin_input(spv::BuiltInNumWorkgroups)]]
static const uint32_t3 NumWorkgroups;
[[vk::ext_builtin_input(spv::BuiltInWorkgroupId)]]
static const uint32_t3 WorkgroupId;
[[vk::ext_builtin_input(spv::BuiltInLocalInvocationId)]]
static const uint32_t3 LocalInvocationId;
[[vk::ext_builtin_input(spv::BuiltInGlobalInvocationId)]]
static const uint32_t3 GlobalInvocationId;
[[vk::ext_builtin_input(spv::BuiltInLocalInvocationIndex)]]
static const uint32_t LocalInvocationIndex;
[[vk::ext_builtin_input(spv::BuiltInSubgroupSize)]]
static const uint32_t SubgroupSize;
[[vk::ext_builtin_input(spv::BuiltInNumSubgroups)]]
static const uint32_t NumSubgroups;
[[vk::ext_builtin_input(spv::BuiltInSubgroupId)]]
static const uint32_t SubgroupId;
[[vk::ext_builtin_input(spv::BuiltInSubgroupLocalInvocationId)]]
static const uint32_t SubgroupLocalInvocationId;
[[vk::ext_builtin_input(spv::BuiltInVertexIndex)]]
static const uint32_t VertexIndex;
[[vk::ext_builtin_input(spv::BuiltInInstanceIndex)]]
static const uint32_t InstanceIndex;
[[vk::ext_builtin_input(spv::BuiltInSubgroupEqMask)]]
static const uint32_t4 SubgroupEqMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupGeMask)]]
static const uint32_t4 SubgroupGeMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupGtMask)]]
static const uint32_t4 SubgroupGtMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupLeMask)]]
static const uint32_t4 SubgroupLeMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupLtMask)]]
static const uint32_t4 SubgroupLtMask;
}

//! Execution Modes
namespace execution_mode
{
	void invocations()
	{
		vk::ext_execution_mode(spv::ExecutionModeInvocations);
	}

	void spacingEqual()
	{
		vk::ext_execution_mode(spv::ExecutionModeSpacingEqual);
	}

	void spacingFractionalEven()
	{
		vk::ext_execution_mode(spv::ExecutionModeSpacingFractionalEven);
	}

	void spacingFractionalOdd()
	{
		vk::ext_execution_mode(spv::ExecutionModeSpacingFractionalOdd);
	}

	void vertexOrderCw()
	{
		vk::ext_execution_mode(spv::ExecutionModeVertexOrderCw);
	}

	void vertexOrderCcw()
	{
		vk::ext_execution_mode(spv::ExecutionModeVertexOrderCcw);
	}

	void pixelCenterInteger()
	{
		vk::ext_execution_mode(spv::ExecutionModePixelCenterInteger);
	}

	void originUpperLeft()
	{
		vk::ext_execution_mode(spv::ExecutionModeOriginUpperLeft);
	}

	void originLowerLeft()
	{
		vk::ext_execution_mode(spv::ExecutionModeOriginLowerLeft);
	}

	void earlyFragmentTests()
	{
		vk::ext_execution_mode(spv::ExecutionModeEarlyFragmentTests);
	}

	void pointMode()
	{
		vk::ext_execution_mode(spv::ExecutionModePointMode);
	}

	void xfb()
	{
		vk::ext_execution_mode(spv::ExecutionModeXfb);
	}

	void depthReplacing()
	{
		vk::ext_execution_mode(spv::ExecutionModeDepthReplacing);
	}

	void depthGreater()
	{
		vk::ext_execution_mode(spv::ExecutionModeDepthGreater);
	}

	void depthLess()
	{
		vk::ext_execution_mode(spv::ExecutionModeDepthLess);
	}

	void depthUnchanged()
	{
		vk::ext_execution_mode(spv::ExecutionModeDepthUnchanged);
	}

	void localSize()
	{
		vk::ext_execution_mode(spv::ExecutionModeLocalSize);
	}

	void localSizeHint()
	{
		vk::ext_execution_mode(spv::ExecutionModeLocalSizeHint);
	}

	void inputPoints()
	{
		vk::ext_execution_mode(spv::ExecutionModeInputPoints);
	}

	void inputLines()
	{
		vk::ext_execution_mode(spv::ExecutionModeInputLines);
	}

	void inputLinesAdjacency()
	{
		vk::ext_execution_mode(spv::ExecutionModeInputLinesAdjacency);
	}

	void triangles()
	{
		vk::ext_execution_mode(spv::ExecutionModeTriangles);
	}

	void inputTrianglesAdjacency()
	{
		vk::ext_execution_mode(spv::ExecutionModeInputTrianglesAdjacency);
	}

	void quads()
	{
		vk::ext_execution_mode(spv::ExecutionModeQuads);
	}

	void isolines()
	{
		vk::ext_execution_mode(spv::ExecutionModeIsolines);
	}

	void outputVertices()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputVertices);
	}

	void outputPoints()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputPoints);
	}

	void outputLineStrip()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputLineStrip);
	}

	void outputTriangleStrip()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputTriangleStrip);
	}

	void vecTypeHint()
	{
		vk::ext_execution_mode(spv::ExecutionModeVecTypeHint);
	}

	void contractionOff()
	{
		vk::ext_execution_mode(spv::ExecutionModeContractionOff);
	}

	void initializer()
	{
		vk::ext_execution_mode(spv::ExecutionModeInitializer);
	}

	void finalizer()
	{
		vk::ext_execution_mode(spv::ExecutionModeFinalizer);
	}

	void subgroupSize()
	{
		vk::ext_execution_mode(spv::ExecutionModeSubgroupSize);
	}

	void subgroupsPerWorkgroup()
	{
		vk::ext_execution_mode(spv::ExecutionModeSubgroupsPerWorkgroup);
	}

	void subgroupsPerWorkgroupId()
	{
		vk::ext_execution_mode(spv::ExecutionModeSubgroupsPerWorkgroupId);
	}

	void localSizeId()
	{
		vk::ext_execution_mode(spv::ExecutionModeLocalSizeId);
	}

	void localSizeHintId()
	{
		vk::ext_execution_mode(spv::ExecutionModeLocalSizeHintId);
	}

	void nonCoherentColorAttachmentReadEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeNonCoherentColorAttachmentReadEXT);
	}

	void nonCoherentDepthAttachmentReadEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeNonCoherentDepthAttachmentReadEXT);
	}

	void nonCoherentStencilAttachmentReadEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeNonCoherentStencilAttachmentReadEXT);
	}

	void subgroupUniformControlFlowKHR()
	{
		vk::ext_execution_mode(spv::ExecutionModeSubgroupUniformControlFlowKHR);
	}

	void postDepthCoverage()
	{
		vk::ext_execution_mode(spv::ExecutionModePostDepthCoverage);
	}

	void denormPreserve()
	{
		vk::ext_execution_mode(spv::ExecutionModeDenormPreserve);
	}

	void denormFlushToZero()
	{
		vk::ext_execution_mode(spv::ExecutionModeDenormFlushToZero);
	}

	void signedZeroInfNanPreserve()
	{
		vk::ext_execution_mode(spv::ExecutionModeSignedZeroInfNanPreserve);
	}

	void roundingModeRTE()
	{
		vk::ext_execution_mode(spv::ExecutionModeRoundingModeRTE);
	}

	void roundingModeRTZ()
	{
		vk::ext_execution_mode(spv::ExecutionModeRoundingModeRTZ);
	}

	void earlyAndLateFragmentTestsAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeEarlyAndLateFragmentTestsAMD);
	}

	void stencilRefReplacingEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefReplacingEXT);
	}

	void coalescingAMDX()
	{
		vk::ext_execution_mode(spv::ExecutionModeCoalescingAMDX);
	}

	void maxNodeRecursionAMDX()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaxNodeRecursionAMDX);
	}

	void staticNumWorkgroupsAMDX()
	{
		vk::ext_execution_mode(spv::ExecutionModeStaticNumWorkgroupsAMDX);
	}

	void shaderIndexAMDX()
	{
		vk::ext_execution_mode(spv::ExecutionModeShaderIndexAMDX);
	}

	void maxNumWorkgroupsAMDX()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaxNumWorkgroupsAMDX);
	}

	void stencilRefUnchangedFrontAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefUnchangedFrontAMD);
	}

	void stencilRefGreaterFrontAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefGreaterFrontAMD);
	}

	void stencilRefLessFrontAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefLessFrontAMD);
	}

	void stencilRefUnchangedBackAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefUnchangedBackAMD);
	}

	void stencilRefGreaterBackAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefGreaterBackAMD);
	}

	void stencilRefLessBackAMD()
	{
		vk::ext_execution_mode(spv::ExecutionModeStencilRefLessBackAMD);
	}

	void quadDerivativesKHR()
	{
		vk::ext_execution_mode(spv::ExecutionModeQuadDerivativesKHR);
	}

	void requireFullQuadsKHR()
	{
		vk::ext_execution_mode(spv::ExecutionModeRequireFullQuadsKHR);
	}

	void outputLinesEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputLinesEXT);
	}

	void outputLinesNV()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputLinesNV);
	}

	void outputPrimitivesEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputPrimitivesEXT);
	}

	void outputPrimitivesNV()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputPrimitivesNV);
	}

	void derivativeGroupQuadsNV()
	{
		vk::ext_execution_mode(spv::ExecutionModeDerivativeGroupQuadsNV);
	}

	void derivativeGroupLinearNV()
	{
		vk::ext_execution_mode(spv::ExecutionModeDerivativeGroupLinearNV);
	}

	void outputTrianglesEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputTrianglesEXT);
	}

	void outputTrianglesNV()
	{
		vk::ext_execution_mode(spv::ExecutionModeOutputTrianglesNV);
	}

	void pixelInterlockOrderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModePixelInterlockOrderedEXT);
	}

	void pixelInterlockUnorderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModePixelInterlockUnorderedEXT);
	}

	void sampleInterlockOrderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeSampleInterlockOrderedEXT);
	}

	void sampleInterlockUnorderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeSampleInterlockUnorderedEXT);
	}

	void shadingRateInterlockOrderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeShadingRateInterlockOrderedEXT);
	}

	void shadingRateInterlockUnorderedEXT()
	{
		vk::ext_execution_mode(spv::ExecutionModeShadingRateInterlockUnorderedEXT);
	}

	void sharedLocalMemorySizeINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeSharedLocalMemorySizeINTEL);
	}

	void roundingModeRTPINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeRoundingModeRTPINTEL);
	}

	void roundingModeRTNINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeRoundingModeRTNINTEL);
	}

	void floatingPointModeALTINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeFloatingPointModeALTINTEL);
	}

	void floatingPointModeIEEEINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeFloatingPointModeIEEEINTEL);
	}

	void maxWorkgroupSizeINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaxWorkgroupSizeINTEL);
	}

	void maxWorkDimINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaxWorkDimINTEL);
	}

	void noGlobalOffsetINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeNoGlobalOffsetINTEL);
	}

	void numSIMDWorkitemsINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeNumSIMDWorkitemsINTEL);
	}

	void schedulerTargetFmaxMhzINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeSchedulerTargetFmaxMhzINTEL);
	}

	void maximallyReconvergesKHR()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaximallyReconvergesKHR);
	}

	void fPFastMathDefault()
	{
		vk::ext_execution_mode(spv::ExecutionModeFPFastMathDefault);
	}

	void streamingInterfaceINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeStreamingInterfaceINTEL);
	}

	void registerMapInterfaceINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeRegisterMapInterfaceINTEL);
	}

	void namedBarrierCountINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeNamedBarrierCountINTEL);
	}

	void maximumRegistersINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaximumRegistersINTEL);
	}

	void maximumRegistersIdINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeMaximumRegistersIdINTEL);
	}

	void namedMaximumRegistersINTEL()
	{
		vk::ext_execution_mode(spv::ExecutionModeNamedMaximumRegistersINTEL);
	}
}

//! Group Operations
namespace group_operation
{
	static const uint32_t Reduce = 0;
	static const uint32_t InclusiveScan = 1;
	static const uint32_t ExclusiveScan = 2;
	static const uint32_t ClusteredReduce = 3;
	static const uint32_t PartitionedReduceNV = 6;
	static const uint32_t PartitionedInclusiveScanNV = 7;
	static const uint32_t PartitionedExclusiveScanNV = 8;
}

//! Instructions
template<typename T, typename P>
[[vk::ext_instruction(spv::OpLoad)]]
enable_if_t<is_spirv_type_v<P>, T> load(P pointer, [[vk::ext_literal]] uint32_t memoryAccess);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpLoad)]]
enable_if_t<is_spirv_type_v<P>, T> load(P pointer, [[vk::ext_literal]] uint32_t memoryAccess, [[vk::ext_literal]] uint32_t memoryAccessParam);

template<typename T, typename P, uint32_t alignment>
[[vk::ext_instruction(spv::OpLoad)]]
enable_if_t<is_spirv_type_v<P>, T> load(P pointer, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpLoad)]]
enable_if_t<is_spirv_type_v<P>, T> load(P pointer);

template<typename T>
[[vk::ext_instruction(spv::OpLoad)]]
T load(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, [[vk::ext_literal]] uint32_t memoryAccess);

template<typename T>
[[vk::ext_instruction(spv::OpLoad)]]
T load(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, [[vk::ext_literal]] uint32_t memoryAccess, [[vk::ext_literal]] uint32_t memoryAccessParam);

template<typename T, uint32_t alignment>
[[vk::ext_instruction(spv::OpLoad)]]
T load(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T>
[[vk::ext_instruction(spv::OpLoad)]]
T load(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpStore)]]
enable_if_t<is_spirv_type_v<P>, void> store(P pointer, T object, [[vk::ext_literal]] uint32_t memoryAccess);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpStore)]]
enable_if_t<is_spirv_type_v<P>, void> store(P pointer, T object, [[vk::ext_literal]] uint32_t memoryAccess, [[vk::ext_literal]] uint32_t memoryAccessParam);

template<typename T, typename P, uint32_t alignment>
[[vk::ext_instruction(spv::OpStore)]]
enable_if_t<is_spirv_type_v<P>, void> store(P pointer, T object, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpStore)]]
enable_if_t<is_spirv_type_v<P>, void> store(P pointer, T object);

template<typename T>
[[vk::ext_instruction(spv::OpStore)]]
void store(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, T object, [[vk::ext_literal]] uint32_t memoryAccess);

template<typename T>
[[vk::ext_instruction(spv::OpStore)]]
void store(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, T object, [[vk::ext_literal]] uint32_t memoryAccess, [[vk::ext_literal]] uint32_t memoryAccessParam);

template<typename T, uint32_t alignment>
[[vk::ext_instruction(spv::OpStore)]]
void store(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, T object, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T>
[[vk::ext_instruction(spv::OpStore)]]
void store(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer, T object);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpGenericPtrMemSemantics)]]
enable_if_t<is_spirv_type_v<P>, T> genericPtrMemSemantics(P pointer);

template<typename T>
[[vk::ext_instruction(spv::OpGenericPtrMemSemantics)]]
T genericPtrMemSemantics(pointer_t<spv::StorageClassPhysicalStorageBuffer, T> pointer);

template<typename T>
[[vk::ext_capability(spv::CapabilityBitInstructions)]]
[[vk::ext_instruction(spv::OpBitFieldInsert)]]
T bitFieldInsert(T base, T insert, uint32_t offset, uint32_t count);

[[vk::ext_capability(spv::CapabilityBitInstructions)]]
[[vk::ext_instruction(spv::OpBitFieldSExtract)]]
int32_t bitFieldExtract(int32_t base, uint32_t offset, uint32_t count);

[[vk::ext_capability(spv::CapabilityBitInstructions)]]
[[vk::ext_instruction(spv::OpBitFieldSExtract)]]
int64_t bitFieldExtract(int64_t base, uint32_t offset, uint32_t count);

[[vk::ext_capability(spv::CapabilityBitInstructions)]]
[[vk::ext_instruction(spv::OpBitFieldUExtract)]]
uint32_t bitFieldExtract(uint32_t base, uint32_t offset, uint32_t count);

[[vk::ext_capability(spv::CapabilityBitInstructions)]]
[[vk::ext_instruction(spv::OpBitFieldUExtract)]]
uint64_t bitFieldExtract(uint64_t base, uint32_t offset, uint32_t count);

template<typename T>
[[vk::ext_capability(spv::CapabilityBitInstructions)]]
[[vk::ext_instruction(spv::OpBitReverse)]]
T bitReverse(T base);

template<typename T>
[[vk::ext_instruction(spv::OpBitCount)]]
T bitCount(T base);

[[vk::ext_instruction(spv::OpControlBarrier)]]
void controlBarrier(uint32_t executionScope, uint32_t memoryScope,  uint32_t semantics);

[[vk::ext_instruction(spv::OpMemoryBarrier)]]
void memoryBarrier(uint32_t memoryScope,  uint32_t semantics);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicLoad)]]
T atomicLoad([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicLoad)]]
enable_if_t<is_spirv_type_v<P>, T> atomicLoad(P pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicStore)]]
void atomicStore([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicStore)]]
enable_if_t<is_spirv_type_v<P>, void> atomicStore(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicExchange)]]
T atomicExchange([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicExchange)]]
enable_if_t<is_spirv_type_v<P>, T> atomicExchange(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicCompareExchange)]]
T atomicCompareExchange([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t equal,  uint32_t unequal, T value, T comparator);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicCompareExchange)]]
enable_if_t<is_spirv_type_v<P>, T> atomicCompareExchange(P pointer, uint32_t memoryScope,  uint32_t equal,  uint32_t unequal, T value, T comparator);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicCompareExchangeWeak)]]
T atomicCompareExchangeWeak([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t equal,  uint32_t unequal, T value, T comparator);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicCompareExchangeWeak)]]
enable_if_t<is_spirv_type_v<P>, T> atomicCompareExchangeWeak(P pointer, uint32_t memoryScope,  uint32_t equal,  uint32_t unequal, T value, T comparator);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicIIncrement)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> atomicIIncrement([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicIIncrement)]]
enable_if_t<is_spirv_type_v<P> && (is_signed_v<T> || is_unsigned_v<T>), T> atomicIIncrement(P pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicIDecrement)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> atomicIDecrement([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicIDecrement)]]
enable_if_t<is_spirv_type_v<P> && (is_signed_v<T> || is_unsigned_v<T>), T> atomicIDecrement(P pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> atomicIAdd([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
enable_if_t<is_spirv_type_v<P> && (is_signed_v<T> || is_unsigned_v<T>), T> atomicIAdd(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicISub)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> atomicISub([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicISub)]]
enable_if_t<is_spirv_type_v<P> && (is_signed_v<T> || is_unsigned_v<T>), T> atomicISub(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

[[vk::ext_instruction(spv::OpAtomicSMin)]]
int32_t atomicMin([[vk::ext_reference]] int32_t pointer, uint32_t memoryScope,  uint32_t semantics, int32_t value);

[[vk::ext_instruction(spv::OpAtomicSMin)]]
int64_t atomicMin([[vk::ext_reference]] int64_t pointer, uint32_t memoryScope,  uint32_t semantics, int64_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicSMin)]]
enable_if_t<is_spirv_type_v<P>, int32_t> atomicMin(P pointer, uint32_t memoryScope,  uint32_t semantics, int32_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicSMin)]]
enable_if_t<is_spirv_type_v<P>, int64_t> atomicMin(P pointer, uint32_t memoryScope,  uint32_t semantics, int64_t value);

[[vk::ext_instruction(spv::OpAtomicUMin)]]
uint32_t atomicMin([[vk::ext_reference]] uint32_t pointer, uint32_t memoryScope,  uint32_t semantics, uint32_t value);

[[vk::ext_instruction(spv::OpAtomicUMin)]]
uint64_t atomicMin([[vk::ext_reference]] uint64_t pointer, uint32_t memoryScope,  uint32_t semantics, uint64_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicUMin)]]
enable_if_t<is_spirv_type_v<P>, uint32_t> atomicMin(P pointer, uint32_t memoryScope,  uint32_t semantics, uint32_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicUMin)]]
enable_if_t<is_spirv_type_v<P>, uint64_t> atomicMin(P pointer, uint32_t memoryScope,  uint32_t semantics, uint64_t value);

[[vk::ext_instruction(spv::OpAtomicSMax)]]
int32_t atomicMax([[vk::ext_reference]] int32_t pointer, uint32_t memoryScope,  uint32_t semantics, int32_t value);

[[vk::ext_instruction(spv::OpAtomicSMax)]]
int64_t atomicMax([[vk::ext_reference]] int64_t pointer, uint32_t memoryScope,  uint32_t semantics, int64_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicSMax)]]
enable_if_t<is_spirv_type_v<P>, int32_t> atomicMax(P pointer, uint32_t memoryScope,  uint32_t semantics, int32_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicSMax)]]
enable_if_t<is_spirv_type_v<P>, int64_t> atomicMax(P pointer, uint32_t memoryScope,  uint32_t semantics, int64_t value);

[[vk::ext_instruction(spv::OpAtomicUMax)]]
uint32_t atomicMax([[vk::ext_reference]] uint32_t pointer, uint32_t memoryScope,  uint32_t semantics, uint32_t value);

[[vk::ext_instruction(spv::OpAtomicUMax)]]
uint64_t atomicMax([[vk::ext_reference]] uint64_t pointer, uint32_t memoryScope,  uint32_t semantics, uint64_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicUMax)]]
enable_if_t<is_spirv_type_v<P>, uint32_t> atomicMax(P pointer, uint32_t memoryScope,  uint32_t semantics, uint32_t value);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicUMax)]]
enable_if_t<is_spirv_type_v<P>, uint64_t> atomicMax(P pointer, uint32_t memoryScope,  uint32_t semantics, uint64_t value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicAnd)]]
T atomicAnd([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicAnd)]]
enable_if_t<is_spirv_type_v<P>, T> atomicAnd(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicOr)]]
T atomicOr([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicOr)]]
enable_if_t<is_spirv_type_v<P>, T> atomicOr(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicXor)]]
T atomicXor([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicXor)]]
enable_if_t<is_spirv_type_v<P>, T> atomicXor(P pointer, uint32_t memoryScope,  uint32_t semantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicFlagTestAndSet)]]
T atomicFlagTestAndSet([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpAtomicFlagTestAndSet)]]
enable_if_t<is_spirv_type_v<P>, T> atomicFlagTestAndSet(P pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicFlagClear)]]
void atomicFlagClear([[vk::ext_reference]] T pointer, uint32_t memoryScope,  uint32_t semantics);

template<typename P>
[[vk::ext_instruction(spv::OpAtomicFlagClear)]]
enable_if_t<is_spirv_type_v<P>, void> atomicFlagClear(P pointer, uint32_t memoryScope,  uint32_t semantics);

[[vk::ext_capability(spv::CapabilityGroupNonUniform)]]
[[vk::ext_instruction(spv::OpGroupNonUniformElect)]]
bool groupNonUniformElect(uint32_t executionScope);

[[vk::ext_capability(spv::CapabilityGroupNonUniformVote)]]
[[vk::ext_instruction(spv::OpGroupNonUniformAll)]]
bool groupNonUniformAll(uint32_t executionScope, bool predicate);

[[vk::ext_capability(spv::CapabilityGroupNonUniformVote)]]
[[vk::ext_instruction(spv::OpGroupNonUniformAny)]]
bool groupNonUniformAny(uint32_t executionScope, bool predicate);

[[vk::ext_capability(spv::CapabilityGroupNonUniformVote)]]
[[vk::ext_instruction(spv::OpGroupNonUniformAllEqual)]]
bool groupNonUniformAllEqual(uint32_t executionScope, bool value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBroadcast)]]
T groupNonUniformBroadcast(uint32_t executionScope, T value, uint32_t id);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBroadcastFirst)]]
T groupNonUniformBroadcastFirst(uint32_t executionScope, T value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallot)]]
uint32_t4 groupNonUniformBallot(uint32_t executionScope, bool predicate);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformInverseBallot)]]
bool groupNonUniformInverseBallot(uint32_t executionScope, uint32_t4 value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallotBitExtract)]]
bool groupNonUniformBallotBitExtract(uint32_t executionScope, uint32_t4 value, uint32_t index);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallotBitCount)]]
uint32_t groupNonUniformBallotBitCount(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t4 value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallotFindLSB)]]
uint32_t groupNonUniformBallotFindLSB(uint32_t executionScope, uint32_t4 value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallotFindMSB)]]
uint32_t groupNonUniformBallotFindMSB(uint32_t executionScope, uint32_t4 value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformShuffle)]]
[[vk::ext_instruction(spv::OpGroupNonUniformShuffle)]]
T groupNonUniformShuffle(uint32_t executionScope, T value, uint32_t id);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformShuffle)]]
[[vk::ext_instruction(spv::OpGroupNonUniformShuffleXor)]]
T groupNonUniformShuffleXor(uint32_t executionScope, T value, uint32_t mask);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformShuffleRelative)]]
[[vk::ext_instruction(spv::OpGroupNonUniformShuffleUp)]]
T groupNonUniformShuffleUp(uint32_t executionScope, T value, uint32_t delta);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformShuffleRelative)]]
[[vk::ext_instruction(spv::OpGroupNonUniformShuffleDown)]]
T groupNonUniformShuffleDown(uint32_t executionScope, T value, uint32_t delta);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformIAdd)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> groupNonUniformIAdd_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformIAdd)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> groupNonUniformIAdd_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformIAdd)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> groupNonUniformIAdd_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFAdd)]]
float groupNonUniformAdd_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFAdd)]]
float groupNonUniformAdd_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFAdd)]]
float groupNonUniformAdd_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformIMul)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> groupNonUniformIMul_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformIMul)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> groupNonUniformIMul_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformIMul)]]
enable_if_t<(is_signed_v<T> || is_unsigned_v<T>), T> groupNonUniformIMul_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMul)]]
float groupNonUniformMul_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMul)]]
float groupNonUniformMul_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMul)]]
float groupNonUniformMul_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMin)]]
int32_t groupNonUniformMin_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMin)]]
int32_t groupNonUniformMin_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMin)]]
int32_t groupNonUniformMin_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMin)]]
int64_t groupNonUniformMin_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMin)]]
int64_t groupNonUniformMin_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMin)]]
int64_t groupNonUniformMin_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMin)]]
uint32_t groupNonUniformMin_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMin)]]
uint32_t groupNonUniformMin_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMin)]]
uint32_t groupNonUniformMin_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMin)]]
uint64_t groupNonUniformMin_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMin)]]
uint64_t groupNonUniformMin_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMin)]]
uint64_t groupNonUniformMin_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMin)]]
float groupNonUniformMin_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMin)]]
float groupNonUniformMin_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMin)]]
float groupNonUniformMin_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMax)]]
int32_t groupNonUniformMax_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMax)]]
int32_t groupNonUniformMax_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMax)]]
int32_t groupNonUniformMax_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMax)]]
int64_t groupNonUniformMax_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMax)]]
int64_t groupNonUniformMax_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformSMax)]]
int64_t groupNonUniformMax_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, int64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMax)]]
uint32_t groupNonUniformMax_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMax)]]
uint32_t groupNonUniformMax_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMax)]]
uint32_t groupNonUniformMax_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMax)]]
uint64_t groupNonUniformMax_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMax)]]
uint64_t groupNonUniformMax_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformUMax)]]
uint64_t groupNonUniformMax_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint64_t value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMax)]]
float groupNonUniformMax_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMax)]]
float groupNonUniformMax_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformFMax)]]
float groupNonUniformMax_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, float value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseAnd)]]
T groupNonUniformBitwiseAnd_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseAnd)]]
T groupNonUniformBitwiseAnd_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseAnd)]]
T groupNonUniformBitwiseAnd_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseOr)]]
T groupNonUniformBitwiseOr_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseOr)]]
T groupNonUniformBitwiseOr_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseOr)]]
T groupNonUniformBitwiseOr_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseXor)]]
T groupNonUniformBitwiseXor_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseXor)]]
T groupNonUniformBitwiseXor_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBitwiseXor)]]
T groupNonUniformBitwiseXor_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalAnd)]]
T groupNonUniformLogicalAnd_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalAnd)]]
T groupNonUniformLogicalAnd_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalAnd)]]
T groupNonUniformLogicalAnd_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalOr)]]
T groupNonUniformLogicalOr_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalOr)]]
T groupNonUniformLogicalOr_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalOr)]]
T groupNonUniformLogicalOr_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformArithmetic)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalXor)]]
T groupNonUniformLogicalXor_GroupNonUniformArithmetic(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformClustered)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalXor)]]
T groupNonUniformLogicalXor_GroupNonUniformClustered(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformLogicalXor)]]
T groupNonUniformLogicalXor_GroupNonUniformPartitionedNV(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformQuad)]]
[[vk::ext_instruction(spv::OpGroupNonUniformQuadBroadcast)]]
T groupNonUniformQuadBroadcast(uint32_t executionScope, T value, uint32_t index);

template<typename T>
[[vk::ext_capability(spv::CapabilityQuadControlKHR)]]
[[vk::ext_instruction(spv::OpGroupNonUniformQuadAllKHR)]]
T groupNonUniformQuadAllKHR(bool predicate);

template<typename T>
[[vk::ext_capability(spv::CapabilityQuadControlKHR)]]
[[vk::ext_instruction(spv::OpGroupNonUniformQuadAnyKHR)]]
T groupNonUniformQuadAnyKHR(bool predicate);

template<typename T>
[[vk::ext_capability(spv::CapabilityGroupNonUniformPartitionedNV)]]
[[vk::ext_instruction(spv::OpGroupNonUniformPartitionNV)]]
T groupNonUniformPartitionNV(T value);

[[vk::ext_capability(spv::CapabilityAtomicFloat16MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
float atomicMinEXT_AtomicFloat16MinMaxEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat32MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
float atomicMinEXT_AtomicFloat32MinMaxEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat64MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
float atomicMinEXT_AtomicFloat64MinMaxEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat16VectorNV)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
float atomicMinEXT_AtomicFloat16VectorNV([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat16MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMinEXT_AtomicFloat16MinMaxEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat32MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMinEXT_AtomicFloat32MinMaxEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat64MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMinEXT_AtomicFloat64MinMaxEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat16VectorNV)]]
[[vk::ext_instruction(spv::OpAtomicFMinEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMinEXT_AtomicFloat16VectorNV(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat16MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
float atomicMaxEXT_AtomicFloat16MinMaxEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat32MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
float atomicMaxEXT_AtomicFloat32MinMaxEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat64MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
float atomicMaxEXT_AtomicFloat64MinMaxEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat16VectorNV)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
float atomicMaxEXT_AtomicFloat16VectorNV([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat16MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMaxEXT_AtomicFloat16MinMaxEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat32MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMaxEXT_AtomicFloat32MinMaxEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat64MinMaxEXT)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMaxEXT_AtomicFloat64MinMaxEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat16VectorNV)]]
[[vk::ext_instruction(spv::OpAtomicFMaxEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicMaxEXT_AtomicFloat16VectorNV(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat16AddEXT)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
float atomicAddEXT_AtomicFloat16AddEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat32AddEXT)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
float atomicAddEXT_AtomicFloat32AddEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat64AddEXT)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
float atomicAddEXT_AtomicFloat64AddEXT([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilityAtomicFloat16VectorNV)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
float atomicAddEXT_AtomicFloat16VectorNV([[vk::ext_reference]] float pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat16AddEXT)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicAddEXT_AtomicFloat16AddEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat32AddEXT)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicAddEXT_AtomicFloat32AddEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat64AddEXT)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicAddEXT_AtomicFloat64AddEXT(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

template<typename P>
[[vk::ext_capability(spv::CapabilityAtomicFloat16VectorNV)]]
[[vk::ext_instruction(spv::OpAtomicFAddEXT)]]
enable_if_t<is_spirv_type_v<P>, float> atomicAddEXT_AtomicFloat16VectorNV(P pointer, uint32_t memoryScope,  uint32_t semantics, float value);

[[vk::ext_capability(spv::CapabilitySplitBarrierINTEL)]]
[[vk::ext_instruction(spv::OpControlBarrierArriveINTEL)]]
void controlBarrierArriveINTEL(uint32_t executionScope, uint32_t memoryScope,  uint32_t semantics);

[[vk::ext_capability(spv::CapabilitySplitBarrierINTEL)]]
[[vk::ext_instruction(spv::OpControlBarrierWaitINTEL)]]
void controlBarrierWaitINTEL(uint32_t executionScope, uint32_t memoryScope,  uint32_t semantics);

}

#endif
}
}

#endif
