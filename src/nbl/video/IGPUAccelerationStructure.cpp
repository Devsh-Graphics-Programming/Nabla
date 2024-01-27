#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_CPP_
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

template<class BufferType>
bool IGPUAccelerationStructure::BuildInfo<BufferType>::invalid(const IGPUAccelerationStructure* const src, const IGPUAccelerationStructure* const dst) const
{
	// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03800
	// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03800
	// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03707
	if (!dst || !dst->getCreationParams().bufferRange.buffer->getBoundMemory().isValid())
        return true;

	const auto device = dst->getOriginDevice();
	if (isUpdate)
	{
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-srcAccelerationStructure-04629
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-04630
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03708
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-srcAccelerationStructure-04629
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-04630
		if (!src || src->getOriginDevice()!=device || !src->getCreationParams().bufferRange.buffer->getBoundMemory().isValid())
			return true;
		// TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03668
		if (src!=dst && /*memory aliasing check*/false)
			return true;
	}
	
	if (!scratch.isValid())
        return true;
	if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
	{
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03674
		if (scratch.buffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_STORAGE_BUFFER_BIT))
			return true;
		const auto scratchAddress = scratch.buffer->getDeviceAddress();
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03802
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03803
		if (scratchAddress==0ull)
			return true;
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03710
		if (!core::is_aligned_to(scratchAddress,device->getPhysicalDevice()->getLimits().minAccelerationStructureScratchOffsetAlignment))
			return true;
	}
	else
	{
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03722
		if (device->invalidAccelerationStructureForHostOperations(dst))
			return true;
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03723
		if (isUpdate && device->invalidAccelerationStructureForHostOperations(src))
			return true;
	}

    return false;
}
//extern template class IGPUAccelerationStructure::BuildInfo<IGPUBuffer>;
//extern template class IGPUAccelerationStructure::BuildInfo<asset::ICPUBuffer>;


template<class BufferType>
template<typename T>// requires nbl::is_any_of_v<T,std::conditional_t<std::is_same_v<BufferType,IGPUBuffer>,uint32_t,IGPUBottomLevelAccelerationStructure::BuildRangeInfo>,IGPUBottomLevelAccelerationStructure::BuildRangeInfo>
uint32_t IGPUBottomLevelAccelerationStructure::BuildInfo<BufferType>::valid(const T* const buildRangeInfosOrMaxPrimitiveCounts) const
{
	if (IGPUAccelerationStructure::BuildInfo<BufferType>::invalid(srcAS,dstAS))
		return false;

	const auto* device = dstAS->getOriginDevice();
	if (!validBuildFlags(buildFlags,device->getEnabledFeatures()))
		return {};

	const auto* physDev = device->getPhysicalDevice();
	const auto& limits = physDev->getLimits();

	const uint32_t geometryCount = inputCount();
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03793
    if (geometryCount>limits.maxAccelerationStructureGeometryCount)
        return {};

	const bool isAABB = buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT);

	#ifdef _NBL_DEBUG
	size_t totalPrims = 0ull;
	const auto& bufferUsages = physDev->getBufferFormatUsages();
	for (auto i=0u; i<geometryCount; i++)
	{
		if (!triangles) // its a union so checks aabbs as well
			return false;

		BuildRangeInfo buildRangeInfo;
		constexpr bool IndirectBuild = std::is_same_v<T,uint32_t>;
		if constexpr (IndirectBuild)
		{
			buildRangeInfo.primitiveByteOffset = 0u;
			buildRangeInfo.primitiveCount = buildRangeInfosOrMaxPrimitiveCounts[i];
		}
		else
			buildRangeInfo = buildRangeInfosOrMaxPrimitiveCounts[i];

		if (isAABB)
		{
			if (!validGeometry(totalPrims,aabbs[i],buildRangeInfo))
				return false;
		}
		else
		{
			if constexpr (IndirectBuild)
			{
				buildRangeInfo.firstVertex = 0u;
				buildRangeInfo.transformByteOffset = 0u;
			}
			if (!validGeometry(totalPrims,triangles[i],buildRangeInfo))
				return false;
		}
	}
	// TODO: and not sure of VUID
	if (totalPrims>size_t(limits.maxAccelerationStructurePrimitiveCount))
		return false;
	#endif

	// destination and scratch
	uint32_t retval = 2u;
	if (Base::isUpdate) // source
		retval++;

	uint32_t MaxBuffersPerGeometry = 1u;
	if (!isAABB)
	{
		// on host builds the transforms are "by-value" no BDA ergo no tracking needed
		MaxBuffersPerGeometry = std::is_same_v<BufferType,IGPUBuffer> ? 3u:2u;

		const bool hasMotion = dstAS->getCreationParams().flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT);
		if (hasMotion)
			MaxBuffersPerGeometry++;
	}

	retval += geometryCount*MaxBuffersPerGeometry;
	return retval;
}
template uint32_t IGPUBottomLevelAccelerationStructure::BuildInfo<IGPUBuffer>::template valid<uint32_t>(const uint32_t* const) const;
template uint32_t IGPUBottomLevelAccelerationStructure::BuildInfo<asset::ICPUBuffer>::template valid<uint32_t>(const uint32_t* const) const;
using BuildRangeInfo = hlsl::acceleration_structures::bottom_level::BuildRangeInfo;
template uint32_t IGPUBottomLevelAccelerationStructure::BuildInfo<IGPUBuffer>::template valid<BuildRangeInfo>(const BuildRangeInfo* const) const;
template uint32_t IGPUBottomLevelAccelerationStructure::BuildInfo<asset::ICPUBuffer>::template valid<BuildRangeInfo>(const BuildRangeInfo* const) const;

bool IGPUBottomLevelAccelerationStructure::validVertexFormat(const asset::E_FORMAT format) const
{
	return getOriginDevice()->getPhysicalDevice()->getBufferFormatUsages()[format].accelerationStructureVertex;
}

}