#define _NBL_VIDEO_I_GPU_ACCELERATION_STRUCTURE_CPP_
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

template<class BufferType>
bool IGPUAccelerationStructure::BuildInfo<BufferType>::invalid(const IGPUAccelerationStructure* const src, const IGPUAccelerationStructure* const dst)
{
	// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03800
	// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03707
	if (!dst || !dst->getCreationParams().bufferRange.buffer->getBoundMemory())
        return true;

	const auto device = dst->getOriginDevice();
	if (isUpdate)
	{
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-srcAccelerationStructure-04629
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-04630
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03708
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-srcAccelerationStructure-04629
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-04630
		if (!src || src->getOriginDevice()!=device || !src->getCreationParams().bufferRange.buffer->getBoundMemory())
			return true;
	}
	
	if (!scratchAddr.isValid())
        return true;
	if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
	{
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03674
		if (scratchAddr.buffer->getCreationParams().usage.hasFlags(IGPUBuffer::EUF_STORAGE_BUFFER_BIT))
			return true;
		const auto scratchAddr = scratchAddr.buffer->getDeviceAddress();
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03802
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03803
		if (scratchAddr==0ull)
			return true;
		// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03710
		if (!core::is_aligned_to(scratchAddr,device->getPhysicalDevice()->getLimits().minAccelerationStructureScratchOffsetAlignment))
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


template<class BufferType>
inline uint32_t IGPUBottomLevelAccelerationStructure::BuildInfo<BufferType>::valid(const BuildRangeInfo* const buildRangeInfos) const
{
	if (IGPUAccelerationStructure::BuildInfo<BufferType>::invalid(srcAS,dstAS))
		return false;

	const bool isAABB = buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT);
	const bool hasMotion = dstAS->getCreationFlags().hasFlags(IAccelerationStructure::CREATE_FLAGS::MOTION_BIT);
	#ifdef _NBL_DEBUG
	if (isAABB)
	{
		for (auto i=0u; i<aabbs.size(); i++)
		{
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03811
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03812
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03814
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03774
			if (invalidInputBuffer(aabbs[i].data,buildRangeInfos[i].primitiveOffset,buildRangeInfos[i].primitiveCount,sizeof(AABB_t),8u))
				return false;
		}
	}
	else
	{
		const auto& bufferUsages = getOriginDevice()->getPhysicalDevice()->getBufferFormatUsages();
		for (auto i=0u; i<triangles.size(); i++)
		{
			const auto& geometry = triangles[i];
			//
			if (!bufferUsages[geometry.vertexFormat].accelerationStructureVertex)
				return false;
			const size_t vertexSize = asset::getTexelOrBlockBytesize(geometry.vertexFormat);
			// TODO: improve in line with
			const size_t vertexAlignment = vertexSize;
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03804
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03805
			if (invalidInputBuffer(geometry.vertexData[0],buildRangeInfos[i].firstVertex,geometry.maxVertex,vertexSize,vertexAlignment))
				return false;
			//
			if (hasMotion && invalidInputBuffer(geometry.vertexData[1],buildRangeInfos[i].firstVertex,geometry.maxVertex,vertexSize,vertexAlignment))
				return false;
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03712
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03806
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03807
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03771
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03772
			if (geometry.indexType!=asset::EIT_UNKNOWN)
			{
				const size_t indexSize = indexType==asset::EIT_16BIT ? sizeof(uint16_t):sizeof(uint32_t);
				if (invalidInputBuffer(geometry.indexData,buildRangeInfos[i].primitiveOffset,buildRangeInfos[i].primitiveCount,indexSize,indexSize))
					return false;
			}
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03808
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03809
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03810
			// https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-vkBuildAccelerationStructuresKHR-pInfos-03773
			if (geometry.transformData.buffer && invalidInputBuffer(geometry.transformData,buildRangeInfos[i].primitiveOffset,buildRangeInfos[i].primitiveCount,sizeof(core::matrix3x4SIMD),sizeof(core::vectorSIMDf))) // TODO: check size
				return false;
		}
	}
	#endif

	// destination and scratch
	uint32_t retval = 2u;
	if (isUpdate) // source
		retval++;

	const uint32_t MaxBuffersPerGeometry = isAABB ? 1u:(hasMotion ? 4u:3u);
	retval += MaxBuffersPerGeometry*geometries.size();
	return retval;
}

}