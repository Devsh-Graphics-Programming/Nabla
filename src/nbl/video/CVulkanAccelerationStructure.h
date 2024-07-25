#ifndef _NBL_VIDEO_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/video/IGPUAccelerationStructure.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include "nbl/video/CVulkanCommon.h"


namespace nbl::video
{
class CVulkanLogicalDevice;


template<class GPUAccelerationStructure> //requires std::is_base_of_v<IGPUAccelerationStructure,GPUAccelerationStructure>
class CVulkanAccelerationStructure : public GPUAccelerationStructure
{
	public:
		inline const void* getNativeHandle() const { return &m_vkAccelerationStructure; }
		inline VkAccelerationStructureKHR getInternalObject() const { return m_vkAccelerationStructure; }
	
		bool wasCopySuccessful(const IDeferredOperation* const deferredOp) override;
		bool wasBuildSuccessful(const IDeferredOperation* const deferredOp) override;

		// public because using can't change the privacy scope
		CVulkanAccelerationStructure(core::smart_refctd_ptr<const CVulkanLogicalDevice>&& logicalDevice, GPUAccelerationStructure::SCreationParams&& params, const VkAccelerationStructureKHR accelerationStructure);
	protected:
		~CVulkanAccelerationStructure();

		VkAccelerationStructureKHR m_vkAccelerationStructure;
		VkDeviceAddress m_deviceAddress;
		
};

class CVulkanBottomLevelAccelerationStructure final : public CVulkanAccelerationStructure<IGPUBottomLevelAccelerationStructure>
{
		using Base = CVulkanAccelerationStructure<IGPUBottomLevelAccelerationStructure>;

	public:
		using Base::Base;

		inline device_op_ref_t getReferenceForDeviceOperations() const override {return {m_deviceAddress};}
		inline host_op_ref_t getReferenceForHostOperations() const override {return {reinterpret_cast<uint64_t>(m_vkAccelerationStructure)};}
};

class CVulkanTopLevelAccelerationStructure final : public CVulkanAccelerationStructure<IGPUTopLevelAccelerationStructure>
{
		using Base = CVulkanAccelerationStructure<IGPUTopLevelAccelerationStructure>;

	public:
		using Base::Base;
};


//! all these utilities cannot be nested because of the complex inheritance between `IGPUAccelerationStructure` and the Vulkan classes
inline VkCopyAccelerationStructureModeKHR getVkCopyAccelerationStructureModeFrom(const IGPUAccelerationStructure::COPY_MODE in)
{
	return static_cast<VkCopyAccelerationStructureModeKHR>(in);
}
inline VkCopyAccelerationStructureInfoKHR getVkCopyAccelerationStructureInfoFrom(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
	VkCopyAccelerationStructureInfoKHR info = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,nullptr };
	info.src = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.src->getNativeHandle());
	info.dst = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.dst->getNativeHandle());
	info.mode = getVkCopyAccelerationStructureModeFrom(copyInfo.mode);
	return info;
}

template<typename T>
concept Buffer = is_any_of_v<std::remove_const_t<T>,IGPUBuffer,asset::ICPUBuffer>;

template<Buffer BufferType>
using DeviceOrHostAddress = std::conditional_t<std::is_const_v<BufferType>,VkDeviceOrHostAddressConstKHR,VkDeviceOrHostAddressKHR>;

template<Buffer BufferType>
inline DeviceOrHostAddress<BufferType> getVkDeviceOrHostAddress(const asset::SBufferBinding<BufferType>& binding)
{
	using buffer_t = std::remove_const_t<BufferType>;

	DeviceOrHostAddress<BufferType> addr;
	if constexpr (std::is_same_v<buffer_t,IGPUBuffer>)
		addr.deviceAddress = binding.buffer->getDeviceAddress()+binding.offset;
	else
	{
		static_assert(std::is_same_v<buffer_t,asset::ICPUBuffer>);
		using byte_t = std::conditional_t<std::is_const_v<BufferType>,const uint8_t,uint8_t>;
		addr.hostAddress = reinterpret_cast<byte_t*>(binding.buffer->getPointer())+binding.offset;
	}
	return addr;
}
template<Buffer BufferType>
inline VkCopyAccelerationStructureToMemoryInfoKHR getVkCopyAccelerationStructureToMemoryInfoFrom(const IGPUAccelerationStructure::CopyToMemoryInfo<BufferType>& copyInfo)
{
	VkCopyAccelerationStructureToMemoryInfoKHR info = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR,nullptr };
	info.src = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.src->getNativeHandle());
	info.dst = getVkDeviceOrHostAddress<BufferType>(copyInfo.dst);
	info.mode = getVkCopyAccelerationStructureModeFrom(copyInfo.mode);
	return info;
}
template<Buffer BufferType>
inline VkCopyMemoryToAccelerationStructureInfoKHR getVkCopyMemoryToAccelerationStructureInfoFrom(const IGPUAccelerationStructure::CopyFromMemoryInfo<BufferType>& copyInfo)
{
	VkCopyMemoryToAccelerationStructureInfoKHR info = { VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR,nullptr };
	info.src = getVkDeviceOrHostAddress<const BufferType>(copyInfo.src);
	info.dst = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.dst->getNativeHandle());
	info.mode = getVkCopyAccelerationStructureModeFrom(copyInfo.mode);
	return info;
}

inline VkGeometryFlagsKHR getVkGeometryFlagsFrom(const IGPUBottomLevelAccelerationStructure::GEOMETRY_FLAGS in)
{
	return static_cast<VkGeometryFlagsKHR>(in);
}

// The srcAccelerationStructure, dstAccelerationStructure, and mode members of pBuildInfo are ignored. Any VkDeviceOrHostAddressKHR members of pBuildInfo are ignored by this command
static const VkDeviceOrHostAddressConstKHR NullAddress = { 0x0ull };
template<Buffer BufferType, bool QueryOnly=false>
void getVkASGeometryFrom(const IGPUBottomLevelAccelerationStructure::Triangles<const BufferType>& triangles, VkAccelerationStructureGeometryKHR& outBase)
{
	static const VkDeviceOrHostAddressConstKHR DummyNonNullAddress = { 0xdeadbeefBADC0FFEull };

	outBase = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,nullptr,VK_GEOMETRY_TYPE_TRIANGLES_KHR};
	outBase.geometry.triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,nullptr};
	outBase.geometry.triangles.vertexFormat = getVkFormatFromFormat(triangles.vertexFormat);
	outBase.geometry.triangles.vertexData = QueryOnly ? NullAddress:getVkDeviceOrHostAddress<const BufferType>(triangles.vertexData[0]);
	outBase.geometry.triangles.vertexStride = triangles.vertexStride;
	outBase.geometry.triangles.maxVertex = triangles.maxVertex;
	outBase.geometry.triangles.indexType = static_cast<VkIndexType>(triangles.indexType);
	outBase.geometry.triangles.indexData = QueryOnly ? NullAddress:getVkDeviceOrHostAddress<const BufferType>(triangles.indexData);
	// except that the hostAddress member of VkAccelerationStructureGeometryTrianglesDataKHR::transformData will be examined to check if it is NULL.
	if (!triangles.hasTransform())
		outBase.geometry.triangles.transformData = NullAddress;
	else if (QueryOnly)
		outBase.geometry.triangles.transformData = DummyNonNullAddress;
	else
	{
		if constexpr (triangles.Host)
			outBase.geometry.triangles.transformData.hostAddress = &triangles.transform;
		else
			outBase.geometry.triangles.transformData = getVkDeviceOrHostAddress<const IGPUBuffer>(triangles.transform);
	}
	outBase.flags = getVkGeometryFlagsFrom(triangles.geometryFlags.value);
}
template<Buffer BufferType, bool QueryOnly=false>
void getVkASGeometryFrom(const IGPUBottomLevelAccelerationStructure::Triangles<const BufferType>& triangles, VkAccelerationStructureGeometryKHR& outBase, VkAccelerationStructureGeometryMotionTrianglesDataNV* &p_vertexMotion)
{
	getVkASGeometryFrom<const BufferType>(triangles,outBase);
	if (triangles.vertexData[1].buffer)
	{
		p_vertexMotion->sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV;
		p_vertexMotion->pNext = nullptr; // no micromaps for now 
		p_vertexMotion->vertexData = QueryOnly ? NullAddress:getVkDeviceOrHostAddress<const BufferType>(triangles.vertexData[1]);
		outBase.geometry.triangles.pNext = p_vertexMotion++;
	}
}

template<Buffer BufferType, bool QueryOnly=false>
void getVkASGeometryFrom(const IGPUBottomLevelAccelerationStructure::AABBs<const BufferType>& aabbs, VkAccelerationStructureGeometryKHR& outBase)
{
	outBase = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,nullptr,VK_GEOMETRY_TYPE_AABBS_KHR};
	outBase.geometry.aabbs = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,nullptr};
	outBase.geometry.aabbs.data = QueryOnly ? NullAddress:getVkDeviceOrHostAddress<const BufferType>(aabbs.data);
	outBase.geometry.aabbs.stride = aabbs.stride;
	outBase.flags = getVkGeometryFlagsFrom(aabbs.geometryFlags.value);
}

template<Buffer BufferType, bool QueryOnly=false>
void getVkASGeometryFrom(const IGPUTopLevelAccelerationStructure::BuildInfo<BufferType>& info, VkAccelerationStructureGeometryKHR& outBase)
{
	outBase = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,nullptr,VK_GEOMETRY_TYPE_INSTANCES_KHR};
	outBase.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,nullptr};
	outBase.geometry.instances.arrayOfPointers = info.buildFlags.hasFlags(IGPUTopLevelAccelerationStructure::BUILD_FLAGS::INSTANCE_DATA_IS_POINTERS_TYPE_ENCODED_LSB);
	outBase.geometry.instances.data = QueryOnly ? NullAddress:getVkDeviceOrHostAddress<const BufferType>(info.instanceData);
	// no "geometry flags" are valid for all instances!
	outBase.flags = static_cast<VkGeometryFlagsKHR>(0u);
}

// TODO: do BLASes with vertex motion need `VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV` ?
template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
inline VkBuildAccelerationStructureFlagsKHR getVkASBuildFlagsFrom(const core::bitflag<typename AccelerationStructure::BUILD_FLAGS> in, const bool motionBlur)
{
	auto retval = static_cast<VkBuildAccelerationStructureFlagsKHR>(in.value);
	if (motionBlur)
		retval |= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
	else
		retval &= ~VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
	return retval;
}
template<class AccelerationStructure> requires std::is_base_of_v<IGPUAccelerationStructure,AccelerationStructure>
inline VkBuildAccelerationStructureFlagsKHR getVkASBuildFlagsFrom(const core::bitflag<typename AccelerationStructure::BUILD_FLAGS> in, const AccelerationStructure* as)
{
	return getVkASBuildFlagsFrom<AccelerationStructure>(in,as->getCreationParams().flags.hasFlags(IGPUAccelerationStructure::SCreationParams::FLAGS::MOTION_BIT));
}

template<class BuildInfo> 
inline VkAccelerationStructureBuildGeometryInfoKHR getVkASBuildGeometryInfo(const BuildInfo& info, VkAccelerationStructureGeometryKHR* &p_vk_geometry, VkAccelerationStructureGeometryMotionTrianglesDataNV* &p_vertexMotion)
{
	using acceleration_structure_t = std::remove_pointer_t<decltype(info.dstAS)>;
	constexpr bool IsTLAS = std::is_same_v<acceleration_structure_t,IGPUTopLevelAccelerationStructure>;

	VkAccelerationStructureBuildGeometryInfoKHR vk_info = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,nullptr};
    vk_info.type = IsTLAS ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR:VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vk_info.flags = getVkASBuildFlagsFrom<acceleration_structure_t>(info.buildFlags,info.dstAS);
    vk_info.mode = info.isUpdate ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR:VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    vk_info.srcAccelerationStructure = static_cast<const CVulkanAccelerationStructure<acceleration_structure_t>*>(info.srcAS)->getInternalObject();
    vk_info.dstAccelerationStructure = static_cast<CVulkanAccelerationStructure<acceleration_structure_t>*>(info.dstAS)->getInternalObject();
    vk_info.geometryCount = info.inputCount();
    vk_info.pGeometries = p_vk_geometry;
    vk_info.ppGeometries = nullptr;
	vk_info.scratchData = getVkDeviceOrHostAddress(info.scratch);

	if constexpr (IsTLAS)
	{
		getVkASGeometryFrom(info,*p_vk_geometry);
		p_vk_geometry++;
	}
	else
	for (auto j=0u; j<info.geometryCount; j++)
	{
		auto& vk_geom = *(p_vk_geometry++);
		if (info.buildFlags.hasFlags(IGPUBottomLevelAccelerationStructure::BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
			getVkASGeometryFrom(info.aabbs[j],vk_geom);
		else
			getVkASGeometryFrom(info.triangles[j],vk_geom,p_vertexMotion);
	}
	return vk_info;
}

inline void getVkASBuildRangeInfos(const uint32_t geometryCount, const IGPUBottomLevelAccelerationStructure::BuildRangeInfo*const pBuildRanges, VkAccelerationStructureBuildRangeInfoKHR* &out_vk_infos)
{
	for (auto i=0; i<geometryCount; i++)
		*(out_vk_infos++) = {
			.primitiveCount = pBuildRanges[i].primitiveCount,
			.primitiveOffset = pBuildRanges[i].primitiveByteOffset,
			.firstVertex = pBuildRanges[i].firstVertex,
			.transformOffset = pBuildRanges[i].transformByteOffset
		};
}
inline VkAccelerationStructureBuildRangeInfoKHR getVkASBuildRangeInfo(const IGPUTopLevelAccelerationStructure::BuildRangeInfo& info)
{
	return {
		.primitiveCount = info.instanceCount,
		.primitiveOffset = info.instanceByteOffset,
		.firstVertex = 0, .transformOffset = 0
	};
}

}

#endif
