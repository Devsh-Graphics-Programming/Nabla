#ifndef _NBL_VIDEO_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_


#include "nbl/video/IGPUAccelerationStructure.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include "nbl/video/CVulkanCommon.h"


namespace nbl::video
{

//! all these utilities cannot be nested because of the complex inheritance between `IGPUAccelerationStructure` and the Vulkan classes
static inline VkCopyAccelerationStructureModeKHR getVkCopyAccelerationStructureModeFrom(const IGPUAccelerationStructure::COPY_MODE in)
{
	return static_cast<VkCopyAccelerationStructureModeKHR>(in);
}
static inline VkCopyAccelerationStructureInfoKHR getVkCopyAccelerationStructureInfoFrom(const IGPUAccelerationStructure::CopyInfo& copyInfo)
{
	VkCopyAccelerationStructureInfoKHR info = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR,nullptr };
	info.src = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.src->getNativeHandle());
	info.dst = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.dst->getNativeHandle());
	info.mode = getVkCopyAccelerationStructureModeFrom(copyInfo.mode);
	return info;
}

template<typename BufferType>
static inline VkDeviceOrHostAddressKHR getVkDeviceOrHostAddress(const asset::SBufferBinding<BufferType>& binding)
{
	VkDeviceOrHostAddressKHR addr;
	if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
		addr.deviceAddress = binding.buffer->getDeviceAddress()+binding.offset;
	else
	{
		static_assert(std::is_same_v<BufferType,asset::ICPUBuffer>);
		addr.hostAddress = reinterpret_cast<uint8_t*>(binding.buffer->getPointer())+binding.offset;
	}
	return addr;
}
template<typename BufferType>
static inline VkDeviceOrHostAddressConstKHR getVkDeviceOrHostConstAddress(const asset::SBufferBinding<const BufferType>& binding)
{
	VkDeviceOrHostAddressConstKHR addr;
	if constexpr (std::is_same_v<BufferType,IGPUBuffer>)
		addr.deviceAddress = binding.buffer->getDeviceAddress()+binding.offset;
	else
	{
		static_assert(std::is_same_v<BufferType,asset::ICPUBuffer>);
		addr.hostAddress = reinterpret_cast<const uint8_t*>(binding.buffer->getPointer())+binding.offset;
	}
	return addr;
}
template<typename BufferType>
static inline VkCopyAccelerationStructureToMemoryInfoKHR getVkCopyAccelerationStructureToMemoryInfoFrom(const IGPUAccelerationStructure::CopyToMemoryInfo<BufferType>& copyInfo)
{
	VkCopyAccelerationStructureToMemoryInfoKHR info = { VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR,nullptr };
	info.src = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.src->getNativeHandle());
	info.dst = getVkDeviceOrHostAddress<BufferType>(copyInfo.dst);
	info.mode = getVkCopyAccelerationStructureModeFrom(copyInfo.mode);
	return info;
}
template<typename BufferType>
static inline VkCopyMemoryToAccelerationStructureInfoKHR getVkCopyMemoryToAccelerationStructureInfoFrom(const IGPUAccelerationStructure::CopyFromMemoryInfo<BufferType>& copyInfo)
{
	VkCopyMemoryToAccelerationStructureInfoKHR info = { VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR,nullptr };
	info.src = getVkDeviceOrHostConstAddress<BufferType>(copyInfo.src);
	info.dst = *reinterpret_cast<const VkAccelerationStructureKHR*>(copyInfo.dst->getNativeHandle());
	info.mode = getVkCopyAccelerationStructureModeFrom(copyInfo.mode);
	return info;
}

static inline VkGeometryFlagsKHR getVkGeometryFlagsFrom(const IGPUAccelerationStructure::GEOMETRY_FLAGS in)
{
	return static_cast<VkGeometryFlagsKHR>(in);
}
static inline VkBuildAccelerationStructureFlagsKHR getVkASBuildFlagsFrom(const IGPUAccelerationStructure::BUILD_FLAGS in)
{
	return static_cast<VkBuildAccelerationStructureFlagsKHR>(in);
}


class CVulkanLogicalDevice;


template<class GPUAccelerationStructure>
class CVulkanAccelerationStructure : public GPUAccelerationStructure
{
	public:
		inline const void* getNativeHandle() const { return &m_accelerationStructure; }
		inline VkAccelerationStructureKHR getInternalObject() const { return m_vkAccelerationStructure; }
	
		bool wasCopySuccessful(const IDeferredOperation* const deferredOp) override;

	protected:
		CVulkanAccelerationStructure(core::smart_refctd_ptr<const CVulkanLogicalDevice>&& logicalDevice, IGPUAccelerationStructure::SCreationParams&& params, const VkAccelerationStructureKHR accelerationStructure);
		~CVulkanAccelerationStructure();

		VkAccelerationStructureKHR m_vkAccelerationStructure;
		VkDeviceAddress m_deviceAddress;
		
};

class CVulkanBottomLevelAccelerationStructure final : public CVulkanAccelerationStructure<IGPUBottomLevelAccelerationStructure>
{
	public:
		using CVulkanAccelerationStructure<IGPUBottomLevelAccelerationStructure>::CVulkanAccelerationStructure<IGPUBottomLevelAccelerationStructure>;

		uint64_t getReferenceForDeviceOperations() const override {return m_deviceAddress;}
		inline uint64_t getReferenceForHostOperations() const override {return reinterpret_cast<uint64_t>(m_vkAccelerationStructure);}

	private:
};

class CVulkanTopLevelAccelerationStructure final : public CVulkanAccelerationStructure<IGPUTopLevelAccelerationStructure>
{
	public:
		using CVulkanAccelerationStructure<IGPUTopLevelAccelerationStructure>::CVulkanAccelerationStructure<IGPUTopLevelAccelerationStructure>;
#if 0
	template<typename AddressType>
	static VkAccelerationStructureGeometryKHR getVkASGeometry(VkDevice vk_device, const CVulkanDeviceFunctionTable* vk_devf, const Geometry<AddressType>& geometry)
	{
		VkAccelerationStructureGeometryKHR ret = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR, nullptr};
		ret.geometryType = getVkGeometryTypeFromGeomType(geometry.type);
		ret.flags = getVkGeometryFlagsFromGeometryFlags(geometry.flags);
		if(E_GEOM_TYPE::EGT_TRIANGLES == geometry.type)
		{
			const auto & triangles = geometry.data.triangles;
			ret.geometry.triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR, nullptr};
			ret.geometry.triangles.vertexFormat = getVkFormatFromFormat(triangles.vertexFormat);
			ret.geometry.triangles.vertexData = getVkDeviceOrHostConstAddress(vk_device, vk_devf, triangles.vertexData);
			ret.geometry.triangles.vertexStride = static_cast<VkDeviceSize>(triangles.vertexStride);
			ret.geometry.triangles.maxVertex = triangles.maxVertex;
			ret.geometry.triangles.indexType = static_cast<VkIndexType>(triangles.indexType); // (Erfan): Converter?
			ret.geometry.triangles.indexData = getVkDeviceOrHostConstAddress(vk_device, vk_devf, triangles.indexData);
			ret.geometry.triangles.transformData = getVkDeviceOrHostConstAddress(vk_device, vk_devf, triangles.transformData);
		}
		else if(E_GEOM_TYPE::EGT_AABBS == geometry.type)
		{
			const auto & aabbs = geometry.data.aabbs;
			ret.geometry.aabbs = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR, nullptr};
			ret.geometry.aabbs.data = getVkDeviceOrHostConstAddress(vk_device, vk_devf, aabbs.data);
			ret.geometry.aabbs.stride = static_cast<VkDeviceSize>(aabbs.stride);
		}
		else if(E_GEOM_TYPE::EGT_INSTANCES == geometry.type)
		{
			const auto & instances = geometry.data.instances;
			ret.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR, nullptr};
			ret.geometry.instances.data = getVkDeviceOrHostConstAddress(vk_device, vk_devf, instances.data);
			ret.geometry.instances.arrayOfPointers = VK_FALSE; // (Erfan): Something to expose?
		}
		return ret;
	}

	template<typename AddressType>
	static VkAccelerationStructureBuildGeometryInfoKHR getVkASBuildGeomInfoFromBuildGeomInfo(
		VkDevice vk_device,
		const CVulkanDeviceFunctionTable* vk_devf,
		const BuildGeometryInfo<AddressType>& buildGeomInfo,
		VkAccelerationStructureGeometryKHR* inoutGeomArray) 
	{
		VkAccelerationStructureBuildGeometryInfoKHR ret = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR, nullptr};
		if(inoutGeomArray != nullptr)
		{
			uint32_t geomCount = buildGeomInfo.geometries.size();
			const Geometry<AddressType>* geoms = buildGeomInfo.geometries.begin();
			for(uint32_t g = 0; g < geomCount; ++g) {
				auto & geom = geoms[g];
				VkAccelerationStructureGeometryKHR vk_geom = getVkASGeometry<AddressType>(vk_device, vk_devf, geom);
				inoutGeomArray[g] = vk_geom;
			}

			VkAccelerationStructureKHR vk_srcAS = (buildGeomInfo.srcAS) ? static_cast<CVulkanAccelerationStructure *>(buildGeomInfo.srcAS)->getInternalObject() : VK_NULL_HANDLE;
			VkAccelerationStructureKHR vk_dstAS = (buildGeomInfo.dstAS) ? static_cast<CVulkanAccelerationStructure *>(buildGeomInfo.dstAS)->getInternalObject() : VK_NULL_HANDLE;

			ret.type = getVkASTypeFromASType(buildGeomInfo.type);
			ret.flags = getVkASBuildFlagsFromASBuildFlags(buildGeomInfo.buildFlags);
			ret.mode = getVkASBuildModeFromASBuildMode(buildGeomInfo.buildMode);
			ret.srcAccelerationStructure = vk_srcAS;
			ret.dstAccelerationStructure = vk_dstAS;
			ret.geometryCount = geomCount;
			ret.pGeometries = inoutGeomArray;
			ret.ppGeometries = nullptr;
			ret.scratchData = getVkDeviceOrHostAddress(vk_device, vk_devf, buildGeomInfo.scratchAddr);
		}
		return ret;
	}
#endif
};

}

#endif
