#ifndef _NBL_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_C_VULKAN_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/video/IGPUAccelerationStructure.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

#include "CVulkanCommon.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanAccelerationStructure final : public IGPUAccelerationStructure
{
public:
	CVulkanAccelerationStructure(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
		SCreationParams&& _params, VkAccelerationStructureKHR accelerationStructure)
		: IGPUAccelerationStructure(std::move(logicalDevice), std::move(_params)), m_vkAccelerationStructure(accelerationStructure)
	{}

	~CVulkanAccelerationStructure();
	
	uint64_t getReferenceForDeviceOperations() const override;
	uint64_t getReferenceForHostOperations() const override;

	inline VkAccelerationStructureKHR getInternalObject() const { return m_vkAccelerationStructure; }
	
public:
	
	static inline VkAccelerationStructureBuildGeometryInfoKHR getVkASBuildGeomInfoFromBuildGeomInfo(
		VkDevice vk_device,
		const BuildGeometryInfo<DeviceAddressType> & buildGeomInfo,
		VkAccelerationStructureGeometryKHR * inoutGeomArray) 
	{
		// TODO
		uint32_t geomCount = buildGeomInfo.geometries.size();
		const Geometry<DeviceAddressType>* geoms = buildGeomInfo.geometries.begin();
		for(uint32_t g = 0; g < geomCount; ++g) {
			auto & geom = geoms[g];
			VkAccelerationStructureGeometryKHR vk_geom = getVkASGeometry(vk_device, geom);
			inoutGeomArray[g] = vk_geom;
		}
	}

	template<typename AddressType>
	static VkAccelerationStructureGeometryKHR getVkASGeometry(VkDevice vk_device, const Geometry<AddressType> & geometry)
	{
		VkAccelerationStructureGeometryKHR ret = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR, nullptr};
		ret.geometryType = getVkGeometryTypeFromGeomType(geometry.type);
		ret.flags = getVkGeometryFlagsFromGeometryFlags(geometry.flags);
		if(E_GEOM_TYPE::EGT_TRIANGLES == geometry.type)
		{
			const auto & triangles = geometry.data.triangles;
			ret.geometry.triangles = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR, nullptr};
			ret.geometry.triangles.vertexFormat = getVkFormatFromFormat(triangles.vertexFormat);
			ret.geometry.triangles.vertexData = getVkDeviceOrHostConstAddr(vk_device, triangles.vertexData);
			ret.geometry.triangles.vertexStride = static_cast<VkDeviceSize>(triangles.vertexStride);
			ret.geometry.triangles.maxVertex = triangles.maxVertex;
			ret.geometry.triangles.indexType = static_cast<VkIndexType>(triangles.indexType); // (Erfan): Converter?
			ret.geometry.triangles.indexData = getVkDeviceOrHostConstAddr(vk_device, triangles.indexData);
			ret.geometry.triangles.transformData = getVkDeviceOrHostConstAddr(vk_device, triangles.transformData);
		}
		else if(E_GEOM_TYPE::EGT_AABBS == geometry.type)
		{
			const auto & aabbs = geometry.data.aabbs;
			ret.geometry.aabbs = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR, nullptr};
			ret.geometry.aabbs.data = getVkDeviceOrHostConstAddr(vk_device, aabbs.data);
			ret.geometry.aabbs.stride = static_cast<VkDeviceSize>(aabbs.stride);
		}
		else if(E_GEOM_TYPE::EGT_INSTANCES == geometry.type)
		{
			const auto & instances = geometry.data.instances;
			ret.geometry.instances = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR, nullptr};
			ret.geometry.instances.data = getVkDeviceOrHostConstAddr(vk_device, instances.data);
			ret.geometry.instances.arrayOfPointers = VK_FALSE; // (Erfan): Something to expose?
		}
		return ret;
	}

	template<typename AddressType>
	static VkDeviceOrHostAddressKHR getVkDeviceOrHostAddr(VkDevice vk_device, const AddressType & addr);
	
	template<typename AddressType>
	static VkDeviceOrHostAddressConstKHR getVkDeviceOrHostConstAddr(VkDevice vk_device, const AddressType & addr);

	static inline VkAccelerationStructureTypeKHR getVkASTypeFromASType(IAccelerationStructure::E_TYPE in) {
		return static_cast<VkAccelerationStructureTypeKHR>(in);
	}
	static inline VkAccelerationStructureCreateFlagsKHR getVkASCreateFlagsFromASCreateFlags(IAccelerationStructure::E_CREATE_FLAGS in) {
		return static_cast<VkAccelerationStructureCreateFlagsKHR>(in);
	}
	static inline VkGeometryTypeKHR getVkGeometryTypeFromGeomType(IAccelerationStructure::E_GEOM_TYPE in) {
		return static_cast<VkGeometryTypeKHR>(in);
	}
	static inline VkGeometryFlagsKHR getVkGeometryFlagsFromGeometryFlags(IAccelerationStructure::E_GEOM_FLAGS in) {
		return static_cast<VkGeometryFlagsKHR>(in);
	}
	static inline VkBuildAccelerationStructureFlagsKHR getVkASBuildFlagsFromASBuildFlags(IAccelerationStructure::E_BUILD_FLAGS in) {
		return static_cast<VkBuildAccelerationStructureFlagsKHR>(in);
	}
	static inline VkBuildAccelerationStructureModeKHR getVkASBuildModeFromASBuildMode(IAccelerationStructure::E_BUILD_MODE in) {
		return static_cast<VkBuildAccelerationStructureModeKHR>(in);
	}


private:
	VkAccelerationStructureKHR m_vkAccelerationStructure;
};

}

#endif
