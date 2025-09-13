#ifndef _NBL_VIDEO_C_VULKAN_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_QUERY_POOL_H_INCLUDED_

#include "nbl/video/IQueryPool.h"

#define VK_NO_PROTOTYPES
#include <volk.h>

namespace nbl::video
{

class CVulkanQueryPool : public IQueryPool
{
	public:
		CVulkanQueryPool(const ILogicalDevice* dev, const SCreationParams& params, const VkQueryPool vkQueryPool)
			: IQueryPool(core::smart_refctd_ptr<const ILogicalDevice>(dev),params), m_queryPool(vkQueryPool) {}
	
		inline VkQueryPool getInternalObject() const { return m_queryPool; }
	
		static inline VkQueryType getVkQueryTypeFrom(const IQueryPool::TYPE in)
		{
			switch(in)
			{
				case TYPE::OCCLUSION: return VK_QUERY_TYPE_OCCLUSION;
				case TYPE::PIPELINE_STATISTICS: return VK_QUERY_TYPE_PIPELINE_STATISTICS;
				case TYPE::TIMESTAMP: return VK_QUERY_TYPE_TIMESTAMP;
//				case TYPE::PERFORMANCE_QUERY: return VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
				case TYPE::ACCELERATION_STRUCTURE_COMPACTED_SIZE: return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
				case TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
				case TYPE::ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS: return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR;
				case TYPE::ACCELERATION_STRUCTURE_SIZE: return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR;
				default:
					assert(false);
					break;
			}
			return VK_QUERY_TYPE_MAX_ENUM;
		}
	
		static inline VkQueryPipelineStatisticFlags getVkPipelineStatisticsFlagsFrom(const IQueryPool::PIPELINE_STATISTICS_FLAGS in)
		{
			return static_cast<VkQueryPipelineStatisticFlags>(in);
		}
	
		static inline VkQueryResultFlags getVkQueryResultsFlagsFrom(const IQueryPool::RESULTS_FLAGS in)
		{
			return static_cast<VkQueryResultFlags>(in);
		}

		static inline VkQueryControlFlags getVkQueryControlFlagsFrom(const IGPUCommandBuffer::QUERY_CONTROL_FLAGS in)
		{
			return static_cast<VkQueryControlFlags>(in);
		}

	private:
		~CVulkanQueryPool();

		VkQueryPool m_queryPool;
};

}

#endif
