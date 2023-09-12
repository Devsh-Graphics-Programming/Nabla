#ifndef _NBL_VIDEO_C_VULKAN_QUERY_POOL_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_QUERY_POOL_H_INCLUDED_

#include "nbl/video/IQueryPool.h"

#define VK_NO_PROTOTYPES
#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanQueryPool : public IQueryPool
{
public:
	CVulkanQueryPool(core::smart_refctd_ptr<ILogicalDevice>&& dev, SCreationParams&& _params, VkQueryPool vkQueryPool)
		: IQueryPool(std::move(dev), std::move(_params)), m_queryPool(vkQueryPool)
	{ }

	~CVulkanQueryPool();

public:
	
	inline VkQueryPool getInternalObject() const { return m_queryPool; }
	
	static inline VkQueryType getVkQueryTypeFromQueryType(IQueryPool::E_QUERY_TYPE in)
	{
		switch(in)
		{
			case EQT_OCCLUSION: return VK_QUERY_TYPE_OCCLUSION;
			case EQT_PIPELINE_STATISTICS: return VK_QUERY_TYPE_PIPELINE_STATISTICS;
			case EQT_TIMESTAMP: return VK_QUERY_TYPE_TIMESTAMP;
			case EQT_PERFORMANCE_QUERY: return VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
			case EQT_ACCELERATION_STRUCTURE_COMPACTED_SIZE: return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
			case EQT_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE: return VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
			default:
				assert(false);
				return VK_QUERY_TYPE_MAX_ENUM;
		}
	}
	
	static inline VkQueryPipelineStatisticFlags getVkPipelineStatisticsFlagsFromPipelineStatisticsFlags(IQueryPool::E_PIPELINE_STATISTICS_FLAGS in)
	{
		return static_cast<VkQueryPipelineStatisticFlags>(in);
	}
	
	static inline VkQueryResultFlags getVkQueryResultsFlagsFromQueryResultsFlags(IQueryPool::E_QUERY_RESULTS_FLAGS in)
	{
		return static_cast<VkQueryResultFlags>(in);
	}
	
	static inline VkQueryControlFlags getVkQueryControlFlagsFromQueryControlFlags(IQueryPool::E_QUERY_CONTROL_FLAGS in)
	{
		return static_cast<VkQueryControlFlags>(in);
	}

	static inline VkQueryPoolCreateInfo getVkCreateInfoFromCreationParams(IQueryPool::SCreationParams&& params)
	{
		VkQueryPoolCreateInfo ret = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr};
		ret.flags = 0; // "flags is reserved for future use."
		ret.queryType = getVkQueryTypeFromQueryType(params.queryType);
		ret.queryCount = params.queryCount;
		ret.pipelineStatistics = getVkPipelineStatisticsFlagsFromPipelineStatisticsFlags(params.pipelineStatisticsFlags.value);
		return ret;
	}

private:
	VkQueryPool m_queryPool;
};

}

#endif
