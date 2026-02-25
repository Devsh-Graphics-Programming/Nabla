#ifndef _NBL_VIDEO_C_VULKAN_PIPELINE_EXECUTABLE_INFO_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_PIPELINE_EXECUTABLE_INFO_H_INCLUDED_

#include "nbl/video/IGPUPipeline.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"

#include <volk.h>

namespace nbl::video
{

inline void populateExecutableInfoFromVulkan(core::vector<IGPUPipelineBase::SExecutableInfo>& outInfo, const CVulkanDeviceFunctionTable* vk, VkDevice vkDevice, VkPipeline vkPipeline, bool includeInternalRepresentations)
{
	VkPipelineInfoKHR pipelineInfo = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr};
	pipelineInfo.pipeline = vkPipeline;

	// Enumerate executables
	uint32_t executableCount = 0;
	vk->vk.vkGetPipelineExecutablePropertiesKHR(vkDevice, &pipelineInfo, &executableCount, nullptr);

	if (executableCount == 0)
		return;

	core::vector<VkPipelineExecutablePropertiesKHR> properties(executableCount);
	for (uint32_t i = 0; i < executableCount; ++i)
		properties[i] = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR, nullptr};
	vk->vk.vkGetPipelineExecutablePropertiesKHR(vkDevice, &pipelineInfo, &executableCount, properties.data());

	outInfo.resize(executableCount);

	for (uint32_t i = 0; i < executableCount; ++i)
	{
		const auto& prop = properties[i];
		auto& info = outInfo[i];

		info.name = prop.name;
		info.description = prop.description;
		info.stages = static_cast<hlsl::ShaderStage>(prop.stages);
		info.subgroupSize = prop.subgroupSize;

		VkPipelineExecutableInfoKHR execInfo = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr};
		execInfo.pipeline = vkPipeline;
		execInfo.executableIndex = i;

		uint32_t statCount = 0;
		vk->vk.vkGetPipelineExecutableStatisticsKHR(vkDevice, &execInfo, &statCount, nullptr);

		if (statCount > 0)
		{
			core::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
			for (uint32_t s = 0; s < statCount; ++s)
				stats[s] = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR, nullptr};
			vk->vk.vkGetPipelineExecutableStatisticsKHR(vkDevice, &execInfo, &statCount, stats.data());

			// First pass: format name:value pairs and find max width for alignment
			core::vector<std::string> nameValues(statCount);
			size_t maxNameValueLen = 0;
			for (uint32_t s = 0; s < statCount; ++s)
			{
				const auto& stat = stats[s];
				std::string value;
				switch (stat.format)
				{
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
						value = stat.value.b32 ? "true" : "false";
						break;
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
						value = std::to_string(stat.value.i64);
						break;
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
						value = std::to_string(stat.value.u64);
						break;
					case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
						value = std::to_string(stat.value.f64);
						break;
					default:
						value = "<unknown format>";
						break;
				}
				nameValues[s] = std::string(stat.name) + ": " + value;
				maxNameValueLen = std::max(maxNameValueLen, nameValues[s].size());
			}

			// Second pass: emit with aligned columns
			std::string& statsStr = info.statistics;
			for (uint32_t s = 0; s < statCount; ++s)
			{
				statsStr += nameValues[s];
				statsStr.append(maxNameValueLen - nameValues[s].size() + 4, ' ');
				statsStr += "// ";
				statsStr += stats[s].description;
				statsStr += "\n";
			}
		}

		// Internal representations
		if (includeInternalRepresentations)
		{
			uint32_t irCount = 0;
			vk->vk.vkGetPipelineExecutableInternalRepresentationsKHR(vkDevice, &execInfo, &irCount, nullptr);

			if (irCount > 0)
			{
				core::vector<VkPipelineExecutableInternalRepresentationKHR> irs(irCount);
				for (uint32_t r = 0; r < irCount; ++r)
					irs[r] = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR, nullptr};

				// First call to get sizes
				vk->vk.vkGetPipelineExecutableInternalRepresentationsKHR(vkDevice, &execInfo, &irCount, irs.data());

				// Allocate data buffers and second call to get data
				core::vector<core::vector<char>> irData(irCount);
				for (uint32_t r = 0; r < irCount; ++r)
				{
					irData[r].resize(irs[r].dataSize);
					irs[r].pData = irData[r].data();
				}

				vk->vk.vkGetPipelineExecutableInternalRepresentationsKHR(vkDevice, &execInfo, &irCount, irs.data());

				std::string& irStr = info.internalRepresentations;
				for (uint32_t r = 0; r < irCount; ++r)
				{
					irStr += "---- ";
					irStr += irs[r].name;
					irStr += " ----\n";
					irStr += irs[r].description;
					irStr += "\n";
					if (irs[r].isText)
					{
						auto* str = static_cast<const char*>(irs[r].pData);
						irStr.append(str, irs[r].dataSize > 0 ? irs[r].dataSize - 1 : 0);
						irStr += "\n";
					}
					else
					{
						irStr += "[binary data, ";
						irStr += std::to_string(irs[r].dataSize);
						irStr += " bytes]\n";
					}
				}
			}
		}
	}
}

}

#endif
