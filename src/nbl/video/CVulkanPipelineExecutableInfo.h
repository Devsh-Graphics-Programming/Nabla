#ifndef _NBL_VIDEO_C_VULKAN_PIPELINE_EXECUTABLE_INFO_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_PIPELINE_EXECUTABLE_INFO_H_INCLUDED_

#include "nbl/video/IGPUPipeline.h"
#include "nbl/video/CVulkanDeviceFunctionTable.h"

#include <volk.h>

#include <cstring>

namespace nbl::video
{

inline void populateExecutableInfoFromVulkan(core::vector<IGPUPipelineBase::SExecutableInfo>& outInfo, const CVulkanDeviceFunctionTable* vk, VkDevice vkDevice, VkPipeline vkPipeline, bool includeInternalRepresentations)
{
	VkPipelineInfoKHR pipelineInfo = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr};
	pipelineInfo.pipeline = vkPipeline;

	// Enumerate executables
	uint32_t executableCount = 0;
	if (vk->vk.vkGetPipelineExecutablePropertiesKHR(vkDevice, &pipelineInfo, &executableCount, nullptr) != VK_SUCCESS)
		return;

	if (executableCount == 0)
		return;

	core::vector<VkPipelineExecutablePropertiesKHR> properties(executableCount);
	for (uint32_t i = 0; i < executableCount; ++i)
		properties[i] = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR, nullptr};
	if (vk->vk.vkGetPipelineExecutablePropertiesKHR(vkDevice, &pipelineInfo, &executableCount, properties.data()) != VK_SUCCESS)
		return;

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
		if (vk->vk.vkGetPipelineExecutableStatisticsKHR(vkDevice, &execInfo, &statCount, nullptr) != VK_SUCCESS)
			statCount = 0;

		if (statCount > 0)
		{
			core::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
			for (uint32_t s = 0; s < statCount; ++s)
				stats[s] = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR, nullptr};
			if (vk->vk.vkGetPipelineExecutableStatisticsKHR(vkDevice, &execInfo, &statCount, stats.data()) != VK_SUCCESS)
				statCount = 0;

			if (statCount > 0)
			{
				info.structuredStatistics.resize(statCount);

				// First pass: format name:value pairs (for the human-readable string) and
				// fill structuredStatistics in lockstep so callers can pick whichever view
				// they need without re-parsing.
				core::vector<std::string> nameValues(statCount);
				size_t maxNameValueLen = 0;
				for (uint32_t s = 0; s < statCount; ++s)
				{
					const auto& stat = stats[s];
					auto& outStat = info.structuredStatistics[s];
					outStat.name        = stat.name;
					outStat.description = stat.description;

					std::string value;
					switch (stat.format)
					{
						case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
							outStat.format    = IGPUPipelineBase::SExecutableStatistic::FORMAT::BOOL32;
							outStat.value.b32 = stat.value.b32 != VK_FALSE;
							value = outStat.value.b32 ? "true" : "false";
							break;
						case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
							outStat.format    = IGPUPipelineBase::SExecutableStatistic::FORMAT::INT64;
							outStat.value.i64 = stat.value.i64;
							value = std::to_string(stat.value.i64);
							break;
						case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
							outStat.format    = IGPUPipelineBase::SExecutableStatistic::FORMAT::UINT64;
							outStat.value.u64 = stat.value.u64;
							value = std::to_string(stat.value.u64);
							break;
						case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
							outStat.format    = IGPUPipelineBase::SExecutableStatistic::FORMAT::FLOAT64;
							outStat.value.f64 = stat.value.f64;
							value = std::to_string(stat.value.f64);
							break;
						default:
							// Unknown format: leave structured value zero, keep raw text marker
							value = "<unknown format>";
							break;
					}
					nameValues[s] = std::string(stat.name) + ": " + value;
					maxNameValueLen = std::max(maxNameValueLen, nameValues[s].size());
				}

				// Second pass: emit with aligned columns (unchanged human-readable format)
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
		}

		// Internal representations
		if (includeInternalRepresentations)
		{
			uint32_t irCount = 0;
			if (vk->vk.vkGetPipelineExecutableInternalRepresentationsKHR(vkDevice, &execInfo, &irCount, nullptr) != VK_SUCCESS)
				irCount = 0;

			if (irCount > 0)
			{
				core::vector<VkPipelineExecutableInternalRepresentationKHR> irs(irCount);
				for (uint32_t r = 0; r < irCount; ++r)
					irs[r] = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR, nullptr};

				// First call to get sizes
				if (vk->vk.vkGetPipelineExecutableInternalRepresentationsKHR(vkDevice, &execInfo, &irCount, irs.data()) != VK_SUCCESS)
					continue;

				// Allocate data buffers and second call to get data
				core::vector<core::vector<char>> irData(irCount);
				for (uint32_t r = 0; r < irCount; ++r)
				{
					irData[r].resize(irs[r].dataSize);
					irs[r].pData = irData[r].data();
				}

				if (vk->vk.vkGetPipelineExecutableInternalRepresentationsKHR(vkDevice, &execInfo, &irCount, irs.data()) != VK_SUCCESS)
					continue;

				info.structuredInternalRepresentations.resize(irCount);

				std::string& irStr = info.internalRepresentations;
				for (uint32_t r = 0; r < irCount; ++r)
				{
					auto& outIr = info.structuredInternalRepresentations[r];
					outIr.name        = irs[r].name;
					outIr.description = irs[r].description;
					outIr.isText      = irs[r].isText != VK_FALSE;
					// Text payloads include a trailing NUL per the spec; drop it from the
					// structured copy so asText().size() matches the textual length.
					const size_t rawSize  = irs[r].dataSize;
					const size_t copySize = outIr.isText && rawSize > 0 ? rawSize - 1 : rawSize;
					outIr.data.resize(copySize);
					if (copySize > 0)
						std::memcpy(outIr.data.data(), irs[r].pData, copySize);

					irStr += "---- ";
					irStr += irs[r].name;
					irStr += " ----\n";
					irStr += irs[r].description;
					irStr += "\n";
					if (outIr.isText)
					{
						auto* str = static_cast<const char*>(irs[r].pData);
						irStr.append(str, copySize);
						irStr += "\n";
					}
					else
					{
						irStr += "[binary data, ";
						irStr += std::to_string(rawSize);
						irStr += " bytes]\n";
					}
				}
			}
		}
	}
}

}

#endif
