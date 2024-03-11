// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_S_PIPELINE_CREATION_PARAMS_H_INCLUDED_
#define _NBL_VIDEO_S_PIPELINE_CREATION_PARAMS_H_INCLUDED_


#include "nbl/video/IGPUPipelineLayout.h"


namespace nbl::video
{

// For now, due to API design we implicitly satisfy:
// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-stage-08771
// to:
// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html#VUID-VkPipelineShaderStageCreateInfo-pSpecializationInfo-06849
template<typename PipelineType>
struct SPipelineCreationParams
{
	struct SSpecializationValidationResult
	{
		constexpr static inline uint32_t Invalid = ~0u;
		inline operator bool() const
		{
			return count!=Invalid && dataSize!=Invalid;
		}

		inline SSpecializationValidationResult& operator+=(const SSpecializationValidationResult& other)
		{
			// TODO: check for overflow before adding
			if (*this && other)
			{
				count += other.count;
				dataSize += other.dataSize;
			}
			else
				*this = {};
			return *this;
		}

		uint32_t count = Invalid;
		uint32_t dataSize = Invalid;
	};
	constexpr static inline int32_t NotDerivingFromPreviousPipeline = -1;

	inline bool isDerivative() const
	{
		return basePipelineIndex!=NotDerivingFromPreviousPipeline || basePipeline;
	}

	// If you set this, then we don't take `basePipelineIndex` into account, the pointer takes precedence
	const PipelineType* basePipeline = nullptr;
	int32_t basePipelineIndex = NotDerivingFromPreviousPipeline;
};

}
#endif