// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_FFT_INCLUDED_
#define _NBL_EXT_FFT_INCLUDED_

#include "nabla.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ICPUShader.h"

namespace nbl
{
namespace ext
{
namespace FFT
{

class FFT : public core::TotalInterface
{
	public:

		enum class Direction : uint32_t {
			X = 0,
			Y = 1,
			Z = 2,
		};
		
		enum class PaddingType : uint32_t {
			CLAMP_TO_EDGE = 0,
			FILL_WITH_ZERO = 1,
		};

		enum class DataType {
			SSBO,
			TEXTURE2D,
		};

		struct DispatchInfo_t
		{
			uint32_t workGroupDims[3];
			uint32_t workGroupCount[3];
		};

		struct alignas(16) Uniforms_t 
		{
			uint32_t dims[3];
		};

		// returns dispatch size and fills the uniform data
		static inline DispatchInfo_t buildParameters(
			asset::VkExtent3D const & paddedInputDimensions,
			Direction direction,
			uint32_t num_channels)
		{
			assert(num_channels > 0);
			assert(core::isPoT(paddedInputDimensions.width) && core::isPoT(paddedInputDimensions.height) && core::isPoT(paddedInputDimensions.depth));
			DispatchInfo_t ret = {};

			ret.workGroupDims[0] = DEFAULT_WORK_GROUP_X_DIM;
			ret.workGroupDims[1] = 1;
			ret.workGroupDims[2] = 1;

			if(direction == Direction::X)
			{
				ret.workGroupCount[0] = num_channels;
				ret.workGroupCount[1] = paddedInputDimensions.height;
				ret.workGroupCount[2] = paddedInputDimensions.depth;
			}
			else if(direction == Direction::Y) {
				ret.workGroupCount[0] = paddedInputDimensions.width;
				ret.workGroupCount[1] = num_channels;
				ret.workGroupCount[2] = paddedInputDimensions.depth;
			}
			else if(direction == Direction::Z)
			{
				ret.workGroupCount[0] = paddedInputDimensions.width;
				ret.workGroupCount[1] = paddedInputDimensions.height;
				ret.workGroupCount[2] = num_channels;
			}

			return ret;
		}

		//
		static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

		//
		static core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver, DataType inputType);
		
		//
		static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::IVideoDriver* driver, DataType inputType)
		{
			auto pcRange = getDefaultPushConstantRanges();
			auto bindings = getDefaultBindings(driver, inputType);
			return driver->createGPUPipelineLayout(
				pcRange.begin(),pcRange.end(),
				driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
			);
		}
		
		//
		static inline size_t getOutputBufferSize(asset::VkExtent3D const & paddedInputDimensions, uint32_t dataPointBytes)
		{
			assert(core::isPoT(paddedInputDimensions.width) && core::isPoT(paddedInputDimensions.height) && core::isPoT(paddedInputDimensions.depth));
			return 2 * (paddedInputDimensions.width * paddedInputDimensions.height * paddedInputDimensions.depth) * dataPointBytes; // x2 because -> output is a complex number
		}
		

		//
		static inline size_t getOutputBufferSize(asset::VkExtent3D const & paddedInputDimensions, asset::E_FORMAT format)
		{
			assert(core::isPoT(paddedInputDimensions.width) && core::isPoT(paddedInputDimensions.height) && core::isPoT(paddedInputDimensions.depth));
			return getOutputBufferSize(paddedInputDimensions, asset::getTexelOrBlockBytesize(format));
		}
		

		static core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(video::IVideoDriver* driver, DataType inputType, asset::E_FORMAT format, uint32_t maxPaddedDimensionSize);
		
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MAX_DESCRIPTOR_COUNT = 2u;
		static inline void updateDescriptorSet(
			video::IVideoDriver * driver,
			video::IGPUDescriptorSet * set,
			core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor)
		{
			video::IGPUDescriptorSet::SDescriptorInfo pInfos[MAX_DESCRIPTOR_COUNT];
			video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[MAX_DESCRIPTOR_COUNT];

			for (auto i=0; i< MAX_DESCRIPTOR_COUNT; i++)
			{
				pWrites[i].dstSet = set;
				pWrites[i].arrayElement = 0u;
				pWrites[i].count = 1u;
				pWrites[i].info = pInfos+i;
			}

			// Input Buffer 
			pWrites[0].binding = 0;
			pWrites[0].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[0].count = 1;
			pInfos[0].desc = inputBufferDescriptor;
			pInfos[0].buffer.size = inputBufferDescriptor->getSize();
			pInfos[0].buffer.offset = 0u;

			// Output Buffer 
			pWrites[1].binding = 1;
			pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[1].count = 1;
			pInfos[1].desc = outputBufferDescriptor;
			pInfos[1].buffer.size = outputBufferDescriptor->getSize();
			pInfos[1].buffer.offset = 0u;

			driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
		}

		static inline void updateDescriptorSet(
			video::IVideoDriver * driver,
			video::IGPUDescriptorSet * set,
			core::smart_refctd_ptr<video::IGPUImageView> inputImageDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor)
		{
			video::IGPUDescriptorSet::SDescriptorInfo pInfos[MAX_DESCRIPTOR_COUNT];
			video::IGPUDescriptorSet::SWriteDescriptorSet pWrites[MAX_DESCRIPTOR_COUNT];

			for (auto i=0; i< MAX_DESCRIPTOR_COUNT; i++)
			{
				pWrites[i].dstSet = set;
				pWrites[i].arrayElement = 0u;
				pWrites[i].count = 1u;
				pWrites[i].info = pInfos+i;
			}

			// Input Buffer 
			pWrites[0].binding = 0;
			pWrites[0].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
			pWrites[0].count = 1;
			pInfos[0].desc = inputImageDescriptor;
			pInfos[0].image.sampler = nullptr;
			pInfos[0].image.imageLayout = static_cast<asset::E_IMAGE_LAYOUT>(0u);;

			// Output Buffer 
			pWrites[1].binding = 1;
			pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[1].count = 1;
			pInfos[1].desc = outputBufferDescriptor;
			pInfos[1].buffer.size = outputBufferDescriptor->getSize();
			pInfos[1].buffer.offset = 0u;

			driver->updateDescriptorSets(2u, pWrites, 0u, nullptr);
		}

		static inline void dispatchHelper(
			video::IVideoDriver* driver,
			const DispatchInfo_t& dispatchInfo,
			bool issueDefaultBarrier=true)
		{
			driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

		static inline void pushConstants(
			video::IVideoDriver* driver,
			video::IGPUPipelineLayout * pipelineLayout,
			asset::VkExtent3D const & inputDimension,
			asset::VkExtent3D const & paddedInputDimension,
			Direction direction,
			bool isInverse, 
			PaddingType paddingType)
		{
			uint32_t is_inverse_u = isInverse;
			driver->pushConstants(pipelineLayout, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, 0u, sizeof(uint32_t) * 3, &inputDimension);
			driver->pushConstants(pipelineLayout, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 4, sizeof(uint32_t) * 3, &paddedInputDimension);
			driver->pushConstants(pipelineLayout, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 8, sizeof(uint32_t), &direction);
			driver->pushConstants(pipelineLayout, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 9, sizeof(uint32_t), &is_inverse_u);
			driver->pushConstants(pipelineLayout, nbl::video::IGPUSpecializedShader::ESS_COMPUTE, sizeof(uint32_t) * 10, sizeof(uint32_t), &paddingType);
		}

	private:
		FFT() = delete;
		//~FFT() = delete;

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_X_DIM = 256u;

		static void defaultBarrier();
};


}
}
}

#endif
