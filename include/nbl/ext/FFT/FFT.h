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
		static inline DispatchInfo_t buildParameters(Uniforms_t * uniform, asset::VkExtent3D const & inputDimensions, uint32_t workGroupXdim = DEFAULT_WORK_GROUP_X_DIM)
		{
			DispatchInfo_t ret = {};
			if(nullptr != uniform) {
				uniform->dims[0] = inputDimensions.width;
				uniform->dims[1] = inputDimensions.height;
				uniform->dims[2] = inputDimensions.depth;
			}

			ret.workGroupDims[0] = workGroupXdim;
			ret.workGroupDims[1] = 1;
			ret.workGroupDims[2] = 1;

			ret.workGroupCount[0] = core::ceil((inputDimensions.width) / float(workGroupXdim));
			ret.workGroupCount[1] = inputDimensions.height;
			ret.workGroupCount[2] = inputDimensions.depth;
			return ret;
		}

		//
		static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

		//
		static core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver);
		
		//
		static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::IVideoDriver* driver)
		{
			auto pcRange = getDefaultPushConstantRanges();
			auto bindings = getDefaultBindings(driver);
			return driver->createGPUPipelineLayout(
				pcRange.begin(),pcRange.end(),
				driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
			);
		}
		
		//
		static inline size_t getInputBufferSize(asset::VkExtent3D const & inputDimensions, uint32_t data_point_bytes)
		{
			return (inputDimensions.width * inputDimensions.height * inputDimensions.depth) * data_point_bytes; // x2 because -> output is a complex number
		}

		//
		static inline size_t getInputBufferSize(asset::VkExtent3D const & inputDimensions, asset::E_FORMAT format)
		{
			return getInputBufferSize(inputDimensions, asset::getTexelOrBlockBytesize(format));
		}
		
		//
		static inline size_t getOutputBufferSize(asset::VkExtent3D const & inputDimensions, asset::E_FORMAT format)
		{
			return 2 * getInputBufferSize(inputDimensions, format); // x2 because -> output is a complex number
		}
		
		//
		static inline size_t getOutputBufferSize(asset::VkExtent3D const & inputDimensions, uint32_t data_point_bytes)
		{
			return 2 * getInputBufferSize(inputDimensions, data_point_bytes); // x2 because -> output is a complex number
		}
		

		static core::smart_refctd_ptr<video::IGPUSpecializedShader> createShader(video::IVideoDriver* driver, asset::E_FORMAT format);
		
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MAX_DESCRIPTOR_COUNT = 4u;
		static inline void updateDescriptorSet(
			video::IVideoDriver * driver,
			video::IGPUDescriptorSet * set,
			asset::VkExtent3D const & inputDimensions,
			core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> uniformBufferDescriptor)
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

			// Uniform Buffer
			pWrites[0].binding = 0;
			pWrites[0].descriptorType = asset::EDT_UNIFORM_BUFFER;
			pWrites[0].count = 1;
			pInfos[0].desc = uniformBufferDescriptor;
			pInfos[0].buffer.size = sizeof(Uniforms_t);
			pInfos[0].buffer.offset = 0u;

			uint32_t input_count = inputDimensions.width * inputDimensions.height * inputDimensions.depth;

			// Input Buffer 
			pWrites[1].binding = 1;
			pWrites[1].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[2].count = 1;
			pInfos[1].desc = inputBufferDescriptor;
			pInfos[1].buffer.size = inputBufferDescriptor->getSize();
			pInfos[1].buffer.offset = 0u;

			// Output Buffer 
			pWrites[2].binding = 2;
			pWrites[2].descriptorType = asset::EDT_STORAGE_BUFFER;
			pWrites[2].count = 1;
			pInfos[2].desc = outputBufferDescriptor;
			pInfos[2].buffer.size = outputBufferDescriptor->getSize();
			pInfos[2].buffer.offset = 0u;

			driver->updateDescriptorSets(3u, pWrites, 0u, nullptr);
		}

		// we expect user binds correct pipeline, descriptor sets and pushes the push constants by themselves
		static inline void dispatchHelper(video::IVideoDriver* driver, const DispatchInfo_t& dispatchInfo, bool issueDefaultBarrier=true)
		{
			driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);

			if (issueDefaultBarrier)
				defaultBarrier();
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
