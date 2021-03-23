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

typedef uint32_t uint;
struct alignas(16) uvec3 {
	uint x,y,z;
};
struct alignas(16) uvec4 {
	uint x,y,z,w;
};
#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl";

class FFT : public core::TotalInterface
{
	public:
		struct Parameters_t alignas(16) : nbl_glsl_ext_FFT_Parameters_t
		{
		};
		
		enum class PaddingType : uint8_t
		{
			CLAMP_TO_EDGE = 0,
			FILL_WITH_ZERO = 1,
			// TODO: mirror?
		};

		struct DispatchInfo_t
		{
			uint32_t workGroupCount[3];
		};

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;

		// returns how many dispatches necessary for computing the FFT and fills the uniform data
		static inline uint32_t buildParameters(bool isInverse, uint32_t numChannels, const asset::VkExtent3D& inputDimensions, Parameters_t* outParams, DispatchInfo_t* outInfos, const PaddingType* paddingType)
		{
			uint32_t passesRequired = 0u;

			if (numChannels)
			{
				const auto paddedInputDimensions = padDimensionToNextPOT(inputDimensions);
				for (uint32_t i=0u; i<3u; i++)
				if ((&inputDimensions.width)[i]>1u)
				{
					// TODO: rework
					auto& dispatch = outInfos[passesRequired];
					dispatch.workGroupCount[0] = paddedInputDimensions.width;
					dispatch.workGroupCount[1] = paddedInputDimensions.height;
					dispatch.workGroupCount[2] = paddedInputDimensions.depth;
					dispatch.workGroupCount[i] = 1u;

					auto& params = outParams[passesRequired];
					params.input_dimensions.x = inputDimensions.width;
					params.input_dimensions.y = inputDimensions.height;
					params.input_dimensions.z = inputDimensions.depth;
					{
						// round up to workgroup size if too small
						const uint32_t fftSize = core::max(DEFAULT_WORK_GROUP_SIZE,(&paddedInputDimensions.width)[i]);

						params.input_dimensions.w = (isInverse ? 0x80000000u:0x0u)|
													(i<<28u)| // direction
													((numChannels-1u)<<26u)| // max channel
													(core::findMSB(fftSize)<<3u)| // log2(fftSize)
													uint32_t(paddingType[i]);
					}
					params.input_strides.x = 1u;
					params.input_strides.y = paddedInputDimensions.width;
					params.input_strides.z = params.input_strides.y*paddedInputDimensions.height;
					params.input_strides.w = params.input_strides.z*paddedInputDimensions.depth;
					params.output_strides = params.input_strides;

					passesRequired++;
				}
			}

			if (passesRequired)
				outParams[passesRequired-1u].output_strides = outParams[0].input_strides;

			return passesRequired;
		}

		// TODO: remove?
		static inline asset::VkExtent3D padDimensionToNextPOT(asset::VkExtent3D dimension, asset::VkExtent3D const & minimum_dimension = asset::VkExtent3D{ 1, 1, 1 })
		{
			if(dimension.width < minimum_dimension.width)
				dimension.width = minimum_dimension.width;
			if(dimension.height < minimum_dimension.height)
				dimension.height = minimum_dimension.height;
			if(dimension.depth < minimum_dimension.depth)
				dimension.depth = minimum_dimension.depth;

			dimension.width = core::roundUpToPoT(dimension.width);
			dimension.height = core::roundUpToPoT(dimension.height);
			dimension.depth = core::roundUpToPoT(dimension.depth);

			return dimension;
		}

		//
		static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

		//
		static core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> getDefaultDescriptorSetLayout(video::IVideoDriver* driver);
		
		//
		static core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::IVideoDriver* driver);
		
		// TODO: rework?
		static inline size_t getOutputBufferSize(asset::VkExtent3D const & paddedInputDimensions, uint32_t numChannels)
		{
			assert(core::isPoT(paddedInputDimensions.width) && core::isPoT(paddedInputDimensions.height) && core::isPoT(paddedInputDimensions.depth));
			return (paddedInputDimensions.width * paddedInputDimensions.height * paddedInputDimensions.depth * numChannels) * (sizeof(float) * 2);
		}
		
		static core::smart_refctd_ptr<video::IGPUComputePipeline> getDefaultPipeline(video::IVideoDriver* driver, uint32_t maxDimensionSize);

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

		static inline void dispatchHelper(
			video::IVideoDriver* driver,
			const video::IGPUPipelineLayout* pipelineLayout,
			const Parameters_t& params,
			const DispatchInfo_t& dispatchInfo,
			bool issueDefaultBarrier=true)
		{
			driver->pushConstants(pipelineLayout,video::IGPUSpecializedShader::ESS_COMPUTE,0u,sizeof(Parameters_t),&params);
			driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

		static void defaultBarrier();

	private:
		FFT() = delete;
		//~FFT() = delete;
};


}
}
}

#endif
