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

class FFT final : public core::IReferenceCounted
{
	public:
		struct Parameters_t alignas(16) : nbl_glsl_ext_FFT_Parameters_t
		{
		};

		struct DispatchInfo_t
		{
			uint32_t workGroupCount[3];
		};

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
		FFT(video::IDriver* driver, uint32_t maxDimensionSize, bool useHalfStorage = false);

		// returns how many dispatches necessary for computing the FFT and fills the uniform data
		static inline uint32_t buildParameters(
			bool isInverse, uint32_t numChannels, const asset::VkExtent3D& inputDimensions, 
			Parameters_t* outParams, DispatchInfo_t* outInfos, const asset::ISampler::E_TEXTURE_CLAMP* paddingType,
			const asset::VkExtent3D& extraPaddedInputDimensions
		)
		{
			uint32_t passesRequired = 0u;

			if (numChannels)
			{
				const auto paddedInputDimensions = padDimensions(extraPaddedInputDimensions);
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
						const uint32_t fftSize = (&paddedInputDimensions.width)[i];
						assert(paddingType[i]<=asset::ISampler::ETC_MIRROR);
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
		static inline uint32_t buildParameters(
			bool isInverse, uint32_t numChannels, const asset::VkExtent3D& inputDimensions,
			Parameters_t* outParams, DispatchInfo_t* outInfos, const asset::ISampler::E_TEXTURE_CLAMP* paddingType
		)
		{
			return buildParameters(isInverse,numChannels,inputDimensions,outParams,outInfos,paddingType,inputDimensions);
		}

		static inline asset::VkExtent3D padDimensions(asset::VkExtent3D dimension)
		{
			static_assert(core::isPoT(MINIMUM_FFT_SIZE),"MINIMUM_FFT_SIZE needs to be Power of Two!");
			for (auto i=0u; i<3u; i++)
			{
				auto& coord = (&dimension.width)[i];
				if (coord<=1u)
					continue;
				coord = core::max(core::roundUpToPoT(coord),MINIMUM_FFT_SIZE);
			}
			return dimension;
		}

		//
		static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

		//
		inline auto getDefaultDescriptorSetLayout() const {return m_dsLayout.get();}
		
		//
		inline auto getDefaultPipelineLayout() const {return m_pplnLayout.get();}

		//
		inline auto getDefaultPipeline() const {return m_ppln.get();}

		//
		inline uint32_t getMaxFFTLength() const { return m_maxFFTLen; }
		inline bool usesHalfFloatStorage() const { return m_halfFloatStorage; }
		
		//
		static inline size_t getOutputBufferSize(bool _halfFloatStorage, const asset::VkExtent3D& inputDimensions, uint32_t numChannels, bool realInput=false)
		{
			size_t retval = getOutputBufferSize_impl(inputDimensions,numChannels);
			if (!realInput)
				retval <<= 1u;
			return retval*(_halfFloatStorage ? sizeof(uint16_t):sizeof(uint32_t));
		}
		inline size_t getOutputBufferSize(const asset::VkExtent3D& inputDimensions, uint32_t numChannels, bool realInput = false)
		{
			return getOutputBufferSize(m_halfFloatStorage,inputDimensions,numChannels,realInput);
		}

		static void updateDescriptorSet(
			video::IVideoDriver* driver,
			video::IGPUDescriptorSet* set,
			core::smart_refctd_ptr<video::IGPUBuffer> inputBufferDescriptor,
			core::smart_refctd_ptr<video::IGPUBuffer> outputBufferDescriptor);

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
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MINIMUM_FFT_SIZE = DEFAULT_WORK_GROUP_SIZE<<1u;
		~FFT() {}

		//
		static inline size_t getOutputBufferSize_impl(const asset::VkExtent3D& inputDimensions, uint32_t numChannels)
		{
			const auto paddedInputDimensions = padDimensions(inputDimensions);
			return paddedInputDimensions.width*paddedInputDimensions.height*paddedInputDimensions.depth*numChannels;
		}

		core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_dsLayout;
		core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pplnLayout;
		core::smart_refctd_ptr<video::IGPUComputePipeline> m_ppln;
		uint32_t m_maxFFTLen;
		bool m_halfFloatStorage;
};


}
}
}

#endif
