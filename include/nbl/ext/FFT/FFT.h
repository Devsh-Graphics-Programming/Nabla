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
struct alignas(16) uvec3
{
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
			inline uint getLog2FFTSize()
			{
				return (input_dimensions.w>>3u)&0x1fu;
			}
		};

		struct DispatchInfo_t
		{
			uint32_t workGroupCount[3];
		};

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_SIZE = 256u;
		FFT(video::IDriver* driver, uint32_t maxDimensionSize, bool useHalfStorage = false);

		// returns how many dispatches necessary for computing the FFT and fills the uniform data
		template<bool unconstrainedAxisOrder=true>
		static inline uint32_t buildParameters(
			bool isInverse, uint32_t numChannels, const asset::VkExtent3D& inputDimensions, 
			Parameters_t* outParams, DispatchInfo_t* outInfos, const asset::ISampler::E_TEXTURE_CLAMP* paddingType,
			const asset::VkExtent3D& extraPaddedInputDimensions, bool realInput = false
		)
		{
			uint32_t passesRequired = 0u;

			const auto paddedInputDimensions = padDimensions(extraPaddedInputDimensions);

			using SizeAxisPair = std::tuple<float,uint8_t,uint8_t>;
			std::array<SizeAxisPair,3u> passes;
			if (numChannels)
			{
				for (uint32_t i=0u; i<3u; i++)
				{
					auto dim = (&paddedInputDimensions.width)[i];
					if (dim<2u)
						continue;
					passes[passesRequired++] = {float(dim)/float((&inputDimensions.width)[i]),i,paddingType[i]};
				}
				if (unconstrainedAxisOrder)
					std::sort(passes.begin(),passes.begin()+passesRequired);
			}

			auto computeOutputStride = [](const uvec3& output_dimensions, const auto axis, const auto nextAxis) -> uvec4
			{
				// coord[axis] = 1u
				// coord[nextAxis] = fftLen;
				// coord[otherAxis] = fftLen*dimension[nextAxis];
				uvec4 stride; 
				stride.w = output_dimensions.x*output_dimensions.y*output_dimensions.z;
				for (auto i=0u; i<3u; i++)
				{
					auto& coord = (&stride.x)[i];
					if (i!=axis)
					{
						coord = (&output_dimensions.x)[axis];
						if (i!=nextAxis)
							coord *= (&output_dimensions.x)[nextAxis];
					}
					else
						coord = 1u;
				}
				return stride;
			};

			if (passesRequired)
			{
				uvec3 output_dimensions = {inputDimensions.width,inputDimensions.height,inputDimensions.depth};
				for (uint32_t i=0u; i<passesRequired; i++)
				{
					auto& params = outParams[i];
					params.input_dimensions.x = output_dimensions.x;
					params.input_dimensions.y = output_dimensions.y;
					params.input_dimensions.z = output_dimensions.z;

					const auto passAxis = std::get<1u>(passes[i]);
					const auto paddedAxisLen = (&paddedInputDimensions.width)[passAxis];
					{
						assert(paddingType[i]<=asset::ISampler::ETC_MIRROR);
						params.input_dimensions.w = (isInverse ? 0x80000000u:0x0u)|
													(passAxis<<28u)| // direction
													((numChannels-1u)<<26u)| // max channel
													(hlsl::findMSB(paddedAxisLen)<<3u)| // log2(fftSize)
													uint32_t(std::get<2u>(passes[i]));
					}

					(&output_dimensions.x)[passAxis] = paddedAxisLen;
					if (i)
						params.input_strides = outParams[i-1u].output_strides;
					else // TODO provide an override for input strides
					{
						params.input_strides.x = 1u;
						params.input_strides.y = inputDimensions.width;
						params.input_strides.z = params.input_strides.y * inputDimensions.height;
						params.input_strides.w = params.input_strides.z * inputDimensions.depth;
					}
					params.output_strides = computeOutputStride(output_dimensions,passAxis,std::get<1u>(passes[(i+1u)%passesRequired]));

					auto& dispatch = outInfos[i];
					dispatch.workGroupCount[0] = output_dimensions.x;
					dispatch.workGroupCount[1] = output_dimensions.y;
					dispatch.workGroupCount[2] = output_dimensions.z;
					dispatch.workGroupCount[passAxis] = 1u;
				}
			}

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
			driver->dispatch(dispatchInfo.workGroupCount[0],dispatchInfo.workGroupCount[1],dispatchInfo.workGroupCount[2]);

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
