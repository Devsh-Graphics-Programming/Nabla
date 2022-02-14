// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_
#define _NBL_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_

#include "nabla.h"
#include "../../../nbl/ext/LumaMeter/CLumaMeter.h"

namespace nbl::ext::ToneMapper
{

class CToneMapper : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORKGROUP_DIM = 16u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_MAX_DESCRIPTOR_COUNT = 4u;

		enum E_OPERATOR
		{
			EO_REINHARD,
			EO_ACES, // its not full ACES, its one particular ACES from Stephen Hill
			EO_COUNT,
		};
		//
		struct ParamsBase
		{
				inline void setAdaptationFactorFromFrameDelta(float frameDeltaSeconds, float upAdaptationPerSecondLog2=-1.1f, float downAdaptationPerSecondLog2=-0.2f)
				{
					float up = core::exp2(upAdaptationPerSecondLog2*frameDeltaSeconds);
					float down = core::exp2(downAdaptationPerSecondLog2*frameDeltaSeconds);

					upExposureAdaptationFactorAsHalf = core::Float16Compressor::compress(up);
					downExposureAdaptationFactorAsHalf = core::Float16Compressor::compress(down);
				}

				uint16_t lastFrameExtraEVAsHalf[2] = { 0u,0u };
			protected:
				// target+(current-target)*exp(-k*t) == mix(target,current,factor)
				uint16_t upExposureAdaptationFactorAsHalf = 0u;
				uint16_t downExposureAdaptationFactorAsHalf = 0u;
		};
		//
		template<E_OPERATOR _operator>
		struct Params_t;
		template<>
		struct alignas(256) Params_t<EO_REINHARD> : ParamsBase
		{
				Params_t(float EV, float key=0.18f, float WhitePointRelToEV=16.f)
				{
					keyAndLinearExposure = key*exp2(EV);
					rcpWhite2 = 1.f/(WhitePointRelToEV*WhitePointRelToEV);
				}

				float keyAndLinearExposure; // usually 0.18*exp2(exposure)
				float rcpWhite2; // the white is relative to post-exposed luma
		};
		template<>
		struct alignas(256) Params_t<EO_ACES> : ParamsBase
		{
				Params_t(float EV, float key=0.18f, float Contrast=1.f) : gamma(Contrast)
				{
					setExposure(EV,key);
				}

				inline void setExposure(float EV, float key=0.18f)
				{
					constexpr float reinhardMatchCorrection = 0.77321666f; // middle grays get exposed to different values between tonemappers given the same key
					exposure = EV+log2(key*reinhardMatchCorrection);
				}

				float gamma; // 1.0
			private:
				float exposure; // actualExposure+midGrayLog2
		};

		//
		static inline core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges(bool usingLumaMeter=false)
		{
			if (usingLumaMeter)
				return LumaMeter::CLumaMeter::getDefaultPushConstantRanges();
			else
				return {nullptr,nullptr};
		}

		//
		static core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::ILogicalDevice* device, bool usingLumaMeter=false);

		//
		static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::ILogicalDevice* device, bool usingLumaMeter=false)
		{
			auto pcRange = getDefaultPushConstantRanges(usingLumaMeter);
			auto bindings = getDefaultBindings(device,usingLumaMeter);
			return device->createGPUPipelineLayout(
				pcRange.begin(),pcRange.end(),
				device->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
			);
		}

		//
		template<E_OPERATOR _operator, LumaMeter::CLumaMeter::E_METERING_MODE MeterMode=LumaMeter::CLumaMeter::EMM_COUNT>
		static inline size_t getParameterBufferSize(uint32_t arrayLayers=1u)
		{
			size_t retval = sizeof(Params_t<_operator>);
			if (MeterMode<LumaMeter::CLumaMeter::EMM_COUNT)
				retval += LumaMeter::CLumaMeter::getOutputBufferSize(MeterMode,arrayLayers);
			return retval;
		}

		//
		template <ext::LumaMeter::CLumaMeter::E_METERING_MODE MeterMode = ext::LumaMeter::CLumaMeter::EMM_COUNT>
		static inline void updateDescriptorSet(
			video::ILogicalDevice* logicalDevice,
			video::IGPUDescriptorSet* ds,
			const core::smart_refctd_ptr<video::IGPUImageView> outImageView,
			const core::smart_refctd_ptr<video::IGPUBuffer> paramsSSBO,
			const core::smart_refctd_ptr<video::IGPUImageView> inputImageView,
			const core::smart_refctd_ptr<video::IGPUBuffer> lumaParamsUbo = nullptr)
		{
			video::IGPUDescriptorSet::SWriteDescriptorSet writes[DEFAULT_MAX_DESCRIPTOR_COUNT] = {};
			video::IGPUDescriptorSet::SDescriptorInfo infos[DEFAULT_MAX_DESCRIPTOR_COUNT] = {};

			const bool usingLumaMeter = MeterMode < ext::LumaMeter::CLumaMeter::EMM_COUNT;

			uint32_t descriptorCount = ~0u;
			asset::E_DESCRIPTOR_TYPE descriptorTypes[DEFAULT_MAX_DESCRIPTOR_COUNT] = {};
			if (!usingLumaMeter)
			{
				descriptorCount = 3u;

				// output image
				descriptorTypes[0] = asset::EDT_STORAGE_IMAGE;
				infos[0].desc = outImageView;
				infos[0].image.sampler = nullptr;
				infos[0].image.imageLayout = asset::EIL_GENERAL;

				// parameter buffer
				descriptorTypes[1] = asset::EDT_STORAGE_BUFFER;
				infos[1].desc = paramsSSBO;
				infos[1].buffer.offset = 0ull;
				infos[1].buffer.size = paramsSSBO->getCachedCreationParams().declaredSize;

				// input image
				descriptorTypes[2] = asset::EDT_COMBINED_IMAGE_SAMPLER;
				infos[2].desc = inputImageView;
				infos[2].image.sampler = nullptr;
				infos[2].image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;
			}
			else
			{
				const auto& lumaMeterBindings = ext::LumaMeter::CLumaMeter::getDefaultBindings(logicalDevice);
				const uint32_t lumaMeterBindingCount = static_cast<uint32_t>(lumaMeterBindings.size());
				assert(lumaMeterBindingCount < DEFAULT_MAX_DESCRIPTOR_COUNT);
				assert(lumaParamsUbo);

				descriptorCount = lumaMeterBindingCount + 1u;

				// luma meter input params
				descriptorTypes[0] = lumaMeterBindings.begin()[0].type;
				infos[0].desc = lumaParamsUbo;
				infos[0].buffer.offset = 0ull;
				infos[0].buffer.size = lumaParamsUbo->getCachedCreationParams().declaredSize;

				// tonemapping params and luma output buffer
				descriptorTypes[1] = lumaMeterBindings.begin()[1].type;
				infos[1].desc = paramsSSBO;
				infos[1].buffer.offset = 0ull;
				infos[1].buffer.size = paramsSSBO->getCachedCreationParams().declaredSize;

				// input image
				descriptorTypes[2] = lumaMeterBindings.begin()[2].type;
				infos[2].desc = inputImageView;
				infos[2].image.sampler = nullptr;
				infos[2].image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;

				// output image
				descriptorTypes[3] = asset::EDT_STORAGE_IMAGE;
				infos[3].desc = outImageView;
				infos[3].image.sampler = nullptr;
				infos[3].image.imageLayout = asset::EIL_GENERAL;
			}

			for (uint32_t binding = 0u; binding < descriptorCount; ++binding)
			{
				writes[binding].dstSet = ds;
				writes[binding].binding = binding;
				writes[binding].count = 1u;
				writes[binding].descriptorType = descriptorTypes[binding];
				writes[binding].arrayElement = 0u;
				writes[binding].info = infos + binding;
			}

			logicalDevice->updateDescriptorSets(descriptorCount, writes, 0u, nullptr);
		}
		
		//
		static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createShader(
			asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::OPTICO_ELECTRICAL_TRANSFER_FUNCTION>& outputColorSpace,
			E_OPERATOR _operator,
			bool usingLumaMeter=false, LumaMeter::CLumaMeter::E_METERING_MODE meterMode=LumaMeter::CLumaMeter::EMM_COUNT, float minLuma=core::nan<float>(), float maxLuma=core::nan<float>(),
			bool usingTemporalAdaptation=false
		);

		// we expect user binds correct pipeline, descriptor sets and pushes the push constants by themselves
		static inline void dispatchHelper(
			video::IGPUCommandBuffer* cmdbuf,
			const video::IGPUImageView* outputView,
			const asset::E_PIPELINE_STAGE_FLAGS srcStageMask,
			const uint32_t srcImageBarrierCount,
			const video::IGPUCommandBuffer::SImageMemoryBarrier* srcImageBarriers,
			const asset::E_PIPELINE_STAGE_FLAGS dstStageMask,
			const uint32_t dstImageBarrierCount,
			const video::IGPUCommandBuffer::SImageMemoryBarrier* dstImageBarriers,
			const bool usingTemporalAdaptation = false,
			uint32_t workGroupSizeX=DEFAULT_WORKGROUP_DIM,
			uint32_t workGroupSizeY=DEFAULT_WORKGROUP_DIM)
		{
			const auto& params = outputView->getCreationParameters();
			auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
			imgViewSize.w = params.subresourceRange.layerCount;
			
			const core::vectorSIMDu32 workgroupSize(workGroupSizeX,workGroupSizeY,1,1);
			auto groups = (imgViewSize+workgroupSize-core::vectorSIMDu32(1,1,1,1))/workgroupSize;

			if (srcStageMask != asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT && srcImageBarrierCount)
				cmdbuf->pipelineBarrier(srcStageMask, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, srcImageBarrierCount, srcImageBarriers);
			cmdbuf->dispatch(groups.x, groups.y, groups.w);
			if (dstStageMask != asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT && dstImageBarrierCount)
				cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, dstStageMask, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, dstImageBarrierCount, dstImageBarriers);
		}

    // private:
		// this can probably be removed now
		static inline asset::E_FORMAT getInputViewFormat(asset::E_FORMAT imageFormat)
		{
			// before adding any more formats to the support list consult the `createShader` function
			switch (imageFormat)
			{
				case asset::EF_B10G11R11_UFLOAT_PACK32:
				case asset::EF_E5B9G9R9_UFLOAT_PACK32:
					return asset::EF_R32_UINT;
					break;
				case asset::EF_R16G16B16A16_SFLOAT:
					return asset::EF_R32G32_UINT;
					break;
				case asset::EF_R32G32B32A32_SFLOAT:
				case asset::EF_R64G64B64A64_SFLOAT:
				case asset::EF_BC6H_SFLOAT_BLOCK:
				case asset::EF_BC6H_UFLOAT_BLOCK:
					return imageFormat;
					break;
				default:
					break;
			}
			// the input format has to be HDR for ths to make sense!
			_NBL_DEBUG_BREAK_IF(true);
			return asset::EF_UNKNOWN;
		}
		static inline asset::E_FORMAT getOutputViewFormat(asset::E_FORMAT imageFormat)
		{
			// before adding any more formats to the support list consult the `createShader` function
			if (asset::isBlockCompressionFormat(imageFormat))
			{
				// you don't know what you're doing, do you?
				_NBL_DEBUG_BREAK_IF(true);
				return asset::EF_UNKNOWN;
			}
			switch (imageFormat)
			{
				case asset::EF_R8G8B8A8_UNORM:
				case asset::EF_R8G8B8A8_SRGB:
				case asset::EF_B8G8R8A8_SRGB:
				case asset::EF_A2B10G10R10_UNORM_PACK32:
					return asset::EF_R32_UINT;
				case asset::EF_R16G16B16A16_UNORM:
				case asset::EF_R16G16B16A16_SFLOAT:
					return asset::EF_R32G32_UINT;
				default:
					break;
			}
			// other formats not supported yet
			_NBL_DEBUG_BREAK_IF(true);
			return asset::EF_UNKNOWN;
		}
};

}

#endif
