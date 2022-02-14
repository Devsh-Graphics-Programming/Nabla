// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_LUMA_METER_C_LUMA_METER_INCLUDED_
#define _NBL_EXT_LUMA_METER_C_LUMA_METER_INCLUDED_

#include "nabla.h"

namespace nbl
{
namespace ext
{
namespace LumaMeter
{

	
/**
- Overridable Tonemapping Parameter preparation (for OptiX and stuff)
**/
class CLumaMeter : public core::TotalInterface
{
    public:
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_BIN_COUNT = 256u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_BIN_GLOBAL_REPLICATION = 4u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_DESCRIPTOR_COUNT = 3u;

		enum E_METERING_MODE
		{
			EMM_GEOM_MEAN,
			EMM_MEDIAN,
			EMM_COUNT
		};

		//
		struct alignas(16) UniformsBase
		{
			float meteringWindowScale[2];
			float meteringWindowOffset[2];
		};
		template<E_METERING_MODE mode>
		struct Uniforms_t;
		template<>
		struct alignas(256) Uniforms_t<EMM_MEDIAN> : UniformsBase
		{
			uint32_t lowerPercentile;
			uint32_t upperPercentile;
		};
		template<>
		struct alignas(256) Uniforms_t<EMM_GEOM_MEAN> : UniformsBase
		{
			float rcpFirstPassWGCount;
		};

		struct DispatchInfo_t
		{
			uint32_t workGroupDims[3];
			uint32_t workGroupCount[3];
		};
		// returns dispatch size (and wg size in x)
		static inline DispatchInfo_t buildParameters(
			Uniforms_t<EMM_GEOM_MEAN>& uniforms,
			const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor=2.f,
			uint32_t workGroupXdim=DEFAULT_WORK_GROUP_X_DIM
		)
		{
			auto retval = commonBuildParameters(uniforms,imageSize,meteringMinUV,meteringMaxUV,samplingFactor,workGroupXdim);

			uniforms.rcpFirstPassWGCount = 1.f/float(retval.workGroupCount[0]*retval.workGroupCount[1]);

			return retval;
		}
		// previous implementation had percentiles 0.72 and 0.96
		static inline DispatchInfo_t buildParameters(
			Uniforms_t<EMM_MEDIAN>& uniforms,
			const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor=2.f,
			float lowerPercentile=0.45f, float upperPercentile=0.55f,
			uint32_t workGroupXdim=DEFAULT_WORK_GROUP_X_DIM
		)
		{
			auto retval = commonBuildParameters(uniforms,imageSize,meteringMinUV,meteringMaxUV,samplingFactor,workGroupXdim);

			uint32_t totalSampleCount = retval.workGroupCount[0]*retval.workGroupDims[0]*retval.workGroupCount[1]*retval.workGroupDims[1];
			uniforms.lowerPercentile = lowerPercentile*float(totalSampleCount);
			uniforms.upperPercentile = upperPercentile*float(totalSampleCount);

			return retval;
		}

		//
		static core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges();

		//
		static core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::ILogicalDevice* device);

		//
		static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::ILogicalDevice* device)
		{
			auto pcRange = getDefaultPushConstantRanges();
			auto bindings = getDefaultBindings(device);
			return device->createGPUPipelineLayout(
				pcRange.begin(), pcRange.end(),
				device->createGPUDescriptorSetLayout(bindings.begin(), bindings.end()), nullptr, nullptr, nullptr
			);
		}

		//
		static inline size_t getOutputBufferSize(E_METERING_MODE meterMode, uint32_t arrayLayers=1u)
		{
			size_t retval = 0ull;
			switch (meterMode)
			{
				case EMM_GEOM_MEAN:
					retval = 1ull;
					break;
				case EMM_MEDIAN:
					retval = DEFAULT_BIN_COUNT*DEFAULT_BIN_GLOBAL_REPLICATION;
					break;
				default:
					_NBL_DEBUG_BREAK_IF(true);
					break;
			}
			return core::roundUp(2ull*arrayLayers*sizeof(uint32_t)*retval,16ull);
		}

		// Special Note for Optix: minLuma>=0.00000001 and std::get<E_COLOR_PRIMARIES>(inputColorSpace)==ECP_SRGB
		static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createShader(
			asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
			E_METERING_MODE meterMode, float minLuma=1.f/2048.f, float maxLuma=65536.f
		);

		template <E_METERING_MODE mode>
		static inline void updateDescriptorSet(
			video::IGPUDescriptorSet* ds,
			const core::smart_refctd_ptr<video::IGPUBuffer> paramUbo,
			const asset::SBufferRange<video::IGPUBuffer>& outputSSBORange,
			const core::smart_refctd_ptr<video::IGPUImageView> inputImageView,
			video::ILogicalDevice* logicalDevice)
		{
			static_assert(mode < EMM_COUNT, "Invalide meter mode!");

			video::IGPUDescriptorSet::SWriteDescriptorSet writes[DEFAULT_DESCRIPTOR_COUNT] = {};
			video::IGPUDescriptorSet::SDescriptorInfo infos[DEFAULT_DESCRIPTOR_COUNT] = {};

			const asset::E_DESCRIPTOR_TYPE descriptorTypes[DEFAULT_DESCRIPTOR_COUNT] =
			{
				asset::EDT_UNIFORM_BUFFER, // luma input params
				asset::EDT_STORAGE_BUFFER, // luma output buffer
				asset::EDT_COMBINED_IMAGE_SAMPLER // input image
			};

			infos[0].desc = paramUbo;
			infos[0].buffer.offset = 0ull;
			infos[0].buffer.size = sizeof(Uniforms_t<mode>);

			infos[1].desc = outputSSBORange.buffer;
			infos[1].buffer.offset = outputSSBORange.offset;
			infos[1].buffer.size = outputSSBORange.size - infos[1].buffer.offset;

			infos[2].desc = inputImageView;
			infos[2].image.sampler = nullptr;
			infos[2].image.imageLayout = asset::EIL_SHADER_READ_ONLY_OPTIMAL;

			for (uint32_t binding = 0u; binding < DEFAULT_DESCRIPTOR_COUNT; ++binding)
			{
				writes[binding].dstSet = ds;
				writes[binding].binding = binding;
				writes[binding].arrayElement = 0u;
				writes[binding].count = 1u;
				writes[binding].descriptorType = descriptorTypes[binding];
				writes[binding].info = infos + binding;
			}

			logicalDevice->updateDescriptorSets(DEFAULT_DESCRIPTOR_COUNT, writes, 0u, nullptr);
		}

		// we expect user binds correct pipeline, descriptor sets and pushes the push constants by themselves
		static inline void dispatchHelper(
			video::IGPUCommandBuffer* cmdbuf, 
			const DispatchInfo_t& dispatchInfo,
			const asset::E_PIPELINE_STAGE_FLAGS srcStageMask,
			const uint32_t srcBarrierCount,
			const video::IGPUCommandBuffer::SBufferMemoryBarrier* srcBarriers,
			const asset::E_PIPELINE_STAGE_FLAGS dstStageMask,
			const uint32_t dstBarrierCount,
			const video::IGPUCommandBuffer::SBufferMemoryBarrier* dstBarriers)
		{
			if (srcStageMask != asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT && srcBarrierCount)
				cmdbuf->pipelineBarrier(srcStageMask, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, srcBarrierCount, srcBarriers, 0u, nullptr);
			cmdbuf->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);
			if (dstStageMask != asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT && dstBarrierCount)
				cmdbuf->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, dstStageMask, asset::EDF_NONE, 0u, nullptr, dstBarrierCount, dstBarriers, 0u, nullptr);
		}

    private:
		CLumaMeter() = delete;
        //~CLumaMeter() = delete;

		_NBL_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_X_DIM = 16u;

		template<E_METERING_MODE mode>
		static inline DispatchInfo_t commonBuildParameters(Uniforms_t<mode>& uniforms, const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor, uint32_t workGroupXdim=DEFAULT_WORK_GROUP_X_DIM)
		{
			assert(core::isPoT(workGroupXdim));

			DispatchInfo_t retval;
			retval.workGroupDims[0] = {workGroupXdim};
			retval.workGroupDims[1] = {DEFAULT_BIN_COUNT/workGroupXdim};
			retval.workGroupDims[2] = 1;
			retval.workGroupCount[2] = imageSize.depth;
			for (auto i=0; i<2; i++)
			{
				const auto imageDim = float((&imageSize.width)[i]);
				const float range = meteringMaxUV[i]-meteringMinUV[i];

				retval.workGroupCount[i] = core::ceil(imageDim*range/(float(retval.workGroupDims[i]*samplingFactor)));

				uniforms.meteringWindowScale[i] = range/(retval.workGroupCount[i]*retval.workGroupDims[i]);
				uniforms.meteringWindowOffset[i] = meteringMinUV[i];
			}
			return retval;
		}
};


}
}
}

#endif
