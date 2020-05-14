#ifndef _IRR_EXT_LUMA_METER_C_LUMA_METER_INCLUDED_
#define _IRR_EXT_LUMA_METER_C_LUMA_METER_INCLUDED_

#include "irrlicht.h"
#include "../ext/LumaMeter/CGLSLLumaBuiltinIncludeLoader.h"

namespace irr
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
		enum E_METERING_MODE
		{
			EMM_GEOM_MEAN,
			EMM_MODE,
			EMM_COUNT
		};

		//
		struct alignas(16) Uniforms_t
		{
			float meteringWindowScale[2];
			float meteringWindowOffset[2];
		};
		template<E_METERING_MODE mode>
		struct PassInfo_t;
		template<>
		struct alignas(16) PassInfo_t<EMM_MODE>
		{
			uint32_t lowerPercentile;
			uint32_t upperPercentile;
		};
		static_assert(sizeof(PassInfo_t<EMM_MODE>)<=sizeof(Uniforms_t), "PassInfo_t<EMM_MODE> cannot be larger than Uniforms_t!");
		template<>
		struct alignas(16) PassInfo_t<EMM_GEOM_MEAN>
		{
			float rcpFirstPassWGCount;
		};
		static_assert(sizeof(PassInfo_t<EMM_GEOM_MEAN>)<=sizeof(Uniforms_t), "PassInfo_t<EMM_GEOM_MEAN> cannot be larger than Uniforms_t!");

		struct DispatchInfo_t
		{
			uint32_t workGroupDims[3];
			uint32_t workGroupCount[3];
		};
		// returns dispatch size (and wg size in x)
		static inline DispatchInfo_t buildParameters(
			Uniforms_t& uniforms, PassInfo_t<EMM_GEOM_MEAN>& info,
			const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor=2.f,
			uint32_t workGroupXdim=DEFAULT_WORK_GROUP_X_DIM
		)
		{
			auto retval = commonBuildParameters(uniforms,imageSize,meteringMinUV,meteringMaxUV,samplingFactor,workGroupXdim);

			info.rcpFirstPassWGCount = 1.f/float(retval.workGroupCount[0]*retval.workGroupCount[1]);

			return retval;
		}
		// previous implementation had percentiles 0.72 and 0.96
		static inline DispatchInfo_t buildParameters(
			Uniforms_t& uniforms, PassInfo_t<EMM_MODE>& info,
			const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor=2.f,
			float lowerPercentile=0.45f, float upperPercentile=0.55f,
			uint32_t workGroupXdim=DEFAULT_WORK_GROUP_X_DIM
		)
		{
			auto retval = commonBuildParameters(uniforms,imageSize,meteringMinUV,meteringMaxUV,samplingFactor,workGroupXdim);

			uint32_t totalSampleCount = retval.workGroupCount[0]*retval.workGroupDims[0]*retval.workGroupCount[1]*retval.workGroupDims[1];
			info.lowerPercentile = lowerPercentile*float(totalSampleCount);
			info.upperPercentile = upperPercentile*float(totalSampleCount);

			return retval;
		}

		//
		static void registerBuiltinGLSLIncludes(asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo);

		//
		static inline core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges()
		{
			return CGLSLLumaBuiltinIncludeLoader::getDefaultPushConstantRanges();
		}

		//
		static inline core::SRange<const video::IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver)
		{
			return CGLSLLumaBuiltinIncludeLoader::getDefaultBindings(driver);
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
				case EMM_MODE:
					// TODO: should be DEFAULT_BIN_COUNT instead of invocation count
					retval = CGLSLLumaBuiltinIncludeLoader::DEFAULT_INVOCATION_COUNT;
					break;
				default:
					_IRR_DEBUG_BREAK_IF(true);
					break;
			}
			return 2ull*arrayLayers*sizeof(uint32_t)*retval;
		}

		// Special Note for Optix: minLuma>=0.00000001 and std::get<E_COLOR_PRIMARIES>(inputColorSpace)==ECP_SRGB
		static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createShader(
			asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
			E_METERING_MODE meterMode, float minLuma=1.f/2048.f, float maxLuma=65536.f
		);

		// we expect user binds correct pipeline, descriptor sets and pushes the push constants by themselves
		static inline void dispatchHelper(video::IVideoDriver* driver, const DispatchInfo_t& dispatchInfo, bool issueDefaultBarrier=true)
		{
			driver->dispatch(dispatchInfo.workGroupCount[0], dispatchInfo.workGroupCount[1], dispatchInfo.workGroupCount[2]);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

    private:
		CLumaMeter() = delete;
        //~CLumaMeter() = delete;

		_IRR_STATIC_INLINE_CONSTEXPR uint32_t DEFAULT_WORK_GROUP_X_DIM = 16u;

		static inline DispatchInfo_t commonBuildParameters(Uniforms_t& uniforms, const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor, uint32_t workGroupXdim=DEFAULT_WORK_GROUP_X_DIM)
		{
			assert(core::isPoT(workGroupXdim));

			DispatchInfo_t retval;
			retval.workGroupDims[0] = {workGroupXdim};
			retval.workGroupDims[1] = {CGLSLLumaBuiltinIncludeLoader::DEFAULT_INVOCATION_COUNT/workGroupXdim};
			retval.workGroupDims[2] = 1;
			retval.workGroupCount[2] = imageSize.depth;
			for (auto i=0; i<2; i++)
			{
				const auto imageDim = float((&imageSize.width)[i]);
				const float windowSizeUnnorm = imageDim*(meteringMaxUV[i]-meteringMinUV[i]);

				retval.workGroupCount[i] = core::ceil(windowSizeUnnorm/(float(retval.workGroupDims[i]*samplingFactor)));

				uniforms.meteringWindowScale[i] = windowSizeUnnorm/(retval.workGroupCount[i]*retval.workGroupDims[i]);
				uniforms.meteringWindowOffset[i] = meteringMinUV[i];
			}
			return retval;
		}

		static void defaultBarrier();
};


}
}
}

#endif
