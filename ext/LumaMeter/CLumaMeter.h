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
		struct alignas(8) PassInfo_t<EMM_MODE>
		{
			uint32_t lowerPercentile;
			uint32_t upperPercentile;
		};
		template<>
		struct PassInfo_t<EMM_GEOM_MEAN>
		{
			float rcpFirstPassWGCount;
		};

		//
		static std::pair<Uniforms_t,PassInfo_t<EMM_GEOM_MEAN> > buildParameters(const asset::VkExtent3D& imageSize,
																				const float meteringMinUV[2], const float meteringMaxUV[2],
																				float samplingFactor=2.f);
		// previous implementation had percentiles 0.72 and 0.96
		static std::pair<Uniforms_t,PassInfo_t<EMM_MODE> >		buildParameters(const asset::VkExtent3D& imageSize,
																				const float meteringMinUV[2], float samplingFactor=2.f,
																				float lowerPercentile=0.45f, float upperPercentile=0.55f);

		//
		static void registerBuiltinGLSLIncludes(asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo);

		// Special Note for Optix: minLuma>=0.00000001 and std::get<E_COLOR_PRIMARIES>(inputColorSpace)==ECP_SRGB
		static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createShader(
			asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
			E_METERING_MODE meterMode, float minLuma=1.f/2048.f, float maxLuma=65536.f
		);

		//
		static core::SRange<IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver);

		//
		static inline void dispatchHelper(	video::IVideoDriver* driver, const Uniforms_t& uniformData,
											const video::IGPUImageView* inputView, const float meteringMaxUV[2],
											bool issueDefaultBarrier=true)
		{
			const auto& params = inputView->getCreationParameters();
			auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
			imgViewSize.w = params.subresourceRange.layerCount;
			
			auto groups = getWorkGroupCounts(uniformData,meteringMaxUV,imgViewSize);
			driver->dispatch(groups.x, groups.y, groups.z);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

    private:
		CLumaMeter() = delete;
        //~CLumaMeter() = delete;

		static inline Uniforms_t commonBuildParameters(const asset::VkExtent3D& imageSize, const float meteringMinUV[2], float samplingFactor)
		{
			Uniforms_t uniforms;
			for (auto i=0; i<2; i++)
			{
				uniforms.meteringWindowScale[i] = samplingFactor/float((&imageSize.width)[i]);
				uniforms.meteringWindowOffset[i] = meteringMinUV[i];
			}
		}

		static inline core::vector3du32_SIMD getWorkGroupCounts(const Uniforms_t& uniformData, const float meteringMaxUV[2], const core::vectorSIMDu32& extentAndLayers)
		{
			core::vector3du32_SIMD retval(extentAndLayers);
			retval.makeSafe2D();
			for (auto i=0; i<2; i++)
				retval[i] = core::ceil<float>((meteringMaxUV[i]-uniformData.meteringWindowOffset[i])/(float(CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE)*uniformData.meteringWindowScale[i]));
			retval.z = extentAndLayers.w;
			return retval;
		}

		static void defaultBarrier();
};

inline std::pair<CLumaMeter::Uniforms_t,CLumaMeter::PassInfo_t<CLumaMeter::EMM_GEOM_MEAN> > CLumaMeter::buildParameters(
	const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor
)
{
	auto uniforms = commonBuildParameters(imageSize,meteringMinUV,samplingFactor);

	auto groups = getWorkGroupCounts(uniforms,meteringMaxUV,core::vectorSIMDu32(imageSize.width,imageSize.height));

	PassInfo_t<EMM_GEOM_MEAN> info;
	info.rcpFirstPassWGCount = groups.x*groups.y;
	return {uniforms,info};
}

inline std::pair<CLumaMeter::Uniforms_t,CLumaMeter::PassInfo_t<CLumaMeter::EMM_MODE> >	CLumaMeter::buildParameters(
	const asset::VkExtent3D& imageSize, const float meteringMinUV[2], float samplingFactor,
	float lowerPercentile, float upperPercentile
)
{
	PassInfo_t<EMM_MODE> info;
	info.lowerPercentile = lowerPercentile*float(totalSampleCount);
	info.upperPercentile = upperPercentile*float(totalSampleCount);
	return {commonBuildParameters(imageSize,meteringMinUV,samplingFactor),info};
}

}
}
}

#endif
