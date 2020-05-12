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
																				const float meteringMinUV[2], const float meteringMaxUV[2],
																				float samplingFactor=2.f,
																				float lowerPercentile=0.45f, float upperPercentile=0.55f);

		//
		static void registerBuiltinGLSLIncludes(asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo);

		//
		static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createShader(
			asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
			E_METERING_MODE meterMode, float minLuma=1.f/2048.f, float maxLuma=65536.f
		);

		//
		static core::SRange<IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver);

		//
		static inline void dispatchHelper(video::IVideoDriver* driver, const video::IGPUImageView* inputView, bool issueDefaultBarrier=true)
		{
			const auto& params = inputView->getCreationParameters();
			auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
			imgViewSize.w = params.subresourceRange.layerCount;

			imgViewSize += core::vectorSIMDu32(CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE-1,CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE-1,0,0);
			imgViewSize /= core::vectorSIMDu32(CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE,  CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE,1,1);
			
			driver->dispatch(imgViewSize.x, imgViewSize.y, imgViewSize.w);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

    private:
		CLumaMeter() = delete;
        //~CLumaMeter() = delete;

		static inline Uniforms_t commonBuildParameters(	const asset::VkExtent3D& imageSize,
														const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor)
		{
			Uniforms_t uniforms;
			for (auto i=0; i<2; i++)
			{
				uniforms.meteringWindowScale[i] = (meteringMaxUV[i]-meteringMinUV[i])*samplingFactor/float((&imageSize.width)[i]);
				uniforms.meteringWindowOffset[i] = meteringMinUV[i];
			}
		}

		static void defaultBarrier();
};

inline std::pair<CLumaMeter::Uniforms_t,CLumaMeter::PassInfo_t<CLumaMeter::EMM_GEOM_MEAN> > CLumaMeter::buildParameters(
	const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor
)
{
	PassInfo_t<EMM_GEOM_MEAN> info;
	info.rcpFirstPassWGCount = ;
	return {commonBuildParameters(imageSize,meteringMinUV,meteringMaxUV,samplingFactor),info};
}

inline std::pair<CLumaMeter::Uniforms_t,CLumaMeter::PassInfo_t<CLumaMeter::EMM_MODE> >	CLumaMeter::buildParameters(
	const asset::VkExtent3D& imageSize, const float meteringMinUV[2], const float meteringMaxUV[2], float samplingFactor,
	float lowerPercentile, float upperPercentile
)
{
	PassInfo_t<EMM_MODE> info;
	info.lowerPercentile = lowerPercentile*float(CGLSLLumaBuiltinIncludeLoader::MODE_BIN_COUNT);
	info.upperPercentile = upperPercentile*float(CGLSLLumaBuiltinIncludeLoader::MODE_BIN_COUNT);
	return {commonBuildParameters(imageSize,meteringMinUV,meteringMaxUV,samplingFactor),info};
}

}
}
}

#endif
