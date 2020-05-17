#ifndef _IRR_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_
#define _IRR_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_

#include "irrlicht.h"
#include "../ext/LumaMeter/CLumaMeter.h"
#include "../ext/ToneMapper/CGLSLToneMappingBuiltinIncludeLoader.h"

namespace irr
{
namespace ext
{
namespace ToneMapper
{


class CToneMapper : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		enum E_OPERATOR
		{
			EO_REINHARD,
			EO_ACES, // its not full ACES, its one particular ACES from Stephen Hill
			EO_COUNT,
		};
		//
		struct ParamsBase
		{
			inline void setAdaptationFactorFromFrameDelta(float frameDeltaSeconds, float upAdaptationPerSecondLog2=-0.5f, float downAdaptationPerSecondLog2=-0.1f)
			{
				float up = core::exp2(upAdaptationPerSecondLog2*frameDeltaSeconds);
				float down = core::exp2(downAdaptationPerSecondLog2*frameDeltaSeconds);

				upExposureAdaptationFactorAsHalf = core::Float16Compressor::compress(up);
				downExposureAdaptationFactorAsHalf = core::Float16Compressor::compress(down);
			}

			// target+(current-target)*exp(-k*t) == mix(target,current,factor)
			uint16_t upExposureAdaptationFactorAsHalf = 0u;
			uint16_t downExposureAdaptationFactorAsHalf = 0u;
			float lastFrameExtraEV = 0.f;
		};
		struct alignas(16) ReinhardParams_t : ParamsBase
		{
			static inline ReinhardParams_t fromExposure(float EV, float key=0.18f, float WhitePointRelToEV=16.f)
			{
				ReinhardParams_t retval;
				retval.keyAndLinearExposure = key*exp2(EV);
				retval.rcpWhite2 = 1.f/(WhitePointRelToEV*WhitePointRelToEV);
				return retval;
			}
			static inline ReinhardParams_t fromKeyAndBurn(float key, float burn, float AvgLuma, float MaxLuma)
			{
				ReinhardParams_t retval;
				retval.keyAndLinearExposure = key/AvgLuma;
				retval.rcpWhite2 = 1.f/(MaxLuma*retval.keyAndLinearExposure*burn*burn);
				retval.rcpWhite2 *= retval.rcpWhite2;
				return retval;
			}

			float keyAndLinearExposure; // usually 0.18*exp2(exposure)
			float rcpWhite2; // the white is relative to post-exposed luma
		};
		//
		struct alignas(16) ACESParams_t : ParamsBase
		{
			ACESParams_t(float EV, float key=0.18f, float Contrast=1.f) : preGamma(Contrast)
			{
				setExposure(EV,key);
			}

			inline void setExposure(float EV, float key=0.18f)
			{
				exposure = exp2(EV)-log2(key)*(preGamma-1.f);
			}

			float preGamma; // 1.0
		private:
			float exposure; // actualExposure-midGrayLog2*(preGamma-1.0)
		};

		//
		static void registerBuiltinGLSLIncludes(asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo);

		//
		static inline core::SRange<const asset::SPushConstantRange> getDefaultPushConstantRanges(bool usingLumaMeter=false)
		{
			if (usingLumaMeter)
				return CGLSLLumaBuiltinIncludeLoader::getDefaultPushConstantRanges();
			else
				return {nullptr,nullptr};
		}

		//
		static core::SRange<const IGPUDescriptorSetLayout::SBinding> getDefaultBindings(video::IVideoDriver* driver, bool usingLumaMeter=false);

		//
		static inline core::smart_refctd_ptr<video::IGPUPipelineLayout> getDefaultPipelineLayout(video::IVideoDriver* driver, bool usingLumaMeter=false)
		{
			auto pcRange = getDefaultPushConstantRanges(usingLumaMeter);
			auto bindings = getDefaultBindings(driver,usingLumaMeter);
			return driver->createGPUPipelineLayout(
				pcRange.begin(),pcRange.end(),
				driver->createGPUDescriptorSetLayout(bindings.begin(),bindings.end()),nullptr,nullptr,nullptr
			);
		}
		
		//
		static core::smart_refctd_ptr<asset::ICPUSpecializedShader> createShader(
			asset::IGLSLCompiler* compilerToAddBuiltinIncludeTo,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION>& inputColorSpace,
			const std::tuple<asset::E_FORMAT,asset::E_COLOR_PRIMARIES,asset::OPTICO_ELECTRICAL_TRANSFER_FUNCTION>& outputColorSpace,
			E_OPERATOR _operator, bool usingLumaMeter=false, LumaMeter::CLumaMeter::E_METERING_MODE meterMode=LumaMeter::CLumaMeter::EMM_UKNONWN, bool usingTemporalAdaptation=false
		);

		//
		static inline core::smart_refctd_ptr<video::IGPUImageView> createViewForImage(
			video::IVideoDriver* driver, bool usedAsInput,
			core::smart_refctd_ptr<video::IGPUImage>&& image,
			const asset::IImage::SSubresourceRange& subresource
		)
		{
			if (!driver || !image)
				return nullptr;

			auto nativeFormat = image->getCreationParams().format;

			video::IGPUImageView::SCreationParams params;
			params.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
			params.image = std::move(image);
			params.type = video::IGPUImageView::ET_2D_ARRAY;
			params.format = usedAsInput ? getInputViewFormat(nativeFormat):getOutputViewFormat(nativeFormat);
			params.components = {};
			params.subresourceRange = subresource;
			return driver->createGPUImageView(std::move(params));
		}

		// we expect user binds correct pipeline, descriptor sets and pushes the push constants by themselves
		static inline void dispatchHelper(video::IVideoDriver* driver, const vide::IGPUImageView* outputView, bool issueDefaultBarrier=true)
		{
			const auto& params = inputView->getCreationParameters();
			auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
			imgViewSize.w = params.subresourceRange.layerCount;
			
			const core::vectorSIMDu32 workgroupSize(CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE,CGLSLLumaBuiltinIncludeLoader::DISPATCH_SIZE,1,1);
			auto groups = (imgViewSize+workgroupSize-core::vectorSIMDu32(1,1,1,1))/workgroupSize;
			driver->dispatch(groups.x, groups.y, groups.w);

			if (issueDefaultBarrier)
				defaultBarrier();
		}

    private:
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
			_IRR_DEBUG_BREAK_IF(true);
			return asset::EF_UNKNOWN;
		}
		static inline asset::E_FORMAT getOutputViewFormat(asset::E_FORMAT imageFormat)
		{
			// before adding any more formats to the support list consult the `createShader` function
			if (asset::isBlockCompressionFormat(imageFormat))
			{
				// you don't know what you're doing, do you?
				_IRR_DEBUG_BREAK_IF(true);
				return asset::EF_UNKNOWN;
			}
			switch (imageFormat)
			{
				case asset::EF_R8G8B8A8_UNORM:
				case asset::EF_R8G8B8A8_SRGB:
				case asset::EF_A2B10G10R10_UNORM_PACK32:
					return asset::EF_R32_UINT;
					break;
				case asset::EF_R16G16B16A16_UNORM:
				case asset::EF_R16G16B16A16_SFLOAT:
					return asset::EF_R32G32_UINT;
					break;
				default:
					break;
			}
			// other formats not supported yet
			_IRR_DEBUG_BREAK_IF(true);
			return asset::EF_UNKNOWN;
		}

		static void defaultBarrier();
};

}
}
}

#endif
