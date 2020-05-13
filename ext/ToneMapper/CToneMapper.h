#ifndef _IRR_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_
#define _IRR_EXT_TONE_MAPPER_C_TONE_MAPPER_INCLUDED_

#include "irrlicht.h"
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
		struct alignas(8) ReinhardParams_t
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
		struct alignas(8) ACESParams_t
		{
			ACESParams_t(float EV, float key=0.18f, float Contrast=1.f) : preGamma(Contrast)
			{
				exposure = exp2(EV)-log2(key)*(preGamma-1.f);
			}

			float preGamma; // 1.0
		private:
			float exposure; // actualExposure-midGrayLog2*(preGamma-1.0)
		};
#if 0
		//
        static core::smart_refctd_ptr<CToneMapper> create(video::IVideoDriver* _driver, asset::E_FORMAT inputFormat, const asset::IGLSLCompiler* compiler);

		bool tonemap(video::IGPUImageView* inputThatsInTheSet, video::IGPUDescriptorSet* set, uint32_t parameterUBOOffset)
		{
			const auto& params = inputThatsInTheSet->getCreationParameters();
			if (params.format != viewFormat)
				return false;
			if (params.image->getCreationParameters().format != format)
				return false;

			auto offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint32_t> >(1u);
			offsets->operator[](0u) = parameterUBOOffset;

			m_driver->bindComputePipeline(computePipeline.get());
			m_driver->bindDescriptorSets(video::EPBP_COMPUTE, pipelineLayout.get(), 3, 1, const_cast<const video::IGPUDescriptorSet**>(&set), &offsets);

			auto imgViewSize = params.image->getMipSize(params.subresourceRange.baseMipLevel);
			imgViewSize /= DISPATCH_SIZE;
			m_driver->dispatch(imgViewSize.x, imgViewSize.y, 1u);
			return true;
		}
    private:
        CToneMapper(video::IVideoDriver* _driver, asset::E_FORMAT inputFormat,
					core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>&& _dsLayout,
					core::smart_refctd_ptr<video::IGPUPipelineLayout>&& _pipelineLayout,
					core::smart_refctd_ptr<video::IGPUComputePipeline>&& _computePipeline);
        ~CToneMapper() = default;

        video::IVideoDriver* m_driver;
		asset::E_FORMAT format;
		asset::E_FORMAT viewFormat;
#endif
};

}
}
}

#endif
