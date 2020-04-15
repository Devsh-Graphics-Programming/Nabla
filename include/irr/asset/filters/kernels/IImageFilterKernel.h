// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_I_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_I_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/ICPUImage.h"

namespace irr
{
namespace asset
{


class IImageFilterKernel
{
	public:
		IImageFilterKernel(const float* _negative_support, const float* _positive_support) :
			negative_support( _negative_support[0],_negative_support[1],_negative_support[2]),
			positive_support( _positive_support[0],_positive_support[1],_positive_support[2]),
			window_size(core::ceil<core::vectorSIMDf>(negative_support+positive_support)),
			window_strides(1,window_size[0],window_size[0]*window_size[1])
		{}
		IImageFilterKernel(const std::initializer_list<float>& _negative_support, const std::initializer_list<float>& _positive_support) :
			IImageFilterKernel(_negative_support.begin(),_positive_support.begin())
		{}

		virtual bool pIsSeparable() const = 0;
		virtual bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const = 0;

		virtual void pEvaluate(const core::vectorSIMDf& globalPos) const = 0;


		// does conversion from corner sampled to center sampled as well
		inline core::vectorSIMDi32 getWindowMinCoord(const core::vectorSIMDf& unnormCeterSampledCoord, core::vectorSIMDf& cornerSampledCoord) const
		{
			cornerSampledCoord = unnormCeterSampledCoord-core::vectorSIMDf(0.5f,0.5f,0.5f,0.f);
			return core::vectorSIMDi32(core::ceil<core::vectorSIMDf>(cornerSampledCoord-negative_support));
		}
		inline core::vectorSIMDi32 getWindowMinCoord(const core::vectorSIMDf& unnormCeterSampledCoord) const
		{
			core::vectorSIMDf dummy;
			return getWindowMinCoord(unnormCeterSampledCoord,dummy);
		}

		inline const auto& getWindowSize() const
		{
			return window_size;
		}
		inline int32_t getWindowVolume() const
		{
			return window_size[0]*window_size[1]*window_size[2];
		}


		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;
		const core::vectorSIMDf		negative_support;
		const core::vectorSIMDf		positive_support;
		const core::vectorSIMDi32	window_size;
		const core::vectorSIMDi32	window_strides;
};

template<class CRTP, typename value_type>
class CImageFilterKernel : public IImageFilterKernel
{
	public:
		using IImageFilterKernel::IImageFilterKernel;

		inline bool pIsSeparable() const override
		{
			return CRTP::is_separable;
		}
		inline bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const override
		{
			return CRTP::validate(inImage,outImage);
		}


		//
		struct default_sample_functor_t
		{
			inline void operator()(const value_type* windowSample, const core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord)
			{
			}
		};

		inline void pEvaluate(const core::vectorSIMDf& globalPos) const override
		{
			default_sample_functor_t void_functor;
			evaluate(globalPos,void_functor,void_functor);
		}
		
		template<class PreFilter=const default_sample_functor_t, class PostFilter=const default_sample_functor_t>
		inline void evaluate(const core::vectorSIMDf& globalPos, PreFilter& preFilter, PostFilter& postFilter) const
		{
			core::vectorSIMDf offsetGlobalPos;
			const auto startCoord = getWindowMinCoord(globalPos,offsetGlobalPos);
			const auto endCoord = startCoord+window_size;

			value_type windowSample[MaxChannels];

			core::vectorSIMDi32 windowCoord(0,0,0,-1);
			for (auto& z=(windowCoord.z=startCoord.z); z!=endCoord.z; z++)
			for (auto& y=(windowCoord.y=startCoord.y); y!=endCoord.y; y++)
			for (auto& x=(windowCoord.x=startCoord.x); x!=endCoord.x; x++)
			{
				// get position relative to kernel origin, note that it is in reverse (tau-x), in accordance with Mathematical Convolution
				auto relativePosAndFactor = offsetGlobalPos-core::vectorSIMDf(windowCoord);
				evaluateImpl<PreFilter,PostFilter>(preFilter,postFilter,windowSample,relativePosAndFactor,windowCoord);
			}
		}

		template<class PreFilter, class PostFilter>
		void evaluateImpl(PreFilter& preFilter, PostFilter& postFilter, value_type* windowSample, core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord) const;
};

} // end namespace asset
} // end namespace irr

#endif