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
			window_strides(1,window_strides[0],window_strides[0]*window_strides[1])
		{}
		IImageFilterKernel(const std::initializer_list<float>& _negative_support, const std::initializer_list<float>& _positive_support) :
			IImageFilterKernel(_negative_support.begin(),_positive_support.begin())
		{}

		virtual bool pIsSeparable() const = 0;
		virtual bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const = 0;

		virtual void pEvaluate(void* windowData, const core::vectorSIMDf& inPos) const = 0;


		inline core::vectorSIMDi32 getWindowMinCoord(const core::vectorSIMDf& unnormCoord) const
		{
			return core::vectorSIMDi32(core::ceil<core::vectorSIMDf>(unnormCoord-negative_support));
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
			inline void operator()(value_type* windowSample, value_type* filteredSample) const
			{
				std::copy(filteredSample,filteredSample+IImageFilterKernel::MaxChannels,windowSample);
			}
		};

	
		template<class PerSampleFunctor=default_sample_functor_t>
		void evaluate(value_type* windowData, const core::vectorSIMDf& inPos, const PerSampleFunctor& perSample=PerSampleFunctor()) const;
		inline void pEvaluate(void* windowData, const core::vectorSIMDf& inPos) const override
		{
			evaluate(reinterpret_cast<value_type*>(windowData),inPos);
		}
};

} // end namespace asset
} // end namespace irr

#endif