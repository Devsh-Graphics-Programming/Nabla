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
		inline int32_t computeWindowSize(int32_t index)
		{
			return int32_t(core::ceil<double>(negative_support[index]+positive_support[index]));
		}

	public:
		IImageFilterKernel(const float* _negative_support, const float* _positive_support) :
			negative_support{ _negative_support[0],_negative_support[1],_negative_support[2]},
			positive_support{ _positive_support[0],_positive_support[1],_positive_support[2]},
			window_size{computeWindowSize(0),computeWindowSize(1),computeWindowSize(2)},
			negative_support_as_vec3(negative_support[0],negative_support[1],negative_support[2],0.f)
		{}
		IImageFilterKernel(const std::initializer_list<float>& _negative_support, const std::initializer_list<float>& _positive_support) :
			IImageFilterKernel(_negative_support.begin(),_positive_support.begin())
		{}

		virtual bool pIsSeparable() const = 0;
		virtual bool pValidate(ICPUImage* inImage, ICPUImage* outImage) const = 0;

		virtual void pEvaluate(void* out, const core::vectorSIMDf& inPos, const void*** slices) const = 0;


		inline core::vectorSIMDi32 getWindowMinCoord(const core::vectorSIMDf& unnormCoord) const
		{
			return core::vectorSIMDi32(core::ceil<core::vectorSIMDf>(unnormCoord-negative_support_as_vec3));
		}


		const float				negative_support[3];
		const float				positive_support[3];
		const int32_t			window_size[3];

	protected:
		const core::vectorSIMDf	negative_support_as_vec3;
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
	
		void evaluate(value_type* out, const core::vectorSIMDf& inPos, const value_type*** slices) const;
		inline void pEvaluate(void* out, const core::vectorSIMDf& inPos, const void*** slices) const override
		{
			evaluate(reinterpret_cast<value_type*>(out),inPos,reinterpret_cast<const value_type***>(slices));
		}
};

} // end namespace asset
} // end namespace irr

#endif