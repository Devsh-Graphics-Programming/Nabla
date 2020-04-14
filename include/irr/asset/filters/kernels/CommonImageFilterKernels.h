// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_COMMON_IMAGE_FILTER_KERNELS_H_INCLUDED__
#define __IRR_COMMON_IMAGE_FILTER_KERNELS_H_INCLUDED__


#include "irr/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>

namespace irr
{
namespace asset
{

class CFloatingPointOnlyImageFilterKernelBase
{
	public:
		using value_type = double;

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			const auto& outParams = inImage->getCreationParameters();
			return !(isIntegerFormat(inParams.format)||isIntegerFormat(outParams.format));
		}

	protected:
		CFloatingPointOnlyImageFilterKernelBase() {}
};

template<class CRTP,class Ratio=std::ratio<1,1> >
class CFloatingPointIsotropicSeparableImageFilterKernelBase : public CImageFilterKernel<CRTP,CFloatingPointOnlyImageFilterKernelBase::value_type>, public CFloatingPointOnlyImageFilterKernelBase
{
		using StaticPolymorphicBase = CImageFilterKernel<CRTP,value_type>;

	public:
		_IRR_STATIC_INLINE_CONSTEXPR bool is_separable = true;
		_IRR_STATIC_INLINE_CONSTEXPR float isotropic_support = float(Ratio::num)/float(Ratio::den);
		_IRR_STATIC_INLINE_CONSTEXPR float symmetric_support[3] = { isotropic_support,isotropic_support,isotropic_support };

		CFloatingPointIsotropicSeparableImageFilterKernelBase() : StaticPolymorphicBase(symmetric_support,symmetric_support) {}
		

		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const CRTP* __this, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(__this), preFilter(_preFilter), postFilter(_postFilter) {}

				void operator()(value_type* windowSample, core::vectorSIMDf& relativePosAndFactor, const core::vectorSIMDi32& globalTexelCoord);

			private:
				const CRTP* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t(reinterpret_cast<const CRTP*>(this),preFilter,postFilter);
		}

	protected:
		inline bool inDomain(const core::vectorSIMDf& inPos) const
		{
			return (abs(inPos)<core::vectorSIMDf(isotropic_support,isotropic_support,isotropic_support,FLT_MAX)).all();
		}
};


class CBoxImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CBoxImageFilterKernel,std::ratio<1,2> >
{
	public:
		inline float weight(const core::vectorSIMDf& inPos) const
		{
			return inDomain(inPos) ? 1.0:0.0;
		}
};

class CTriangleImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CTriangleImageFilterKernel,std::ratio<1,1> >
{
	public:
		inline float weight(const core::vectorSIMDf& inPos) const
		{
			if (inDomain(inPos))
			{
				auto hats = core::vectorSIMDf(1.f,1.f,1.f)-abs(inPos);
				return hats.x * hats.y * hats.z;
			}
			return 0.f;
		}
};

template<uint32_t support=3u>
class CKaiserImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CKaiserImageFilterKernel<support>,std::ratio<support,1> >
{
	public:
		const float alpha = 3.f;

		inline float weight(const core::vectorSIMDf& inPos) const
		{
			if (inDomain(inPos))
			{
				const auto PI = core::PI<core::vectorSIMDf>();
				const auto x = core::abs(inPos);
				auto axisVal = core::sinc(x*PI)*core::KaiserWindow(x,core::vectorSIMDf(alpha),core::vectorSIMDf(isotropic_support));
				return axisVal.x * axisVal.y * axisVal.z;
			}
			return 0.f;
		}
};

template<class B=std::ratio<1,3>, class C=std::ratio<1,3> >
class CMitchellImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CMitchellImageFilterKernel<B,C>,std::ratio<2,1> >
{
	public:
		inline float weight(const core::vectorSIMDf& inPos) const
		{
			if (inDomain(inPos))
			{
				const auto x = core::abs(inPos);
				auto axisVal = core::mix(
									core::vectorSIMDf(p0)+x*x*(core::vectorSIMDf(p2)+x*p3),
									core::vectorSIMDf(q0)+x*(core::vectorSIMDf(q1)+x*(core::vectorSIMDf(q2)+x*q3)),
									x>=core::vectorSIMDf(1.f)
								);
				return axisVal.x * axisVal.y * axisVal.z;
			}
			return 0.f;
		}

	protected:
		_IRR_STATIC_INLINE_CONSTEXPR float b = float(B::num)/float(B::den);
		_IRR_STATIC_INLINE_CONSTEXPR float c = float(C::num)/float(C::den);
		_IRR_STATIC_INLINE_CONSTEXPR float p0 = (6.0f - 2.0f * b) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float p2 = (-18.0f + 12.0f * b + 6.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float p3 = (12.0f - 9.0f * b - 6.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q0 = (8.0f * b + 24.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q1 = (-12.0f * b - 48.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q2 = (6.0f * b + 30.0f * c) / 6.0f;
		_IRR_STATIC_INLINE_CONSTEXPR float q3 = (-b - 6.0f * c) / 6.0f;
};
#undef IRR_ASSET_DECLARE_ISOTROPIC_FILTER

} // end namespace asset
} // end namespace irr

#endif