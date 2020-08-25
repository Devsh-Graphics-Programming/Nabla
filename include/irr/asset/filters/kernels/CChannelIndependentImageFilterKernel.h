// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_C_DERIVATIVE_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "irr/core/core.h"

#include "irr/asset/filters/kernels/CommonImageFilterKernels.h"

#include <type_traits>

namespace irr
{
namespace asset
{

namespace impl
{

class CChannelIndependentImageFilterKernelBase
{
		const bool haveScale;
		template<class... Kernels>
		static inline bool doesHaveScale(const Kernels&... kernels)
		{
			return IImageFilterKernel::ScaleFactorUserData::cast(kernels.getUserData())&&...;
		}

		const IImageFilterKernel::ScaleFactorUserData scale;
		template<class... Kernels>
		static inline IImageFilterKernel::ScaleFactorUserData computeScale(bool bother, const Kernels&... kernels)
		{
			IImageFilterKernel::ScaleFactorUserData retval(1.f);
			if (bother)
			{
				std::array<const IImageFilterKernel::ScaleFactorUserData,sizeof...(kernels)> userData = {IImageFilterKernel::ScaleFactorUserData::cast(kernels.getUserData())...};
				for (auto i=0; i<userData.size(); i++)
				{
					retval.factor[i] = userData[i]->factor[i];
				}
			}
			return retval;
		}

	public:
		template<class... Kernels>
		CChannelIndependentImageFilterKernelBase(const Kernels&... kernels) : haveScale(doesHaveScale(kernels...)), scale(computeScale(haveScale,kernels...))
		{
		}
};

}

// kernel that is composed of different kernels for each color channel, you can disable the final channels by passing `void`
template<class KernelR, class KernelG, class KernelB, class KernelA>
class CChannelIndependentImageFilterKernel;


template<class KernelR>
class CChannelIndependentImageFilterKernel<KernelR,void,void,void> : public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,void,void,void>>, private KernelR
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 1;

		// pass on any scale
		inline const IImageFilterKernel::UserData* getUserData() const { return haveScale ? &scale:nullptr; }

		inline float weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::weight(x,0);
				default:
					break;
			}
			return 0.f;
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative;
		inline std::enable_if_t<has_derivative,float> d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::d_weight(x,0);
				default:
					break;
			}
			return 0.f;
		}
};

template<class KernelR, class KernelG>
class CChannelIndependentImageFilterKernel<KernelR,KernelG,void,void> : public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,KernelG,void,void>>, private KernelR,KernelG
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 2;

		inline float weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::weight(x,0);
				case 1:
					return KernelG::weight(x,1);
				default:
					break;
			}
			return 0.f;
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative&&KernelG::has_derivative;
		inline std::enable_if_t<has_derivative,float> d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::d_weight(x,0);
				case 1:
					return KernelG::d_weight(x,1);
				default:
					break;
			}
			return 0.f;
		}
};

template<class KernelR, class KernelG, class KernelB>
class CChannelIndependentImageFilterKernel<KernelR,KernelG,KernelB,void> : public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,KernelG,KernelB,void>>, private KernelR,KernelG,KernelB
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 3;

		inline float weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::weight(x, 0);
				case 1:
					return KernelG::weight(x,1);
				case 2:
					return KernelB::weight(x,2);
				default:
					break;
			}
			return 0.f;
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative&&KernelG::has_derivative&&KernelB::has_derivative;
		inline std::enable_if_t<has_derivative,float> d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::d_weight(x,0);
				case 1:
					return KernelG::d_weight(x,1);
				case 2:
					return KernelB::d_weight(x,2);
				default:
					break;
			}
			return 0.f;
		}
};

template<class KernelR, class KernelG, class KernelB, class KernelA>
class CChannelIndependentImageFilterKernel : public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,KernelG,KernelB,KernelA>>, private KernelR,KernelG,KernelB,KernelA
{
	public:
		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;

		inline float weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::weight(x, 0);
				case 1:
					return KernelG::weight(x,1);
				case 2:
					return KernelB::weight(x,2);
				case 3:
					return KernelA::weight(x,3);
				default:
					break;
			}
			#ifdef _IRR_DEBUG
				assert(false);
			#endif
			return 0.f;
		}

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative&&KernelG::has_derivative&&KernelB::has_derivative&&KernelA::has_derivative;
		inline std::enable_if_t<has_derivative,float> d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					return KernelR::d_weight(x,0);
				case 1:
					return KernelG::d_weight(x,1);
				case 2:
					return KernelB::d_weight(x,2);
				case 3:
					return KernelA::d_weight(x,3);
				default:
					break;
			}
			#ifdef _IRR_DEBUG
				assert(false);
			#endif
			return 0.f;
		}
};


} // end namespace asset
} // end namespace irr

#endif