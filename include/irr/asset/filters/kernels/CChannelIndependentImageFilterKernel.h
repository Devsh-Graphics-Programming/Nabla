// Copyright (C) 2020- Mateusz 'DevSH' Kielan
// This file is part of the "IrrlichtBAW" engine.
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __IRR_C_CHANNEL_INDEPENDENT_IMAGE_FILTER_KERNEL_H_INCLUDED__

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
	protected:
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
class CChannelIndependentImageFilterKernel<KernelR,void,void,void> : 
	public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,void,void,void>>, public impl::CChannelIndependentImageFilterKernelBase, private KernelR
{
		using base_t = CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR, void, void, void>>;
	public:
		using value_type = typename base_t::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 1;

		CChannelIndependentImageFilterKernel(float _negative_support, float _positive_support, KernelR&& kernel_r) : base_t(_negative_support, _positive_support), KernelR(std::move(kernel_r)) {}

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
		inline float d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					if constexpr (KernelR::has_derivative)
					{
						return KernelR::d_weight(x, 0);
					}
					_IRR_DEBUG_BREAK_IF(!KernelR::has_derivative);
					break;
				default:
					break;
			}
			return 0.f;
		}

		IRR_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(base_t)
};

template<class KernelR, class KernelG>
class CChannelIndependentImageFilterKernel<KernelR,KernelG,void,void> : 
	public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,KernelG,void,void>>, public impl::CChannelIndependentImageFilterKernelBase, private KernelR,KernelG
{
		using base_t = CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR, KernelG, void, void>>;

	public:
		using value_type = typename base_t::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 2;

		CChannelIndependentImageFilterKernel(float _negative_support, float _positive_support, KernelR&& kernel_r, KernelG&& kernel_g) : 
			base_t(_negative_support, _positive_support), KernelR(std::move(kernel_r)), KernelG(std::move(kernel_g)) {}

		// pass on any scale
		inline const IImageFilterKernel::UserData* getUserData() const { return haveScale ? &scale : nullptr; }

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

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative||KernelG::has_derivative;
		inline float d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					if constexpr (KernelR::has_derivative)
					{
						return KernelR::d_weight(x, 0);
					}
					_IRR_DEBUG_BREAK_IF(!KernelR::has_derivative);
					break;
				case 1:
					if constexpr (KernelG::has_derivative)
					{
						return KernelG::d_weight(x, 1);
					}
					_IRR_DEBUG_BREAK_IF(!KernelG::has_derivative);
					break;
				default:
					break;
			}
			return 0.f;
		}

		IRR_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(base_t)
};

template<class KernelR, class KernelG, class KernelB>
class CChannelIndependentImageFilterKernel<KernelR,KernelG,KernelB,void> : 
	public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,KernelG,KernelB,void>>, public impl::CChannelIndependentImageFilterKernelBase, private KernelR,KernelG,KernelB
{
		using base_t = CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR, KernelG, KernelB, void>>;
	public:
		using value_type = typename base_t::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 3;

		CChannelIndependentImageFilterKernel(float _negative_support, float _positive_support, KernelR&& kernel_r, KernelG&& kernel_g, KernelB&& kernel_b) :
			base_t(_negative_support, _positive_support), KernelR(std::move(kernel_r)), KernelG(std::move(kernel_g)), KernelB(std::move(kernel_b)) {}

		// pass on any scale
		inline const IImageFilterKernel::UserData* getUserData() const { return haveScale ? &scale : nullptr; }

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

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative||KernelG::has_derivative||KernelB::has_derivative;
		inline float d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					if constexpr (KernelR::has_derivative)
					{
						return KernelR::d_weight(x, 0);
					}
					_IRR_DEBUG_BREAK_IF(!KernelR::has_derivative);
					break;
				case 1:
					if constexpr (KernelG::has_derivative)
					{
						return KernelG::d_weight(x, 1);
					}
					_IRR_DEBUG_BREAK_IF(!KernelG::has_derivative);
					break;
				case 2:
					if constexpr (KernelB::has_derivative)
					{
						return KernelB::d_weight(x, 2);
					}
					_IRR_DEBUG_BREAK_IF(!KernelB::has_derivative);
					break;
				default:
					break;
			}
			return 0.f;
		}

		IRR_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(base_t)
};

template<class KernelR, class KernelG, class KernelB, class KernelA>
class CChannelIndependentImageFilterKernel : 
	public CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR,KernelG,KernelB,KernelA>>, public impl::CChannelIndependentImageFilterKernelBase, private KernelR,KernelG,KernelB,KernelA
{
		using base_t = CFloatingPointSeparableImageFilterKernelBase<CChannelIndependentImageFilterKernel<KernelR, KernelG, KernelB, KernelA>>;

	public:
		using value_type = typename base_t::value_type;

		_IRR_STATIC_INLINE_CONSTEXPR auto MaxChannels = 4;

		CChannelIndependentImageFilterKernel(float _negative_support, float _positive_support, KernelR&& kernel_r, KernelG&& kernel_g, KernelB&& kernel_b, KernelA&& kernel_a) :
			base_t(_negative_support, _positive_support), KernelR(std::move(kernel_r)), KernelG(std::move(kernel_g)), KernelB(std::move(kernel_b)), KernelA(std::move(kernel_a)) {}

		// pass on any scale
		inline const IImageFilterKernel::UserData* getUserData() const { return haveScale ? &scale : nullptr; }

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

		_IRR_STATIC_INLINE_CONSTEXPR bool has_derivative = KernelR::has_derivative||KernelG::has_derivative||KernelB::has_derivative||KernelA::has_derivative;
		inline float d_weight(float x, int32_t channel) const
		{
			switch (channel)
			{
				case 0:
					if constexpr (KernelR::has_derivative)
					{
						return KernelR::d_weight(x, 0);
					}
					_IRR_DEBUG_BREAK_IF(!KernelR::has_derivative);
					break;
				case 1:
					if constexpr (KernelG::has_derivative)
					{
						return KernelG::d_weight(x, 1);
					}
					_IRR_DEBUG_BREAK_IF(!KernelG::has_derivative);
					break;
				case 2:
					if constexpr (KernelB::has_derivative)
					{
						return KernelB::d_weight(x, 2);
					}
					_IRR_DEBUG_BREAK_IF(!KernelB::has_derivative);
					break;
				case 3:
					if constexpr (KernelA::has_derivative)
					{
						return KernelA::d_weight(x, 3);
					}
					_IRR_DEBUG_BREAK_IF(!KernelA::has_derivative);
					break;
				default:
					break;
			}
			#ifdef _IRR_DEBUG
				assert(false);
			#endif
			return 0.f;
		}

		IRR_DECLARE_DEFINE_CIMAGEFILTER_KERNEL_PASS_THROUGHS(base_t)
};


} // end namespace asset
} // end namespace irr

#endif