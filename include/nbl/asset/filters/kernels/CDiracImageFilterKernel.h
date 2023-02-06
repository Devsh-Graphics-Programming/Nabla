#ifndef __NBL_ASSET_C_DIRAC_IMAGE_FILTER_KERNEL_H_INCLUDED__
#define __NBL_ASSET_C_DIRAC_IMAGE_FILTER_KERNEL_H_INCLUDED__

#include "nbl/asset/filters/kernels/IImageFilterKernel.h"

#include <ratio>
#include <limits>

namespace nbl::asset
{

class NBL_API CDiracImageFilterKernel : public CFloatingPointIsotropicSeparableImageFilterKernelBase<CDiracImageFilterKernel>
{
	using Base = CFloatingPointIsotropicSeparableImageFilterKernelBase<CDiracImageFilterKernel>;

public:
	CDiracImageFilterKernel() : Base(0.f) {}

	template <uint32_t derivative = 0>
	inline float weight(float x, int32_t channel) const
	{
		if (x == 0.f)
		{
			if constexpr (derivative == 0)
			{
				std::numeric_limits<float>::infinity();
			}
			else
			{
				static_assert(false);
				return core::nan<float>();
			}
		}
		return 0.f;
	}

	static inline constexpr bool has_derivative = false;
};

} // end namespace nbl::asset
#endif