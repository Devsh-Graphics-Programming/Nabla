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

	inline float weight(float x, int32_t channel) const
	{
		return (x==0.f) ? std::numeric_limits<float>::infinity() : 0.f;
	}

	static inline constexpr bool has_derivative = false;
};

} // end namespace nbl::asset
#endif