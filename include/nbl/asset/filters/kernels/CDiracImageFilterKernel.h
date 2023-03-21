#ifndef _NBL_ASSET_C_DIRAC_IMAGE_FILTER_KERNEL_H_INCLUDED_
#define _NBL_ASSET_C_DIRAC_IMAGE_FILTER_KERNEL_H_INCLUDED_

#include <limits>

namespace nbl::asset
{

struct CDiracFunction
{
	constexpr static inline uint32_t k_smoothness = 0;
	constexpr static inline float min_support = std::nextafter<float>(0.f,-1.f);
	constexpr static inline float max_support = std::nextafter<float>(0.f,+1.f);

	template<int32_t derivative=0>
	inline float operator(float x, uint32_t channel) const
	{
		if (x!=0.f)
			return 0.f;

		if constexpr (derivative == 0)
			return std::numeric_limits<float>::infinity();
		else
		{
			static_assert(false);
			return core::nan<float>();
		}
	}
};

} // end namespace nbl::asset
#endif
