// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_FLOATING_POINT_SEPARABLE_IMAGE_FILTER_KERNEL_H_INCLUDED_
#define _NBL_ASSET_C_FLOATING_POINT_SEPARABLE_IMAGE_FILTER_KERNEL_H_INCLUDED_

namespace nbl::asset
{

namespace impl
{

template<class WeightFunction1D>
struct weight_function_value_type : protected WeightFunction1D
{
	using type = decltype(std::declval<WeightFunction1D>().operator()(0.f,0));
};

template<class WeightFunction1D>
using weight_function_value_type_t = typename weight_function_value_type<WeightFunction1D>::type;

}

// TODO(achal): It would probably be nice to make a concept here which ensures:
// 1. WeightFunction1D has {min/max}_support and/or their getters.
// kernels that requires pixels and arithmetic to be done in precise floats AND are separable AND have the same kernel function in each dimension AND have a rational support
template<class WeightFunction1D>
class CFloatingPointSeparableImageFilterKernel : public impl::weight_function_value_type_t<WeightFunction1D>
{
public:
	// All kernels are by default, defined on max 4 channels
	static inline constexpr auto MaxChannels = 4;

	// TODO(achal): I don't know why but this makes value_type a float.
	// using value_type = impl::weight_function_value_type_t<WeightFunction1D>; 
	using value_type = double;
	static_assert(std::is_same_v<value_type,double>,"should probably allow `float`s at some point!");

	CFloatingPointSeparableImageFilterKernel(WeightFunction1D&& _func) : func(std::move(_func))
	{
		// The reason we use a ceil for window_size:
		// For a convolution operation, depending upon where you place the kernel center in the output image it can encompass different number of input pixel centers.
		// For example, assume you have a 1D kernel with supports [-3/4, 3/4) and you place this at x=0.5, then kernel weights will be
		// non-zero in [-3/4 + 0.5, 3/4 + 0.5) so there will be only one pixel center (at x=0.5) in the non-zero kernel domain, hence window_size will be 1.
		// But if you place the same kernel at x=0, then the non-zero kernel domain will become [-3/4, 3/4) which now encompasses two pixel centers
		// (x=-0.5 and x=0.5), that is window_size will be 2.
		// Note that the window_size can never exceed 2, in the above case, because for that to happen there should be more than 2 pixel centers in non-zero
		// kernel domain which is not possible given that two pixel centers are always separated by a distance of 1.
		m_windowSize = static_cast<int32_t>(core::ceil<float>(func.getMaxSupport() - func.getMinSupport()));
	}

	static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
	{
		const auto& inParams = inImage->getCreationParameters();
		const auto& outParams = inImage->getCreationParameters();
		return !(isIntegerFormat(inParams.format)||isIntegerFormat(outParams.format));
	}

	//template<typename... Args> // TODO: later if needed
	inline value_type weight(const float x, const uint32_t channel) const
	{
		if ((x >= getMinSupport()) && (x < getMaxSupport()))
			return func.operator()(x, channel);
		else
			return 0.f;
	}

	// given an unnormalized (measured in pixels), center sampled (sample at the center of the pixel) coordinate (origin is at the center of the first pixel),
	// return corner sampled coordinate (origin at the very edge of the first pixel) as well as the
	// corner sampled coordinate of the first pixel that lays inside the kernel's support when centered on the given pixel
	inline int32_t getWindowMinCoord(const float unnormCenterSampledCoord, float& cornerSampledCoord) const
	{
		cornerSampledCoord = unnormCenterSampledCoord - 0.5f;
		return static_cast<int32_t>(core::ceil<float>(cornerSampledCoord + func.getMinSupport()));
	}

	// overload that does not return the cornern sampled coordinate of the given center sampled coordinate
	inline int32_t getWindowMinCoord(const float unnormCeterSampledCoord) const
	{
		float dummy;
		return getWindowMinCoord(unnormCeterSampledCoord, dummy);
	}

	// get the kernel support (measured in pixels)
	inline const int32_t getWindowSize() const { return m_windowSize; }
	inline float getMinSupport() const { return func.getMinSupport(); }
	inline float getMaxSupport() const { return func.getMaxSupport(); }

protected:
	CFloatingPointSeparableImageFilterKernel() {}
		
	WeightFunction1D func;

private:
	int32_t m_windowSize;
};

} // end namespace nbl::asset

#endif
