// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_FLOATING_POINT_SEPARABLE_IMAGE_FILTER_KERNEL_H_INCLUDED_
#define _NBL_ASSET_C_FLOATING_POINT_SEPARABLE_IMAGE_FILTER_KERNEL_H_INCLUDED_


#include "nbl/asset/filters/kernels/IImageFilterKernel.h"


namespace nbl::asset
{

namespace impl
{
template<class Weight1DFunction>
struct weight_function_value_type : protected Weight1DFunction
{
	using type = decltype(std::declval<Weight1DFunction>().weight<0>(0.f,0));
};
template<class Weight1DFunction>
using weight_function_value_type_t = typename weight_function_value_type::type;
}

// kernels that requires pixels and arithmetic to be done in precise floats AND are separable AND have the same kernel function in each dimension AND have a rational support
template<class Weight1DFunction, uint32_t derivative_order=0>
class CFloatingPointSeparableImageFilterKernel : public CImageFilterKernel<CFloatingPointSeparableImageFilterKernel<Weight1DFunction>,weight_function_value_type_t<Weight1DFunction>>, protected Weight1DFunction
{
	public:
		using value_type = weight_function_value_type_t<Weight1DFunction>;
	private:
		using this_t = CFloatingPointSeparableImageFilterKernel<Weight1DFunction>;
		using base_t = CImageFilterKernel<this_t,weight_function_value_type_t<Weight1DFunction>>;

	public:
		CFloatingPointSeparableImageFilterKernel(Weight1DFunction&& func) : StaticPolymorphicBase({_negative_support,_negative_support,_negative_support},{_positive_support,_positive_support,_positive_support}) {}

		static_assert(std::is_same_v<value_type,double>,"should probably allow `float`s at some point!");
	
		inline bool pIsSeparable() const override final
		{
			return true;
		}

		static inline bool validate(ICPUImage* inImage, ICPUImage* outImage)
		{
			const auto& inParams = inImage->getCreationParameters();
			const auto& outParams = inImage->getCreationParameters();
			return !(isIntegerFormat(inParams.format)||isIntegerFormat(outParams.format));
		}

		//
		template<class PreFilter, class PostFilter>
		struct sample_functor_t
		{
				sample_functor_t(const this_t* _this_, PreFilter& _preFilter, PostFilter& _postFilter) :
					_this(_this_), preFilter(_preFilter), postFilter(_postFilter) {}

				inline void operator()(value_type* windowSample, core::vectorSIMDf& relativePos, const core::vectorSIMDi32& globalTexelCoord, const core::vectorSIMDf& scale)
				{
					// this is programmable, but usually in the case of a convolution filter it would be loading the values from a temporary and decoded copy of the input image
					preFilter(windowSample, relativePos, globalTexelCoord, multipliedScale);

					// by default there's no optimization so operation is O(SupportExtent^3) even though the filter is separable
					for (int32_t i=0; i<CRTP::MaxChannels; i++)
					{
						if constexpr(derivative_order<0)
						{
							windowSample[i] = core::nan<value_type>();
							continue;
						}
						// its possible that the original kernel which defines the `weight` function was stretched or modified, so a correction factor is applied
						windowSample[i] *= (_this->weight(relativePos.x,i)*_this->weight(relativePos.y,i)*_this->weight(relativePos.z,i))* multipliedScale[i];
						//
						if constexpr(derivative_order>0)
							windowSample[i] *= multipliedStretch[i];
					}

					// this is programmable, but usually in the case of a convolution filter it would be summing the values
					postFilter(windowSample, relativePos, globalTexelCoord, multipliedScale);
				}

			private:
				const this_t* _this;
				PreFilter& preFilter;
				PostFilter& postFilter;
		};

		// this is the function that must be defined for each kernel
		template<class PreFilter, class PostFilter>
		inline auto create_sample_functor_t(PreFilter& preFilter, PostFilter& postFilter) const
		{
			return sample_functor_t<PreFilter,PostFilter>(static_cast<const CRTP*>(this),preFilter,postFilter);
		}

		template<typename... Args>
		inline core::vectorSIMDi32 getWindowMinCoord(Args&&... args) const
		{
			return base_t::getWindowMinCoord(std::forward<Args>(args)...);
		}
		inline const auto& getWindowSize() const
		{
			return base_t::getWindowSize();
		}
		
		template<class PreFilter, class PostFilter>
		inline void evaluate(const core::vectorSIMDf& globalPos, PreFilter& preFilter, PostFilter& postFilter) const
		{
			base_t::evaluate(globalPos, preFilter, postFilter);
		}

		// TODO: virtual overloads of `stretch`, `stretchAndScale` so they appropriately accumulate into `multipliedStretch`

	protected:
		CFloatingPointSeparableImageFilterKernel() {}
		
		core::vectorSIMDf multipliedStretch;
		
	private:
		template<class PreFilter, class PostFilter>
		friend struct sample_functor_t;
		
		template<typename... Args>
		inline value_type weight(Args&&... args)
		{
			// we dont evaluate `weight` function in children outside the support and just are able to return 0.f
			if (x>=base_t::positive_support.x || (-x)>base_t::negative_support.x) // TODO: unscrew the convention for negative supports?
				return 0.f;
			return Weight1DFunction::operator()<derivative_order,Args...>(std::forward<Args>(args)...);
		}
};

} // end namespace nbl::asset

#endif
