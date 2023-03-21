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
template<class Weight1DFunction, int32_t derivative_order=0>
class CFloatingPointSeparableImageFilterKernel : public CImageFilterKernel<CFloatingPointSeparableImageFilterKernel<Weight1DFunction,derivative_order>,weight_function_value_type_t<Weight1DFunction>>
{
	public:
		using value_type = weight_function_value_type_t<Weight1DFunction>;
		static_assert(std::is_same_v<value_type,double>,"should probably allow `float`s at some point!");
		
	private:
		using this_t = CFloatingPointSeparableImageFilterKernel<Weight1DFunction,derivative_order>;
		using base_t = CImageFilterKernel<this_t,value_type>;

	public:
		CFloatingPointSeparableImageFilterKernel(Weight1DFunction&& _func) :
			StaticPolymorphicBase(
				{func::min_support,func::min_support,func::min_support},
				{func::max_support,func::max_support,func::max_support}
			), func(std::move(_func))
		{}


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
						// we don't support integration (yet)
						if constexpr(derivative_order<0)
						{
							windowSample[i] = core::nan<value_type>();
							continue;
						}
						// its possible that the original kernel which defines the `weight` function was stretched or modified, so a correction factor is applied
						windowSample[i] *= (_this->weight<0>(relativePos,i)*_this->weight<1>(relativePos,i)*_this->weight<2>(relativePos,i))* multipliedScale[i];
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

		virtual inline void stretch(const core::vectorSIMDf&/*vec3*/ s) override
		{
			m_invStretch /= s;
			if constexpr(derivative_order) // a `core::pow` could be useful
				scale(core::vectorSIMDf(pow(s.x,derivative_order),pow(s.y,derivative_order),pow(s.z,derivative_order),1.f));
			base_t::stretch(s);
		}

	protected:
		CFloatingPointSeparableImageFilterKernel() {}
		
		Weight1DFunction func;
		core::vectorSIMDf m_invStretch = {1.f,1.f,1.f,0.f};
		
	private:
		template<class PreFilter, class PostFilter>
		friend struct sample_functor_t;
		
		template<uint32_t dim>
		//template<typename... Args> // TODO: later if needed
		inline value_type weight(const core::vectorSIMDf& relativePos, const uint32_t channel)
		{
			const float ax = relativePos[dim];
			// we dont evaluate `weight` function in children outside the support and just are able to return 0.f
			if (ax<base_t::min_support[dim] || ax>=base_t::max_support[dim]) // TODO: unscrew the convention for negative supports?
				return 0.f;
			const float x = ax*m_invStretch[dim];
			return func.operator()<derivative_order>(x,channel);
		}
};

//
template<class Weight1DFunction>
using CDerivativeImageFilterKernel = CFloatingPointSeparableImageFilterKernel<Weight1DFunction,1>;

} // end namespace nbl::asset

#endif
