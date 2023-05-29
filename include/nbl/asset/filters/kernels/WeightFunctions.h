#ifndef _NBL_ASSET_MATH_FUNCTIONS_H_INCLUDED_
#define _NBL_ASSET_MATH_FUNCTIONS_H_INCLUDED_

#include <limits>
#include <cstdint>

// move later if its useful for anything else
namespace nbl::core::impl
{
template<uint32_t order>
class polynomial_t final : public std::array<int64_t,order+1>
{
        using base_t = std::array<int64_t,order+1>;

    public:
        constexpr polynomial_t() : base_t() {}
        constexpr polynomial_t(base_t _list) : base_t(_list) {}

        template<typename T>
        constexpr T evaluate(const T x) const
        {
            T result = T(base_t::back());
            for (int64_t i=order-1; i>=0ll; i--)
            {
                result *= x;
                result += base_t::operator[](i);
            }
            return result;
        }

        constexpr auto differentiate() const
        {
            polynomial_t<order-1> f_prime;
            for (uint32_t power=1; power<=order; power++)
                f_prime[power-1] = base_t::operator[](power)*power;
            return f_prime;
        }
};
}

namespace nbl::asset
{

struct SDiracFunction final
{
	// We can use std::nextafter after C++23
	constexpr static inline float min_support = -1e-6f;
	constexpr static inline float max_support = +1e-6f;
	constexpr static inline uint32_t k_smoothness = 0;
	constexpr static inline double k_energy[1] = {1.0};

	template<int32_t derivative=0>
	static inline double weight(float x)
	{
		if (x != 0.f)
			return 0.0;

		if constexpr (derivative == 0)
			return std::numeric_limits<double>::infinity();
		else
			return core::nan<double>();
	}
};

// Standard Box function, symmetric, value in the support is 1, integral is 1, so support must be [-1/2,1/2)
struct SBoxFunction final
{
	constexpr static inline float min_support = -0.5f;
	constexpr static inline float max_support = +0.5f;
	constexpr static inline uint32_t k_smoothness = 0;
	constexpr static inline double k_energy[1] = {1.0};

	template <int32_t derivative = 0>
	static inline double weight(float x)
	{
		if (x >= min_support && x < max_support)
		{
			if constexpr (derivative == 0)
				return 1.0;
			else
				return core::nan<double>(); // a bit overkill, but better people don't cut themselves
		}
		return 0.0;
	}
};

// Standard Triangle function, symmetric, peak in the support is 1 and at origin, integral is 1, so support must be [-1,1)
struct STriangleFunction final
{
	constexpr static inline float min_support = -1.f;
	constexpr static inline float max_support = +1.f;
	// Derivative at 0 not defined.
	constexpr static inline uint32_t k_smoothness = 0;
	constexpr static inline double k_energy[1] = {1.0};

	template <int32_t derivative = 0>
	static inline double weight(float x)
	{
		if (x >= min_support && x < max_support)
		{
			if constexpr (derivative == 0)
				return 1.0 - core::abs(x);
			else
				return core::nan<double>(); // a bit overkill, but better people don't cut themselves
		}
		return 0.0;
	}
};

// Truncated Gaussian function, with stddev = 1.0, if you want a different stddev then you need to scale it.
template<class support=std::ratio<3,1>>
struct SGaussianFunction final
{
	constexpr static inline float max_support = double(support::num)/double(support::den);
	constexpr static inline float min_support = -max_support;
	// normally it would be INF, but we overflow on the compile-time polynomials
	constexpr static inline uint32_t k_smoothness = 32;
	// even though function is smooth, integrals of derivatives are always 0 by construction
	constexpr static inline double k_energy[1] = {1.0};

	template <int32_t derivative = 0>
	static inline double weight(float x)
	{
		if (x >= min_support && x < max_support)
		{
			double retval = exp2(-0.72134752*x*x)*core::inversesqrt(2.0*core::PI<double>()) / std::erff(core::inversesqrt<double>(2.0)*max_support);
			if constexpr (derivative != 0)
				retval *= differentialPolynomialFactor<derivative>().evaluate(x);
			return retval;
		}
		return 0.f;
	}
	
private:
	template<int32_t derivative>
	constexpr static polynomial_t<derivative> differentialPolynomialFactor()
	{
		static_assert(derivative>0,"DIFFERENTIATION ONLY!");
		if constexpr (derivative>1)
		{
		    constexpr auto f = differentialPolynomialFactor<derivative-1>();
		    constexpr auto f_prime = f.differentiate();
		    polynomial_t<derivative> retval;
		    for (uint32_t power=0; power<f_prime.size(); power++)
			retval[power] = f_prime[power];
		    for (uint32_t power=0; power<f.size(); power++)
			retval[power+1] -= f[power];
		    return retval;
		}
		else
		    return std::array<int64_t,2>{0,-1};
	}
};

// A standard Mitchell function, the standard has a support of [-2,2] the B and C template parameters are the same ones from the paper
template<class B = std::ratio<1, 3>, class C = std::ratio<1, 3>>
struct SMitchellFunction final
{
	// TODO: are the supports independent of B and C ?
	constexpr static inline float min_support = -2.f;
	constexpr static inline float max_support = +2.f;
	constexpr static inline uint32_t k_smoothness = std::numeric_limits<uint32_t>::max();
	// even though function is infinitely smooth, its 0 valued after the 3rd derivative
	// also any function with finite support will cause its derivatives to have 0 infinite integrals
	constexpr static inline double k_energy[1] = {1.0};

	template <int32_t derivative = 0>
	static inline double weight(float x)
	{
		if (x >= min_support && x < max_support)
		{
			bool neg = x < 0.f;
			x = core::abs(x);

			float retval;
			if constexpr (derivative == 0)
			{
				return core::mix(p0 + x * x * (p2 + x * p3), q0 + x * (q1 + x * (q2 + x * q3)), x >= 1.0);
			}
			else if constexpr (derivative == 1)
			{
				retval = core::mix(x * (2.f * p2 + 3.f * x * p3), q1 + x * (2.f * q2 + 3.f * x * q3), x >= 1.0);
			}
			else if constexpr (derivative == 2)
			{
				retval = core::mix(2.f * p2 + 6.f * p3 * x, 2.f * q2 + 6.f * q3 * x, x >= 1.0);
			}
			else if constexpr (derivative == 3)
			{
				retval = core::mix(6.f * p3, 6.f * q3, x >= 1.0);
			}
			else
				retval = 0.f;

			return neg ? -retval : retval;
		}
		return 0.0;
	}

private:
	static inline constexpr double b = double(B::num) / double(B::den);
	static inline constexpr double c = double(C::num) / double(C::den);
	static inline constexpr double p0 = (6.0 - 2.0 * b) / 6.0;
	static inline constexpr double p2 = (-18.0 + 12.0 * b + 6.0 * c) / 6.0;
	static inline constexpr double p3 = (12.0 - 9.0 * b - 6.0 * c) / 6.0;
	static inline constexpr double q0 = (8.0 * b + 24.0 * c) / 6.0;
	static inline constexpr double q1 = (-12.0 * b - 48.0 * c) / 6.0;
	static inline constexpr double q2 = (6.0 * b + 30.0 * c) / 6.0;
	static inline constexpr double q3 = (-b - 6.0 * c) / 6.0;
};

// Kaiser filter, basically a windowed sinc.
struct SKaiserFunction final
{
	constexpr static inline float min_support = -3.f;
	constexpr static inline float max_support = +3.f;
	// we only implemented the derivative once
	constexpr static inline uint32_t k_smoothness = 1;
	// any function with finite support will cause its derivatives to have 0 infinite integrals
	constexpr static inline double k_energy[1] = {1.0};

	template <int32_t derivative = 0>
	static inline double weight(float x)
	{
		if (x >= min_support && x < max_support)
		{
			const float absMinSupport = core::abs(min_support);

			if constexpr (derivative == 0)
			{
				const auto PI = core::PI<double>();
				return core::sinc(x * PI) * core::KaiserWindow(x, alpha, absMinSupport);
			}
			else if constexpr (derivative == 1)
			{
				const auto PIx = core::PI<double>() * x;
				double f = core::sinc(PIx);
				double df = core::PI<double>() * core::d_sinc(PIx);
				double g = core::KaiserWindow(x, alpha, absMinSupport);
				double dg = core::d_KaiserWindow(x, alpha, absMinSupport);
				return df * g + f * dg;
			}
			else
				return core::nan<double>();
		}
		return 0.0;
	}
	
private:
	// important constant, do not touch, do not tweak
	static inline constexpr float alpha = 3.f;
};

// This is the interface for canonical unscaled 1D weight functions that can be used to create a `CWeightFunction1D`
template<typename T>
concept Function1D = requires(T t, const float x)
{
	{ T::min_support }          -> std::same_as<const float&>;
	{ T::max_support }          -> std::same_as<const float&>;
	{ T::k_smoothness }         -> std::same_as<const uint32_t&>;
	{ T::template weight<0>(x) }-> std::floating_point;
	{ T::k_energy[0] }	    -> std::same_as<const decltype(T::template weight<0>(x))&>;
};


namespace impl
{

template<typename value_type>
class IWeightFunction1D
{
	public:
		using value_t = value_type;
		
		//
		inline void scale(const value_t s)
		{
			assert(s != 0.f);
			m_totalScale *= s;
		}

		// getters
		inline float getMinSupport() const { return m_minSupport; }
		inline float getMaxSupport() const { return m_maxSupport; }
		inline float getInvStretch() const { return m_invStretch; }
		inline value_t getTotalScale() const { return m_totalScale; }

	private:
		inline IWeightFunction1D(const float _minSupport, const float _maxSupport) : m_minSupport(_minSupport), m_maxSupport(_maxSupport) {}
		
		inline float impl_stretch(const float s)
		{
			assert(s != 0.f);

			m_minSupport *= s;
			m_maxSupport *= s;

			const float rcp_s = 1.f / s;
			m_invStretch *= rcp_s;
			
			return rcp_s;
		}


		float m_minSupport;
		float m_maxSupport;
		float m_invStretch = 1.f;
		value_t m_totalScale = 1.f;
};

}

template <Function1D _function_t, uint32_t derivative = 0>
class CWeightFunction1D final : public impl::IWeightFunction1D<decltype(std::declval<function_t>().weight(0.f))>
{
	public:
		using function_t = _function_t;
		constexpr static inline uint32_t k_derivative = derivative;

		static_assert(function_t::k_smoothness>k_derivative);
		constexpr static inline uint32_t k_smoothness = function_t::k_smoothness-k_derivative;

		CWeightFunction1D() : impl::IWeightFunction1D<value_t>(function_t::min_support,function_t::max_support) {}
		
		// Calling: f(x).stretch(2) will obviously give you f(x/2)
		inline void stretch(const float s)
		{
			const auto rcp_s = impl_stretch(s);

			if constexpr (derivative != 0)
				scale(pow(rcp_s,k_derivative));
		}

		// This method will keep the integral of the weight function without derivatives constant.
		inline void stretchAndScale(const float stretchFactor)
		{
			stretch(stretchFactor);
			scale(value_t(1)/stretchFactor);
		}

		inline value_t weight(const float x) const
		{
			return static_cast<double>(m_totalScale*function_t::weight<derivative>(x*m_invStretch));
		}

		// Integral of `weight(x) dx` from -INF to +INF
		inline value_t energy() const
		{
			if constexpr(sizeof(function_t::k_energy)/sizeof(value_t)>k_derivative)
			{
				// normally it would be `scale*invStretch^(derivative+1)*k_energy[k_derivative]`
				// but `scale` already contains precomputed `invStretch^derivative` factor when we call `stretch`
				return m_totalScale*m_invStretch*function_t::k_energy[k_derivative];
			}
			return 0.0;
		}
};

// This is the interface for 1D weight functions that can be used to create a `CChannelIndependentWeightFunction`.
// Current implementations of this interface are:
// - `CWeightFunction1D`
// - `CConvolutionWeightFunction1D`
template<typename T>
concept WeightFunction1D = requires(T t, const float x, const T::value_t s)
{
	std::derived_from<T,IWeightFunction1D<T::value_t>>;

	{ T::k_smoothness }	-> std::same_as<const uint32_t&>;

	{ t.stretch(x) }	-> std::same_as<void>;

	{ t.weight(x) }		-> std::same_as<typename T::value_t>;
};

// This is for detecting `CWeightFunction1D` only
template<typename T>
concept SimpleWeightFunction1D = requires(T t)
{
	WeightFunction1D<T>;
	
	Function1D<typename T::function_t>;
	{ T::k_derivative }	-> std::same_as<const uint32_t&>;
	std::same_as<CWeightFunction1D<typename T::function_t,T::k_derivative>,T>;

	{ t.energy() }		-> std::same_as<typename T::value_t>;
};

} // end namespace nbl::asset
#endif
