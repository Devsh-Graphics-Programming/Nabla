#ifndef _NBL_ASSET_MATH_FUNCTIONS_H_INCLUDED_
#define _NBL_ASSET_MATH_FUNCTIONS_H_INCLUDED_

#include <limits>
#include <cstdint>

namespace nbl::asset
{

struct SDiracFunction
{
	// We can use std::nextafter after C++23
	constexpr static inline float min_support = -1e-6f;
	constexpr static inline float max_support = +1e-6f;
	constexpr static inline uint32_t k_smoothness = 0;

	template<int32_t derivative=0>
	inline double weight(float x, uint32_t channel) const
	{
		if (x != 0.f)
			return 0.0;

		if constexpr (derivative == 0)
		{
			return std::numeric_limits<double>::infinity();
		}
		else
		{
			static_assert(false);
			return core::nan<double>();
		}
	}
};

// Standard Box function, symmetric, value in the support is 1, integral is 1, so support must be [-1/2,1/2)
struct SBoxFunction
{
	constexpr static inline float min_support = -0.5f;
	constexpr static inline float max_support = +0.5f;
	constexpr static inline uint32_t k_smoothness = 0;

	template <int32_t derivative = 0>
	inline double weight(float x, uint32_t channel) const
	{
		if (x >= min_support && x < max_support)
		{
			if constexpr (derivative == 0)
			{
				return 1.0;
			}
			else
			{
				static_assert(false);
				return core::nan<double>();
			}
		}
		return 0.0;
	}
};

// Standard Triangle function, symmetric, peak in the support is 1 and at origin, integral is 1, so support must be [-1,1)
struct STriangleFunction
{
	constexpr static inline float min_support = -1.f;
	constexpr static inline float max_support = +1.f;
	constexpr static inline uint32_t k_smoothness = 0;

	template <int32_t derivative = 0>
	static inline double weight(float x, uint32_t channel)
	{
		if (x >= min_support && x < max_support)
		{
			if constexpr (derivative > 0)
			{
				// Derivative at 0 not defined.
				static_assert(false);
				return core::nan<double>();
			}
			else
			{
				return 1.0 - core::abs(x);
			}
		}
		return 0.0;
	}
};

// Truncated Gaussian function, with stddev = 1.0, if you want a different stddev then you need to scale it.
struct SGaussianFunction
{
	constexpr static inline float min_support = -3.f;
	constexpr static inline float max_support = +3.f;
	constexpr static inline uint32_t k_smoothness = std::numeric_limits<uint32_t>::max();

	template <int32_t derivative = 0>
	static inline double weight(float x, uint32_t channel)
	{
		if (x >= min_support && x < max_support)
		{
			if constexpr (derivative == 0)
			{
				const double normalizationFactor = core::inversesqrt(2.0 * core::PI<double>()) / std::erff(core::sqrt<double>(2.0) * double(min_support));
				return normalizationFactor * exp2(-0.72134752 * x * x);
			}
			else if constexpr (derivative == 1)
			{
				return -x * SGaussianFunction::weight(x, channel);
			}
			else if constexpr (derivative == 2)
			{
				return x * (x + 1.0) * SGaussianFunction::weight(x, channel);
			}
			else if constexpr (derivative == 3)
			{
				return (1.0 - (x - 1.0) * (x + 1.0) * (x + 1.0)) * SGaussianFunction::weight(x, channel);
			}
			else
			{
				static_assert(false, "TODO");
				return core::nan<double>();
			}
		}
		return 0.f;
	}
};

// A standard Mitchell function, the standard has a support of [-2,2] the B and C template parameters are the same ones from the paper
template<class B = std::ratio<1, 3>, class C = std::ratio<1, 3>>
struct SMitchellFunction
{
	constexpr static inline float min_support = -2.f;
	constexpr static inline float max_support = +2.f;
	constexpr static inline uint32_t k_smoothness = 3;

	template <int32_t derivative = 0>
	static inline double weight(float x, uint32_t channel)
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
			{
				static_assert(false);
				return core::nan<double>();
			}

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
struct SKaiserFunction
{
	constexpr static inline float min_support = -3.f;
	constexpr static inline float max_support = +3.f;
	// important constant, do not touch, do not tweak
	static inline constexpr float alpha = 3.f;

	template <int32_t derivative = 0>
	static inline double weight(float x, uint32_t channel)
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
			{
				static_assert(false, "TODO");
				return core::nan<double>();
			}
		}
		return 0.0;
	}
};

// CConvolutionWeightFunction1D<CWeightFunction1D<SMitchellFunction>, CWeightFunction1D<SMitchellFunction>>
template <typename Function1D, int32_t derivative = 0>
class CWeightFunction1D /*final*/ // TODO(achal): Cannot make it final just yet because `weight_function_value_type` inherits from this.
{
public:
	constexpr static inline uint32_t k_smoothness = Function1D::k_smoothness;
	constexpr static inline float k_energy[4] = { 0.f, 0.f, 0.f, 0.f };

	inline void stretch(const float s)
	{
		assert(s != 0.f);

		m_minSupport *= s;
		m_maxSupport *= s;

		m_invStretch /= s;

		if constexpr (derivative != 0)
			scale(pow(s, derivative));
	}

	inline void scale(const float s)
	{
		assert(s != 0.f);
		m_totalScale *= s;
	}

	// This method will keep the integral of the weight function constant.
	inline void stretchAndScale(const float stretchFactor)
	{
		stretch(stretchFactor);
		scale(1.f / stretchFactor);
	}

	// TODO(achal): It makes no sense to me as to why we take a channel param here.
	inline double operator()(const float x, const uint32_t channel) const
	{
		return static_cast<double>(m_totalScale*Function1D::weight<derivative>(x*m_invStretch, channel));
	}

	inline float getMinSupport() const { m_minSupport; }
	inline float getMaxSupport() const { m_maxSupport; }
	inline float getInvStretch() const { m_invStretch; }

private:
	float m_minSupport = Function1D::min_support;
	float m_maxSupport = Function1D::max_support;
	float m_invStretch = 1.f;
	float m_totalScale = 1.f;
};

} // end namespace nbl::asset
#endif
