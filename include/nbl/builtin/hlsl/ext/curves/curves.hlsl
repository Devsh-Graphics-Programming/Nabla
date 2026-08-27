#ifndef _NBL_BUILTIN_HLSL_EXT_CURVES_CURVES_INCLUDED_
#define _NBL_BUILTIN_HLSL_EXT_CURVES_CURVES_INCLUDED_

// TODO: make this file HLSL compatible
#ifndef __HLSL_VERSION

#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO: do we want to keep it in the nbl::hlsl::ext namespace? this code is not generalized and used for specific scenario so maybe it should stay in the `ext` namespace?
namespace ext
{
namespace curves
{

// Base class for all our curves
template<typename float_t>
struct ParametricCurve
{
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	//! compute position at t
	virtual float_t2 computePosition(float_t t) const = 0;

	//! compute unnormalized tangent vector at t
	virtual float_t2 computeTangent(float_t t) const = 0;

	//! compute differential arc length at t
	virtual float_t differentialArcLen(float_t t) const
	{
		return nbl::hlsl::length(computeTangent(t));
	}

	struct ArcLenIntegrand
	{
		const ParametricCurve* m_curve;

		ArcLenIntegrand(const ParametricCurve* curve)
			: m_curve(curve)
		{}

		inline float_t operator()(const float_t t) const
		{
			return m_curve->differentialArcLen(t);
		}
	};

	//! compute arc length by gauss legendere integration
	float_t arcLen(float_t t0, float_t t1) const
	{
		constexpr uint16_t IntegrationOrder = 10u;
		return nbl::hlsl::math::quadrature::GaussLegendreIntegration<IntegrationOrder, float_t, ArcLenIntegrand>::calculateIntegral(ArcLenIntegrand(this), t0, t1);
	}

	//! compute inverse arc len using bisection search
	float_t inverseArcLen_BisectionSearch(float_t targetLen, float_t min, float_t max, const float_t cdfAccuracyThreshold = 1e-4, const uint16_t iterationThreshold = 16u)
	{
		float_t xi = 0.0;
		float_t low = min;
		float_t high = max;
		for (uint16_t i = 0; i < iterationThreshold; ++i)
		{
			xi = (low + high) / 2.0;
			float_t sum = arcLen(min, xi);
			float_t integral = sum + arcLen(xi, max);

			// we could've done sum/integral - targetLen, but this is more robust as it avoids a divsion
			float_t valueAtParamGuess = sum - targetLen * integral;

			if (abs(valueAtParamGuess) < cdfAccuracyThreshold * integral)
				return xi; // we found xi value that gives us a cdf of targetLen within cdfAccuracyThreshold
			else
			{
				if (valueAtParamGuess > 0.0)
					high = xi;
				else
					low = xi;
			}
		}

		return xi;
	}

	//! compute inverse arc len  
	float_t inverseArcLen(float_t targetLen, float_t min, float_t max, const float_t cdfAccuracyThreshold = 1e-4)
	{
		return inverseArcLen_BisectionSearch(targetLen, min, max, cdfAccuracyThreshold);
	}

    virtual float_t2 computeSecondOrderDifferential(float_t t) const
	{
		return float_t2(numeric_limits<float_t>::quiet_NaN, numeric_limits<float_t>::quiet_NaN);
	}

    virtual float_t computeInflectionPoint(float_t errorThreshold) const
	{
		return numeric_limits<float_t>::quiet_NaN;
	}
};

// TODO: make an `ExplicitCurve` concept, it should require for a type to define `y` and `derivative`

// It's when t = x in a Parametric Curve
#define DEFINE_EXPLICIT_CURVE_FUNCTIONS \
float_t differentialArcLen(float_t x) const override\
{ \
	float_t deriv = derivative(x); \
	return sqrt(1.0 + deriv * deriv); \
} \
 \
nbl::hlsl::portable_vector_t2<float_t> computeTangent(float_t x) const override\
{ \
	const float_t deriv = derivative(x); \
	nbl::hlsl::portable_vector_t2<float_t> v = nbl::hlsl::portable_vector_t2<float_t>(1.0, deriv); \
	if (nbl::hlsl::isinf(deriv)) \
		v = nbl::hlsl::portable_vector_t2<float_t>(0.0, 1.0); \
	return v; \
} \
 \
inline nbl::hlsl::portable_vector_t2<float_t> computePosition(float_t x) const override { return nbl::hlsl::portable_vector_t2<float_t>(x, y(x)); }


template<typename float_t>
struct Parabola : ParametricCurve<float_t>
{
	DEFINE_EXPLICIT_CURVE_FUNCTIONS

	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	float_t a, b, c;

	static Parabola create(float_t a, float_t b, float_t c)
    {
        Parabola output;
        output.a = a;
        output.b = b;
        output.c = c;

        return output;
    }

	static Parabola fromThreePoints(NBL_CONST_REF_ARG(float_t2) P0, NBL_CONST_REF_ARG(float_t2) P1, NBL_CONST_REF_ARG(float_t2) P2)
	{
		glm::dmat3 X = glm::dmat3(
			glm::dvec3(P0.x * P0.x, P0.x, 1.0),
			glm::dvec3(P1.x * P1.x, P1.x, 1.0),
			glm::dvec3(P2.x * P2.x, P2.x, 1.0)
		);
		glm::dvec3 M = inverse(transpose(X)) * glm::dvec3(P0.y, P1.y, P2.y);
		return Parabola(M[0], M[1], M[2]);
	}

	float_t y(float_t x) const
	{
		return ((a * x) + b) * x + c;
	}

	float_t derivative(float_t x) const
	{
		return 2.0 * a * x + b;
	}
};

template<typename float_t>
struct CubicCurve : ParametricCurve<float_t>
{
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;
	using float_t4 = nbl::hlsl::portable_vector_t4<float_t>;

	float_t4 X;
	float_t4 Y;

	static CubicCurve create(NBL_CONST_REF_ARG(float_t4) X, NBL_CONST_REF_ARG(float_t4) Y)
	{
		CubicCurve output;
		output.X = X;
		output.Y = Y;

		return output;
	}

	float_t2 computePosition(float_t t) const override
	{
		return float_t2(
			((X[0] * t + X[1]) * t + X[2]) * t + X[3],
			((Y[0] * t + Y[1]) * t + Y[2]) * t + Y[3]
		);
	}

	//! compute unnormalized tangent vector at t
	float_t2 computeTangent(float_t t) const override
	{
		return float_t2(
			(3.0 * X[0] * t + 2.0 * X[1]) * t + X[2],
			(3.0 * Y[0] * t + 2.0 * Y[1]) * t + Y[2]
		);
	}

	//! compute second order differential at t
	float_t2 computeSecondOrderDifferential(float_t t) const override
	{
		return float_t2(
			6.0 * X[0] * t + 2.0 * X[1],
			6.0 * Y[0] * t + 2.0 * Y[1]
		);
	}

	float_t differentialArcLen(float_t t) const override
	{
		return nbl::hlsl::length(computeTangent(t));
	}

	float_t computeInflectionPoint(float_t errorThreshold) const override
	{
		// solve for signed curvature root 
		// when x'*y''-x''*y' = 0
		// https://www.wolframalpha.com/input?i=cross+product+%283*x0*t%5E2%2B2*x1%2Bx2%2C3*y0*t%5E2%2B2*y1%2By2%29+and+%286*x0*t%2B2*x1%2C6*y0*t%2B2*y1%29
		const float_t a = 6.0 * (X[0] * Y[1] - X[1] * Y[0]);
		const float_t b = 6.0 * (2.0 * X[1] * Y[0] - 2.0 * X[0] * Y[1] + X[2] * Y[0] - X[0] * Y[2]);
		const float_t c = 2.0 * (X[2] * Y[1] - X[1] * Y[2]);

		nbl::hlsl::math::equations::Quadratic<float_t> quadratic = nbl::hlsl::math::equations::Quadratic<float_t>::construct(a, b, c);
		const float_t2 roots = quadratic.computeRoots();
		if (roots[0] <= 1.0 && roots[0] >= 0.0)
			return roots[0];
		if (roots[1] <= 1.0 && roots[1] >= 0.0)
			return roots[1];
		return numeric_limits<float_t>::quiet_NaN;
	}
};

// specialized circular arc for the purpose of mixing it with another curve of the same type later
// (r*cos(t*sweep+start), r*sin(t*sweep+start) + originY)
template<typename float_t>
struct CircularArc : ParametricCurve<float_t>
{
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	float_t r;
	float_t originY; // originX is 0
	float_t startAngle;
	float_t sweepAngle;

	static CircularArc create(float_t r, float_t originY, float_t startAngle, float_t sweepAngle)
	{
	    CircularArc output;
	    output.r = r;
	    output.originY = originY;
	    output.startAngle = startAngle;
	    output.sweepAngle = sweepAngle;

	    return output;
	}

	// from circle center (0, -v.y) to start pos (v.x, 0)
	static CircularArc create(float_t2 v, float_t sweepAngle)
	{
	    CircularArc output;
	    output.originY = -v.y;
	    output.sweepAngle = sweepAngle;

	    output.r = length(v);
	    output.startAngle = getSign(v.y) * acos(v.x / output.r);

	    return output;
	}

	// from circle center (0, -v.y) to start pos (v.x, 0)
	static CircularArc create(float_t2 v)
	{
	    CircularArc output;
	    output.originY = -v.y;

	    output.r = length(v);
	    output.startAngle = getSign(v.y) * acos(v.x / output.r);
	    output.sweepAngle = -2.0 * getSign(v.y) * acos(abs(output.originY) / output.r);

	    return output;
	}

	float_t2 computePosition(float_t t) const override
	{
		const float_t actualT = t * sweepAngle + startAngle;
		return float_t2(
			r * cos(actualT),
			r * sin(actualT) + originY
		);
	}

	//! compute unnormalized tangent vector at t
	float_t2 computeTangent(float_t t) const override
	{
		const float_t actualT = t * sweepAngle + startAngle;
		return float_t2(
			-1.0 * r * sweepAngle * sin(actualT),
			+1.0 * r * sweepAngle * cos(actualT)
		);
	}

	float_t2 computeSecondOrderDifferential(float_t t) const override
	{
		const float_t actualT = t * sweepAngle + startAngle;
		return float_t2(
			-1.0 * r * sweepAngle * sweepAngle * cos(actualT),
			-1.0 * r * sweepAngle * sweepAngle * sin(actualT)
		);
	}

	static float_t getSign(float_t x)
	{
		return static_cast<float_t>((x > 0.0)) - static_cast<float_t>((x <= 0.0));
	}
};

// Centered at (0,0), aligned with x axis
template<typename float_t>
struct ExplicitEllipse : ParametricCurve<float_t>
{
	DEFINE_EXPLICIT_CURVE_FUNCTIONS

	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	float_t a, b;
	static ExplicitEllipse create(float_t a, float_t b)
	{
	    ExplicitEllipse output;
	    output.a = a;
	    output.b = b;

	    return output;
	}

	float_t y(float_t x) const
	{
		return a * sqrt(static_cast<float_t>(1.0) - pow((x / b), static_cast<float_t>(2.0)));
	}

	float_t derivative(float_t x) const
	{
		return (-a * x) / ((b * b) * sqrt(static_cast<float_t>(1.0) - pow((x / b), static_cast<float_t>(2.0))));
	}
};

// Centered at (0,0), aligned with x 
template<typename float_t>
struct AxisAlignedEllipse : ParametricCurve<float_t>
{
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	float_t a, b;
	float_t start, end;
	static AxisAlignedEllipse create(float_t a, float_t b, float_t start, float_t end)
	{
	    AxisAlignedEllipse output;
	    output.a = a;
	    output.b = b;
	    output.start = start;
	    output.end = end;

	    return output;
	}

	float_t2 computePosition(float_t t) const override
	{
		const float_t theta = start + (end - start) * t;
		return float_t2(a * cos(theta), b * sin(theta));
	}

	float_t2 computeTangent(float_t t) const override
	{
		const float_t theta = start + (end - start) * t;
		const float_t dThetaDt = end - start;
		return float_t2(-a * dThetaDt * sin(theta), b * dThetaDt * cos(theta));
	}
};

template<typename float_t>
struct EllipticalArcInfo
{
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	float_t2 majorAxis;
	float_t2 center;
	float_t2 angleBounds; // [0, 2Pi)
	float_t eccentricity; // (0, 1]

	inline bool isValid()
	{
		if (eccentricity > 1.0 || eccentricity <= 0.0)
			return false;
		if (angleBounds.y == angleBounds.x)
			return false;
		if (abs(angleBounds.y - angleBounds.x) > 2.0 * nbl::core::PI<float_t>())
			return false;
		return true;
	}
};

template<typename float_t>
struct OffsettedBezier : ParametricCurve<float_t>
{
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;

	nbl::hlsl::shapes::Quadratic<float_t> quadratic;
	float_t offset;

	static OffsettedBezier create(NBL_CONST_REF_ARG(nbl::hlsl::shapes::QuadraticBezier<float_t>) quadBezier, float_t offset)
	{
	    OffsettedBezier output;
	    output.offset = offset;

	    output.quadratic = nbl::hlsl::shapes::Quadratic<float_t>::constructFromBezier(quadBezier.P0, quadBezier.P1, quadBezier.P2);

	    return output;
	}

	float_t2 computePosition(float_t t) const override
	{
		const float_t2 deriv = quadratic.derivative(t);
		const float_t2 normal = normalize(float_t2(deriv.y, -deriv.x));
		return quadratic.evaluate(t) + offset * normal;
	}

	//! compute unnormalized tangent vector at t
	float_t2 computeTangent(float_t t) const override
	{
		const float_t2 ddt = quadratic.derivative(t);
		const float_t2 d2dt2 = quadratic.secondDerivative(t);
		const float_t g = offset * (ddt.x * d2dt2.y - ddt.y * d2dt2.x);
		return ddt + (ddt * g) / glm::length(ddt);
	}

	//! if offset is more than minimum radius of curvature then we get an unwanted gouging/cusp
	float_t2 findCusps()
	{
		// we're basically solving for t in "offset = radiusOfCurvature(t)"
		const float_t lhs = pow(offset * 2.0 * abs(quadratic.B.x * quadratic.A.y - quadratic.B.y * quadratic.A.x), 2.0 / 3.0);
		const float_t a = 4.0 * (quadratic.A.x * quadratic.A.x + quadratic.A.y * quadratic.A.y);
		const float_t b = 4.0 * (quadratic.A.x * quadratic.B.x + quadratic.A.y * quadratic.B.y);
		const float_t c = quadratic.B.x * quadratic.B.x + quadratic.B.y * quadratic.B.y - lhs;
		nbl::hlsl::math::equations::Quadratic<float_t> findCuspsQuadratic = nbl::hlsl::math::equations::Quadratic<float_t>::construct(a, b, c);
		return findCuspsQuadratic.computeRoots();
	}
};

template<typename float_t>
class QuadraticBezierFitter final
{
public:
	using float_t2 = nbl::hlsl::portable_vector_t2<float_t>;
	using float_t3 = nbl::hlsl::portable_vector_t3<float_t>;
	using float_t4 = nbl::hlsl::portable_vector_t4<float_t>;
	using float_t2x2 = nbl::hlsl::portable_matrix_t2x2<float_t>;

	typedef std::function<void(nbl::hlsl::shapes::QuadraticBezier<float_t>&&)> AddBezierFunc;

	//! this subdivision algorithm works/converges for any x-monotonic curve (only 1 y for each x) over the [min, max] range and will continue until hits the `maxDepth` or `targetMaxError` threshold
	//! this function will call the AddBezierFunc when the bezier is finalized, whether to render it directly, write it to file, add it to a vector, etc.. is up to the user.
	//! the subdivision samples the points based on arc length and the error is computed by distance in y direction, so pre and post transform may be needed for your curve and the outputted beziers
	//! it will first split at inflection point of the curve; curves are assumed to have at most 1 inflection point, and will get the best convergence rates. but it will work for curves with more inflection points as well.
	static void adaptive(const ParametricCurve<float_t>& curve, float_t min, float_t max, float_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12)
	{
		// The curves we're working with will have at most 1 inflection point.
		const float_t inflectX = curve.computeInflectionPoint(targetMaxError); // if no inflection point then this will return NaN and the adaptive subdivision will continue as normal (from min to max)
		if (inflectX > min && inflectX < max)
		{
			adaptive_impl(curve, min, inflectX, targetMaxError, addBezierFunc, maxDepth);
			adaptive_impl(curve, inflectX, max, targetMaxError, addBezierFunc, maxDepth);
		}
		else
			adaptive_impl(curve, min, max, targetMaxError, addBezierFunc, maxDepth);
	}

	static void adaptive(const EllipticalArcInfo<float_t>& ellipse, float_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12)
	{
		using namespace nbl::hlsl;

		if (!ellipse.isValid())
		{
			_NBL_DEBUG_BREAK_IF(true);
			return;
		}

		float_t lenghtMajor = length(ellipse.majorAxis);
		float_t lenghtMinor = lenghtMajor * ellipse.eccentricity;
		float_t2 normalizedMajor = ellipse.majorAxis / lenghtMajor;

		float_t2x2 rotate = float_t2x2({
			float_t2(normalizedMajor.x, -normalizedMajor.y),
			float_t2(normalizedMajor.y, normalizedMajor.x)
			});

		AddBezierFunc addTransformedBezier = [&](nbl::hlsl::shapes::QuadraticBezier<float_t>&& quadBezier)
			{
				quadBezier.P0 = mul(rotate, quadBezier.P0);
				quadBezier.P1 = mul(rotate, quadBezier.P1);
				quadBezier.P2 = mul(rotate, quadBezier.P2);
				quadBezier.P0 += ellipse.center;
				quadBezier.P1 += ellipse.center;
				quadBezier.P2 += ellipse.center;
				addBezierFunc(std::move(quadBezier));
			};

		if (ellipse.angleBounds.x != ellipse.angleBounds.y)
		{
			AxisAlignedEllipse aaEllipse(lenghtMajor, lenghtMinor, ellipse.angleBounds.x, ellipse.angleBounds.y);
			adaptive(aaEllipse, 0.0, 1.0, targetMaxError, addTransformedBezier, maxDepth);
		}
	}
		
	static void adaptive(const OffsettedBezier<float_t>& curve, float_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12)
	{
		const float_t2 cusps = curve.findCusps();

		const float_t t0 = nbl::core::min(cusps[0], cusps[1]);
		const float_t t1 = nbl::core::max(cusps[0], cusps[1]);

		const bool firstCusp = t0 > 0.0 && t0 < 1.0;
		const bool secondCusp = t1 > 0.0 && t1 < 1.0;

		// if there are two cusps (offset = radius of curvature) then we have that unwanted gouging and we prefer to seperately subdivide those three sections
		if (firstCusp && secondCusp)
		{
			adaptive_impl(curve, 0.0, t0, targetMaxError, addBezierFunc, maxDepth);
			adaptive_impl(curve, t0, t1, targetMaxError, addBezierFunc, maxDepth);
			adaptive_impl(curve, t1, 1.0, targetMaxError, addBezierFunc, maxDepth);
		}
		// otherwise just subdivide from start/0.0 to end/1.0
		else
		{
			adaptive_impl(curve, 0.0, 1.0, targetMaxError, addBezierFunc, maxDepth);
		}
	}

private:
	// Fix Bezier Hack for when P1 is "outside" P0 -> P2
	// We project P1 into P0->P2 line and see whether it lies inside.
	// Because our curves shouldn't go back on themselves in the direction of the chord
	static void fixBezierMidPoint(nbl::hlsl::shapes::QuadraticBezier<float_t>& bezier)
	{
		const float_t2 localChord = bezier.P2 - bezier.P0;
		const float_t localX = dot(normalize(localChord), bezier.P1 - bezier.P0);
		const bool outside = localX<0 || localX>length(localChord);
		if (outside || nbl::core::isnan(bezier.P1.x) || nbl::core::isnan(bezier.P1.y))
		{
			// _NBL_DEBUG_BREAK_IF(true); // this shouldn't happen but we fix it just in case anyways
			bezier.P1 = bezier.P0 * 0.4 + bezier.P2 * 0.6;
		}
	}

	static void adaptive_impl(const ParametricCurve<float_t>& curve, float_t min, float_t max, float_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t depth)
	{
		if (min == max)
			return;
		assert(min < max);

		float_t split = curve.inverseArcLen_BisectionSearch(0.5, min, max);

		// Shouldn't happen but may happen if we use NewtonRaphson for non convergent inverse CDF
		if (split <= min || split >= max)
		{
			_NBL_DEBUG_BREAK_IF(split < min || split > max);
			split = (min + max) / 2.0;
		}

		const float_t2 P0 = curve.computePosition(min);
		const float_t2 V0 = curve.computeTangent(min);
		const float_t2 P2 = curve.computePosition(max);
		const float_t2 V2 = curve.computeTangent(max);
		nbl::hlsl::shapes::QuadraticBezier<float_t> bezier = nbl::hlsl::shapes::QuadraticBezier<float_t>::constructBezierWithTwoPointsAndTangents(P0, V0, P2, V2);

		bool shouldSubdivide = false;

		// TODO: compare with certain threshold
		if (depth > 0u && normalize(V0) == normalize(V2))
		{
			shouldSubdivide = true;
		}
		else
		{
			fixBezierMidPoint(bezier);
			if (depth > 0u)
			{
				if (glm::distance(P0, P2) < targetMaxError)
				{
					const float_t2 posAtSplit = curve.computePosition(split);
					// If it came down to a bezier small that causes P0 P2 and the position at split smaller than targetMaxError then we stop
					if (glm::distance(posAtSplit, P0) < targetMaxError)
						shouldSubdivide = false;
					// But sometimes when P0 and P2 are close together a split will fix them, like a full circle and needs further subdivision
					else
						shouldSubdivide = true;
				}
				else
				{
					const float_t2 curvePositionAtSplit = curve.computePosition(split);
					float_t bezierYAtSplit = bezier.calcYatX(curvePositionAtSplit.x);
					//_NBL_DEBUG_BREAK_IF(nbl::core::isnan(bezierYAtSplit)); 
					// TODO: maybe a better error comaprison is find the normal at split and intersect with the bezier
					if (nbl::core::isnan(bezierYAtSplit) || abs(curvePositionAtSplit.y - bezierYAtSplit) > targetMaxError)
						shouldSubdivide = true;
				}
			}
		}

		if (shouldSubdivide)
		{
			adaptive_impl(curve, min, split, targetMaxError, addBezierFunc, depth - 1u);
			adaptive_impl(curve, split, max, targetMaxError, addBezierFunc, depth - 1u);
		}
		else
		{
			const bool degenerate = (bezier.P0 == bezier.P2);
			if (!degenerate)
				addBezierFunc(std::move(bezier));
		}
	}
};

} // namespace curves
} // namespace ext
} // namespace hlsl
} // namespace nbl

#undef DEFINE_EXPLICIT_CURVE_FUNCTIONS

#endif

#endif