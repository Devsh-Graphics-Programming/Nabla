#ifndef _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_IMPL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/tgmath/output_structs.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>

// C++ includes
#ifndef __HLSL_VERSION
#include <cmath>
#include <tgmath.h>
#endif

namespace nbl
{
namespace hlsl
{
namespace tgmath_impl
{

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct erf_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct erfInv_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct isnan_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct isinf_helper;
template<typename V NBL_STRUCT_CONSTRAINABLE>
struct floor_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct pow_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct exp_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct exp2_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct log_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct log2_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct abs_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cos_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sin_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct acos_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct tan_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct asin_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct atan_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sinh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct cosh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct tanh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct asinh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct acosh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct atanh_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct atan2_helper;

template<typename T NBL_STRUCT_CONSTRAINABLE>
struct sqrt_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct modf_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct round_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct roundEven_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct trunc_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct ceil_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct fma_helper;
template<typename T, typename U NBL_STRUCT_CONSTRAINABLE>
struct ldexp_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct modfStruct_helper;
template<typename T NBL_STRUCT_CONSTRAINABLE>
struct frexpStruct_helper;

#ifdef __HLSL_VERSION

#define DECLVAL(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) experimental::declval<_T>()
#define DECL_ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) const _T arg##i
#define WRAP(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) _T
#define ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) arg##i

// the template<> needs to be written ourselves
// return type is __VA_ARGS__ to protect against `,` in templated return types
#define AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(HELPER_NAME, SPIRV_FUNCTION_NAME, ARG_TYPE_LIST, ARG_TYPE_SET, ...)\
NBL_PARTIAL_REQ_TOP(is_same_v<decltype(spirv::SPIRV_FUNCTION_NAME<T>(BOOST_PP_SEQ_FOR_EACH_I(DECLVAL, _, ARG_TYPE_SET))), __VA_ARGS__ >) \
struct HELPER_NAME<BOOST_PP_SEQ_FOR_EACH_I(WRAP, _, ARG_TYPE_LIST) NBL_PARTIAL_REQ_BOT(is_same_v<decltype(spirv::SPIRV_FUNCTION_NAME<T>(BOOST_PP_SEQ_FOR_EACH_I(DECLVAL, _, ARG_TYPE_SET))), __VA_ARGS__ >) >\
{\
	using return_t = __VA_ARGS__;\
	static inline return_t __call( BOOST_PP_SEQ_FOR_EACH_I(DECL_ARG, _, ARG_TYPE_SET) )\
	{\
		return spirv::SPIRV_FUNCTION_NAME<T>( BOOST_PP_SEQ_FOR_EACH_I(ARG, _, ARG_TYPE_SET) );\
	}\
};

template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sin_helper, sin, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cos_helper, cos, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acos_helper, acos, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tan_helper, tan, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asin_helper, asin, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan_helper, atan, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sinh_helper, sinh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cosh_helper, cosh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tanh_helper, tanh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asinh_helper, asinh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acosh_helper, acosh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atanh_helper, atanh, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan2_helper, atan2, (T), (T)(T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, sAbs, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, fAbs, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sqrt_helper, sqrt, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log_helper, log, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log2_helper, log2, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp2_helper, exp2, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp_helper, exp, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(floor_helper, floor, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(round_helper, round, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(roundEven_helper, roundEven, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(trunc_helper, trunc, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(ceil_helper, ceil, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(pow_helper, pow, (T), (T)(T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(fma_helper, fma, (T), (T)(T)(T), T)
template<typename T, typename U> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(ldexp_helper, ldexp, (T)(U), (T)(U), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(modfStruct_helper, modfStruct, (T), (T), ModfOutput<T>)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(frexpStruct_helper, frexpStruct, (T), (T), FrexpOutput<T>)

#define ISINF_AND_ISNAN_RETURN_TYPE conditional_t<is_vector_v<T>, vector<bool, vector_traits<T>::Dimension>, bool>
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(isinf_helper, isInf, (T), (T), ISINF_AND_ISNAN_RETURN_TYPE)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(isnan_helper, isNan, (T), (T), ISINF_AND_ISNAN_RETURN_TYPE)
#undef ISINF_AND_ISNAN_RETURN_TYPE 

#undef DECLVAL
#undef DECL_ARG
#undef WRAP
#undef ARG
#undef AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T>)
struct modf_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T>) >
{
	using return_t = T;
	static inline return_t __call(const T x)
	{
		ModfOutput<T> output = modfStruct_helper<T>::__call(x);
		return output.fractionalPart;
	}
};

template<typename T> NBL_PARTIAL_REQ_TOP(concepts::FloatingPoint<T> && is_vector_v<T>)
struct modf_helper<T NBL_PARTIAL_REQ_BOT(concepts::FloatingPoint<T> && is_vector_v<T>) >
{
	using return_t = T;
	static inline return_t __call(const T x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, modf_helper<typename traits::scalar_type>::__call(getter(x, i)));

		return output;
	}
};

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erf_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) _x)
	{
		// glibc implementation
		const float64_t tiny = NBL_FP64_LITERAL(1e-300),
			one = NBL_FP64_LITERAL(1.00000000000000000000e+00), /* 0x3FF00000, 0x00000000 */
			erx = NBL_FP64_LITERAL(8.45062911510467529297e-01); /* 0x3FEB0AC1, 0x60000000 */

		// Coefficients for approximation to erf in [0,0.84375]
		const float64_t efx = NBL_FP64_LITERAL(1.28379167095512586316e-01); /* 0x3FC06EBA, 0x8214DB69 */
		const float64_t pp0 = NBL_FP64_LITERAL(1.28379167095512558561e-01); /* 0x3FC06EBA, 0x8214DB68 */
		const float64_t pp1 = NBL_FP64_LITERAL(-3.25042107247001499370e-01); /* 0xBFD4CD7D, 0x691CB913 */
		const float64_t pp2 = NBL_FP64_LITERAL(-2.84817495755985104766e-02); /* 0xBF9D2A51, 0xDBD7194F */
		const float64_t pp3 = NBL_FP64_LITERAL(-5.77027029648944159157e-03); /* 0xBF77A291, 0x236668E4 */
		const float64_t pp4 = NBL_FP64_LITERAL(-2.37630166566501626084e-05); /* 0xBEF8EAD6, 0x120016AC */
		const float64_t qq1 = NBL_FP64_LITERAL(3.97917223959155352819e-01); /* 0x3FD97779, 0xCDDADC09 */
		const float64_t qq2 = NBL_FP64_LITERAL(6.50222499887672944485e-02); /* 0x3FB0A54C, 0x5536CEBA */
		const float64_t qq3 = NBL_FP64_LITERAL(5.08130628187576562776e-03); /* 0x3F74D022, 0xC4D36B0F */
		const float64_t qq4 = NBL_FP64_LITERAL(1.32494738004321644526e-04); /* 0x3F215DC9, 0x221C1A10 */
		const float64_t qq5 = NBL_FP64_LITERAL(-3.96022827877536812320e-06); /* 0xBED09C43, 0x42A26120 */

		//Coefficients for approximation to erf in [0.84375,1.25]
		const float64_t pa0 = NBL_FP64_LITERAL(-2.36211856075265944077e-03); /* 0xBF6359B8, 0xBEF77538 */
		const float64_t pa1 = NBL_FP64_LITERAL(4.14856118683748331666e-01); /* 0x3FDA8D00, 0xAD92B34D */
		const float64_t pa2 = NBL_FP64_LITERAL(-3.72207876035701323847e-01); /* 0xBFD7D240, 0xFBB8C3F1 */
		const float64_t pa3 = NBL_FP64_LITERAL(3.18346619901161753674e-01); /* 0x3FD45FCA, 0x805120E4 */
		const float64_t pa4 = NBL_FP64_LITERAL(-1.10894694282396677476e-01); /* 0xBFBC6398, 0x3D3E28EC */
		const float64_t pa5 = NBL_FP64_LITERAL(3.54783043256182359371e-02); /* 0x3FA22A36, 0x599795EB */
		const float64_t pa6 = NBL_FP64_LITERAL(-2.16637559486879084300e-03); /* 0xBF61BF38, 0x0A96073F */
		const float64_t qa1 = NBL_FP64_LITERAL(1.06420880400844228286e-01); /* 0x3FBB3E66, 0x18EEE323 */
		const float64_t qa2 = NBL_FP64_LITERAL(5.40397917702171048937e-01); /* 0x3FE14AF0, 0x92EB6F33 */
		const float64_t qa3 = NBL_FP64_LITERAL(7.18286544141962662868e-02); /* 0x3FB2635C, 0xD99FE9A7 */
		const float64_t qa4 = NBL_FP64_LITERAL(1.26171219808761642112e-01); /* 0x3FC02660, 0xE763351F */
		const float64_t qa5 = NBL_FP64_LITERAL(1.36370839120290507362e-02); /* 0x3F8BEDC2, 0x6B51DD1C */
		const float64_t qa6 = NBL_FP64_LITERAL(1.19844998467991074170e-02); /* 0x3F888B54, 0x5735151D */

		// Coefficients for approximation to erfc in [1.25,1/0.35]
		const float64_t ra0 = NBL_FP64_LITERAL(-9.86494403484714822705e-03); /* 0xBF843412, 0x600D6435 */
		const float64_t ra1 = NBL_FP64_LITERAL(-6.93858572707181764372e-01); /* 0xBFE63416, 0xE4BA7360 */
		const float64_t ra2 = NBL_FP64_LITERAL(-1.05586262253232909814e+01); /* 0xC0251E04, 0x41B0E726 */
		const float64_t ra3 = NBL_FP64_LITERAL(-6.23753324503260060396e+01); /* 0xC04F300A, 0xE4CBA38D */
		const float64_t ra4 = NBL_FP64_LITERAL(-1.62396669462573470355e+02); /* 0xC0644CB1, 0x84282266 */
		const float64_t ra5 = NBL_FP64_LITERAL(-1.84605092906711035994e+02); /* 0xC067135C, 0xEBCCABB2 */
		const float64_t ra6 = NBL_FP64_LITERAL(-8.12874355063065934246e+01); /* 0xC0545265, 0x57E4D2F2 */
		const float64_t ra7 = NBL_FP64_LITERAL(-9.81432934416914548592e+00); /* 0xC023A0EF, 0xC69AC25C */
		const float64_t sa1 = NBL_FP64_LITERAL(1.96512716674392571292e+01); /* 0x4033A6B9, 0xBD707687 */
		const float64_t sa2 = NBL_FP64_LITERAL(1.37657754143519042600e+02); /* 0x4061350C, 0x526AE721 */
		const float64_t sa3 = NBL_FP64_LITERAL(4.34565877475229228821e+02); /* 0x407B290D, 0xD58A1A71 */
		const float64_t sa4 = NBL_FP64_LITERAL(6.45387271733267880336e+02); /* 0x40842B19, 0x21EC2868 */
		const float64_t sa5 = NBL_FP64_LITERAL(4.29008140027567833386e+02); /* 0x407AD021, 0x57700314 */
		const float64_t sa6 = NBL_FP64_LITERAL(1.08635005541779435134e+02); /* 0x405B28A3, 0xEE48AE2C */
		const float64_t sa7 = NBL_FP64_LITERAL(6.57024977031928170135e+00); /* 0x401A47EF, 0x8E484A93 */
		const float64_t sa8 = NBL_FP64_LITERAL(-6.04244152148580987438e-02); /* 0xBFAEEFF2, 0xEE749A62 */

		// Coefficients for approximation to erfc in [1/.35,28]
		const float64_t rb0 = NBL_FP64_LITERAL(-9.86494292470009928597e-03); /* 0xBF843412, 0x39E86F4A */
		const float64_t rb1 = NBL_FP64_LITERAL(-7.99283237680523006574e-01); /* 0xBFE993BA, 0x70C285DE */
		const float64_t rb2 = NBL_FP64_LITERAL(-1.77579549177547519889e+01); /* 0xC031C209, 0x555F995A */
		const float64_t rb3 = NBL_FP64_LITERAL(-1.60636384855821916062e+02); /* 0xC064145D, 0x43C5ED98 */
		const float64_t rb4 = NBL_FP64_LITERAL(-6.37566443368389627722e+02); /* 0xC083EC88, 0x1375F228 */
		const float64_t rb5 = NBL_FP64_LITERAL(-1.02509513161107724954e+03); /* 0xC0900461, 0x6A2E5992 */
		const float64_t rb6 = NBL_FP64_LITERAL(-4.83519191608651397019e+02); /* 0xC07E384E, 0x9BDC383F */
		const float64_t sb1 = NBL_FP64_LITERAL(3.03380607434824582924e+01); /* 0x403E568B, 0x261D5190 */
		const float64_t sb2 = NBL_FP64_LITERAL(3.25792512996573918826e+02); /* 0x40745CAE, 0x221B9F0A */
		const float64_t sb3 = NBL_FP64_LITERAL(1.53672958608443695994e+03); /* 0x409802EB, 0x189D5118 */
		const float64_t sb4 = NBL_FP64_LITERAL(3.19985821950859553908e+03); /* 0x40A8FFB7, 0x688C246A */
		const float64_t sb5 = NBL_FP64_LITERAL(2.55305040643316442583e+03); /* 0x40A3F219, 0xCEDF3BE6 */
		const float64_t sb6 = NBL_FP64_LITERAL(4.74528541206955367215e+02); /* 0x407DA874, 0xE79FE763 */
		const float64_t sb7 = NBL_FP64_LITERAL(-2.24409524465858183362e+01); /* 0xC03670E2, 0x42712D62 */

		float64_t x = float64_t(_x);
		int32_t hx, ix;
		float64_t s, y, z, r;
		hx = int32_t(bit_cast<uint64_t, float64_t>(x) >> 32);
		ix = hx & 0x7fffffff;
		if (ix >= 0x7ff00000)           // erf(nan)=nan, erf(+-inf)=+-1
		{
			int32_t i = ((uint32_t)hx >> 31) << 1;
			return (float64_t)(1.0 - i) + one / x;
		}

		float64_t P, Q;
		if (ix < 0x3feb0000)            // |x| < 0.84375
		{
			if (ix < 0x3e300000)        // |x| < 2**-28
			{
				if (ix < 0x00800000)
				{
					// avoid underflow
					return FloatingPoint(0.0625 * (16.0 * x + (16.0 * efx) * x));
				}
				return FloatingPoint(x + efx * x);
			}
			z = x * x;
			r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
			s = one + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
			y = r / s;
			return FloatingPoint(x + x * y);
		}
		if (ix < 0x3ff40000)            // 0.84375 <= |x| < 1.25
		{
			s = abs_helper<float64_t>::__call(x) - one;
			P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
			Q = one + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * (qa5 + s * qa6))))));
			if (hx >= 0)
				return FloatingPoint(erx + P / Q);
			else
				return FloatingPoint(-erx - P / Q);
		}
		if (ix >= 0x40180000)           // inf > |x| >= 6
		{
			if (hx >= 0)
				return FloatingPoint(one - tiny);
			else
				return FloatingPoint(tiny - one);
		}

		x = abs_helper<float64_t>::__call(x);
		s = one / (x * x);
		float64_t R, S;
		if (ix < 0x4006DB6E)            // |x| < 1/0.35     ~2.85714
		{
			R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
			S = one + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * sa7))))));
		}
		else                            // |x| >= 1/0.35
		{
			R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * rb5))));
			S = one + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
		}
		z = x;
		uint64_t z1 = bit_cast<uint64_t, float64_t>(x);
		z1 &= 0xffffffff00000000;
		z = bit_cast<float64_t, uint64_t>(z1);
		r = exp_helper<float64_t>::__call(-z * z - 0.5625) * exp_helper<float64_t>::__call((z - x) * (z + x) + R / S);
		if (hx >= 0)
			return FloatingPoint(one - r / x);
		else
			return FloatingPoint(r / x - one);
	}
};

template<>
struct erf_helper<float32_t>
{
	static float32_t __call(NBL_CONST_REF_ARG(float32_t) _x)
	{
		// A&S approximation to 1.5x10-7
		const float32_t a1 = float32_t(NBL_FP64_LITERAL(0.254829592));
		const float32_t a2 = float32_t(NBL_FP64_LITERAL(-0.284496736));
		const float32_t a3 = float32_t(NBL_FP64_LITERAL(1.421413741));
		const float32_t a4 = float32_t(NBL_FP64_LITERAL(-1.453152027));
		const float32_t a5 = float32_t(NBL_FP64_LITERAL(1.061405429));
		const float32_t p = float32_t(NBL_FP64_LITERAL(0.3275911));

		float32_t _sign = float32_t(sign(_x));
		float32_t x = abs(_x);

		float32_t t = float32_t(NBL_FP64_LITERAL(1.0)) / (float32_t(NBL_FP64_LITERAL(1.0)) + p * x);
		float32_t y = float32_t(NBL_FP64_LITERAL(1.0)) - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

		return _sign * y;
	}
};


#else // C++ only specializations

#define DECL_ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) const _T arg##i
#define WRAP(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) _T
#define ARG(r,data,i,_T) BOOST_PP_COMMA_IF(BOOST_PP_NOT_EQUAL(i,0)) arg##i

// not giving an explicit template parameter to std function below because not every function used here is templated
#define AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(HELPER_NAME, STD_FUNCTION_NAME, REQUIREMENT, ARG_TYPE_LIST, ARG_TYPE_SET, ...)\
requires REQUIREMENT \
struct HELPER_NAME<BOOST_PP_SEQ_FOR_EACH_I(WRAP, _, ARG_TYPE_LIST)>\
{\
	using return_t = __VA_ARGS__;\
	static inline return_t __call( BOOST_PP_SEQ_FOR_EACH_I(DECL_ARG, _, ARG_TYPE_SET) )\
	{\
		return std::STD_FUNCTION_NAME( BOOST_PP_SEQ_FOR_EACH_I(ARG, _, ARG_TYPE_SET) );\
	}\
};

template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cos_helper, cos, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sin_helper, sin, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tan_helper, tan, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asin_helper, asin, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acos_helper, acos, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan_helper, atan, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sinh_helper, sinh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(cosh_helper, cosh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(tanh_helper, tanh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(asinh_helper, asinh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(acosh_helper, acosh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atanh_helper, atanh, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(atan2_helper, atan2, concepts::FloatingPointScalar<T>, (T), (T)(T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(sqrt_helper, sqrt, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(abs_helper, abs, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log_helper, log, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(log2_helper, log2, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp2_helper, exp2, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(exp_helper, exp, concepts::Scalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(floor_helper, floor, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(round_helper, round, concepts::FloatingPointScalar<T>, (T), (T), T)
// TODO: uncomment when C++23
//template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(roundEven_helper, roundeven, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(trunc_helper, trunc, concepts::FloatingPointScalar<T>, (T), (T), T)
template<typename T> AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER(ceil_helper, ceil, concepts::FloatingPointScalar<T>, (T), (T), T)

#undef DECL_ARG
#undef WRAP
#undef ARG
#undef AUTO_SPECIALIZE_TRIVIAL_CASE_HELPER

template<typename T>
requires concepts::FloatingPointScalar<T>
struct pow_helper<T>
{
	using return_t = T;
	static inline return_t __call(const T x, const T y)
	{
		return std::pow<T>(x, y);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct modf_helper<T>
{
	using return_t = T;
	static inline return_t __call(const T x)
	{
		T tmp;
		return std::modf(x, &tmp);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct isinf_helper<T>
{
	using return_t = bool;
	static inline return_t __call(const T arg)
	{
		// GCC and Clang will always return false with call to std::isinf when fast math is enabled,
		// this implementation will always return appropriate output regardless is fast math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return cpp_compat_intrinsics_impl::isinf_uint_impl(reinterpret_cast<const AsUint&>(arg));
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct isnan_helper<T>
{
	using return_t = bool;
	static inline return_t __call(const T arg)
	{
		// GCC and Clang will always return false with call to std::isnan when fast math is enabled,
		// this implementation will always return appropriate output regardless is fast math is enabled or not
		using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
		return cpp_compat_intrinsics_impl::isnan_uint_impl(reinterpret_cast<const AsUint&>(arg));
	}
};

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erf_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
		return std::erf(x);
	}
};

// TODO: remove when C++23
template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct roundEven_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x)
	{
		// TODO: no way this is optimal, find a better implementation
		float tmp;
		if (std::abs(std::modf(x, &tmp)) == 0.5f)
		{
			int32_t result = static_cast<int32_t>(x);
			if (result % 2 != 0)
				result >= 0 ? ++result : --result;
			return result;
		}

		return std::round(x);
	}
};

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct fma_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) x, NBL_CONST_REF_ARG(FloatingPoint) y, NBL_CONST_REF_ARG(FloatingPoint) z)
	{
		return std::fma(x, y, z);
	}
};

template<typename T, typename U>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<T> && concepts::IntegralScalar<U>)
struct ldexp_helper<T, U NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<T> && concepts::IntegralScalar<U>) >
{
	static T __call(NBL_CONST_REF_ARG(T) arg, NBL_CONST_REF_ARG(U) exp)
	{
		return std::ldexp(arg, exp);
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct modfStruct_helper<T>
{
	using return_t = ModfOutput<T>;
	static inline return_t __call(const T val)
	{
		return_t output;
		output.fractionalPart = std::modf(val, &output.wholeNumberPart);

		return output;
	}
};

template<typename T>
requires concepts::FloatingPointScalar<T>
struct frexpStruct_helper<T>
{
	using return_t = FrexpOutput<T>;
	static inline return_t __call(const T val)
	{
		return_t output;
		output.significand = std::frexp(val, &output.exponent);

		return output;
	}
};

#endif // C++ only specializations

// C++ and HLSL specializations

template<>
struct erf_helper<float16_t>
{
	static float16_t __call(float16_t _x)
	{
		// A&S approximation to 2.5x10-5
		const float16_t a1 = float16_t(0.3480242f);
		const float16_t a2 = float16_t(-0.0958798f);
		const float16_t a3 = float16_t(0.7478556f);
		const float16_t p = float16_t(0.47047f);

		float16_t _sign = float16_t(sign<float16_t>(_x));
		float16_t x = abs_helper<float16_t>::__call(_x);

		float16_t t = float16_t(1.f) / (float16_t(1.f) + p * x);
		float16_t y = float16_t(1.f) - (((a3 * t + a2) * t) + a1) * t * exp(-x * x);

		return _sign * y;
	}
};

template<typename FloatingPoint>
NBL_PARTIAL_REQ_TOP(concepts::FloatingPointScalar<FloatingPoint>)
struct erfInv_helper<FloatingPoint NBL_PARTIAL_REQ_BOT(concepts::FloatingPointScalar<FloatingPoint>) >
{
	static FloatingPoint __call(NBL_CONST_REF_ARG(FloatingPoint) _x)
	{
		FloatingPoint x = clamp<FloatingPoint>(_x, FloatingPoint(NBL_FP64_LITERAL(-0.99999)), FloatingPoint(NBL_FP64_LITERAL(0.99999)));

		FloatingPoint w = -log_helper<FloatingPoint>::__call((FloatingPoint(NBL_FP64_LITERAL(1.0)) - x) * (FloatingPoint(NBL_FP64_LITERAL(1.0)) + x));
		FloatingPoint p;
		if (w < 5.0)
		{
			w -= FloatingPoint(NBL_FP64_LITERAL(2.5));
			p = FloatingPoint(NBL_FP64_LITERAL(2.81022636e-08));
			p = FloatingPoint(NBL_FP64_LITERAL(3.43273939e-07)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-3.5233877e-06)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-4.39150654e-06)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00021858087)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.00125372503)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.00417768164)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.246640727)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(1.50140941)) + p * w;
		}
		else
		{
			w = sqrt_helper<FloatingPoint>::__call(w) - FloatingPoint(NBL_FP64_LITERAL(3.0));
			p = FloatingPoint(NBL_FP64_LITERAL(-0.000200214257));
			p = FloatingPoint(NBL_FP64_LITERAL(0.000100950558)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00134934322)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.00367342844)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00573950773)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(-0.0076224613)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(0.00943887047)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(1.00167406)) + p * w;
			p = FloatingPoint(NBL_FP64_LITERAL(2.83297682)) + p * w;
		}
		return p * x;
	}
};

#ifdef __HLSL_VERSION
// SPIR-V already defines specializations for builtin vector types
#define VECTOR_SPECIALIZATION_CONCEPT concepts::Vectorial<T> && !is_vector_v<T>
#else
#define VECTOR_SPECIALIZATION_CONCEPT concepts::Vectorial<T>
#endif

#define AUTO_SPECIALIZE_HELPER_FOR_VECTOR(HELPER_NAME, CONCEPT, RETURN_TYPE)\
template<typename T>\
NBL_PARTIAL_REQ_TOP(CONCEPT)\
struct HELPER_NAME<T NBL_PARTIAL_REQ_BOT(CONCEPT) >\
{\
	using return_t = RETURN_TYPE;\
	static return_t __call(NBL_CONST_REF_ARG(T) vec)\
	{\
		using traits = hlsl::vector_traits<T>;\
		using return_t_traits = hlsl::vector_traits<return_t>;\
		array_get<T, typename traits::scalar_type> getter;\
		array_set<return_t, typename return_t_traits::scalar_type> setter;\
\
		return_t output;\
		for (uint32_t i = 0; i < traits::Dimension; ++i)\
			setter(output, i, HELPER_NAME<typename traits::scalar_type>::__call(getter(vec, i)));\
\
		return output;\
	}\
};

AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sqrt_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(abs_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(log_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(log2_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(exp2_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(exp_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(floor_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
#define INT_VECTOR_RETURN_TYPE vector<int32_t, vector_traits<T>::Dimension>
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(isinf_helper, VECTOR_SPECIALIZATION_CONCEPT, INT_VECTOR_RETURN_TYPE)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(isnan_helper, VECTOR_SPECIALIZATION_CONCEPT, INT_VECTOR_RETURN_TYPE)
#undef INT_VECTOR_RETURN_TYPE
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(cos_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sin_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(acos_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(tan_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(asin_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(atan_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(sinh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(cosh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(tanh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(asinh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(acosh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(atanh_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(modf_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(round_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(roundEven_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(trunc_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(ceil_helper, VECTOR_SPECIALIZATION_CONCEPT, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(erf_helper, concepts::Vectorial<T>, T)
AUTO_SPECIALIZE_HELPER_FOR_VECTOR(erfInv_helper, concepts::Vectorial<T>, T)


#undef AUTO_SPECIALIZE_HELPER_FOR_VECTOR

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct pow_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;
		
		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, pow_helper<typename traits::scalar_type>::__call(getter(x, i), getter(y, i)));
	
		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct fma_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(T) z)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, fma_helper<typename traits::scalar_type>::__call(getter(x, i), getter(y, i), getter(z, i)));

		return output;
	}
};

template<typename T, typename U>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT && (vector_traits<T>::Dimension == vector_traits<U>::Dimension))
struct ldexp_helper<T, U NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT && (vector_traits<T>::Dimension == vector_traits<U>::Dimension)) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) arg, NBL_CONST_REF_ARG(U) exp)
	{
		using arg_traits = hlsl::vector_traits<T>;
		using exp_traits = hlsl::vector_traits<U>;
		array_get<T, typename arg_traits::scalar_type> argGetter;
		array_get<U, typename exp_traits::scalar_type> expGetter;
		array_set<T, typename arg_traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < arg_traits::Dimension; ++i)
			setter(output, i, ldexp_helper<typename arg_traits::scalar_type, typename exp_traits::scalar_type>::__call(argGetter(arg, i), expGetter(exp, i)));

		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct modfStruct_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = ModfOutput<T>;
	static return_t __call(NBL_CONST_REF_ARG(T) x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		T fracPartOut;
		T intPartOut;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
		{
			using component_return_t = ModfOutput<typename vector_traits<T>::scalar_type>;
			component_return_t result = modfStruct_helper<typename traits::scalar_type>::__call(getter(x, i));

			setter(fracPartOut, i, result.fractionalPart);
			setter(intPartOut, i, result.wholeNumberPart);
		}

		return_t output;
		output.fractionalPart = fracPartOut;
		output.wholeNumberPart = intPartOut;

		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct frexpStruct_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = FrexpOutput<T>;
	static return_t __call(NBL_CONST_REF_ARG(T) x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> significandSetter;
		array_set<T, typename traits::scalar_type> exponentSetter;

		T significandOut;
		T exponentOut;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
		{
			using component_return_t = FrexpOutput<typename vector_traits<T>::scalar_type>;
			component_return_t result = frexpStruct_helper<typename traits::scalar_type>::__call(getter(x, i));

			significandSetter(significandOut, i, result.significand);
			exponentSetter(exponentOut, i, result.exponent);
		}

		return_t output;
		output.significand = significandOut;
		output.exponent = exponentOut;

		return output;
	}
};

template<typename T>
NBL_PARTIAL_REQ_TOP(VECTOR_SPECIALIZATION_CONCEPT)
struct atan2_helper<T NBL_PARTIAL_REQ_BOT(VECTOR_SPECIALIZATION_CONCEPT) >
{
	using return_t = T;
	static return_t __call(NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(T) x)
	{
		using traits = hlsl::vector_traits<T>;
		array_get<T, typename traits::scalar_type> getter;
		array_set<T, typename traits::scalar_type> setter;

		return_t output;
		for (uint32_t i = 0; i < traits::Dimension; ++i)
			setter(output, i, atan2_helper<typename traits::scalar_type>::__call(getter(y, i), getter(x, i)));

		return output;
	}
};

#undef VECTOR_SPECIALIZATION_CONCEPT

}
}
}

#endif
