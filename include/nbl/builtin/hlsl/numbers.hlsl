#ifndef _NBL_BUILTIN_HLSL_MATH_NUMBERS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_NUMBERS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace numbers
{

template <typename float_t>
NBL_CONSTEXPR float_t e = float_t(2.718281828459045);
template <typename float_t>
NBL_CONSTEXPR float_t log2e = float_t(1.4426950408889634);
template <typename float_t>
NBL_CONSTEXPR float_t log10e = float_t(0.4342944819032518);
template <typename float_t>
NBL_CONSTEXPR float_t pi = float_t(3.141592653589793);
template <typename float_t>
NBL_CONSTEXPR float_t inv_pi = float_t(0.3183098861837907);
template <typename float_t>
NBL_CONSTEXPR float_t inv_sqrtpi = float_t(0.5641895835477563);
template <typename float_t>
NBL_CONSTEXPR float_t ln2 = float_t(0.6931471805599453);
template <typename float_t>
NBL_CONSTEXPR float_t ln10 = float_t(2.302585092994046);
template <typename float_t>
NBL_CONSTEXPR float_t sqrt2 = float_t(1.4142135623730951);
template <typename float_t>
NBL_CONSTEXPR float_t sqrt3 = float_t(1.7320508075688772);
template <typename float_t>
NBL_CONSTEXPR float_t inv_sqrt3 = float_t(0.5773502691896257);
template <typename float_t>
NBL_CONSTEXPR float_t egamma = float_t(0.5772156649015329);
template <typename float_t>
NBL_CONSTEXPR float_t phi = float_t(1.618033988749895);              

}
}
}

#endif
