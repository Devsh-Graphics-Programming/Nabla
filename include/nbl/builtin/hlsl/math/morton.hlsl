#ifndef _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/concepts/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace morton
{

template<typename I, uint16_t D NBL_PRIMARY_REQUIRES(concepts::IntegralScalar<I>)
struct code
{
    using this_t = code<I, D>;
    using U = make_unsigned<I>;

    static this_t create(vector<I, D> cartesian)
    {
        //... TODO ...
        return this_t();
    }

    //operator+, operator-, operator>>, operator<<, and other bitwise ops

    U value;
};

} //namespace morton
} //namespace hlsl
} //namespace nbl



#endif