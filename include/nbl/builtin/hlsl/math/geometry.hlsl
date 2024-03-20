#ifndef _NBL_BUILTIN_HLSL_MATH_GEOMETRY_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_GEOMETRY_INCLUDED_

namespace nbl
{
namespace hlsl
{
// TODO: add NBL_CONST_REF_ARG()
template<typename float_t>
float_t cross2D(vector<float_t, 2> lhs, vector<float_t, 2> rhs) { return lhs.x*rhs.y - lhs.y*rhs.x; }    
}
}

#endif