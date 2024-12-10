#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_

#include "nbl/builtin/hlsl/math/functions.hlsl"

namespace nbl
{
namespace hlsl
{

template<typename T>
vector<T,2> concentricMapping(vector<T,2> _u)
{
    //map [0;1]^2 to [-1;1]^2
    vector<T,2> u = 2.0 * _u - 1.0;
    
    vector<T,2> p;
    if (u == (vector<T,2>)(0.0))
        p = (vector<T,2>)(0.0);
    else
    {
        T r;
        T theta;
        if (abs(u.x) > abs(u.y)) {
            r = u.x;
            theta = 0.25 * numbers::pi<float> * (u.y / u.x);
        } else {
            r = u.y;
            theta = 0.5 * numbers::pi<float> - 0.25 * numbers::pi<float> * (u.x / u.y);
        }
		
        p = r * vector<T,2>(cos(theta), sin(theta));
    }

    return p;
}

}
}

#endif