#ifndef _NBL_BUILTIN_HLSL_PATH_TRACING_BASIC_RAYGEN_INCLUDED_
#define _NBL_BUILTIN_HLSL_PATH_TRACING_BASIC_RAYGEN_INCLUDED_

#include <nbl/builtin/hlsl/path_tracing/gaussian_filter.hlsl>

namespace nbl
{
namespace hlsl
{
namespace path_tracing
{

template<class Ray>
struct BasicRayGenerator
{
    using this_t = BasicRayGenerator<Ray>;
    using ray_type = Ray;
    using scalar_type = typename Ray::scalar_type;
    using vector3_type = typename Ray::vector3_type;
    
    using vector2_type = vector<scalar_type, 2>;
    using vector4_type = vector<scalar_type, 4>;
    using matrix4x4_type = matrix<scalar_type, 4, 4>;

    ray_type generate(const vector3_type randVec)
    {
        vector4_type tmp = NDC;
        GaussianFilter<scalar_type> filter = GaussianFilter<scalar_type>::create(2.5, 1.5); // stochastic reconstruction filter
        tmp.xy += pixOffsetParam * filter.sample(randVec);
        // for depth of field we could do another stochastic point-pick
        tmp = nbl::hlsl::mul(invMVP, tmp);

        ray_type ray;
        ray.initData(camPos, hlsl::normalize(tmp.xyz / tmp.w - camPos), hlsl::promote<vector3_type>(0.0), false);

        return ray;
    }

    vector2_type pixOffsetParam;
    vector3_type camPos;
    vector4_type NDC;
    matrix4x4_type invMVP;
};

}
}
}

#endif
