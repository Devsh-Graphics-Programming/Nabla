#ifndef _NBL_BUILTIN_HLSL_PATH_TRACING_GAUSSIAN_FILTER_INCLUDED_
#define _NBL_BUILTIN_HLSL_PATH_TRACING_GAUSSIAN_FILTER_INCLUDED_

#include <nbl/builtin/hlsl/sampling/box_muller_transform.hlsl>

namespace nbl
{
namespace hlsl
{
namespace path_tracing
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeScalar<T>)
struct GaussianFilter
{
    using this_t = GaussianFilter<T>;
    using scalar_type = T;
    
    using vector2_type = vector<scalar_type, 2>;

    static this_t create(const scalar_type gaussianFilterCutoff, const scalar_type stddev)
    {
        this_t retval;
        retval.truncation = hlsl::exp(-0.5 * gaussianFilterCutoff * gaussianFilterCutoff);
        retval.boxMuller.stddev = stddev;
        return retval;
    }

    vector2_type sample(const vector2_type randVec)
    {
        vector2_type remappedRand = randVec;
        remappedRand.x *= 1.0 - truncation;
        remappedRand.x += truncation;
        return boxMuller(remappedRand);
    }

    scalar_type truncation;
    nbl::hlsl::sampling::BoxMullerTransform<scalar_type> boxMuller;
};

}
}
}

#endif
