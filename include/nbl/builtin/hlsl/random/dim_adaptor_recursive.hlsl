#ifndef _NBL_HLSL_RANDOM_DIM_ADAPTOR_RECURSIVE_INCLUDED_
#define _NBL_HLSL_RANDOM_DIM_ADAPTOR_RECURSIVE_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace random
{

// adapts PRNG for multiple dimensions using recursive calls, rather than hash based
template<typename RNG, uint32_t DIM=1>
struct DimAdaptorRecursive
{
    using rng_type = RNG;
    using return_type = vector<uint32_t, DIM>;

    static DimAdaptorRecursive<RNG, DIM> construct(rng_type rng)
    {
        DimAdaptorRecursive<RNG, DIM> retval;
        retval.rng = rng;
        return retval;
    }

    return_type operator()()
    {
        array_set<return_type, uint32_t> setter;

        return_type retval;
#ifdef __HLSL_VERSION
        [unroll]
#endif
        for (uint32_t i = 0; i < DIM; i++)
            setter(retval, i, rng());
        return retval;
    }

    rng_type rng;
};

}
}
}

#endif