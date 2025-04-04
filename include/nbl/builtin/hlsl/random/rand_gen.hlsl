#ifndef _NBL_HLSL_RANDOM_RANDGEN_INCLUDED_
#define _NBL_HLSL_RANDOM_RANDGEN_INCLUDED_

#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace random
{

template<typename RNG, uint32_t SEED_DIM=1>
struct Uniform2D
{
    using rng_type = RNG;
    using seed_type = conditional_t<SEED_DIM==1, uint32_t, vector<uint32_t, SEED_DIM> >;

    static Uniform2D<RNG> construct(seed_type seed)
    {
        Uniform2D<RNG> retval;
        retval.rng = rng_type::construct(seed);
        return retval;
    }

    uint32_t2 operator()()
    {
        return uint32_t2(rng(), rng());
    }

    rng_type rng;
};

template<typename RNG, uint32_t SEED_DIM=1>
struct Uniform3D
{
    using rng_type = RNG;
    using seed_type = conditional_t<SEED_DIM==1, uint32_t, vector<uint32_t, SEED_DIM> >;

    static Uniform3D<RNG> construct(seed_type seed)
    {
        Uniform3D<RNG> retval;
        retval.rng = rng_type::construct(seed);
        return retval;
    }

    uint32_t3 operator()()
    {
        return uint32_t3(rng(), rng(), rng());
    }

    rng_type rng;
};

}
}
}

#endif