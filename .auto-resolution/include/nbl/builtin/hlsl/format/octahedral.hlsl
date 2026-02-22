#ifndef _NBL_HLSL_FORMAT_OCTAHEDRAL_HLSL_
#define _NBL_HLSL_FORMAT_OCTAHEDRAL_HLSL_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/limits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace format
{

template<typename UintT, uint16_t Bits=sizeof(UintT)*4>
struct octahedral// : enable_if_t<Bits*2>sizeof(UintT)||Bits*4<sizeof(UintT)> need a way to static_assert in SPIRV!
{
    using this_t = octahedral<UintT,Bits>;
    using storage_t = UintT;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsUsed = Bits;

    inline bool operator==(const this_t other)
    {
        return storage==other.storage;
    }
    inline bool operator!=(const this_t other)
    {
        return storage==other.storage;
    }

    storage_t storage;
};

}

// https://www.shadertoy.com/view/Mtfyzl
namespace impl
{
// decode
template<typename float_t, typename UintT, uint16_t Bits>
struct static_cast_helper<vector<float_t,3>,format::octahedral<UintT,Bits> >
{
    using T = vector<float_t,3>;
    using U = format::octahedral<UintT,Bits>;

    static inline T cast(U val)
    {
        using storage_t = typename U::storage_t;
        const storage_t MaxVal = (storage_t(1)<<U::BitsUsed)-1u;
    
        // NOTE: We Assume the top unused bits are clean!
        const vector<float_t,2> v = vector<float_t,2>(val.storage&MaxVal,val.storage>>U::BitsUsed) / (vector<float_t,2>(MaxVal,MaxVal)*0.5) - vector<float_t,2>(1,1);

        // Rune Stubbe's version, much faster than original
        vector<float_t,3> nor = vector<float_t,3>(v,float_t(1)-abs(v.x)-abs(v.y));
        const float_t t = max(-nor.z,float_t(0));
        // TODO: improve the copysign with `^` and a sign mask
        nor.x += (nor.x>0.0) ? -t:t;
        nor.y += (nor.y>0.0) ? -t:t;

        return normalize(nor);
    }
};
// encode
template<typename UintT, uint16_t Bits, typename float_t>
struct static_cast_helper<format::octahedral<UintT,Bits>,vector<float_t,3> >
{
    using T = format::octahedral<UintT,Bits>;
    using U = vector<float_t,3>;

    static inline T cast(U nor)
    {
        nor /= (abs(nor.x) + abs(nor.y) + abs(nor.z));
        if (nor.z<float_t(0)) // TODO: faster sign copy
            nor.xy = (float_t(1)-abs(nor.yx))*sign(nor.xy);

        vector<float_t,2> v = nor.xy*float_t(0.5)+vector<float_t,2>(0.5,0.5);

        using storage_t = typename T::storage_t;
        const storage_t MaxVal = (storage_t(1)<<T::BitsUsed)-1u;
        const vector<storage_t,2> d = vector<storage_t,2>(v*float_t(MaxVal)+vector<float_t,2>(0.5,0.5));

        T retval;
        retval.storage = (d.y<<T::BitsUsed)|d.x;
        return retval;
    }
};
}

}
}
#endif