// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl
{
namespace hlsl
{

namespace bxdf
{

namespace impl
{
template<typename T>
struct orientedEtas;

template<>
struct orientedEtas<float>
{
    static bool __call(NBL_REF_ARG(float) orientedEta, NBL_REF_ARG(float) rcpOrientedEta, float NdotI, float eta)
    {
        const bool backside = NdotI < 0.0;
        const float rcpEta = 1.0 / eta;
        orientedEta = backside ? rcpEta : eta;
        rcpOrientedEta = backside ? eta : rcpEta;
        return backside;
    }
};

template<>
struct orientedEtas<float32_t3>
{
    static bool __call(NBL_REF_ARG(float32_t3) orientedEta, NBL_REF_ARG(float32_t3) rcpOrientedEta, float NdotI, float32_t3 eta)
    {
        const bool backside = NdotI < 0.0;
        const float32_t3 rcpEta = (float32_t3)1.0 / eta;
        orientedEta = backside ? rcpEta:eta;
        rcpOrientedEta = backside ? eta:rcpEta;
        return backside;
    }
};
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T> || is_vector_v<T>)
bool getOrientedEtas(NBL_REF_ARG(T) orientedEta, NBL_REF_ARG(T) rcpOrientedEta, scalar_type_t<T> NdotI, T eta)
{
    return impl::orientedEtas<T>::__call(orientedEta, rcpOrientedEta, NdotI, eta);
}


template <typename T NBL_FUNC_REQUIRES(concepts::Vectorial<T>)
T reflect(T I, T N, typename vector_traits<T>::scalar_type NdotI)
{
    return N * 2.0f * NdotI - I;
}

template<typename T NBL_PRIMARY_REQUIRES(vector_traits<T>::Dimension == 3)
struct refract
{
    using this_t = refract<T>;
    using vector_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, bool backside, scalar_type NdotI, scalar_type NdotI2, scalar_type rcpOrientedEta, scalar_type rcpOrientedEta2)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.backside = backside;
        retval.NdotI = NdotI;
        retval.NdotI2 = NdotI2;
        retval.rcpOrientedEta = rcpOrientedEta;
        retval.rcpOrientedEta2 = rcpOrientedEta2;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type NdotI, scalar_type eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        scalar_type orientedEta;
        retval.backside = getOrientedEtas<scalar_type>(orientedEta, retval.rcpOrientedEta, NdotI, eta);
        retval.NdotI = NdotI;
        retval.NdotI2 = NdotI * NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(vector_type) I, NBL_CONST_REF_ARG(vector_type) N, scalar_type eta)
    {
        this_t retval;
        retval.I = I;
        retval.N = N;
        retval.NdotI = dot<vector_type>(N, I);
        scalar_type orientedEta;
        retval.backside = getOrientedEtas<scalar_type>(orientedEta, retval.rcpOrientedEta, retval.NdotI, eta);
        retval.NdotI2 = retval.NdotI * retval.NdotI;
        retval.rcpOrientedEta2 = retval.rcpOrientedEta * retval.rcpOrientedEta;
        return retval;
    }

    static scalar_type computeNdotT(bool backside, scalar_type NdotI2, scalar_type rcpOrientedEta2)
    {
        scalar_type NdotT2 = rcpOrientedEta2 * NdotI2 + 1.0 - rcpOrientedEta2;
        scalar_type absNdotT = sqrt<scalar_type>(NdotT2);
        return backside ? absNdotT : -(absNdotT);
    }

    vector_type doRefract()
    {
        return N * (NdotI * rcpOrientedEta + computeNdotT(backside, NdotI2, rcpOrientedEta2)) - rcpOrientedEta * I;
    }

    static vector_type doReflectRefract(bool _refract, NBL_CONST_REF_ARG(vector_type) _I, NBL_CONST_REF_ARG(vector_type) _N, scalar_type _NdotI, scalar_type _NdotTorR, scalar_type _rcpOrientedEta)
    {
        return _N * (_NdotI * (_refract ? _rcpOrientedEta : 1.0f) + _NdotTorR) - _I * (_refract ? _rcpOrientedEta : 1.0f);
    }

    vector_type doReflectRefract(bool r)
    {
        const T NdotTorR = r ? computeNdotT(backside, NdotI2, rcpOrientedEta2) : NdotI;
        return doReflectRefract(r, I, N, NdotI, NdotTorR, rcpOrientedEta);
    }

    vector_type I;
    vector_type N;
    bool backside;
    scalar_type NdotI;
    scalar_type NdotI2;
    scalar_type rcpOrientedEta;
    scalar_type rcpOrientedEta2;
};

}

}
}

#endif
