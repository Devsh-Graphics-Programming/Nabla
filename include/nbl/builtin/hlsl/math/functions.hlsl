// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_


#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/concepts.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl
{
namespace hlsl
{
namespace math
{

namespace impl
{

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
static vector<Scalar, 3> reflect_refract(bool _refract, vector<Scalar, 3> I, vector<Scalar, 3> N, Scalar NdotI, Scalar NdotTorR, Scalar rcpOrientedEta)
{    
    return N * (NdotI* (_refract ? rcpOrientedEta : 1.0) + NdotTorR) - I * (_refract ? rcpOrientedEta : 1.0);
}

}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> reflect_refract(bool _refract, vector<Scalar, 3> I, vector<Scalar, 3> N, bool backside, Scalar NdotI, Scalar NdotI2, Scalar rcpOrientedEta, Scalar rcpOrientedEta2)
{
    const Scalar NdotTorR = _refract ? refract_compute_NdotT(backside, NdotI2, rcpOrientedEta2) : NdotI;
    return impl::reflect_refract(_refract, I, N, NdotI, NdotTorR, rcpOrientedEta);
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> reflect_refract(bool _refract, vector<Scalar, 3> I, vector<Scalar, 3> N, Scalar NdotI, Scalar NdotI2, Scalar eta)
{
    Scalar orientedEta, rcpOrientedEta;
    const bool backside = getOrientedEtas(orientedEta, rcpOrientedEta, NdotI, eta);
    return reflect_refract(_refract, I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> reflect(vector<Scalar, 3> I, vector<Scalar, 3> N, Scalar NdotI)
{
    return N * 2.0 * NdotI - I;
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> reflect(vector<Scalar, 3> I, vector<Scalar, 3> N)
{
    Scalar NdotI = dot(N,I);
    return reflect(I, N, NdotI);
}

// for refraction the orientation of the normal matters, because a different IoR will be used
template <typename Scalar, uint32_t Dims>
    NBL_REQUIRES(concepts::scalar<Scalar>)
bool getOrientedEtas(vector<Scalar, Dims> orientedEta, NBL_REF_ARG(vector<Scalar, Dims>) rcpOrientedEta, Scalar NdotI, vector<Scalar, Dims> eta)
{
    const bool backside = NdotI < 0.0;
    const vector<Scalar, Dims> rcpEta = 1.0 / eta;
    orientedEta = backside ? rcpEta : eta;
    rcpOrientedEta = backside ? eta : rcpEta;
    return backside;
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
Scalar refractComputeNdotT2(Scalar NdotI2, Scalar rcpOrientedEta2)
{
    return rcpOrientedEta2 * NdotI2 + 1.0 - rcpOrientedEta2;
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
Scalar refractComputeNdotT(bool backside, Scalar NdotI2, Scalar rcpOrientedEta2)
{
    Scalar absNdotT = sqrt(refractComputeNdotT2(NdotI2,rcpOrientedEta2));
    return backside ? absNdotT : (-absNdotT);
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> refract(vector<Scalar, 3> I, vector<Scalar, 3> N, bool backside, Scalar NdotI, Scalar NdotI2, Scalar rcpOrientedEta, Scalar rcpOrientedEta2)
{
    return N * (NdotI * rcpOrientedEta + refractcomputeNdotT(backside, NdotI2, rcpOrientedEta2)) - rcpOrientedEta * I;
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> refract(vector<Scalar, 3> I, vector<Scalar,3> N, Scalar NdotI, Scalar eta)
{
    Scalar orientedEta, rcpOrientedEta;
    const bool backside = getOrientedEtas(orientedEta, rcpOrientedEta, NdotI, eta);
    return refract(I, N, backside, NdotI, NdotI*NdotI, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> refract(vector<Scalar, 3> I, vector<Scalar, 3> N, Scalar eta)
{
    const Scalar NdotI = dot(N, I);
    return refract(I, N, NdotI, eta);
}


template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> computeUnnormalizedMicrofacetNormal(bool _refract, vector<Scalar, 3> V, vector<Scalar, 3> L, Scalar orientedEta)
{
    const Scalar etaFactor = (_refract ? orientedEta:1.0);
    const vector<Scalar, 3> tmpH = V+L*etaFactor;
    return _refract ? (-tmpH):tmpH;
}

template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
vector<Scalar, 3> computeMicrofacetNormal(bool _refract, vector<Scalar, 3> V, vector<Scalar, 3> L, Scalar orientedEta)
{
    const vector<Scalar, 3> H = computeUnnormalizedMicrofacetNormal(_refract, V, L, orientedEta);
    const Scalar unnormRcpLen = rsqrt(dot(H, H));
    return H * unnormRcpLen;
}

// if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
bool isTransmissionPath(Scalar NdotV, Scalar NdotL)
{
    return bool((asuint(NdotV) ^ asuint(NdotL)) & 0x80000000u);
}


template <typename Scalar>
    NBL_REQUIRES(concepts::scalar<Scalar>)
matrix<Scalar, 3, 2> frisvad(vector<Scalar, 3> n)
{
    const float a = 1.0 / (1.0 + n.z);
    const float b = -n.x * n.y * a;
    return (n.z < -0.9999999)?
        matrix<Scalar, 3, 2>(vector<Scalar, 2>(0.0 ,-1.0), vector<Scalar, 2>(-1.0, 0.0),vector<Scalar, 2>(0.0, 0.0)) : 
        matrix<Scalar, 3, 2>(vector<Scalar, 2>(1.0 - n.x * n.x *a, b), vector<Scalar, 2>(b, 1.0 - n.y * n.y * a), vector<Scalar, 2>(-n.x, -n.y));
}


}
}
}

#endif