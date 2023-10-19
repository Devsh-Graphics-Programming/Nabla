
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_FUNCTIONS_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/common.hlsl>


namespace nbl
{
namespace hlsl
{
namespace math
{

namespace impl
{

static float3 reflect_refract(in bool _refract, in float3 I, in float3 N, in float NdotI, in float NdotTorR, in float rcpOrientedEta)
{    
    return N*(NdotI*(_refract ? rcpOrientedEta:1.0)+NdotTorR) - I*(_refract ? rcpOrientedEta:1.0);
}

static uint bitfieldInsert(in uint base, in uint shifted_masked_value, in uint lo, in uint count)
{
    const uint hi = base^lo;
    return (hi<<count)|shifted_masked_value|lo;
}

}


int dot(in int2 a, in int2 b)    { return a.x*b.x+a.y*b.y; }
uint dot(in uint2 a, in uint2 b) { return a.x*b.x+a.y*b.y; }
int dot(in int3 a, in int3 b)    { return a.x*b.x+a.y*b.y+a.z*b.z; }
uint dot(in uint3 a, in uint3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
int dot(in int4 a, in int4 b)    { return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w; }
uint dot(in uint4 a, in uint4 b) { return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w; }


template<typename vector_t>
vector_t erf(in vector_t _x)
{
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;
    const float p  = 0.3275911;

    vector_t _sign = sign(_x);
    vector_t x = abs(_x);
    
    vector_t t = 1.0 / (1.0 + p*x);
    vector_t y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return _sign*y;
}



float erfInv(in float _x)
{
    float x = clamp(_x, -0.99999, 0.99999);
    float w = -log((1.0-x) * (1.0+x));
    float p;
    if (w<5.0)
    {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p*w;
        p = -3.5233877e-06 + p*w;
        p = -4.39150654e-06 + p*w;
        p = 0.00021858087 + p*w;
        p = -0.00125372503 + p*w;
        p = -0.00417768164 + p*w;
        p = 0.246640727 + p*w;
        p = 1.50140941 + p*w;
    }
    else
    {
        w = sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p*w;
        p = 0.00134934322 + p*w;
        p = -0.00367342844 + p*w;
        p = 0.00573950773 + p*w;
        p = -0.0076224613 + p*w;
        p = 0.00943887047 + p*w;
        p = 1.00167406 + p*w;
        p = 2.83297682 + p*w;
    }
    return p*x;
}

float lengthManhattan(float v)
{
    return abs(v);
}
float lengthManhattan(float2 v)
{
    v = abs(v);
    return v.x + v.y;
}
float lengthManhattan(float3 v)
{
    v = abs(v);
    return v.x + v.y + v.z;
}
float lengthManhattan(float4 v)
{
    v = abs(v);
    return v.x + v.y + v.z + v.w;
}


float lengthSq(in float v)
{
    return v * v;
}
template<typename vector_t>
float lengthSq(in vector_t v)
{
    return dot(v, v);
}


float3 reflect(in float3 I, in float3 N, in float NdotI)
{
    return N*2.0*NdotI - I;
}
float3 reflect(in float3 I, in float3 N)
{
    float NdotI = dot(N,I);
    return reflect(I, N, NdotI);
}

template<typename vector_t>
// for refraction the orientation of the normal matters, because a different IoR will be used
bool getOrientedEtas(out vector_t orientedEta, out vector_t rcpOrientedEta, in float NdotI, in vector_t eta)
{
    const bool backside = NdotI<0.0;
    const vector_t rcpEta = 1.0/eta;
    orientedEta = backside ? rcpEta:eta;
    rcpOrientedEta = backside ? eta:rcpEta;
    return backside;
}


float refract_compute_NdotT2(in float NdotI2, in float rcpOrientedEta2)
{
    return rcpOrientedEta2*NdotI2 + 1.0 - rcpOrientedEta2;
}
float refract_compute_NdotT(in bool backside, in float NdotI2, in float rcpOrientedEta2)
{
    const float abs_NdotT = sqrt(refract_compute_NdotT2(NdotI2,rcpOrientedEta2));
    return backside ? abs_NdotT:(-abs_NdotT);
}
float3 refract(in float3 I, in float3 N, in bool backside, in float NdotI, in float NdotI2, in float rcpOrientedEta, in float rcpOrientedEta2)
{
    return N*(NdotI*rcpOrientedEta + refract_compute_NdotT(backside,NdotI2,rcpOrientedEta2)) - rcpOrientedEta*I;
}
float3 refract(in float3 I, in float3 N, in float NdotI, in float eta)
{
    float orientedEta, rcpOrientedEta;
    const bool backside = getOrientedEtas(orientedEta, rcpOrientedEta, NdotI, eta);
    return refract(I, N, backside, NdotI, NdotI*NdotI, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
}
float3 refract(in float3 I, in float3 N, in float eta)
{
    const float NdotI = dot(N, I);
    return refract(I, N, NdotI, eta);
}



float3 reflect_refract(in bool _refract, in float3 I, in float3 N, in bool backside, in float NdotI, in float NdotI2, in float rcpOrientedEta, in float rcpOrientedEta2)
{
    const float NdotTorR = _refract ? refract_compute_NdotT(backside,NdotI2,rcpOrientedEta2):NdotI;
    return impl::reflect_refract(_refract,I,N,NdotI,NdotTorR,rcpOrientedEta);
}
float3 reflect_refract(in bool _refract, in float3 I, in float3 N, in float NdotI, in float NdotI2, in float eta)
{
    float orientedEta, rcpOrientedEta;
    const bool backside = getOrientedEtas(orientedEta, rcpOrientedEta, NdotI, eta);
    return reflect_refract(_refract, I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
}

// returns unnormalized floattor
float3 computeUnnormalizedMicrofacetNormal(in bool _refract, in float3 V, in float3 L, in float orientedEta)
{
    const float etaFactor = (_refract ? orientedEta:1.0);
    const float3 tmpH = V+L*etaFactor;
    return _refract ? (-tmpH):tmpH;
}
// returns normalized floattor, but NaN when 
float3 computeMicrofacetNormal(in bool _refract, in float3 V, in float3 L, in float orientedEta)
{
    const float3 H = computeUnnormalizedMicrofacetNormal(_refract,V,L,orientedEta);
    const float unnormRcpLen = rsqrt(dot(H,H));
    return H*unnormRcpLen;
}

// if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
bool isTransmissionPath(in float NdotV, in float NdotL)
{
    return bool((asuint(NdotV)^asuint(NdotL)) & 0x80000000u);
}

// valid only for `theta` in [-PI,PI]
void sincos(in float theta, out float s, out float c)
{
    c = cos(theta);
    s = sqrt(1.0-c*c);
    s = theta<0.0 ? -s:s; // TODO: test with XOR
    //s = asfloat(asuint(s)^(asuint(theta)&0x80000000u));
}

float3x2 frisvad(in float3 n)
{
   const float a = 1.0/(1.0 + n.z);
   const float b = -n.x*n.y*a;
   return (n.z<-0.9999999) ? float3x2(float2(0.0,-1.0),float2(-1.0,0.0),float2(0.0,0.0)):float3x2(float2(1.0-n.x*n.x*a, b), float2(b, 1.0-n.y*n.y*a), float2(-n.x, -n.y));
}

// @return if picked right choice
bool partitionRandVariable(in float leftProb, inout float xi, out float rcpChoiceProb)
{
    const float NEXT_ULP_AFTER_UNITY = asfloat(0x3f800001u);
    const bool pickRight = xi>=leftProb*NEXT_ULP_AFTER_UNITY;

    // This is all 100% correct taking into account the above NEXT_ULP_AFTER_UNITY
    xi -= pickRight ? leftProb:0.0;

    rcpChoiceProb = 1.0/(pickRight ? (1.0-leftProb):leftProb);
    xi *= rcpChoiceProb;

    return pickRight;
}

// @ return abs(x) if cond==true, max(x,0.0) otherwise
template<typename vector_t>
vector_t conditionalAbsOrMax(in bool cond, in vector_t x, in vector_t limit)
{
    const vector_t condAbs = asfloat(asuint(x) & (cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}


//! Integer
uint rotl(in uint x, in uint k)
{
   return (x<<k) | (x>>(32u-k));
}

// Count Leading Zeroes
uint clz(in uint x) 
{
    return 31u - firstbithigh(x);
}

// GLSL's builtin is badly named
uint bitfieldOverwrite(in uint base, in uint value, in uint offset, in uint count)
{
    return impl::bitfieldInsert(base,value,int(offset),int(count));
}


uint bitfieldInsert(in uint base, uint value, in uint offset, in uint count)
{
    const uint shifted_masked_value = (value&((0x1u<<count)-1u))<<offset;
    return impl::bitfieldInsert(base,shifted_masked_value,base&((0x1u<<offset)-1u),count);
}

//! Trig

// returns `acos(acos(A)+acos(B)+acos(C))-PI` but requires `sinA,sinB,sinC` are all positive
float getArccosSumofABC_minus_PI(in float cosA, in float cosB, in float cosC, in float sinA, in float sinB, in float sinC)
{
    // sorry about the naming of `something` I just can't seem to be able to give good name to the variables that is consistent with semantics
   const bool something0 = cosA<(-cosB);
    const float cosSumAB = cosA*cosB-sinA*sinB;
   const bool something1 = cosSumAB<(-cosC);
   const bool something2 = cosSumAB<cosC;
   // apply triple angle formula
   const float absArccosSumABC = acos(clamp(cosSumAB*cosC-(cosA*sinB+sinA*cosB)*sinC,-1.f,1.f));
   return ((something0 ? something2:something1) ? (-absArccosSumABC):absArccosSumABC)+(something0||something1 ? PI:(-PI));
}

float2 combineCosForSumOfAcos(in float2 cosA, in float2 cosB) 
{
   const float bias = cosA.y + cosB.y;
   const float a = cosA.x;
   const float b = cosB.x;
   const bool reverse = abs(min(a,b)) > max(a,b);
   const float c = a*b - sqrt((1.0f-a*a)*(1.0f-b*b));
   return reverse ? float2(-c, bias + PI) : float2(c, bias);
}

// returns acos(a) + acos(b)
float getSumofArccosAB(in float cosA, in float cosB) 
{
   const float2 combinedCos = combineCosForSumOfAcos(float2(cosA, 0.0f), float2(cosB, 0.0f));
   return acos(combinedCos.x) + combinedCos.y;
}

// returns acos(a) + acos(b) + acos(c) + acos(d)
float getSumofArccosABCD(in float cosA, in float cosB, in float cosC, in float cosD) 
{
   const float2 combinedCos0 = combineCosForSumOfAcos(float2(cosA, 0.0f), float2(cosB, 0.0f));
   const float2 combinedCos1 = combineCosForSumOfAcos(float2(cosC, 0.0f), float2(cosD, 0.0f));
   const float2 combinedCos = combineCosForSumOfAcos(combinedCos0, combinedCos1);
   return acos(combinedCos.x) + combinedCos.y;
}

//! MVC


// return dFdR (TODO: a completely separate include for this)
float applyChainRule1D(in float dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
float2 applyChainRule1D(in float2 dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
float3 applyChainRule1D(in float3 dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
float4 applyChainRule1D(in float4 dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
/* TODO*/
float1x2 applyChainRule1D(in float dFdG, in float1x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float2x2 applyChainRule1D(in float2x1 dFdG, in float1x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x2 applyChainRule1D(in float3x1 dFdG, in float1x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x2 applyChainRule1D(in float4x1 dFdG, in float1x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float1x3 applyChainRule1D(in float1x1 dFdG, in float1x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float2x3 applyChainRule1D(in float2x1 dFdG, in float1x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x3 applyChainRule1D(in float3x1 dFdG, in float1x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x3 applyChainRule1D(in float4x1 dFdG, in float1x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float1x4 applyChainRule1D(in float1x1 dFdG, in float1x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float2x4 applyChainRule1D(in float2x1 dFdG, in float1x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x4 applyChainRule1D(in float3x1 dFdG, in float1x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x4 applyChainRule1D(in float4x1 dFdG, in float1x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float applyChainRule2D(in float1x2 dFdG, in float2 dGdR)
{
   return mul(dFdG, dGdR);
}
float2 applyChainRule2D(in float2x2 dFdG, in float2 dGdR)
{
   return mul(dFdG, dGdR);
}
float3 applyChainRule2D(in float3x2 dFdG, in float2 dGdR)
{
   return mul(dFdG, dGdR);
}
float4 applyChainRule2D(in float4x2 dFdG, in float2 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x2 applyChainRule2D(in float1x2 dFdG, in float2x2 dGdR) // needed for deriv map
{
   return mul(dFdG, dGdR);
}

float2x2 applyChainRule2D(in float2x2 dFdG, in float2x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x2 applyChainRule2D(in float3x2 dFdG, in float2x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x2 applyChainRule2D(in float4x2 dFdG, in float2x2 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x3 applyChainRule2D(in float1x2 dFdG, in float2x3 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x3 applyChainRule2D(in float2x2 dFdG, in float2x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x3 applyChainRule2D(in float3x2 dFdG, in float2x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x4 applyChainRule2D(in float4x2 dFdG, in float2x4 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x4 applyChainRule2D(in float1x2 dFdG, in float2x4 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x4 applyChainRule2D(in float2x2 dFdG, in float2x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x4 applyChainRule2D(in float3x2 dFdG, in float2x4 dGdR)
{
   return mul(dFdG, dGdR);
}



float applyChainRule3D(in float1x3 dFdG, in float3 dGdR)
{
   return mul(dFdG, dGdR);
}

float2 applyChainRule3D(in float2x3 dFdG, in float3 dGdR)
{
   return mul(dFdG, dGdR);
}
float3 applyChainRule3D(in float3x3 dFdG, in float3 dGdR)
{
   return mul(dFdG, dGdR);
}
float4 applyChainRule3D(in float4x3 dFdG, in float3 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x2 applyChainRule3D(in float1x3 dFdG, in float3x2 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x2 applyChainRule3D(in float2x3 dFdG, in float3x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x2 applyChainRule3D(in float3x3 dFdG, in float3x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x2 applyChainRule3D(in float4x3 dFdG, in float3x2 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x3 applyChainRule3D(in float1x3 dFdG, in float3x3 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x3 applyChainRule3D(in float2x3 dFdG, in float3x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x3 applyChainRule3D(in float3x3 dFdG, in float3x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x3 applyChainRule3D(in float4x3 dFdG, in float3x3 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x4 applyChainRule3D(in float1x3 dFdG, in float3x4 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x4 applyChainRule3D(in float2x3 dFdG, in float3x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x4 applyChainRule3D(in float3x3 dFdG, in float3x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x4 applyChainRule3D(in float4x3 dFdG, in float3x4 dGdR)
{
   return mul(dFdG, dGdR);
}



float applyChainRule4D(in float1x4 dFdG, in float4 dGdR)
{
   return mul(dFdG, dGdR);
}

float2 applyChainRule4D(in float2x4 dFdG, in float4 dGdR)
{
   return mul(dFdG, dGdR);
}
float3 applyChainRule4D(in float3x4 dFdG, in float4 dGdR)
{
   return mul(dFdG, dGdR);
}
float4 applyChainRule4D(in float4x4 dFdG, in float4 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x2 applyChainRule4D(in float1x4 dFdG, in float4x2 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x2 applyChainRule4D(in float2x4 dFdG, in float4x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x2 applyChainRule4D(in float3x4 dFdG, in float4x2 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x2 applyChainRule4D(in float4x4 dFdG, in float4x2 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x3 applyChainRule4D(in float1x4 dFdG, in float4x3 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x3 applyChainRule4D(in float2x4 dFdG, in float4x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x3 applyChainRule4D(in float3x4 dFdG, in float4x3 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x3 applyChainRule4D(in float4x4 dFdG, in float4x3 dGdR)
{
   return mul(dFdG, dGdR);
}

float1x4 applyChainRule4D(in float1x4 dFdG, in float4x4 dGdR)
{
   return mul(dFdG, dGdR);
}

float2x4 applyChainRule4D(in float2x4 dFdG, in float4x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float3x4 applyChainRule4D(in float3x4 dFdG, in float4x4 dGdR)
{
   return mul(dFdG, dGdR);
}
float4x4 applyChainRule4D(in float4x4 dFdG, in float4x4 dGdR)
{
   return mul(dFdG, dGdR);
}


}
}
}

#endif