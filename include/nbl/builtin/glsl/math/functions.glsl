// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATH_FUNCTIONS_INCLUDED_
#define _NBL_MATH_FUNCTIONS_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>

int nbl_glsl_dot(in ivec2 a, in ivec2 b) {return a.x*b.x+a.y*b.y;}
uint nbl_glsl_dot(in uvec2 a, in uvec2 b) {return a.x*b.x+a.y*b.y;}
int nbl_glsl_dot(in ivec3 a, in ivec3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
uint nbl_glsl_dot(in uvec3 a, in uvec3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
int nbl_glsl_dot(in ivec4 a, in ivec4 b) {return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;}
uint nbl_glsl_dot(in uvec4 a, in uvec4 b) {return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;}
float nbl_glsl_cross( in vec2 a, in vec2 b ) {return a.x*b.y-a.y*b.x;}

//
float nbl_glsl_erf(in float _x)
{
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;
    const float p = 0.3275911;

    float sign = sign(_x);
    float x = abs(_x);
    
    float t = 1.0 / (1.0 + p*x);
    float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return sign*y;
}

float nbl_glsl_erfInv(in float _x)
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

float nbl_glsl_lengthManhattan(float v)
{
    return abs(v);
}
float nbl_glsl_lengthManhattan(vec2 v)
{
    v = abs(v);
    return v.x + v.y;
}
float nbl_glsl_lengthManhattan(vec3 v)
{
    v = abs(v);
    return v.x + v.y + v.z;
}
float nbl_glsl_lengthManhattan(vec4 v)
{
    v = abs(v);
    return v.x + v.y + v.z + v.w;
}

float nbl_glsl_lengthSq(in float v)
{
    return v * v;
}
float nbl_glsl_lengthSq(in vec2 v)
{
    return dot(v, v);
}
float nbl_glsl_lengthSq(in vec3 v)
{
    return dot(v, v);
}
float nbl_glsl_lengthSq(in vec4 v)
{
    return dot(v, v);
}

vec3 nbl_glsl_reflect(in vec3 I, in vec3 N, in float NdotI)
{
    return N*2.0*NdotI - I;
}
vec3 nbl_glsl_reflect(in vec3 I, in vec3 N)
{
    float NdotI = dot(N,I);
    return nbl_glsl_reflect(I, N, NdotI);
}

// for refraction the orientation of the normal matters, because a different IoR will be used
bool nbl_glsl_getOrientedEtas(out float orientedEta, out float rcpOrientedEta, in float NdotI, in float eta)
{
    const bool backside = NdotI<0.0;
    const float rcpEta = 1.0/eta;
    orientedEta = backside ? rcpEta:eta;
    rcpOrientedEta = backside ? eta:rcpEta;
    return backside;
}
bool nbl_glsl_getOrientedEtas(out vec3 orientedEta, out vec3 rcpOrientedEta, in float NdotI, in vec3 eta)
{
    const bool backside = NdotI<0.0;
    const vec3 rcpEta = vec3(1.0) / eta;
    orientedEta = backside ? rcpEta:eta;
    rcpOrientedEta = backside ? eta:rcpEta;
    return backside;
}

float nbl_glsl_refract_compute_NdotT2(in float NdotI2, in float rcpOrientedEta2)
{
    return rcpOrientedEta2*NdotI2 + 1.0 - rcpOrientedEta2;
}
float nbl_glsl_refract_compute_NdotT(in bool backside, in float NdotI2, in float rcpOrientedEta2)
{
    const float abs_NdotT = sqrt(nbl_glsl_refract_compute_NdotT2(NdotI2,rcpOrientedEta2));
    return backside ? abs_NdotT:(-abs_NdotT);
}
vec3 nbl_glsl_refract(in vec3 I, in vec3 N, in bool backside, in float NdotI, in float NdotI2, in float rcpOrientedEta, in float rcpOrientedEta2)
{
    return N*(NdotI*rcpOrientedEta + nbl_glsl_refract_compute_NdotT(backside,NdotI2,rcpOrientedEta2)) - rcpOrientedEta*I;
}
vec3 nbl_glsl_refract(in vec3 I, in vec3 N, in float NdotI, in float eta)
{
    float orientedEta, rcpOrientedEta;
    const bool backside = nbl_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, NdotI, eta);
    return nbl_glsl_refract(I, N, backside, NdotI, NdotI*NdotI, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
}
vec3 nbl_glsl_refract(in vec3 I, in vec3 N, in float eta)
{
    const float NdotI = dot(N, I);
    return nbl_glsl_refract(I, N, NdotI, eta);
}

vec3 nbl_glsl_reflect_refract_impl(in bool _refract, in vec3 I, in vec3 N, in float NdotI, in float NdotTorR, in float rcpOrientedEta)
{    
    return N*(NdotI*(_refract ? rcpOrientedEta:1.0)+NdotTorR) - I*(_refract ? rcpOrientedEta:1.0);
}
vec3 nbl_glsl_reflect_refract(in bool _refract, in vec3 I, in vec3 N, in bool backside, in float NdotI, in float NdotI2, in float rcpOrientedEta, in float rcpOrientedEta2)
{
    const float NdotTorR = _refract ? nbl_glsl_refract_compute_NdotT(backside,NdotI2,rcpOrientedEta2):NdotI;
    return nbl_glsl_reflect_refract_impl(_refract,I,N,NdotI,NdotTorR,rcpOrientedEta);
}
vec3 nbl_glsl_reflect_refract(in bool _refract, in vec3 I, in vec3 N, in float NdotI, in float NdotI2, in float eta)
{
    float orientedEta, rcpOrientedEta;
    const bool backside = nbl_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, NdotI, eta);
    return nbl_glsl_reflect_refract(_refract, I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta*rcpOrientedEta);
}

// returns unnormalized vector
vec3 nbl_glsl_computeUnnormalizedMicrofacetNormal(in bool _refract, in vec3 V, in vec3 L, in float orientedEta)
{
    const float etaFactor = (_refract ? orientedEta:1.0);
    const vec3 tmpH = V+L*etaFactor;
    return _refract ? (-tmpH):tmpH;
}
// returns normalized vector, but NaN when 
vec3 nbl_glsl_computeMicrofacetNormal(in bool _refract, in vec3 V, in vec3 L, in float orientedEta)
{
    const vec3 H = nbl_glsl_computeUnnormalizedMicrofacetNormal(_refract,V,L,orientedEta);
    const float unnormRcpLen = inversesqrt(dot(H,H));
    return H*unnormRcpLen;
}

// if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
bool nbl_glsl_isTransmissionPath(in float NdotV, in float NdotL)
{
    return bool((floatBitsToUint(NdotV)^floatBitsToUint(NdotL)) & 0x80000000u);
}

// valid only for `theta` in [-PI,PI]
void nbl_glsl_sincos(in float theta, out float s, out float c)
{
    c = cos(theta);
    s = sqrt(1.0-c*c);
    s = theta<0.0 ? -s:s; // TODO: test with XOR
    //s = uintBitsToFloat(floatBitsToUint(s)^(floatBitsToUint(theta)&0x80000000u));
}

mat2x3 nbl_glsl_frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}

// @return if picked right choice
bool nbl_glsl_partitionRandVariable(in float leftProb, inout float xi, out float rcpChoiceProb)
{
    const float NEXT_ULP_AFTER_UNITY = uintBitsToFloat(0x3f800001u);
    const bool pickRight = xi>=leftProb*NEXT_ULP_AFTER_UNITY;

    // This is all 100% correct taking into account the above NEXT_ULP_AFTER_UNITY
    xi -= pickRight ? leftProb:0.0;

    rcpChoiceProb = 1.0/(pickRight ? (1.0-leftProb):leftProb);
    xi *= rcpChoiceProb;

    return pickRight;
}

// @ return abs(x) if cond==true, max(x,0.0) otherwise
float nbl_glsl_conditionalAbsOrMax(in bool cond, in float x, in float limit)
{
    const float condAbs = uintBitsToFloat(floatBitsToUint(x) & uint(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}
vec2 nbl_glsl_conditionalAbsOrMax(in bool cond, in vec2 x, in vec2 limit)
{
    const vec2 condAbs = uintBitsToFloat(floatBitsToUint(x) & uvec2(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}
vec3 nbl_glsl_conditionalAbsOrMax(in bool cond, in vec3 x, in vec3 limit)
{
    const vec3 condAbs = uintBitsToFloat(floatBitsToUint(x) & uvec3(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}
vec4 nbl_glsl_conditionalAbsOrMax(in bool cond, in vec4 x, in vec4 limit)
{
    const vec4 condAbs = uintBitsToFloat(floatBitsToUint(x) & uvec4(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}

//! Integer
uint nbl_glsl_rotl(in uint x, in uint k)
{
	return (x<<k) | (x>>(32u-k));
}

// Count Leading Zeroes
uint nbl_glsl_clz(in uint x) 
{
    return 31u - findMSB(x);
}

// GLSL's builtin is badly named
uint nbl_glsl_bitfieldOverwrite(in uint base, in uint value, in uint offset, in uint count)
{
    return bitfieldInsert(base,value,int(offset),int(count));
}

uint nbl_glsl_bitfieldInsert_impl(in uint base, in uint shifted_masked_value, in uint lo, in uint count)
{
    const uint hi = base^lo;
    return (hi<<count)|shifted_masked_value|lo;
}
uint nbl_glsl_bitfieldInsert(in uint base, uint value, in uint offset, in uint count)
{
    const uint shifted_masked_value = (value&((0x1u<<count)-1u))<<offset;
    return nbl_glsl_bitfieldInsert_impl(base,shifted_masked_value,base&((0x1u<<offset)-1u),count);
}

//! Trig

// returns `acos(acos(A)+acos(B)+acos(C))-PI` but requires `sinA,sinB,sinC` are all positive
float nbl_glsl_getArccosSumofABC_minus_PI(in float cosA, in float cosB, in float cosC, in float sinA, in float sinB, in float sinC)
{
    // sorry about the naming of `something` I just can't seem to be able to give good name to the variables that is consistent with semantics
	const bool something0 = cosA<(-cosB);
    const float cosSumAB = cosA*cosB-sinA*sinB;
	const bool something1 = cosSumAB<(-cosC);
	const bool something2 = cosSumAB<cosC;
	// apply triple angle formula
	const float absArccosSumABC = acos(clamp(cosSumAB*cosC-(cosA*sinB+sinA*cosB)*sinC,-1.f,1.f));
	return ((something0 ? something2:something1) ? (-absArccosSumABC):absArccosSumABC)+(something0||something1 ? nbl_glsl_PI:(-nbl_glsl_PI));
}


//! MVC

// return dFdR (TODO: a completely separate include for this)
float nbl_glsl_applyChainRule1D(in float dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
vec2 nbl_glsl_applyChainRule1D(in vec2 dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
vec3 nbl_glsl_applyChainRule1D(in vec3 dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
vec4 nbl_glsl_applyChainRule1D(in vec4 dFdG, in float dGdR)
{
   return dFdG*dGdR;
}
/* TODO
mat2x1 nbl_glsl_applyChainRule1D(in float dFdG, in mat2x1 dGdR)
{
   return dFdG*dGdR;
}
mat2 nbl_glsl_applyChainRule1D(in vec2 dFdG, in mat2x1 dGdR)
{
   return dFdG*dGdR;
}
mat2x3 nbl_glsl_applyChainRule1D(in vec3 dFdG, in mat2x1 dGdR)
{
   return dFdG*dGdR;
}
mat2x4 nbl_glsl_applyChainRule1D(in vec4 dFdG, in mat2x1 dGdR)
{
   return dFdG*dGdR;
}
mat3x1 nbl_glsl_applyChainRule1D(in float dFdG, in mat3x1 dGdR)
{
   return dFdG*dGdR;
}
mat3x2 nbl_glsl_applyChainRule1D(in vec2 dFdG, in mat3x1 dGdR)
{
   return dFdG*dGdR;
}
mat3x3 nbl_glsl_applyChainRule1D(in vec3 dFdG, in mat3x1 dGdR)
{
   return dFdG*dGdR;
}
mat3x4 nbl_glsl_applyChainRule1D(in vec4 dFdG, in mat3x1 dGdR)
{
   return dFdG*dGdR;
}
mat4x1 nbl_glsl_applyChainRule1D(in float dFdG, in mat4x1 dGdR)
{
   return dFdG*dGdR;
}
mat4x2 nbl_glsl_applyChainRule1D(in vec2 dFdG, in mat4x1 dGdR)
{
   return dFdG*dGdR;
}
mat4x3 nbl_glsl_applyChainRule1D(in vec3 dFdG, in mat4x1 dGdR)
{
   return dFdG*dGdR;
}
mat4 nbl_glsl_applyChainRule1D(in vec4 dFdG, in mat4x1 dGdR)
{
   return dFdG*dGdR;
}


float nbl_glsl_applyChainRule2D(in mat2x1 dFdG, in vec2 dGdR)
{
   return dFdG*dGdR;
}*/
vec2 nbl_glsl_applyChainRule2D(in mat2 dFdG, in vec2 dGdR)
{
   return dFdG*dGdR;
}
vec3 nbl_glsl_applyChainRule2D(in mat2x3 dFdG, in vec2 dGdR)
{
   return dFdG*dGdR;
}
vec4 nbl_glsl_applyChainRule2D(in mat2x4 dFdG, in vec2 dGdR)
{
   return dFdG*dGdR;
}
/*
mat2x1 nbl_glsl_applyChainRule2D(in mat2x1 dFdG, in mat2 dGdR) // needed for deriv map
{
   return dFdG*dGdR;
}
*/
mat2 nbl_glsl_applyChainRule2D(in mat2 dFdG, in mat2 dGdR)
{
   return dFdG*dGdR;
}
mat2x3 nbl_glsl_applyChainRule2D(in mat2x3 dFdG, in mat2 dGdR)
{
   return dFdG*dGdR;
}
mat2x4 nbl_glsl_applyChainRule2D(in mat2x4 dFdG, in mat2 dGdR)
{
   return dFdG*dGdR;
}
/*
mat3x1 nbl_glsl_applyChainRule2D(in mat2x1 dFdG, in mat3x2 dGdR)
{
   return dFdG*dGdR;
}
*/
mat3x2 nbl_glsl_applyChainRule2D(in mat2 dFdG, in mat3x2 dGdR)
{
   return dFdG*dGdR;
}
mat3 nbl_glsl_applyChainRule2D(in mat2x3 dFdG, in mat3x2 dGdR)
{
   return dFdG*dGdR;
}
mat3x4 nbl_glsl_applyChainRule2D(in mat2x4 dFdG, in mat3x2 dGdR)
{
   return dFdG*dGdR;
}
/*
mat4x1 nbl_glsl_applyChainRule2D(in mat2x1 dFdG, in mat4x2 dGdR)
{
   return dFdG*dGdR;
}
*/
mat4x2 nbl_glsl_applyChainRule2D(in mat2 dFdG, in mat4x2 dGdR)
{
   return dFdG*dGdR;
}
mat4x3 nbl_glsl_applyChainRule2D(in mat2x3 dFdG, in mat4x2 dGdR)
{
   return dFdG*dGdR;
}
mat4 nbl_glsl_applyChainRule2D(in mat2x4 dFdG, in mat4x2 dGdR)
{
   return dFdG*dGdR;
}


/*
float nbl_glsl_applyChainRule3D(in mat3x1 dFdG, in vec3 dGdR)
{
   return dFdG*dGdR;
}
*/
vec2 nbl_glsl_applyChainRule3D(in mat3x2 dFdG, in vec3 dGdR)
{
   return dFdG*dGdR;
}
vec3 nbl_glsl_applyChainRule3D(in mat3 dFdG, in vec3 dGdR)
{
   return dFdG*dGdR;
}
vec4 nbl_glsl_applyChainRule3D(in mat3x4 dFdG, in vec3 dGdR)
{
   return dFdG*dGdR;
}
/*
mat2x1 nbl_glsl_applyChainRule3D(in mat3x1 dFdG, in mat2x3 dGdR)
{
   return dFdG*dGdR;
}
*/
mat2 nbl_glsl_applyChainRule3D(in mat3x2 dFdG, in mat2x3 dGdR)
{
   return dFdG*dGdR;
}
mat2x3 nbl_glsl_applyChainRule3D(in mat3 dFdG, in mat2x3 dGdR)
{
   return dFdG*dGdR;
}
mat2x4 nbl_glsl_applyChainRule3D(in mat3x4 dFdG, in mat2x3 dGdR)
{
   return dFdG*dGdR;
}
/*
mat3x1 nbl_glsl_applyChainRule3D(in mat3x1 dFdG, in mat3 dGdR)
{
   return dFdG*dGdR;
}
*/
mat3x2 nbl_glsl_applyChainRule3D(in mat3x2 dFdG, in mat3 dGdR)
{
   return dFdG*dGdR;
}
mat3 nbl_glsl_applyChainRule3D(in mat3 dFdG, in mat3 dGdR)
{
   return dFdG*dGdR;
}
mat3x4 nbl_glsl_applyChainRule3D(in mat3x4 dFdG, in mat3 dGdR)
{
   return dFdG*dGdR;
}
/*
mat4x1 nbl_glsl_applyChainRule3D(in mat3x1 dFdG, in mat4x3 dGdR)
{
   return dFdG*dGdR;
}
*/
mat4x2 nbl_glsl_applyChainRule3D(in mat3x2 dFdG, in mat4x3 dGdR)
{
   return dFdG*dGdR;
}
mat4x3 nbl_glsl_applyChainRule3D(in mat3 dFdG, in mat4x3 dGdR)
{
   return dFdG*dGdR;
}
mat4 nbl_glsl_applyChainRule3D(in mat3x4 dFdG, in mat4x3 dGdR)
{
   return dFdG*dGdR;
}


/*
float nbl_glsl_applyChainRule4D(in mat4x1 dFdG, in vec4 dGdR)
{
   return dFdG*dGdR;
}
*/
vec2 nbl_glsl_applyChainRule4D(in mat4x2 dFdG, in vec4 dGdR)
{
   return dFdG*dGdR;
}
vec3 nbl_glsl_applyChainRule4D(in mat4x3 dFdG, in vec4 dGdR)
{
   return dFdG*dGdR;
}
vec4 nbl_glsl_applyChainRule4D(in mat4 dFdG, in vec4 dGdR)
{
   return dFdG*dGdR;
}
/*
mat2x1 nbl_glsl_applyChainRule4D(in mat4x1 dFdG, in mat2x4 dGdR)
{
   return dFdG*dGdR;
}
*/
mat2 nbl_glsl_applyChainRule4D(in mat4x2 dFdG, in mat2x4 dGdR)
{
   return dFdG*dGdR;
}
mat2x3 nbl_glsl_applyChainRule4D(in mat4x3 dFdG, in mat2x4 dGdR)
{
   return dFdG*dGdR;
}
mat2x4 nbl_glsl_applyChainRule4D(in mat4 dFdG, in mat2x4 dGdR)
{
   return dFdG*dGdR;
}
/*
mat3x1 nbl_glsl_applyChainRule4D(in mat4x1 dFdG, in mat3x4 dGdR)
{
   return dFdG*dGdR;
}
*/
mat3x2 nbl_glsl_applyChainRule4D(in mat4x2 dFdG, in mat3x4 dGdR)
{
   return dFdG*dGdR;
}
mat3 nbl_glsl_applyChainRule4D(in mat4x3 dFdG, in mat3x4 dGdR)
{
   return dFdG*dGdR;
}
mat3x4 nbl_glsl_applyChainRule4D(in mat4 dFdG, in mat3x4 dGdR)
{
   return dFdG*dGdR;
}
/*
mat4x1 nbl_glsl_applyChainRule4D(in mat4x1 dFdG, in mat4 dGdR)
{
   return dFdG*dGdR;
}
*/
mat4x2 nbl_glsl_applyChainRule4D(in mat4x2 dFdG, in mat4 dGdR)
{
   return dFdG*dGdR;
}
mat4x3 nbl_glsl_applyChainRule4D(in mat4x3 dFdG, in mat4 dGdR)
{
   return dFdG*dGdR;
}
mat4 nbl_glsl_applyChainRule4D(in mat4 dFdG, in mat4 dGdR)
{
   return dFdG*dGdR;
}

#endif
