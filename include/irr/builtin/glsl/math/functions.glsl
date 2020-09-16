#ifndef _IRR_MATH_FUNCTIONS_INCLUDED_
#define _IRR_MATH_FUNCTIONS_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>

float irr_glsl_erf(in float _x)
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

float irr_glsl_erfInv(in float _x)
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

float irr_glsl_lengthManhattan(float v)
{
    return abs(v);
}
float irr_glsl_lengthManhattan(vec2 v)
{
    v = abs(v);
    return v.x + v.y;
}
float irr_glsl_lengthManhattan(vec3 v)
{
    v = abs(v);
    return v.x + v.y + v.z;
}
float irr_glsl_lengthManhattan(vec4 v)
{
    v = abs(v);
    return v.x + v.y + v.z + v.w;
}

float irr_glsl_lengthSq(in float v)
{
    return v * v;
}
float irr_glsl_lengthSq(in vec2 v)
{
    return dot(v, v);
}
float irr_glsl_lengthSq(in vec3 v)
{
    return dot(v, v);
}
float irr_glsl_lengthSq(in vec4 v)
{
    return dot(v, v);
}

vec3 irr_glsl_reflect(in vec3 I, in vec3 N, in float NdotI)
{
    return N*2.0*NdotI - I;
}
vec3 irr_glsl_reflect(in vec3 I, in vec3 N)
{
    float NdotI = dot(N,I);
    return irr_glsl_reflect(I, N, NdotI);
}

// for refraction the orientation of the normal matters, because a different IoR will be used
bool irr_glsl_getOrientedEtas(out float rcpOrientedEta, out float orientedEta2, out float rcpOrientedEta2, in float NdotI, in float eta)
{
    const bool backside = NdotI<0.0;
    const float rcpEta = 1.0/eta;
    const float orientedEta = backside ? rcpEta:eta;
    rcpOrientedEta = backside ? eta:rcpEta;
    orientedEta2 = orientedEta * orientedEta;
    rcpOrientedEta2 = rcpOrientedEta*rcpOrientedEta;
    return backside;
}
bool irr_glsl_getOrientedEtas(out vec3 rcpOrientedEta, out vec3 orientedEta2, out vec3 rcpOrientedEta2, in float NdotI, in vec3 eta)
{
    const bool backside = NdotI<0.0;
    const vec3 rcpEta = 1.0 / eta;
    const vec3 orientedEta = backside ? rcpEta:eta;
    rcpOrientedEta = backside ? eta:rcpEta;
    orientedEta2 = orientedEta * orientedEta;
    rcpOrientedEta2 = rcpOrientedEta*rcpOrientedEta;
    return backside;
}

float irr_glsl_refract_compute_NdotT2(in float NdotI2, in float rcpOrientedEta2)
{
    return rcpOrientedEta2*NdotI2 + 1.0 - rcpOrientedEta2;
}
float irr_glsl_refract_compute_NdotT(in bool backside, in float NdotI2, in float rcpOrientedEta2)
{
    const float abs_NdotT = sqrt(irr_glsl_refract_compute_NdotT2(NdotI2,rcpOrientedEta2));
    return backside ? abs_NdotT:(-abs_NdotT);
}
vec3 irr_glsl_refract(in vec3 I, in vec3 N, in bool backside, in float NdotI, in float NdotI2, in float rcpOrientedEta, in float rcpOrientedEta2)
{
    return N*(NdotI*rcpOrientedEta + irr_glsl_refract_compute_NdotT(backside,NdotI2,rcpOrientedEta2)) - rcpOrientedEta*I;
}
vec3 irr_glsl_refract(in vec3 I, in vec3 N, in float NdotI, in float eta)
{
    float rcpOrientedEta, dummy, rcpOrientedEta2;
    const bool backside = irr_glsl_getOrientedEtas(rcpOrientedEta, dummy, rcpOrientedEta2, NdotI, eta);
    return irr_glsl_refract(I, N, backside, NdotI, NdotI*NdotI, rcpOrientedEta, rcpOrientedEta2);
}
vec3 irr_glsl_refract(in vec3 I, in vec3 N, in float eta)
{
    const float NdotI = dot(N, I);
    return irr_glsl_refract(I, N, NdotI, eta);
}

vec3 irr_glsl_reflect_refract(in bool _refract, in vec3 I, in vec3 N, in bool backside, in float NdotI, in float NdotI2, in float rcpOrientedEta, in float rcpOrientedEta2)
{    
    const float k = _refract ? irr_glsl_refract_compute_NdotT(backside,NdotI2,rcpOrientedEta2):0.0;
    return N*(NdotI*(_refract ? rcpOrientedEta:2.0)+k) - I*(_refract ? rcpOrientedEta:1.0);
}
vec3 irr_glsl_reflect_refract(in bool _refract, in vec3 I, in vec3 N, in float NdotI, in float NdotI2, in float eta)
{
    float rcpOrientedEta, dummy, rcpOrientedEta2;
    const bool backside = irr_glsl_getOrientedEtas(rcpOrientedEta, dummy, rcpOrientedEta2, NdotI, eta);
    return irr_glsl_reflect_refract(_refract, I, N, backside, NdotI, NdotI2, rcpOrientedEta, rcpOrientedEta2);
}


// if V and L are on different sides of the surface normal, then their dot product sign bits will differ, hence XOR will yield 1 at last bit
bool irr_glsl_isTransmissionPath(in float NdotV, in float NdotL)
{
    return ((floatBitsToUint(NdotV)^floatBitsToUint(NdotL)) & 0x80000000u) != 0u;
}

// valid only for `theta` in [-PI,PI]
void irr_glsl_sincos(in float theta, out float s, out float c)
{
    c = cos(theta);
    s = sqrt(1.0-c*c);
    s = theta<0.0 ? -s:s; // TODO: do with XOR
}

mat2x3 irr_glsl_frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}

// @return if picked right choice
bool irr_glsl_partitionRandVariable(in float leftProb, inout float xi, out float rcpChoiceProb)
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
float irr_glsl_conditionalAbsOrMax(in bool cond, in float x, in float limit)
{
    const float condAbs = uintBitsToFloat(floatBitsToUint(x) & uint(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}
vec2 irr_glsl_conditionalAbsOrMax(in bool cond, in vec2 x, in vec2 limit)
{
    const vec2 condAbs = uintBitsToFloat(floatBitsToUint(x) & uvec2(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}
vec3 irr_glsl_conditionalAbsOrMax(in bool cond, in vec3 x, in vec3 limit)
{
    const vec3 condAbs = uintBitsToFloat(floatBitsToUint(x) & uvec3(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}
vec4 irr_glsl_conditionalAbsOrMax(in bool cond, in vec4 x, in vec4 limit)
{
    const vec4 condAbs = uintBitsToFloat(floatBitsToUint(x) & uvec4(cond ? 0x7fFFffFFu:0xffFFffFFu));
    return max(condAbs,limit);
}

//
uint irr_glsl_rotl(in uint x, in uint k)
{
	return (x<<k) | (x>>(32u-k));
}

#endif
