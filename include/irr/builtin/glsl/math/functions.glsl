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

// valid only for `theta` in [-PI,PI]
void irr_glsl_sincos(in float theta, out float s, out float c)
{
    c = cos(theta);
    s = sqrt(1.0-c*c);
    s *= theta<0.0 ? -1.0:1.0;
}

#endif
