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

vec3 irr_glsl_reflect(in vec3 I, in vec3 N, in float NdotI)
{
    return N*2.0*NdotI - I;
}
vec3 irr_glsl_reflect(in vec3 I, in vec3 N)
{
    float NdotI = dot(N,I);
    return irr_glsl_reflect(I, N, NdotI);
}

#define irr_glsl_sincos(x,s,c) s=sin(x);c=cos(x);

mat2x3 irr_glsl_frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}

#endif
