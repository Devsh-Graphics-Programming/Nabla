#ifndef _NBL_BUILTIN_GLSL_SAMPLING_BILINEAR_INCLUDED_
#define _NBL_BUILTIN_GLSL_SAMPLING_BILINEAR_INCLUDED_

#include <nbl/builtin/glsl/sampling/linear.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

// The square's vertex values are defined in Z-order, so indices 0,1,2,3 (xyzw) correspond to (0,0),(1,0),(0,1),(1,1)
vec2 nbl_glsl_sampling_generateBilinearSample(out float rcpPdf, in vec4 bilinearCoeffs, vec2 u)
{
    const vec2 twiceAreasUnderXCurve = vec2(bilinearCoeffs[0]+bilinearCoeffs[1],bilinearCoeffs[2]+bilinearCoeffs[3]);
    u.y = nbl_glsl_sampling_generateLinearSample(twiceAreasUnderXCurve,u.y);

    const vec2 ySliceEndPoints = vec2(mix(bilinearCoeffs[0],bilinearCoeffs[2],u.y),mix(bilinearCoeffs[1],bilinearCoeffs[3],u.y));
    u.x = nbl_glsl_sampling_generateLinearSample(ySliceEndPoints,u.x);

    rcpPdf = (twiceAreasUnderXCurve[0]+twiceAreasUnderXCurve[1])/(4.0*mix(ySliceEndPoints[0],ySliceEndPoints[1],u.x));

    return u;
}

float nbl_glsl_sampling_probBilinearSample(in vec4 bilinearCoeffs, vec2 u)
{
    return 4.0*mix(mix(bilinearCoeffs[0],bilinearCoeffs[1],u.x),mix(bilinearCoeffs[2],bilinearCoeffs[3],u.x),u.y)/(bilinearCoeffs[0]+bilinearCoeffs[1]+bilinearCoeffs[2]+bilinearCoeffs[3]);
}

// https://iquilezles.org/www/articles/ibilinear/ibilinear.htm

vec2 nbl_glsl_invBilinear2D(in vec2 p, in vec2 a, in vec2 b, in vec2 c, in vec2 d)
{
    vec2 res = vec2(-1.0);

    vec2 e = b-a;
    vec2 f = d-a;
    vec2 g = a-b+c-d;
    vec2 h = p-a;
        
    float k2 = nbl_glsl_cross( g, f );
    float k1 = nbl_glsl_cross( e, f ) + nbl_glsl_cross( h, g );
    float k0 = nbl_glsl_cross( h, e );
    
    // if edges are parallel, this is a linear equation
    if( abs(k2)<0.001 )
    {
        res = vec2( (h.x*k1+f.x*k0)/(e.x*k1-g.x*k0), -k0/k1 );
    }
    // otherwise, it's a quadratic
    else
    {
        float w = k1*k1 - 4.0*k0*k2;
        if( w<0.0 ) return vec2(-1.0);
        w = sqrt( w );

        float ik2 = 0.5/k2;
        float v = (-k1 - w)*ik2;
        float u = (h.x - f.x*v)/(e.x + g.x*v);
        
        if( u<0.0 || u>1.0 || v<0.0 || v>1.0 )
        {
           v = (-k1 + w)*ik2;
           u = (h.x - f.x*v)/(e.x + g.x*v);
        }
        res = vec2( u, v );
    }
    
    return res;
}

#endif
