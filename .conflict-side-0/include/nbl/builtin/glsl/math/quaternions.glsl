#ifndef _NBL_BUILTIN_GLSL_MATH_QUATERNIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_MATH_QUATERNIONS_INCLUDED_



#include <nbl/builtin/glsl/math/functions.glsl>



struct nbl_glsl_quaternion_t
{
    vec4 data;
};


nbl_glsl_quaternion_t nbl_glsl_quaternion_t_constructFromTruncated(in vec3 first3Components)
{
    nbl_glsl_quaternion_t quat;
    quat.data.xyz = first3Components;
    quat.data.w = sqrt(1.0-dot(first3Components,first3Components));
    return quat;
}

nbl_glsl_quaternion_t nbl_glsl_quaternion_t_lerp(in nbl_glsl_quaternion_t start, in nbl_glsl_quaternion_t end, in float fraction, in float totalPseudoAngle)
{
    const uint negationMask = floatBitsToUint(totalPseudoAngle)&0x80000000u;
    const vec4 adjEnd = uintBitsToFloat(floatBitsToUint(end.data)^negationMask);

    nbl_glsl_quaternion_t quat;
    quat.data = mix(start.data,adjEnd,fraction);
    return quat;
}
nbl_glsl_quaternion_t nbl_glsl_quaternion_t_lerp(in nbl_glsl_quaternion_t start, in nbl_glsl_quaternion_t end, in float fraction)
{
    return nbl_glsl_quaternion_t_lerp(start,end,fraction,dot(start.data,end.data));
}

float nbl_glsl_quaternion_t_flerp_impl_adj_interpolant(in float angle, in float fraction, in float interpolantPrecalcTerm2, in float interpolantPrecalcTerm3)
{
    const float A = 1.0904f + angle * (-3.2452f + angle * (3.55645f - angle * 1.43519f));
    const float B = 0.848013f + angle * (-1.06021f + angle * 0.215638f);
    const float k = A * interpolantPrecalcTerm2 + B;
    return fraction+interpolantPrecalcTerm3*k;
}

nbl_glsl_quaternion_t nbl_glsl_quaternion_t_flerp(in nbl_glsl_quaternion_t start, in nbl_glsl_quaternion_t end, in float fraction)
{
    const float pseudoAngle = dot(start.data,end.data);

    const float interpolantPrecalcTerm = fraction-0.5f;
    const float interpolantPrecalcTerm3 = fraction*interpolantPrecalcTerm*(fraction-1.f);
    const float adjFrac = nbl_glsl_quaternion_t_flerp_impl_adj_interpolant(abs(pseudoAngle),fraction,interpolantPrecalcTerm*interpolantPrecalcTerm,interpolantPrecalcTerm3);
    nbl_glsl_quaternion_t quat = nbl_glsl_quaternion_t_lerp(start,end,adjFrac,pseudoAngle);
    quat.data = normalize(quat.data);
    return quat;
}

mat3 nbl_glsl_quaternion_t_constructMatrix(in nbl_glsl_quaternion_t quat)
{
    mat3 mat;
    mat[0] = quat.data.yzx*quat.data.ywz+quat.data.zxy*quat.data.zyw*vec3( 1.f, 1.f,-1.f);
    mat[1] = quat.data.yzx*quat.data.xzw+quat.data.zxy*quat.data.wxz*vec3(-1.f, 1.f, 1.f);
    mat[2] = quat.data.yzx*quat.data.wyx+quat.data.zxy*quat.data.xwy*vec3( 1.f,-1.f, 1.f);
    mat[0][0] = 0.5f-mat[0][0];
    mat[1][1] = 0.5f-mat[1][1];
    mat[2][2] = 0.5f-mat[2][2];
    mat *= 2.f;
    return mat;
}


vec3 nbl_glsl_slerp_delta_impl(in vec3 start, in vec3 preScaledWaypoint, in float cosAngleFromStart)
{
    vec3 planeNormal = cross(start,preScaledWaypoint);
    
    cosAngleFromStart *= 0.5;
    const float sinAngle = sqrt(0.5-cosAngleFromStart);
    const float cosAngle = sqrt(0.5+cosAngleFromStart);
    
    planeNormal *= sinAngle;
    const vec3 precompPart = cross(planeNormal,start)*2.0;

    return precompPart*cosAngle+cross(planeNormal,precompPart);
}

vec3 nbl_glsl_slerp_impl_impl(in vec3 start, in vec3 preScaledWaypoint, in float cosAngleFromStart)
{
    return start+nbl_glsl_slerp_delta_impl(start,preScaledWaypoint,cosAngleFromStart);
}

#endif
