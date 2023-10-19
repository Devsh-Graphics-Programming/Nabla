// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_DIFFUSE_FRESNEL_CORRECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_DIFFUSE_FRESNEL_CORRECTION_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace brdf
{
namespace diffuse
{

float3 diffuseFresnelCorrectionFactor(in vec3 n, in vec3 n2)
{
    const float C1 = 554.33;
    const float C2 = 380.7;
    const float C3 = 298.25;
    const float C4 = 261.38;
    const float C5 = 138.43;
    const float C6 = 0.8078843897748912;
    const float C7 = -1.67;
    const float C8 = 0.1921156102251088;


    //assert(n*n==n2);
    bool3 TIR = (n < 1.0);
    vec3 invdenum = lerp(float3(1.0,1.0,1.0), float3(1.0,1.0,1.0)/(n2*n2*(vec3(C1,C1,C1) - C2*n)), TIR);
    vec3 num = n*lerp(float3(C8,C8,C8),n*C3 - C4*n2 + C5,TIR);
    num += lerp(float3(C6,C6,C6),float3(C7,C7,C7),TIR);
    return num*invdenum;
}

}
}
}
}
}

#endif
