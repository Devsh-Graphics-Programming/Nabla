
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MTL_LOADER_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MTL_LOADER_COMMON_INCLUDED_


namespace nbl
{
namespace hlsl
{


struct MTLMaterialParameters
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Ke;
    float4 Tf;  // w component doesnt matter
    float Ns;
    float d;
    float bm;
    float Ni;
    float roughness;
    float metallic;
    float sheen;
    float clearcoatThickness;
    float clearcoatRoughness;
    float anisotropy;
    float anisoRotation;

    //extra info
    uint extra;
};


}
}

#endif