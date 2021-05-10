// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BSDF_COMMON_INCLUDED_
#define _NBL_BSDF_COMMON_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>

#include <nbl/builtin/glsl/math/functions.glsl>

// do not use this struct in SSBO or UBO, its wasteful on memory
struct nbl_glsl_DirAndDifferential
{
   vec3 dir;
   // TODO: investigate covariance rendering and maybe kill this struct
};

// do not use this struct in SSBO or UBO, its wasteful on memory
struct nbl_glsl_IsotropicViewSurfaceInteraction
{
   nbl_glsl_DirAndDifferential V; // outgoing direction, NOT NORMALIZED; V.dir can have undef value for lambertian BSDF
   vec3 N; // surface normal, NOT NORMALIZED
   float NdotV;
   float NdotV_squared; // TODO: rename to NdotV2
};
struct nbl_glsl_AnisotropicViewSurfaceInteraction
{
    nbl_glsl_IsotropicViewSurfaceInteraction isotropic;
    vec3 T;
    vec3 B;
    float TdotV;
    float BdotV;
};

vec3 nbl_glsl_getTangentSpaceV(in nbl_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return vec3(interaction.TdotV,interaction.BdotV,interaction.isotropic.NdotV);
}
mat3 nbl_glsl_getTangentFrame(in nbl_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return mat3(interaction.T,interaction.B,interaction.isotropic.N);
}


struct nbl_glsl_LightSample
{
    vec3 L;  // incoming direction, normalized
    float VdotL;

    float TdotL; 
    float BdotL;
    float NdotL;
    float NdotL2;
};

// require tangentSpaceL already be normalized and in tangent space (tangentSpaceL==vec3(TdotL,BdotL,NdotL))
nbl_glsl_LightSample nbl_glsl_createLightSampleTangentSpace(in vec3 tangentSpaceV, in vec3 tangentSpaceL, in mat3 tangentFrame)
{
    nbl_glsl_LightSample s;

    s.L = tangentFrame*tangentSpaceL; // m must be an orthonormal matrix
    s.VdotL = dot(tangentSpaceV,tangentSpaceL);

    s.TdotL = tangentSpaceL.x;
    s.BdotL = tangentSpaceL.y;
    s.NdotL = tangentSpaceL.z;
    s.NdotL2 = s.NdotL*s.NdotL;

    return s;
}

//
nbl_glsl_LightSample nbl_glsl_createLightSample(in vec3 L, in float VdotL, in vec3 N)
{
    nbl_glsl_LightSample s;

    s.L = L;
    s.VdotL = VdotL;

    s.TdotL = nbl_glsl_FLT_NAN;
    s.BdotL = nbl_glsl_FLT_NAN;
    s.NdotL = dot(N,L);
    s.NdotL2 = s.NdotL*s.NdotL;

    return s;
}
nbl_glsl_LightSample nbl_glsl_createLightSample(in vec3 L, in nbl_glsl_IsotropicViewSurfaceInteraction interaction)
{
    return nbl_glsl_createLightSample(L,dot(interaction.V.dir,L),interaction.N);
}
nbl_glsl_LightSample nbl_glsl_createLightSample(in vec3 L, in float VdotL, in vec3 T, in vec3 B, in vec3 N)
{
    nbl_glsl_LightSample s;

    s.L = L;
    s.VdotL = VdotL;

    s.TdotL = dot(T,L);
    s.BdotL = dot(B,L);
    s.NdotL = dot(N,L);
    s.NdotL2 = s.NdotL*s.NdotL;

    return s;
}
nbl_glsl_LightSample nbl_glsl_createLightSample(in vec3 L, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return nbl_glsl_createLightSample(L,dot(interaction.isotropic.V.dir,L),interaction.T,interaction.B,interaction.isotropic.N);
}

vec3 nbl_glsl_getTangentSpaceL(in nbl_glsl_LightSample s)
{
    return vec3(s.TdotL, s.BdotL, s.NdotL);
}

nbl_glsl_IsotropicViewSurfaceInteraction nbl_glsl_calcSurfaceInteractionFromViewVector(in vec3 _View, in vec3 _Normal)
{
    nbl_glsl_IsotropicViewSurfaceInteraction interaction;
    interaction.V.dir = _View;
    interaction.N = _Normal;
    float invlenV2 = inversesqrt(dot(interaction.V.dir, interaction.V.dir));
    float invlenN2 = inversesqrt(dot(interaction.N, interaction.N));
    interaction.V.dir *= invlenV2;
    interaction.N *= invlenN2;
    interaction.NdotV = dot(interaction.N, interaction.V.dir);
    interaction.NdotV_squared = interaction.NdotV * interaction.NdotV;
    return interaction;
}
nbl_glsl_IsotropicViewSurfaceInteraction nbl_glsl_calcSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos, in vec3 _Normal)
{
    vec3 V = _CamPos - _SurfacePos;
    return nbl_glsl_calcSurfaceInteractionFromViewVector(V, _Normal);
}

nbl_glsl_AnisotropicViewSurfaceInteraction nbl_glsl_calcAnisotropicInteraction(in nbl_glsl_IsotropicViewSurfaceInteraction isotropic, in vec3 T, in vec3 B)
{
    nbl_glsl_AnisotropicViewSurfaceInteraction inter;
    inter.isotropic = isotropic;
    inter.T = T;
    inter.B = B;
    inter.TdotV = dot(inter.isotropic.V.dir,inter.T);
    inter.BdotV = dot(inter.isotropic.V.dir,inter.B);

    return inter;
}
nbl_glsl_AnisotropicViewSurfaceInteraction nbl_glsl_calcAnisotropicInteraction(in nbl_glsl_IsotropicViewSurfaceInteraction isotropic, in vec3 T)
{
    return nbl_glsl_calcAnisotropicInteraction(isotropic, T, cross(isotropic.N,T));
}
nbl_glsl_AnisotropicViewSurfaceInteraction nbl_glsl_calcAnisotropicInteraction(in nbl_glsl_IsotropicViewSurfaceInteraction isotropic)
{
    mat2x3 TB = nbl_glsl_frisvad(isotropic.N);
    return nbl_glsl_calcAnisotropicInteraction(isotropic, TB[0], TB[1]);
}


// do not use this struct in SSBO or UBO, its wasteful on memory
struct nbl_glsl_IsotropicMicrofacetCache
{
    float VdotH;
    float LdotH;
    float NdotH;
    float NdotH2;
};

// do not use this struct in SSBO or UBO, its wasteful on memory
struct nbl_glsl_AnisotropicMicrofacetCache
{
    nbl_glsl_IsotropicMicrofacetCache isotropic;
    float TdotH;
    float BdotH;
};

bool nbl_glsl_isValidVNDFMicrofacet(in nbl_glsl_IsotropicMicrofacetCache microfacet, in bool is_bsdf, in bool transmission, in float VdotL, in float eta, in float rcp_eta)
{
    return microfacet.NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -min(eta, rcp_eta)));
}

// returns if the configuration of V and L can be achieved 
bool nbl_glsl_calcIsotropicMicrofacetCache(out nbl_glsl_IsotropicMicrofacetCache _cache, in bool transmitted, in vec3 V, in vec3 L, in vec3 N, in float NdotL, in float VdotL, in float orientedEta, in float rcpOrientedEta, out vec3 H)
{
    H = nbl_glsl_computeMicrofacetNormal(transmitted, V, L, orientedEta);
    
    _cache.VdotH = dot(V,H);
    _cache.LdotH = dot(L,H);
    _cache.NdotH = dot(N,H);
    _cache.NdotH2 = _cache.NdotH*_cache.NdotH;

    // not coming from the medium and exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) ) 
    return !(transmitted && (VdotL > -min(orientedEta,rcpOrientedEta) || _cache.NdotH < 0.0));
}
bool nbl_glsl_calcIsotropicMicrofacetCache(out nbl_glsl_IsotropicMicrofacetCache _cache, in nbl_glsl_IsotropicViewSurfaceInteraction interaction, in nbl_glsl_LightSample _sample, in float eta)
{
    const float NdotV = interaction.NdotV;
    const float NdotL = _sample.NdotL;
    const bool transmitted = nbl_glsl_isTransmissionPath(NdotV,NdotL);
    
    float orientedEta, rcpOrientedEta;
    const bool backside = nbl_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, NdotV, eta);

    const vec3 V = interaction.V.dir;
    const vec3 L = _sample.L;
    const float VdotL = dot(V,L);
    vec3 dummy;
    return nbl_glsl_calcIsotropicMicrofacetCache(_cache,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,dummy);
}
// always valid because its specialized for the reflective case
nbl_glsl_IsotropicMicrofacetCache nbl_glsl_calcIsotropicMicrofacetCache(in float NdotV, in float NdotL, in float VdotL, out float LplusV_rcpLen)
{
    nbl_glsl_IsotropicMicrofacetCache _cache;

    LplusV_rcpLen = inversesqrt(2.0+2.0*VdotL);

    _cache.VdotH = LplusV_rcpLen*VdotL+LplusV_rcpLen;
    _cache.LdotH = _cache.VdotH;
    _cache.NdotH = (NdotL+NdotV)*LplusV_rcpLen;
    _cache.NdotH2 = _cache.NdotH*_cache.NdotH;

    return _cache;
}
nbl_glsl_IsotropicMicrofacetCache nbl_glsl_calcIsotropicMicrofacetCache(in nbl_glsl_IsotropicViewSurfaceInteraction interaction, in nbl_glsl_LightSample _sample)
{
    float dummy;
    return nbl_glsl_calcIsotropicMicrofacetCache(interaction.NdotV,_sample.NdotL,_sample.VdotL,dummy);
}

// get extra stuff for anisotropy, here we actually require T and B to be normalized
bool nbl_glsl_calcAnisotropicMicrofacetCache(out nbl_glsl_AnisotropicMicrofacetCache _cache, in bool transmitted, in vec3 V, in vec3 L, in vec3 T, in vec3 B, in vec3 N, in float NdotL, in float VdotL, in float orientedEta, in float rcpOrientedEta)
{
    vec3 H;
    const bool valid = nbl_glsl_calcIsotropicMicrofacetCache(_cache.isotropic,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);

    _cache.TdotH = dot(T,H);
    _cache.BdotH = dot(B,H);

    return valid;
}
bool nbl_glsl_calcAnisotropicMicrofacetCache(out nbl_glsl_AnisotropicMicrofacetCache _cache, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in nbl_glsl_LightSample _sample, in float eta)
{
    const float NdotV = interaction.isotropic.NdotV;
    const float NdotL = _sample.NdotL;
    const bool transmitted = nbl_glsl_isTransmissionPath(NdotV,NdotL);
    
    float orientedEta, rcpOrientedEta;
    const bool backside = nbl_glsl_getOrientedEtas(orientedEta, rcpOrientedEta, NdotV, eta);
    
    const vec3 V = interaction.isotropic.V.dir;
    const vec3 L = _sample.L;
    const float VdotL = dot(V,L);
    return nbl_glsl_calcAnisotropicMicrofacetCache(_cache,transmitted,V,L,interaction.T,interaction.B,interaction.isotropic.N,NdotL,VdotL,orientedEta,rcpOrientedEta);
}
// always valid because its for the reflective case
nbl_glsl_AnisotropicMicrofacetCache nbl_glsl_calcAnisotropicMicrofacetCache(in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in nbl_glsl_LightSample _sample)
{
    nbl_glsl_AnisotropicMicrofacetCache _cache;

    float LplusV_rcpLen;
    _cache.isotropic = nbl_glsl_calcIsotropicMicrofacetCache(interaction.isotropic.NdotV,_sample.NdotL,dot(interaction.isotropic.V.dir,_sample.L),LplusV_rcpLen);

    _cache.TdotH = (interaction.TdotV+_sample.TdotL)*LplusV_rcpLen;
    _cache.BdotH = (interaction.BdotV+_sample.BdotL)*LplusV_rcpLen;

   return _cache;
}

void nbl_glsl_calcAnisotropicMicrofacetCache_common(out nbl_glsl_AnisotropicMicrofacetCache _cache, in vec3 tangentSpaceV, in vec3 tangentSpaceH)
{
    _cache.isotropic.VdotH = dot(tangentSpaceV,tangentSpaceH);

    _cache.isotropic.NdotH = tangentSpaceH.z;
    _cache.isotropic.NdotH2 = tangentSpaceH.z*tangentSpaceH.z;
    _cache.TdotH = tangentSpaceH.x;
    _cache.BdotH = tangentSpaceH.y;
}
// always valid, by construction
nbl_glsl_AnisotropicMicrofacetCache nbl_glsl_calcAnisotropicMicrofacetCache(in vec3 tangentSpaceV, in vec3 tangentSpaceH, out vec3 tangentSpaceL)
{
    nbl_glsl_AnisotropicMicrofacetCache _cache;
    nbl_glsl_calcAnisotropicMicrofacetCache_common(_cache,tangentSpaceV,tangentSpaceH);

    _cache.isotropic.LdotH = _cache.isotropic.VdotH;
    tangentSpaceL = nbl_glsl_reflect(tangentSpaceV,tangentSpaceH,_cache.isotropic.VdotH);

    return _cache;
}
nbl_glsl_AnisotropicMicrofacetCache nbl_glsl_calcAnisotropicMicrofacetCache(in bool transmitted, in vec3 tangentSpaceV, in vec3 tangentSpaceH, out vec3 tangentSpaceL, in float rcpOrientedEta, in float rcpOrientedEta2)
{
    nbl_glsl_AnisotropicMicrofacetCache _cache;
    nbl_glsl_calcAnisotropicMicrofacetCache_common(_cache,tangentSpaceV,tangentSpaceH);

    const float VdotH = _cache.isotropic.VdotH;
    _cache.isotropic.LdotH = transmitted ? nbl_glsl_refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2):VdotH;
    tangentSpaceL = nbl_glsl_reflect_refract_impl(transmitted, tangentSpaceV,tangentSpaceH, VdotH,_cache.isotropic.LdotH, rcpOrientedEta);

    return _cache;
}

float nbl_glsl_bxdf_remainder_to_eval(in float remainder, in float pdf)
{
    return remainder * pdf;
}
vec3 nbl_glsl_bxdf_remainder_to_eval(in vec3 remainder, in float pdf)
{
    return remainder * pdf;
}

#endif
