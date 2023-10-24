// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/math/functions.glsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{


namespace ray_dir_info
{

// no ray-differentials, nothing
struct Basic
{
  float32_t3 getDirection() {return direction;}

  Basic transmit()
  {
    Basic retval;
    retval.direction = -direction;
    return retval;
  }
  
  Basic reflect(const float32_t3 N, const float32_t directionDotN)
  {
    Basic retval;
    retval.direction = nbl::hlsl::reflect(direction,N,directionDotN);
    return retval;
  }

  float32_t3 direction;
};
// more to come!

}


namespace surface_interactions
{

template<class RayDirInfo>
struct Isotropic
{
  // WARNING: Changed since GLSL, now arguments need to be normalized!
  static Isotropic<RayDirInfo> create(const RayDirInfo normalizedV, const float32_t3 normalizedN)
  {
    Isotropic<RayDirInfo> retval;
    retval.V = normalizedV;
    retval.N = normalizedN;

    retval.NdotV = dot(retval.N,retval.V.getDirection());
    retval.NdotV_squared = retval.NdotV*retval.NdotV;

    return retval;
  }

  RayDirInfo V;
  float32_t3 N;
  float32_t NdotV;
  float32_t NdotV2; // old NdotV_squared
};

template<class RayDirInfo>
struct Anisotropic : Isotropic<RayDirInfo>
{
  // WARNING: Changed since GLSL, now arguments need to be normalized!
  static Anisotropic<RayDirInfo> create(
    const Isotropic<RayDirInfo> isotropic,
    const float32_t3 normalizedT,
    const float32_t normalizedB
  )
  {
    Anisotropic<RayDirInfo> retval;
    retval::Isotropic<RayDirInfo> = isotropic;
    retval.T = normalizedT;
    retval.B = normalizedB;
    
    const float32_t3 V = retval.getDirection();
    retval.TdotV = dot(V,retval.T);
    retval.BdotV = dot(V,retval.B);

    return retval;
  }
  static Anisotropic<RayDirInfo> create(const Isotropic<RayDirInfo> isotropic, const float32_t3 normalizedT)
  {
    return create(isotropic,normalizedT,cross(isotropic.N,normalizedT));
  }
  static Anisotropic<RayDirInfo> create(const Isotropic<RayDirInfo> isotropic)
  {
    float32_t2x3 TB = nbl::hlsl::frisvad(isotropic.N);
    return create(isotropic,TB[0],TB[1]);
  }

  float32_t3 getTangentSpaceV() {return float3(Tdot,BdotV,Isotropic<RayDirInfo>::NdotV);}
  // WARNING: its the transpose of the old GLSL function return value!
  float32_t3x3 getTangentFrame() {return float32_t3x3(T,B,Isotropic<RayDirInfo>::N);}

  float32_t3 T;
  float32_t3 B;
  float32_t3 TdotV;
  float32_t3 BdotV;
};

}


template<class RayDirInfo>
struct LightSample
{
  static LightSample<RayDirInfo> createTangentSpace(
    const float32_t3 tangentSpaceV,
    const RayDirInfo tangentSpaceL,
    const float32_t3x3 tangentFrame // WARNING: its the transpose of the old GLSL function return value!
  )
  {
    LightSample<RayDirInfo> retval;
    
    retval.L = RayDirInfo::transform(tangentSpaceL,tangentFrame);
    retval.VdotL = dot(tangentSpaceV,tangentSpaceL);

    retval.TdotL = tangentSpaceL.x;
    retval.BdotL = tangentSpaceL.y;
    retval.NdotL = tangentSpaceL.z;
    retval.NdotL2 = retval.NdotL*retval.NdotL;
    
    return retval;
  }
  static LightSample<RayDirInfo> create(const RayDirInfo L, const float32_t VdotL, const float32_t3 N)
  {
    LightSample<RayDirInfo> retval;
    
    retval.L = L;
    retval.VdotL = VdotL;

    retval.TdotL = nbl::hlsl::numeric_limits<float32_t>::nan();
    retval.BdotL = nbl::hlsl::numeric_limits<float32_t>::nan();
    retval.NdotL = dot(N,L);
    retval.NdotL2 = retval.NdotL*retval.NdotL;
    
    return retval;
  }
  static LightSample<RayDirInfo> create(const RayDirInfo L, const float32_t VdotL, const float32_t3 T, const float32_t3 B, const float32_t3 N)
  {
    LightSample<RayDirInfo> retval = create(L,VdotL,N);
    
    retval.TdotL = dot(T,L);
    retval.BdotL = dot(B,L);
    
    return retval;
  }
  // overloads for surface_interactions
  template<class ObserverRayDirInfo>
  static LightSample<RayDirInfo> create(const float32_t3 L, const surface_interactions::Isotropic<ObserverRayDirInfo> interaction)
  {
    const float32_t3 V = interaction.V.getDirection();
    const float32_t VdotL = dot(V,L);
    return create(L,VdotL,interaction.N);
  }
  template<class ObserverRayDirInfo>
  static LightSample<RayDirInfo> create(const float32_t3 L, const surface_interactions::Anisotropic<ObserverRayDirInfo> interaction)
  {
    const float32_t3 V = interaction.V.getDirection();
    const float32_t VdotL = dot(V,L);
    return create(L,VdotL,interaction.T,interaction.B,interaction.N);
  }
  //
  float32_t3 getTangentSpaceL()
  {
    return float3(TdotL,BdotL,NdotL);
  }

  RayDirInfo L;
  float32_t VdotL;

  float32_t TdotL; 
  float32_t BdotL;
  float32_t NdotL;
  float32_t NdotL2;
};


//
struct IsotropicMicrofacetCache
{
  // always valid because its specialized for the reflective case
  static IsotropicMicrofacetCache createForReflection(const float32_t NdotV, const float32_t NdotL, const float32_t VdotL, out float32_t LplusV_rcpLen)
  {
    LplusV_rcpLen = inversesqrt(2.0+2.0*VdotL);

    IsotropicMicrofacetCache retval;
    
    retval.VdotH = LplusV_rcpLen*VdotL+LplusV_rcpLen;
    retval.LdotH = retval.VdotH;
    retval.NdotH = (NdotL+NdotV)*LplusV_rcpLen;
    retval.NdotH2 = retval.NdotH*retval.NdotH;
    
    return retval;
  }
  static IsotropicMicrofacetCache createForReflection(const float32_t NdotV, const float32_t NdotL, const float32_t VdotL)
  {
    float32_t dummy;
    return createForReflection(NdotV,NdotL,VdotL,dummy);
  }
  template<class ObserverRayDirInfo, class IncomingRayDirInfo>
  static IsotropicMicrofacetCache createForReflection(
    const surface_interactions::Isotropic<ObserverRayDirInfo> interaction, 
    const LightSample<IncomingRayDirInfo> _sample)
  {
    return createForReflection(interaction.NdotV,_sample.NdotL,_sample.VdotL);
  }
  // transmissive cases need to be checked if the path is valid before usage
  static bool compute(
    out IsotropicMicrofacetCache retval,
    const bool transmitted, const float32_t3 V, const float32_t3 L,
    const float32_t3 N, const float32_t NdotL, const float32_t VdotL,
    const float32_t orientedEta, const float32_t rcpOrientedEta, out float32_t3 H
  )
  {
    // TODO: can we optimize?
    H = computeMicrofacetNormal(transmitted,V,L,orientedEta);
    retval.NdotH = dot(N,H);
    
    // not coming from the medium (reflected) OR
    // exiting at the macro scale AND ( (not L outside the cone of possible directions given IoR with constraint VdotH*LdotH<0.0) OR (microfacet not facing toward the macrosurface, i.e. non heightfield profile of microsurface) ) 
    const bool valid = !transmitted || (VdotL<=-min(orientedEta,rcpOrientedEta) && _cache.NdotH>nbl::hlsl::numeric_limits::min());
    if (valid)
    {
      // TODO: can we optimize?
      retval.VdotH = dot(V,H);
      retval.LdotH = dot(L,H);
      retval.NdotH2 = retval.NdotH*retval.NdotH;
      return true;
    }
    return false;
  }
  template<class ObserverRayDirInfo, class IncomingRayDirInfo>
  static bool compute(
    out IsotropicMicrofacetCache retval,
    const surface_interactions::Isotropic<ObserverRayDirInfo> interaction, 
    const LightSample<IncomingRayDirInfo> _sample,
    const float32_t eta, out float32_t3 H
  )
  {
    const float32_t NdotV = interaction.NdotV;
    const float32_t NdotL = _sample.NdotL;
    const bool transmitted = nbl_glsl_isTransmissionPath(NdotV,NdotL);
    
    float32_t orientedEta, rcpOrientedEta;
    const bool backside = nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,NdotV,eta);

    const vec3 V = interaction.V.getDirection();
    const vec3 L = _sample.L;
    const float32_t VdotL = dot(V,L);
    return nbl_glsl_calcIsotropicMicrofacetCache(_cache,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
  }
  template<class ObserverRayDirInfo, class IncomingRayDirInfo>
  static bool compute(
    out IsotropicMicrofacetCache retval,
    const surface_interactions::Isotropic<ObserverRayDirInfo> interaction, 
    const LightSample<IncomingRayDirInfo> _sample,
    const float32_t eta
  )
  {
    float32_t3 dummy;
    return nbl_glsl_calcIsotropicMicrofacetCache(_cache,transmitted,V,L,interaction.N,NdotL,VdotL,orientedEta,rcpOrientedEta,dummy);
  }

  bool isValidVNDFMicrofacet(const bool is_bsdf, const bool transmission, const float32_t VdotL, const float32_t eta, const float32_t rcp_eta)
  {
    return NdotH >= 0.0 && !(is_bsdf && transmission && (VdotL > -min(eta,rcp_eta)));
  }

  float32_t VdotH;
  float32_t LdotH;
  float32_t NdotH;
  float32_t NdotH2;
};

struct AnisotropicMicrofacetCache : IsotropicMicrofacetCache
{
  // always valid by construction
  static AnisotropicMicrofacetCache create(const float32_t3 tangentSpaceV, const float32_t3 tangentSpaceH)
  {
    AnisotropicMicrofacetCache retval;
    
    retval.VdotH = dot(tangentSpaceV,tangentSpaceH);
    retval.LdotH = retval.VdotH;
    retval.NdotH = tangentSpaceH.z;
    retval.NdotH2 = retval.NdotH*retval.NdotH;
    retval.TdotH = tangentSpaceH.x;
    retval.BdotH = tangentSpaceH.y;
    
    return retval;
  }
  static AnisotropicMicrofacetCache create(
    const float32_t3 tangentSpaceV, 
    const float32_t3 tangentSpaceH,
    const bool transmitted,
    const float32_t rcpOrientedEta,
    const float32_t rcpOrientedEta2
  )
  {
    AnisotropicMicrofacetCache retval = create(tangentSpaceV,tangentSpaceH);
    if (transmitted)
    {
      const float32_t VdotH = retval.VdotH;
      LdotH = transmitted ? refract_compute_NdotT(VdotH<0.0,VdotH*VdotH,rcpOrientedEta2);
    }
    
    return retval;
  }
  // always valid because its specialized for the reflective case
  static AnisotropicMicrofacetCache createForReflection(const float32_t3 tangentSpaceV, const float32_t3 tangentSpaceL, const float32_t VdotL)
  {
    AnisotropicMicrofacetCache retval;
    
    float32_t LplusV_rcpLen;
    retval = createForReflection(tangentSpaceV.z,tangentSpaceL.z,VdotL,LplusV_rcpLen);
    retval.TdotH = (tangentSpaceV.x+tangentSpaceL.x)*LplusV_rcpLen;
    retval.BdotH = (tangentSpaceV.y+tangentSpaceL.y)*LplusV_rcpLen;
    
    return retval;
  }
  template<class ObserverRayDirInfo, class IncomingRayDirInfo>
  static AnisotropicMicrofacetCache createForReflection(
    const surface_interactions::Anisotropic<ObserverRayDirInfo> interaction, 
    const LightSample<IncomingRayDirInfo> _sample)
  {
    return createForReflection(interaction.getTangentSpaceV(),_sample.getTangentSpaceL(),_sample.VdotL);
  }
  // transmissive cases need to be checked if the path is valid before usage
  static bool compute(
    out AnisotropicMicrofacetCache retval,
    const bool transmitted, const float32_t3 V, const float32_t3 L,
    const float32_t3 T, const float32_t3 B, const float32_t3 N,
    const float32_t NdotL, const float32_t VdotL,
    const float32_t orientedEta, const float32_t rcpOrientedEta, out float32_t3 H
  )
  {
    float32_t3 H;
    const bool valid = IsotropicMicrofacetCache::compute(retval,transmitted,V,L,N,NdotL,VdotL,orientedEta,rcpOrientedEta,H);
    if (valid)
    {
      retval.TdotH = dot(T,H);
      retval.BdotH = dot(B,H);
    }
    return valid;
  }
  template<class ObserverRayDirInfo, class IncomingRayDirInfo>
  static bool compute(
    out AnisotropicMicrofacetCache retval,
    const surface_interactions::Anisotropic<ObserverRayDirInfo> interaction, 
    const LightSample<IncomingRayDirInfo> _sample,
    const float32_t eta
  )
  {
    float32_t3 H;
    const bool valid = IsotropicMicrofacetCache::compute(retval,interaction,_sample,eta,H);
    if (valid)
    {
      retval.TdotH = dot(interaction.T,H);
      retval.BdotH = dot(interaction.B,H);
    }
    return valid;
  }

  float32_t TdotH;
  float32_t BdotH;
};


// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename SpectralBins>
struct quotient_and_pdf
{
  quotient_and_pdf<SpectralBins> create(const SpectralBins _quotient, const float32_t _pdf)
  {
    quotient_and_pdf<SpectralBins> retval;
    retval.quotient = _quotient;
    retval.pdf = _pdf;
    return retval;
  }

  SpectralBins value()
  {
    return quotient*pdf;
  }
  
  SpectralBins quotient;
  float32_t pdf;
};


}
}
}

#endif
