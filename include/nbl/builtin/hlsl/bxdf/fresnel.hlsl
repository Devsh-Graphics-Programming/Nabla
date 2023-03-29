
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: rename Eta to something more fitting to let it be known that reciprocal Eta convention is used (ior_dest/ior_src)

#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_

#include <nbl/builtin/hlsl/math/functions.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace fresnel
{

// only works for implied IoR==1.333....
float3 schlick(in float3 F0, in float VdotH)
{
    float x = 1.0 - VdotH;
    return F0 + (1.0 - F0) * x*x*x*x*x;
}

// TODO: provide a `nbl_glsl_fresnel_conductor_impl` that take `Eta` and `EtaLen2` directly
// conductors, only works for `CosTheta>=0`
float3 conductor(in float3 Eta, in float3 Etak, in float CosTheta)
{  
   const float CosTheta2 = CosTheta*CosTheta;
   const float SinTheta2 = 1.0 - CosTheta2;

   const float3 EtaLen2 = Eta*Eta+Etak*Etak;
   const float3 etaCosTwice = Eta*CosTheta*2.0;

   const float3 rs_common = EtaLen2 + (CosTheta2).xxx;
   const float3 rs2 = (rs_common - etaCosTwice)/(rs_common + etaCosTwice);

   const float3 rp_common = EtaLen2*CosTheta2 + (1.0).xxx;
   const float3 rp2 = (rp_common - etaCosTwice)/(rp_common + etaCosTwice);
   
   return (rs2 + rp2)*0.5;
}

float3 conductor_impl(in float3 Eta, in float3 EtaLen2, in float CosTheta)
{  
   const float CosTheta2 = CosTheta*CosTheta;
   const float SinTheta2 = 1.0 - CosTheta2;

   const float3 etaCosTwice = Eta*CosTheta*2.0;

   const float3 rs_common = EtaLen2 + (CosTheta2).xxx;
   const float3 rs2 = (rs_common - etaCosTwice)/(rs_common + etaCosTwice);

   const float3 rp_common = EtaLen2*CosTheta2 + (1.0).xxx;
   const float3 rp2 = (rp_common - etaCosTwice)/(rp_common + etaCosTwice);
   
   return (rs2 + rp2)*0.5;
}


// dielectrics
float dielectric_common(in float orientedEta2, in float AbsCosTheta)
{
    const float SinTheta2 = 1.0-AbsCosTheta*AbsCosTheta;

    // the max() clamping can handle TIR when orientedEta2<1.0
    const float t0 = sqrt(max(orientedEta2-SinTheta2,0.0));
    const float rs = (AbsCosTheta - t0) / (AbsCosTheta + t0);

    const float t2 = orientedEta2 * AbsCosTheta;
    const float rp = (t0 - t2) / (t0 + t2);

    return (rs * rs + rp * rp) * 0.5;
}

float3 dielectric_common(in float3 orientedEta2, in float AbsCosTheta)
{
   const float SinTheta2 = 1.0-AbsCosTheta*AbsCosTheta;

   // the max() clamping can handle TIR when orientedEta2<1.0
   const float3 t0 = sqrt(max(float3(orientedEta2)-SinTheta2, (0.0).xxx));
   const float3 rs = ((AbsCosTheta).xxx - t0) / ((AbsCosTheta).xxx + t0);

   const float3 t2 = orientedEta2*AbsCosTheta;
   const float3 rp = (t0 - t2) / (t0 + t2);

   return (rs*rs + rp*rp)*0.5;
}

float3 dielectric_frontface_only(in float3 Eta, in float CosTheta)
{
    return dielectric_common(Eta*Eta,CosTheta);
}

float3 dielectric(in float3 Eta, in float CosTheta)
{
    float3 orientedEta,rcpOrientedEta;
    math::getOrientedEtas(orientedEta,rcpOrientedEta,CosTheta,Eta);
    return dielectric_common(orientedEta*orientedEta,abs(CosTheta));
}


// gets the sum of all R, T R T, T R^3 T, T R^5 T, ... paths
float3 thindielectric_infinite_scatter(in float3 singleInterfaceReflectance)
{
    const float3 doubleInterfaceReflectance = singleInterfaceReflectance*singleInterfaceReflectance;
        
    return lerp(
        (singleInterfaceReflectance-doubleInterfaceReflectance)/(float3(1.0,1.0,1.0)-doubleInterfaceReflectance)*2.0,
        float3(1.0,1.0,1.0),
        (doubleInterfaceReflectance > float3(0.9999,0.9999,0.9999))
    );
}

float thindielectric_infinite_scatter(in float singleInterfaceReflectance)
{
    const float doubleInterfaceReflectance = singleInterfaceReflectance*singleInterfaceReflectance;
        
    return doubleInterfaceReflectance>0.9999 ? 1.0:((singleInterfaceReflectance-doubleInterfaceReflectance)/(1.0-doubleInterfaceReflectance)*2.0);
}


// Utility class
template <class Spectrum>
struct FresnelBase
{
    // Fresnels must define such typenames:
    using spectrum_t = Spectrum;

    /**
    * Fresnels must define following member functions:
    *
    * spectrum_t operator()(...); // TODO is there some paremeter list that can be universal for all fresnels ever needed?
    */
};


template <class Spectrum>
struct FresnelSchlick : FresnelBase<Spectrum>
{
    spectrum_t F0;

    static FresnelSchlick<spectrum_t> create(in spectrum_t _F0)
    {
        FresnelSchlick<spectrum_t> fs;
        fs.F0 = _F0;
        return fs;
    }

    spectrum_t operator()(in float cosTheta)
    {
        float x = 1.0 - cosTheta;
        return F0 + (1.0 - F0) * x*x*x*x*x;
    }
};
using FresnelSchlickScalar  = FresnelSchlick<float>;
using FresnelSchlickRGB     = FresnelSchlick<float3>;


template <class Spectrum>
struct FresnelConductor : FresnelBase<Spectrum>
{
    spectrum_t eta;
    spectrum_t etak;

    static FresnelConductor<Spectrum> create(in spectrum_t _eta, in spectrum_t _etak)
    {
        FresnelConductor<Spectrum> f;

        f.eta     = _eta;
        f.etak    = _etak;

        return f;
    }

    spectrum_t operator()(in float cosTheta)
    {
       const float CosTheta2 = cosTheta*cosTheta;
       const float SinTheta2 = 1.0 - CosTheta2;

       const spectrum_t EtaLen2 = eta*eta + etak*etak;
       const spectrum_t etaCosTwice = eta*cosTheta*2.0;

       const spectrum_t rs_common = EtaLen2 + (CosTheta2).xxx;
       const spectrum_t rs2 = (rs_common - etaCosTwice)/(rs_common + etaCosTwice);

       const spectrum_t rp_common = EtaLen2*CosTheta2 + spectrum_t(1);
       const spectrum_t rp2 = (rp_common - etaCosTwice)/(rp_common + etaCosTwice);
   
       return (rs2 + rp2)*0.5;
    }
};
using FresnelConductorScalar    = FresnelConductor<float>;
using FresnelConductorRGB       = FresnelConductor<float3>;


template <class Spectrum>
struct FresnelDielectric : FresnelBase<Spectrum>
{
    spectrum_t eta;

    static FresnelDielectric<Spectrum> create(in spectrum_t _eta)
    {
        FresnelDielectric<Spectrum> f;

        f.eta = _eta;

        return f;
    }

    spectrum_t operator()(in float cosTheta)
    {
        spectrum_t orientedEta, rcpOrientedEta;
        math::getOrientedEtas<spectrum_t>(orientedEta, rcpOrientedEta, cosTheta, eta);

        const float AbsCosTheta = abs(cosTheta);
        const spectrum_t orientedEta2 = orientedEta * orientedEta;

        const float SinTheta2 = 1.0-AbsCosTheta*AbsCosTheta;

        // the max() clamping can handle TIR when orientedEta2<1.0
        const spectrum_t t0 = sqrt(max(orientedEta2-SinTheta2, spectrum_t(0)));
        const spectrum_t rs = (AbsCosTheta - t0) / (AbsCosTheta + t0);

        const spectrum_t t2 = orientedEta2*AbsCosTheta;
        const spectrum_t rp = (t0 - t2) / (t0 + t2);

        return (rs*rs + rp*rp)*0.5;
    }
};
using FresnelDielectricScalar = FresnelDielectric<float>;
using FresnelDielectricRGB = FresnelDielectric<float3>;

}
}
}
}


#endif
