
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: rename Eta to something more fitting to let it be known that reciprocal Eta convention is used (ior_dest/ior_src)

#ifndef _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_FRESNEL_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace fresnel
{

// only works for implied IoR==1.333....
float3 schlick(const float3 F0, const float VdotH)
{
    float x = 1.0 - VdotH;
    return F0 + (1.0 - F0) * x*x*x*x*x;
}

// TODO: provide a `nbl_glsl_fresnel_conductor_impl` that take `Eta` and `EtaLen2` directly
// conductors, only works for `CosTheta>=0`
float3 conductor(float3 Eta, float3 Etak, float CosTheta)
{  
   const float CosTheta2 = CosTheta*CosTheta;
   const float SinTheta2 = 1.0 - CosTheta2;

   const float3 EtaLen2 = Eta*Eta+Etak*Etak;
   const float3 etaCosTwice = Eta*CosTheta*2.0;

   const float3 rs_common = EtaLen2+float3(CosTheta2);
   const float3 rs2 = (rs_common - etaCosTwice)/(rs_common + etaCosTwice);

   const float3 rp_common = EtaLen2*CosTheta2+float3(1.0);
   const float3 rp2 = (rp_common - etaCosTwice)/(rp_common + etaCosTwice);
   
   return (rs2 + rp2)*0.5;
}

float3 conductor_impl(float3 Eta, float3 EtaLen2, float CosTheta)
{  
   const float CosTheta2 = CosTheta*CosTheta;
   const float SinTheta2 = 1.0 - CosTheta2;

   const float3 etaCosTwice = Eta*CosTheta*2.0;

   const float3 rs_common = EtaLen2+float3(CosTheta2);
   const float3 rs2 = (rs_common - etaCosTwice)/(rs_common + etaCosTwice);

   const float3 rp_common = EtaLen2*CosTheta2+float3(1.0);
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
   const float3 t0 = sqrt(max(float3(orientedEta2)-SinTheta2,float3(0.0)));
   const float3 rs = (float3(AbsCosTheta) - t0) / (float3(AbsCosTheta) + t0);

   const float3 t2 = orientedEta2*AbsCosTheta;
   const float3 rp = (t0 - t2) / (t0 + t2);

   return (rs*rs + rp*rp)*0.5;
}

float3 dielectric_frontface_only(in float3 Eta, in float CosTheta)
{
    return dielectric_common(Eta*Eta,CosTheta);
}

float3 dielectric(float3 Eta, in float CosTheta)
{
    float3 orientedEta,rcpOrientedEta;
    nbl_glsl_getOrientedEtas(orientedEta,rcpOrientedEta,CosTheta,Eta);
    return dielectric_common(orientedEta*orientedEta,abs(CosTheta));
}



struct thindielectric
{
    // gets the sum of all R, T R T, T R^3 T, T R^5 T, ... paths
    float3 infinite_scatter(in float3 singleInterfaceReflectance)
    {
        const float3 doubleInterfaceReflectance = singleInterfaceReflectance*singleInterfaceReflectance;
        
        return lerp(
            (singleInterfaceReflectance-doubleInterfaceReflectance)/(float3(1.0)-doubleInterfaceReflectance)*2.0,
            float3(1.0),
            greaterThan(doubleInterfaceReflectance,float3(0.9999))
        );
    }

    float infinite_scatter(in float singleInterfaceReflectance)
    {
        const float doubleInterfaceReflectance = singleInterfaceReflectance*singleInterfaceReflectance;
        
        return doubleInterfaceReflectance>0.9999 ? 1.0:((singleInterfaceReflectance-doubleInterfaceReflectance)/(1.0-doubleInterfaceReflectance)*2.0);
    }
};

}
}
}
}


#endif
