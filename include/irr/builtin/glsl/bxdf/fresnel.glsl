#ifndef _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_
#define _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_

// only works for implied IoR==1.333....
vec3 irr_glsl_fresnel_schlick(in vec3 F0, in float VdotH)
{
    float x = 1.0 - VdotH;
    return F0 + (1.0 - F0) * x*x*x*x*x;
}

// conductors, only works for `CosTheta>=0`
vec3 irr_glsl_fresnel_conductor(vec3 Eta, vec3 Etak, float CosTheta)
{  
   const float CosTheta2 = CosTheta*CosTheta;
   const float SinTheta2 = 1.0 - CosTheta2;

   const vec3 EtaLen2 = Eta*Eta+Etak*Etak;
   const vec3 etaCosTwice = Eta*CosTheta*2.0;

   const vec3 rs_common = EtaLen2+vec3(CosTheta2);
   const vec3 rs2 = (rs_common - etaCosTwice)/(rs_common + etaCosTwice);

   const vec3 rp_common = EtaLen2*CosTheta2+vec3(1.0);
   const vec3 rp2 = (rp_common - etaCosTwice)/(rp_common + etaCosTwice);
   
   return (rs2 + rp2)*0.5;
}

// dielectrics
vec3 irr_glsl_fresnel_dielectric_common(in vec3 Eta2, in float AbsCosTheta)
{
   const float SinTheta2 = 1.0-AbsCosTheta*AbsCosTheta;

   const vec3 t0 = sqrt(vec3(Eta2)-SinTheta2);
   const vec3 rs = (vec3(AbsCosTheta) - t0) / (vec3(AbsCosTheta) + t0);

   const vec3 t2 = Eta2*AbsCosTheta;
   const vec3 rp = (t0 - t2) / (t0 + t2);

   return (rs*rs + rp*rp)*0.5;
}
vec3 irr_glsl_fresnel_dielectric_frontface_only(in vec3 Eta, in float CosTheta)
{
    return irr_glsl_fresnel_dielectric_common(Eta*Eta,CosTheta);
}
vec3 irr_glsl_fresnel_dielectric(vec3 Eta, in float CosTheta)
{
    const bool backside = CosTheta<0.0;
    Eta = backside ? (1.0/Eta):Eta;
    return irr_glsl_fresnel_dielectric_common(Eta*Eta,backside ? (-CosTheta):CosTheta);
}

// gets the sum of all R, T R T, T R^3 T, T R^5 T, ... paths
vec3 irr_glsl_thindielectric_infinite_scatter(in vec3 singleInterfaceReflectance)
{
    const vec3 doubleInterfaceReflectance = singleInterfaceReflectance*singleInterfaceReflectance;
    return mix((singleInterfaceReflectance-doubleInterfaceReflectance)/(vec3(1.0)-doubleInterfaceReflectance)*2.0,vec3(1.0),greaterThan(doubleInterfaceReflectance,vec3(0.9999)));
}
float irr_glsl_thindielectric_infinite_scatter(in float singleInterfaceReflectance)
{
    const float doubleInterfaceReflectance = singleInterfaceReflectance*singleInterfaceReflectance;
    return doubleInterfaceReflectance>0.9999 ? 1.0:((singleInterfaceReflectance-doubleInterfaceReflectance)/(1.0-doubleInterfaceReflectance)*2.0);
}

#endif
