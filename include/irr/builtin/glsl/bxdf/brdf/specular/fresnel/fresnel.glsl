#ifndef _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_
#define _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_

// only works for implied IoR==1.333....
vec3 irr_glsl_fresnel_schlick(in vec3 F0, in float VdotH)
{
    float x = 1.0 - VdotH;
    return F0 + (1.0 - F0) * x*x*x*x*x;
}

// code from https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3 irr_glsl_fresnel_conductor(vec3 Eta2, vec3 Etak2, float CosTheta)
{  
   float CosTheta2 = CosTheta * CosTheta;
   float SinTheta2 = 1.0 - CosTheta2;

   vec3 t0 = Eta2 - Etak2 - SinTheta2;
   vec3 a2plusb2 = sqrt(t0 * t0 + 4 * Eta2 * Etak2);
   vec3 t1 = a2plusb2 + CosTheta2;
   vec3 a = sqrt(0.5 * (a2plusb2 + t0));
   vec3 t2 = 2 * a * CosTheta;
   vec3 Rs = (t1 - t2) / (t1 + t2);

   vec3 t3 = CosTheta2 * a2plusb2 + SinTheta2 * SinTheta2;
   vec3 t4 = t2 * SinTheta2;   
   vec3 Rp = Rs * (t3 - t4) / (t3 + t4);

   return 0.5 * (Rp + Rs);
}

vec3 irr_glsl_fresnel_dielectric_common(in vec3 Eta2, in float AbsCosTheta)
{
   const float SinTheta2 = 1.0-AbsCosTheta*AbsCosTheta;

   const vec3 t0 = sqrt(vec3(Eta2)-SinTheta2);
   const vec3 rs = (vec3(AbsCosTheta) - t0) / (vec3(AbsCosTheta) + t0);

   const vec3 t2 = Eta2*AbsCosTheta;
   const vec3 rp = (t0 - t2) / (t0 + t2);

   return 0.5 * (rs * rs + rp * rp);
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

#endif
