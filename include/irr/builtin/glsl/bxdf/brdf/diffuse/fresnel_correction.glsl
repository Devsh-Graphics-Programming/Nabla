#ifndef _IRR_BSDF_BRDF_DIFFUSE_FRESNEL_CORRECTION_INCLUDED_
#define _IRR_BSDF_BRDF_DIFFUSE_FRESNEL_CORRECTION_INCLUDED_

vec3 irr_glsl_diffuseFresnelCorrectionFactor(in vec3 n, in vec3 n2)
{
    //assert(n*n==n2);
    bvec3 TIR = lessThan(n,vec3(1.0));
    vec3 invdenum = mix(vec3(1.0), vec3(1.0)/(n2*n2*(vec3(554.33) - 380.7*n)), TIR);
    vec3 num = n*mix(vec3(0.1921156102251088),n*298.25 - 261.38*n2 + 138.43,TIR);
    num += mix(vec3(0.8078843897748912),vec3(-1.67),TIR);
    return num*invdenum;
}

//R_L should be something like -normalize(reflect(L,N))
vec3 irr_glsl_delta_distribution_specular_cos_eval(in irr_glsl_BSDFIsotropicParams params, in vec3 R_L, in mat2x3 ior2)
{
    return vec3(0.0);
}

#endif
