#ifndef __IRR_C_GLSL_BSDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_BSDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED__


#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr {
namespace asset
{    

class CGLSLBSDFBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/bsdf/"; }

private:
    static std::string getCommons(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_COMMON_INCLUDED_
#define _IRR_BSDF_COMMON_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>

#include <irr/builtin/glsl/limits/numeric.glsl>

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_DirAndDifferential
{
   vec3 dir;
   // differentials at origin, I'd much prefer them to be differentials of barycentrics instead of position in the future
   mat2x3 dPosdScreen;
};

//TODO change name to irr_glsl_IsotropicViewSurfaceInteraction
// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_IsotropicViewSurfaceInteraction
{
   irr_glsl_DirAndDifferential V; // outgoing direction, NOT NORMALIZED; V.dir can have undef value for lambertian BSDF
   vec3 N; // surface normal, NOT NORMALIZED
   float NdotV;
   float NdotV_squared;
};
struct irr_glsl_AnisotropicViewSurfaceInteraction
{
    irr_glsl_IsotropicViewSurfaceInteraction isotropic;
    vec3 T;
    vec3 B;
    float TdotV;
    float BdotV;
};
mat3 irr_glsl_getTangentFrame(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    return mat3(interaction.T,interaction.B,interaction.isotropic.N);
}

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFSample
{
   vec3 L;  // incoming direction, normalized
   float probability; // for a single sample (don't care about number drawn)
};


// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFIsotropicParams
{
   float NdotL;
   float NdotL_squared;
   float VdotL; // same as LdotV
   float NdotH;
   float VdotH; // same as LdotH
   // left over for anisotropic calc and BSDF that want to implement fast bump mapping
   float LplusV_rcpLen;
   // basically metadata
   vec3 L;
   float invlenL2;
   irr_glsl_IsotropicViewSurfaceInteraction interaction;
};

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_BSDFAnisotropicParams
{
   irr_glsl_BSDFIsotropicParams isotropic;
   float TdotL;
   float TdotV;
   float TdotH;
   float BdotL;
   float BdotV;
   float BdotH;
   // useless metadata
   vec3 T;
   vec3 B;
};

// chain rule on various functions (usually vertex attributes and barycentrics)
vec2 irr_glsl_applyScreenSpaceChainRule1D3(in vec3 dFdG, in mat2x3 dGdScreen)
{
   return vec2(dot(dFdG,dGdScreen[0]),dot(dFdG,dGdScreen[1]));
}
mat2 irr_glsl_applyScreenSpaceChainRule2D3(in mat3x2 dFdG, in mat2x3 dGdScreen)
{
   return dFdG*dGdScreen;
}
mat2x3 irr_glsl_applyScreenSpaceChainRule3D3(in mat3 dFdG, in mat2x3 dGdScreen)
{
   return dFdG*dGdScreen;
}
mat2x4 irr_glsl_applyScreenSpaceChainRule4D3(in mat3x4 dFdG, in mat2x3 dGdScreen)
{
   return dFdG*dGdScreen;
}

//TODO move to different glsl header
vec2 irr_glsl_concentricMapping(in vec2 _u)
{
    //map [0;1]^2 to [-1;1]^2
    vec2 u = 2.0*_u - 1.0;
    
    vec2 p;
    if (u==vec2(0.0))
        p = vec2(0.0);
    else
    {
        float r;
        float theta;
        if (abs(u.x)>abs(u.y)) {
            r = u.x;
            theta = 0.25*irr_glsl_PI * (u.y/u.x);
        } else {
            r = u.y;
            theta = 0.5*irr_glsl_PI - 0.25*irr_glsl_PI*(u.x/u.y);
        }
        p = r*vec2(cos(theta),sin(theta));
    }

    return p;
}

//TODO move this to different glsl header
mat2x3 irr_glsl_frisvad(in vec3 n)
{
	const float a = 1.0/(1.0 + n.z);
	const float b = -n.x*n.y*a;
	return (n.z<-0.9999999) ? mat2x3(vec3(0.0,-1.0,0.0),vec3(-1.0,0.0,0.0)):mat2x3(vec3(1.0-n.x*n.x*a, b, -n.x),vec3(b, 1.0-n.y*n.y*a, -n.y));
}

// only in the fragment shader we have access to implicit derivatives
irr_glsl_IsotropicViewSurfaceInteraction irr_glsl_calcFragmentShaderSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos, in vec3 _Normal)
{
   irr_glsl_IsotropicViewSurfaceInteraction interaction;
   interaction.V.dir = _CamPos-_SurfacePos;
   interaction.V.dPosdScreen[0] = dFdx(_SurfacePos);
   interaction.V.dPosdScreen[1] = dFdy(_SurfacePos);
   interaction.N = _Normal;
   float invlenV2 = inversesqrt(dot(interaction.V.dir,interaction.V.dir));
   float invlenN2 = inversesqrt(dot(interaction.N,interaction.N));
   interaction.V.dir *= invlenV2;
   interaction.N *= invlenN2;
   interaction.NdotV = dot(interaction.N,interaction.V.dir);
   interaction.NdotV_squared = interaction.NdotV*interaction.NdotV;
   return interaction;
}
irr_glsl_AnisotropicViewSurfaceInteraction irr_glsl_calcAnisotropicInteraction(in irr_glsl_IsotropicViewSurfaceInteraction isotropic, in vec3 T, in vec3 B)
{
    irr_glsl_AnisotropicViewSurfaceInteraction inter;
    inter.isotropic = isotropic;
    inter.T = T;
    inter.B = B;
    inter.TdotV = dot(inter.isotropic.V.dir,inter.T);
    inter.BdotV = dot(inter.isotropic.V.dir,inter.B);

    return inter;
}
irr_glsl_AnisotropicViewSurfaceInteraction irr_glsl_calcAnisotropicInteraction(in irr_glsl_IsotropicViewSurfaceInteraction isotropic, in vec3 T)
{
    return irr_glsl_calcAnisotropicInteraction(isotropic, T, cross(isotropic.N,T));
}
irr_glsl_AnisotropicViewSurfaceInteraction irr_glsl_calcAnisotropicInteraction(in irr_glsl_IsotropicViewSurfaceInteraction isotropic)
{
    mat2x3 TB = irr_glsl_frisvad(isotropic.N);
    return irr_glsl_calcAnisotropicInteraction(isotropic, TB[0], TB[1]);
}
/*
//TODO it doesnt compile, lots of undefined symbols
// when you know the projected positions of your triangles (TODO: should probably make a function like this that also computes barycentrics)
irr_glsl_IsotropicViewSurfaceInteraction irr_glsl_calcBarycentricSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos[3], in vec3 _Normal[3], in float _Barycentrics[2], in vec2 _ProjectedPos[3])
{
   irr_glsl_IsotropicViewSurfaceInteraction interaction;

   // Barycentric interpolation = b0*attr0+b1*attr1+attr2*(1-b0-b1)
   vec3 b = vec3(_Barycentrics[0],_Barycentrics[1],1.0-_Barycentrics[0]-_Barycentrics[1]);
   mat3 vertexAttrMatrix = mat3(_SurfacePos[0],_SurfacePos[1],_SurfacePos[2]);
   interaction.V.dir = _CamPos-vertexAttrMatrix*b;
   // Schied's derivation - modified
   vec2 to2 = _ProjectedPos[2]-_ProjectedPos[1];
   vec2 to1 = _ProjectedPos[0]-_ProjectedPos[1];
   float d = 1.0/determinant(mat2(to2,to1)); // TODO double check all this
   mat2x3 dBaryd = mat2x3(vec3(v[1].y-v[2].y,to2.y,to0.y)*d,-vec3(v[1].x-v[2].x,to2.x,t0.x)*d);
   //
   interaction.dPosdScreen = irr_glsl_applyScreenSpaceChainRule3D3(vertexAttrMatrix,dBaryd);

   vertexAttrMatrix = mat3(_Normal[0],_Normal[1],_Normal[2]);
   interaction.N = vertexAttrMatrix*b;

   return interaction;
}
// when you know the ray and triangle it hits
irr_glsl_IsotropicViewSurfaceInteraction  irr_glsl_calcRaySurfaceInteraction(in irr_glsl_DirAndDifferential _rayTowardsSurface, in vec3 _SurfacePos[3], in vec3 _Normal[3], in float _Barycentrics[2])
{
   irr_glsl_IsotropicViewSurfaceInteraction interaction;
   // flip ray
   interaction.V.dir = -_rayTowardsSurface.dir;
   // do some hardcore shizz to transform a differential at origin into a differential at surface
   // also in barycentrics preferably (turn world pos diff into bary diff with applyScreenSpaceChainRule3D3)
   //interaction.V.dPosdx = TODO;
   //interaction.V.dPosdy = TODO;

   vertexAttrMatrix = mat3(_Normal[0],_Normal[1],_Normal[2]);
   interaction.N = vertexAttrMatrix*b;

   return interaction;
}
*/

// will normalize all the vectors
irr_glsl_BSDFIsotropicParams irr_glsl_calcBSDFIsotropicParams(in irr_glsl_IsotropicViewSurfaceInteraction interaction, in vec3 L)
{
   float invlenL2 = inversesqrt(dot(L,L));

   irr_glsl_BSDFIsotropicParams params;

   // totally useless vectors, will probably get optimized away by compiler if they don't get used
   // but useful as temporaries
   params.interaction = interaction;
   params.L = L*invlenL2;
   params.invlenL2 = invlenL2;

   // this stuff only works with normalized L,N,V
   params.NdotL = dot(params.interaction.N,params.L);
   params.NdotL_squared = params.NdotL*params.NdotL;

   params.VdotL = dot(params.interaction.V.dir,params.L);
   float LplusV_rcpLen = inversesqrt(2.0 + 2.0*params.VdotL);
   params.LplusV_rcpLen = LplusV_rcpLen;

   // this stuff works unnormalized L,N,V
   params.NdotH = (params.NdotL+params.interaction.NdotV)*LplusV_rcpLen;
   params.VdotH = LplusV_rcpLen + LplusV_rcpLen*params.VdotL;

   return params;
}
// get extra stuff for anisotropy, here we actually require T and B to be normalized
irr_glsl_BSDFAnisotropicParams irr_glsl_calcBSDFAnisotropicParams(in irr_glsl_BSDFIsotropicParams isotropic, in vec3 T, in vec3 B)
{
   irr_glsl_BSDFAnisotropicParams params;
   params.isotropic = isotropic;

   // meat
   params.TdotL = dot(T,isotropic.L);
   params.TdotV = dot(T,isotropic.interaction.V.dir);
   params.TdotH = (params.TdotV+params.TdotL)*isotropic.LplusV_rcpLen;
   params.BdotL = dot(B,isotropic.L);
   params.BdotV = dot(B,isotropic.interaction.V.dir);
   params.BdotH = (params.BdotV+params.BdotL)*isotropic.LplusV_rcpLen;

   // useless stuff we keep just to be complete
   params.T = T;
   params.B = B;

   return params;
}
#endif
)";
    }
    static std::string getCosWeightedSample(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_
#define _IRR_BSDF_BRDF_COS_WEIGHTED_SAMPLE_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample)
{
    vec2 p = irr_glsl_concentricMapping(_sample);
    
    mat3 m = irr_glsl_getTangentFrame(interaction);
    float z = sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y));
    vec3 L = m*vec3(p.x,p.y,z);

    irr_glsl_BSDFSample smpl;
    smpl.L = L;
    smpl.probability = dot(interaction.N,smpl.L)*irr_glsl_RECIPROCAL_PI;

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_cos_weighted_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_cos_weighted_cos_gen_sample(interaction, u);
}

#endif
)";
    }
    static std::string getCommonSamples(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_COMMON_SAMPLES_INCLUDED_
#define _IRR_BSDF_COMMON_SAMPLES_INCLUDED_

#include <irr/builtin/glsl/bsdf/brdf/cos_weighted_sample.glsl>

irr_glsl_BSDFSample irr_glsl_transmission_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = -interaction.isotropic.V.dir;
    smpl.probability = 1.0;
    
    return smpl;
}

irr_glsl_BSDFSample irr_glsl_delta_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    irr_glsl_BSDFSample smpl;
    smpl.L = interaction.isotropic.N*2.0*interaction.isotropic.NdotV - interaction.isotropic.V.dir;
    smpl.probability = 1.0;

    return smpl;
}

// usually  `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
//assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in vec3 eta, in vec3 luminosityContributionHint)
{
    vec3 Fr = irr_glsl_fresnel_dielectric(eta, interaction.isotropic.NdotV);
    float reflectionProb = dot(Fr, luminosityContributionHint);//why dont we just use fresnel as reflection probability? i know its a vec3 but all its components should be equal in case of dielectric
    
    irr_glsl_BSDFSample smpl;
    if (reflectionProb==1.0 || u.x<reflectionProb)
    {
        smpl.L = interaction.isotropic.N*2.0*NdotV - interaction.isotropic.V.dir;
        smpl.probability = reflectionProb;
    }
    else
    {
        //no idea whats going on here and whats `k` a few lines below
        float fittedMonochromeEta = dot(eta, luminosityContributionHint); //????
        //refract
        float NdotL2 = fittedMonochromeEta*fittedMonochromeEta - (1.0-NdotV2);
        /*if (k < 0.0)
            smpl.L = vec3(0.0);
        else*/
            smpl.L = ((NdotV /*+ sqrt(k)*/) * interaction.isotropic.N - V) / fittedMonochromeEta;
        smpl.probability = 1.0-reflectionProb;
    }

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_smooth_dielectric_cos_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _u, in vec3 eta, in vec3 luminosityContributionHint)
{
    vec2 u = vec2(_u)/float(UINT_MAX);
    return irr_glsl_smooth_dielectric_cos_sample(interaction, u, eta, luminosityContributionHint);
}

#endif
)";
    }
    static std::string getDiffuseFresnelCorrectionFactor(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_DIFFUSE_FRESNEL_CORRECTION_INCLUDED_
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
)";
    }
    static std::string getDeltaDistSpecular(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_DELTA_DIST_SPEC_INCLUDED_
#define _IRR_BSDF_BRDF_DELTA_DIST_SPEC_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

vec3 irr_glsl_delta_distribution_specular_cos_eval(in irr_glsl_BSDFIsotropicParams params)
{
    return vec3(0.0);
}

#endif
)";
    }
    static std::string getBlinnPhong(const std::string&)
    {
        return
            R"(#ifndef _IRR_BSDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>

float irr_glsl_blinn_phong(in float NdotH, in float n)
{
    float nom = n*(n + 6.0) + 8.0;
    float denom = pow(0.5, 0.5*n) + n;
    float normalization = 0.125 * irr_glsl_RECIPROCAL_PI * nom/denom;
    return normalization*pow(NdotH, n);
}

vec3 irr_glsl_blinn_phong_fresnel_dielectric_cos_eval(in irr_glsl_BSDFIsotropicParams params, in float n, in vec3 ior)
{
    float denom = 4.0*params.interaction.NdotV;
    return irr_glsl_blinn_phong(params.NdotH, n) * irr_glsl_fresnel_dielectric(ior, params.VdotH) / denom;
}

vec3 irr_glsl_blinn_phong_fresnel_conductor_cos_eval(in irr_glsl_BSDFIsotropicParams params, in float n, in mat2x3 ior2)
{
    float denom = 4.0*params.interaction.NdotV;
    return irr_glsl_blinn_phong(params.NdotH, n) * irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.VdotH) / denom;
}

#endif
)";
    }
    static std::string getAshikhminShirleyNDF(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_SPECULAR_NDF_ASHIKHMIN_SHIRLEY_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_NDF_ASHIKHMIN_SHIRLEY_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

//n is 2 phong-like exponents for anisotropy, can be defined as vec2(1.0/at, 1.0/ab) where at is roughness along tangent direction and ab is roughness along bitangent direction
//sin_cos_phi is sin and cos of azimuth angle of half vector
float irr_glsl_ashikhmin_shirley(in float NdotL, in float NdotV, in float NdotH, in float VdotH, in vec2 n, in vec2 sin_cos_phi)
{
    float nom = sqrt((n.x + 1.0)*(n.y + 1.0)) * pow(NdotH, n.x*sin_cos_phi.x*sin_cos_phi.x + n.y*sin_cos_phi.y*sin_cos_phi.y);
    float denom = 8.0 * irr_glsl_PI * VdotH * max(NdotV,NdotL);

    return NdotL * nom/denom;
}

#endif
)";
    }
    static std::string getAshikhminShirley_cos_eval(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_SPECULAR_ASHIKHMIN_SHIRLEY_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_ASHIKHMIN_SHIRLEY_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/ndf/ashikhmin_shirley.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/geom/smith.glsl>

//n is 2 phong-like exponents for anisotropy, can be defined as vec2(1.0/at, 1.0/ab) where at is roughness along tangent direction and ab is roughness along bitangent direction
//sin_cos_phi is sin and cos of azimuth angle of half vector
vec3 irr_glsl_ashikhmin_shirley_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in vec2 n, in vec2 sin_cos_phi, in vec2 atb, in mat2x3 ior2)
{
    float ndf = irr_glsl_ashikhmin_shirley(params.isotropic.NdotL, params.isotropic.interaction.NdotV, params.isotropic.NdotH, params.isotropic.VdotH, n, sin_cos_phi);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.isotropic.VdotH);
    //using ggx smith shadowing term here is wrong, however for now we're doing it because of lack of any other compatible one
    //Ashikhmin and Shirley came up with their own shadowing term, however implementation of it would be too complex in terms of our current design (https://www.researchgate.net/publication/220721563_A_microfacet-based_BRDF_generator)
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, params.TdotV, params.BdotL, params.BdotV, params.isotropic.NdotL, params.isotropic.interaction.NdotV);

    return g*ndf*fr / (4.0 * params.isotropic.interaction.NdotV);
}

#endif
)";
    }
    static std::string getBeckmann(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_SPECULAR_BECKMANN_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BECKMANN_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

float irr_glsl_beckmann(in float a2, in float NdotH2)
{
    float nom = exp( (NdotH2 - 1.0)/(a2*NdotH2) );
    float denom = a2*NdotH2*NdotH2;

    return irr_glsl_RECIPROCAL_PI * nom/denom;
}

#endif
)";
    }
    static std::string getLambert(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

float irr_glsl_lambertian()
{
    return irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_lambertian_cos_eval_rec_pi_factored_out(in irr_glsl_BSDFIsotropicParams params)
{
   return params.NdotL;
}

float irr_glsl_lambertian_cos_eval(in irr_glsl_BSDFIsotropicParams params)
{
   return irr_glsl_lambertian_cos_eval_rec_pi_factored_out(params)*irr_glsl_lambertian();
}

#endif
)";
    }
    static std::string getOrenNayar(const std::string&)
    {
        return
R"(#ifndef _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_
#define _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

float irr_glsl_oren_nayar(in float _a2, in float VdotL, in float NdotL, in float NdotV)
{
    // theta - polar angles
    // phi - azimuth angles
    float a2 = _a2*0.5; //todo read about this
    vec2 AB = vec2(1.0, 0.0) + vec2(-0.5, 0.45) * vec2(a2, a2)/vec2(a2+0.33, a2+0.09);
    float C = 1.0 / max(NdotL, NdotV);

    // should be equal to cos(phi)*sin(theta_i)*sin(theta_o)
    // where `phi` is the angle in the tangent plane to N, between L and V
    // and `theta_i` is the sine of the angle between L and N, similarily for `theta_o` but with V
    float cos_phi_sin_theta = max(VdotL-NdotL*NdotV,0.0);
    
    return (AB.x + AB.y * cos_phi_sin_theta * C) * irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_oren_nayar_cos_eval(in irr_glsl_BSDFIsotropicParams params, in float a2)
{
    return params.NdotL * irr_glsl_oren_nayar(a2, params.VdotL, params.NdotL, params.interaction.NdotV);
}

#endif
)";
    }
    static std::string getGGX_NDF(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_NDF_GGX_INCLUDED_
#define _BRDF_SPECULAR_NDF_GGX_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

float irr_glsl_ggx_trowbridge_reitz(in float a2, in float NdotH2)
{
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (irr_glsl_PI * denom*denom);
}

float irr_glsl_ggx_burley_aniso(float anisotropy, float a2, float TdotH, float BdotH, float NdotH) {
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab * irr_glsl_RECIPROCAL_PI;
}

#endif
)";
    }
    //https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
    //http://jcgt.org/published/0003/02/03/paper.pdf
    //https://hal.inria.fr/hal-00942452v1/document
    static std::string getSmith_G(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_
#define _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_

float irr_glsl_smith_beckmann_C2(in float NdotX2, in float a2)
{
    return NdotX2 / (a2 * (1.0 - NdotX2));
}
//G1 = 1/(1+_Lambda)
float irr_glsl_smith_beckmann_Lambda(in float c2)
{
    float c = sqrt(c2);
    float nom = 1.0 - 1.259*c + 0.396*c2;
    float denom = 2.181*c2 + 3.535*c;
    return mix(0.0, nom/denom, c<1.6);
}

float irr_glsl_smith_ggx_C2(in float NdotX2, in float a2)
{
    float sin2 = 1.0 - NdotX2;
    return a2 * sin2/NdotX2;
}

float irr_glsl_smith_ggx_Lambda(in float c2)
{
    return 0.5 * (sqrt(1.0+c2)-1.0);
}

float irr_glsl_GGXSmith_G1_(in float a2, in float NdotX)
{
    return (2.0*NdotX) / (NdotX + sqrt(a2 + (1.0 - a2)*NdotX*NdotX));
}
float irr_glsl_GGXSmith_G1_wo_numerator(in float a2, in float NdotX)
{
    return 1.0 / (NdotX + sqrt(a2 + (1.0 - a2)*NdotX*NdotX));
}

float irr_glsl_ggx_smith(in float a2, in float NdotL, in float NdotV)
{
    return irr_glsl_GGXSmith_G1_(a2, NdotL) * irr_glsl_GGXSmith_G1_(a2, NdotV);
}
float irr_glsl_ggx_smith_wo_numerator(in float a2, in float NdotL, in float NdotV)
{
    return irr_glsl_GGXSmith_G1_wo_numerator(a2, NdotL) * irr_glsl_GGXSmith_G1_wo_numerator(a2, NdotV);
}

float irr_glsl_ggx_smith_height_correlated(in float a2, in float NdotL, in float NdotV)
{
    float denom = NdotV*sqrt(a2 + (1.0 - a2)*NdotL*NdotL) + NdotL*sqrt(a2 + (1.0 - a2)*NdotV*NdotV);
    return 2.0*NdotL*NdotV / denom;
}

float irr_glsl_ggx_smith_height_correlated_wo_numerator(in float a2, in float NdotL, in float NdotV)
{
    float denom = NdotV*sqrt(a2 + (1.0 - a2)*NdotL*NdotL) + NdotL*sqrt(a2 + (1.0 - a2)*NdotV*NdotV);
    return 0.5 / denom;
}

// Note a, not a2!
float irr_glsl_ggx_smith_height_correlated_approx(in float a, in float NdotL, in float NdotV)
{
    float num = 2.0*NdotL*NdotV;
    return num / mix(num, NdotL+NdotV, a);
}

// Note a, not a2!
float irr_glsl_ggx_smith_height_correlated_approx_wo_numerator(in float a, in float NdotL, in float NdotV)
{
    return 0.5 / mix(2.0*NdotL*NdotV, NdotL+NdotV, a);
}

//Taken from https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel
float irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(in float at, in float ab, in float TdotL, in float TdotV, in float BdotL, in float BdotV, in float NdotL, in float NdotV)
{
    float Vterm = NdotL * length(vec3(at*TdotV, ab*BdotV, NdotV));
    float Lterm = NdotV * length(vec3(at*TdotL, ab*BdotL, NdotL));
    return 0.5 / (Vterm + Lterm);
}

#endif
)";
    }
    static std::string getGGX_cos_eval(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/ndf/ggx.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/geom/smith.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>

vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in mat2x3 ior2, in float a2, in vec2 atb, in float aniso)
{
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, params.TdotV, params.BdotL, params.BdotV, params.isotropic.NdotL, params.isotropic.interaction.NdotV);
    float ndf = irr_glsl_ggx_burley_aniso(aniso, a2, params.TdotH, params.BdotH, params.isotropic.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.isotropic.VdotH);

    return params.isotropic.NdotL * g*ndf*fr;
}
vec3 irr_glsl_ggx_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in mat2x3 ior2, in float a2)
{
    float g = irr_glsl_ggx_smith_height_correlated_wo_numerator(a2, params.NdotL, params.interaction.NdotV);
    float ndf = irr_glsl_ggx_trowbridge_reitz(a2, params.NdotH*params.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.VdotH);

    return params.NdotL * g*ndf*fr;
}

//Heitz's 2018 paper "Sampling the GGX Distribution of Visible Normals"
//Also: problem is our anisotropic ggx ndf (above) has extremely weird API (anisotropy and a2 instead of ax and ay) and so it's incosistent with sampling function
//  currently using isotropic trowbridge_reitz for PDF
irr_glsl_BSDFSample irr_glsl_ggx_smith_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float _ax, in float _ay)
{
    vec2 u = _sample;

    mat3 m = irr_glsl_getTangentFrame(interaction);

    vec3 V = interaction.isotropic.V.dir;
    V = normalize(V*m);//transform to tangent space
    V = normalize(vec3(_ax*V.x, _ay*V.y, V.z));//stretch view vector so that we're sampling as if roughness=1.0

    float lensq = V.x*V.x + V.y*V.y;
    vec3 T1 = lensq > 0.0 ? vec3(-V.y, V.x, 0.0)*inversesqrt(lensq) : vec3(1.0,0.0,0.0);
    vec3 T2 = cross(V,T1);

    float r = sqrt(u.x);
    float phi = 2.0 * irr_glsl_PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + V.z);
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    
    //reprojection onto hemisphere
    vec3 H = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0-t1*t1-t2*t2))*V;
    //unstretch
    H = normalize(vec3(_ax*H.x, _ay*H.y, max(0.0,H.z)));
    float NdotH = H.z;

    irr_glsl_BSDFSample smpl;
    //==== compute L ====
    H = normalize(m*H);//transform to correct space
    float HdotV = dot(H,interaction.isotropic.V.dir);
    //reflect V on H to actually get L
    smpl.L = H*2.0*HdotV - interaction.isotropic.V.dir;

    //==== compute probability ====
    float a2 = _ax*_ay;
    float lambda = irr_glsl_smith_ggx_Lambda(irr_glsl_smith_ggx_C2(interaction.isotropic.NdotV_squared, a2));
    float G1 = 1.0 / (1.0 + lambda);
    //here using isotropic trowbridge_reitz() instead of irr_glsl_ggx_burley_aniso()
    smpl.probability = irr_glsl_ggx_trowbridge_reitz(a2,NdotH*NdotH) * G1 * abs(dot(interaction.isotropic.V.dir,H)) / interaction.isotropic.NdotV;

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_ggx_smith_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample, in float _ax, in float _ay)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_ggx_smith_cos_gen_sample(interaction, u, _ax, _ay);
}

#endif
)";
    }
    static std::string getBeckmannSmith_cos_eval(const std::string&)
    {
        return
            R"(#ifndef _IRR_BSDF_BRDF_SPECULAR_BECKMANN_SMITH_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BECKMANN_SMITH_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/ndf/beckmann.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/geom/smith.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>
#include <irr/builtin/glsl/math/functions.glsl>

#include <irr/builtin/glsl/math/functions.glsl>

//i wonder where i got irr_glsl_ggx_smith_height_correlated() from because it looks very different from 1/(1+L_v+L_l) form
float irr_glsl_beckmann_smith_height_correlated(in float NdotV2, in float NdotL2, in float a2)
{
    float c2 = irr_glsl_smith_beckmann_C2(NdotV2, a2);
    float L_v = irr_glsl_smith_beckmann_Lambda(c2);
    c2 = irr_glsl_smith_beckmann_C2(NdotL2, a2);
    float L_l = irr_glsl_smith_beckmann_Lambda(c2);
    return 1.0 / (1.0 + L_v + L_l);
}

irr_glsl_BSDFSample irr_glsl_beckmann_smith_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float ax, in float ay)
{
    vec2 u = _sample;
    
    mat3 m = irr_glsl_getTangentFrame(interaction);

    vec3 V = interaction.isotropic.V.dir;
    V = normalize(V*m);//transform to tangent space
    //stretch
    V = normalize(vec3(ax*V.x, ay*V.y, V.z));

    vec2 slope;
    if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
    {
        float r = sqrt(-log(1.0-u.x));
        float sinPhi = sin(2.0*irr_glsl_PI*u.y);
        float cosPhi = cos(2.0*irr_glsl_PI*u.y);
        slope = vec2(r)*vec2(cosPhi,sinPhi);
    }
    else
    {
        float cosTheta = V.z;
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        float tanTheta = sinTheta/cosTheta;
        float cotTheta = 1.0/tanTheta;
        
        float a = -1.0;
        float c = irr_glsl_erf(cosTheta);
        float sample_x = max(u.x, 1.0e-6);
        float theta = acos(cosTheta);
        float fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
        float b = c - (1.0 + c) * pow(1.0-sample_x, fit);
        
        float normalization = 1.0 / (1.0 + c + irr_glsl_SQRT_RECIPROCAL_PI * tanTheta * exp(-cosTheta*cosTheta));

        const int ITER_THRESHOLD = 10;
        int it = 0;
        float value;
        while (++it<ITER_THRESHOLD && abs(value)<1.0e-5)
        {
            if (!(b>=a && b<=c))
                b = 0.5 * (a+c);

            float invErf = irr_glsl_erfInv(b);
            value = normalization * (1.0 + b + irr_glsl_SQRT_RECIPROCAL_PI * tanTheta * exp(-invErf*invErf)) - sample_x;
            float derivative = normalization * (1.0 - invErf*cosTheta);

            if (value > 0.0)
                c = b;
            else
                a = b;

            b -= value/derivative;
        }
        slope.x = irr_glsl_erfInv(b);
        slope.y = irr_glsl_erfInv(2.0 * max(u.y,1.0e-6) - 1.0);
    }
    
    float sinTheta = sqrt(1.0 - V.z*V.z);
    float cosPhi = sinTheta==0.0 ? 1.0 : clamp(V.x/sinTheta, -1.0, 1.0);
    float sinPhi = sinTheta==0.0 ? 0.0 : clamp(V.y/sinTheta, -1.0, 1.0);
    //rotate
    float tmp = cosPhi*slope.x - sinPhi*slope.y;
    slope.y = sinPhi*slope.x + cosPhi*slope.y;
    slope.x = tmp;

    //unstretch
    slope = vec2(ax,ay)*slope;

    //==== compute L ====
    vec3 H = normalize(vec3(-slope, 1.0));
    float NdotH = H.z;
    H = normalize(m*H);//transform to correct space
    //reflect
    float HdotV = dot(H,interaction.isotropic.V.dir);
    irr_glsl_BSDFSample smpl;
    smpl.L = H*2.0*HdotV - interaction.isotropic.V.dir;

    //==== compute probability ====
    //PBRT does it like a2 = cos2phi*ax*ax + sin2phi*ay*ay
    float a2 = ax*ay;
    float lambda = irr_glsl_smith_beckmann_Lambda(irr_glsl_smith_beckmann_C2(interaction.isotropic.NdotV_squared, a2));
    float G1 = 1.0 / (1.0 + lambda);
    smpl.probability = irr_glsl_beckmann(a2,NdotH*NdotH) * G1 * abs(dot(interaction.isotropic.V.dir,H)) / interaction.isotropic.NdotV;

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_beckmann_smith_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample, in float _ax, in float _ay)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_beckmann_smith_cos_gen_sample(interaction, u, _ax, _ay);
}

//TODO get rid of `a` parameter
vec3 irr_glsl_beckmann_smith_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in mat2x3 ior2, in float a, in float a2)
{
    float g = irr_glsl_beckmann_smith_height_correlated(params.interaction.NdotV_squared, params.NdotL_squared, a2);
    float ndf = irr_glsl_beckmann(a2, params.NdotH*params.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.VdotH);
    
    return g*ndf*fr / (4.0 * params.interaction.NdotV);
}

#endif
)";
    }
    static std::string getFresnel(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_
#define _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_

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
vec3 irr_glsl_fresnel_dielectric(in vec3 Eta, in float CosTheta)
{
   float SinTheta2 = 1.0 - CosTheta*CosTheta;

   vec3 t0 = sqrt(vec3(1.0) - (SinTheta2 / (Eta * Eta)));
   vec3 t1 = Eta * t0;
   vec3 t2 = Eta * CosTheta;

   vec3 rs = (vec3(CosTheta) - t1) / (vec3(CosTheta) + t1);
   vec3 rp = (t0 - t2) / (t0 + t2);

   return 0.5 * (rs * rs + rp * rp);
}

#endif
)";
    }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        // TODO: maybe change some paths, like separate out the NDFs out from BRDF/BSDFs and separate BSDF from BRDF
        return {
            { std::regex{"glsl/bsdf/brdf/diffuse/lambert\\.glsl"}, &getLambert },
            { std::regex{"glsl/bsdf/brdf/diffuse/oren_nayar\\.glsl"}, &getOrenNayar },
            { std::regex{"glsl/bsdf/brdf/specular/ndf/ggx\\.glsl"}, &getGGX_NDF },
            { std::regex{"glsl/bsdf/brdf/specular/geom/smith\\.glsl"}, &getSmith_G },
            { std::regex{"glsl/bsdf/brdf/specular/ggx\\.glsl"}, &getGGX_cos_eval },
            { std::regex{"glsl/bsdf/brdf/specular/fresnel/fresnel\\.glsl"}, &getFresnel },
            { std::regex{"glsl/bsdf/brdf/specular/blinn_phong\\.glsl"}, &getBlinnPhong },
            { std::regex{"glsl/bsdf/brdf/specular/ndf/beckmann\\.glsl"}, &getBeckmann },
            { std::regex{"glsl/bsdf/brdf/specular/ndf/ashikhmin_shirley\\.glsl"}, &getAshikhminShirleyNDF },
            { std::regex{"glsl/bsdf/brdf/specular/ashikhmin_shirley\\.glsl"}, &getAshikhminShirley_cos_eval },
            { std::regex{"glsl/bsdf/brdf/specular/beckmann_smith\\.glsl"}, &getBeckmannSmith_cos_eval },
            { std::regex{"glsl/bsdf/common\\.glsl"}, &getCommons },
            { std::regex{"glsl/bsdf/brdf/diffuse/fresnel_correction\\.glsl"}, &getDiffuseFresnelCorrectionFactor },
            { std::regex{"glsl/bsdf/brdf/cos_weighted_sample\\.glsl"}, &getCosWeightedSample },
            { std::regex{"glsl/bsdf/common_samples\\.glsl"}, &getCommonSamples }
        };
    }
};

}}

#endif //__IRR_C_GLSL_BSDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
