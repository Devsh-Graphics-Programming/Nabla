#ifndef __IRR_C_GLSL_BSDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_BSDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

//TODO this file should change name to CGLSLBSDFBuiltinIncludeLoader.h

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

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_DirAndDifferential
{
   vec3 dir;
   // differentials at origin, I'd much prefer them to be differentials of barycentrics instead of position in the future
   mat2x3 dPosdScreen;
};

// do not use this struct in SSBO or UBO, its wasteful on memory
struct irr_glsl_ViewSurfaceInteraction
{
   irr_glsl_DirAndDifferential V; // outgoing direction, NOT NORMALIZED; V.dir can have undef value for lambertian BSDF
   vec3 N; // surface normal, NOT NORMALIZED
};

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
   float NdotV;
   float NdotV_squared;
   float VdotL; // same as LdotV
   float NdotH;
   float VdotH; // same as LdotH
   // left over for anisotropic calc and BSDF that want to implement fast bump mapping
   float LplusV_rcpLen;
   // basically metadata
   vec3 L;
   float invlenL2;
   irr_glsl_ViewSurfaceInteraction interaction;
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

// only in the fragment shader we have access to implicit derivatives
irr_glsl_ViewSurfaceInteraction irr_glsl_calcFragmentShaderSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos, in vec3 _Normal)
{
   irr_glsl_ViewSurfaceInteraction interaction;
   interaction.V.dir = _CamPos-_SurfacePos;
   interaction.V.dPosdScreen[0] = dFdx(_SurfacePos);
   interaction.V.dPosdScreen[1] = dFdy(_SurfacePos);
   interaction.N = _Normal;
   return interaction;
}
/*
//TODO it doesnt compile, lots of undefined symbols
// when you know the projected positions of your triangles (TODO: should probably make a function like this that also computes barycentrics)
irr_glsl_ViewSurfaceInteraction irr_glsl_calcBarycentricSurfaceInteraction(in vec3 _CamPos, in vec3 _SurfacePos[3], in vec3 _Normal[3], in float _Barycentrics[2], in vec2 _ProjectedPos[3])
{
   irr_glsl_ViewSurfaceInteraction interaction;

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
irr_glsl_ViewSurfaceInteraction  irr_glsl_calcRaySurfaceInteraction(in irr_glsl_DirAndDifferential _rayTowardsSurface, in vec3 _SurfacePos[3], in vec3 _Normal[3], in float _Barycentrics[2])
{
   irr_glsl_ViewSurfaceInteraction interaction;
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
irr_glsl_BSDFIsotropicParams irr_glsl_calcBSDFIsotropicParams(in irr_glsl_ViewSurfaceInteraction interaction, in vec3 L)
{
   float invlenV2 = inversesqrt(dot(interaction.V.dir,interaction.V.dir));
   float invlenN2 = inversesqrt(dot(interaction.N,interaction.N));
   float invlenL2 = inversesqrt(dot(L,L));

   irr_glsl_BSDFIsotropicParams params;

   // totally useless vectors, will probably get optimized away by compiler if they don't get used
   // but useful as temporaries
   params.interaction.V.dir = interaction.V.dir*invlenV2;
   params.interaction.N = interaction.N*invlenN2;
   params.L = L*invlenL2;
   params.invlenL2 = invlenL2;

   // this stuff only works with normalized L,N,V
   params.NdotL = dot(params.interaction.N,params.L);
   params.NdotL_squared = params.NdotL*params.NdotL;
   params.NdotV = dot(params.interaction.N,params.interaction.V.dir);
   params.NdotV_squared = params.NdotV*params.NdotV;

   params.VdotL = dot(params.interaction.V.dir,params.L);
   float LplusV_rcpLen = inversesqrt(2.0 + 2.0*params.VdotL);
   params.LplusV_rcpLen = LplusV_rcpLen;

   // this stuff works unnormalized L,N,V
   params.NdotH = (params.NdotL+params.NdotV)*LplusV_rcpLen;
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

#endif
)";
    }
    static std::string getLambert(const std::string&)
    {
        return
R"(#ifndef _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

float irr_glsl_lambert()
{
    return 1.0/3.14159265359;
}

#endif
)";
    }
    static std::string getOrenNayar(const std::string&)
    {
        return
R"(#ifndef _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_
#define _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_

float oren_nayar(in float _a2, in vec3 N, in vec3 L, in vec3 V, in float NdotL, in float NdotV)
{
    // theta - polar angles
    // phi - azimuth angles
    float a2 = _a2*0.5; //todo read about this
    vec2 AB = vec2(1.0, 0.0) + vec2(-0.5, 0.45) * vec2(a2, a2)/vec2(a2+0.33, a2+0.09);
    float C = 1.0 / max(NdotL, NdotV);

    // should be equal to cos(phi)*sin(theta_i)*sin(theta_o)
    // where `phi` is the angle in the tangent plane to N, between L and V
    // and `theta_i` is the sine of the angle between L and N, similarily for `theta_o` but with V
    float cos_phi_sin_theta = max(dot(V,L)-NdotL*NdotV,0.0);
    
    return (AB.x + AB.y * cos_phi_sin_theta * C) / 3.14159265359;
}

#endif
)";
    }
    static std::string getGGXTrowbridgeReitz(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_NDF_GGX_TROWBRIDGE_REITZ_INCLUDED_
#define _BRDF_SPECULAR_NDF_GGX_TROWBRIDGE_REITZ_INCLUDED_

float GGXTrowbridgeReitz(in float a2, in float NdotH)
{
    float denom = NdotH*NdotH * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265359 * denom*denom);
}

float GGXBurleyAnisotropic(float anisotropy, float a2, float TdotH, float BdotH, float NdotH) {
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab / 3.14159265359;
}

#endif
)";
    }
    static std::string getGGXSmith(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_
#define _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_

float _GGXSmith_G1_(in float a2, in float NdotX)
{
    return (2.0*NdotX) / (NdotX + sqrt(a2 + (1.0 - a2)*NdotX*NdotX));
}
float _GGXSmith_G1_wo_numerator(in float a2, in float NdotX)
{
    return 1.0 / (NdotX + sqrt(a2 + (1.0 - a2)*NdotX*NdotX));
}

float GGXSmith(in float a2, in float NdotL, in float NdotV)
{
    return _GGXSmith_G1_(a2, NdotL) * _GGXSmith_G1_(a2, NdotV);
}
float GGXSmith_wo_numerator(in float a2, in float NdotL, in float NdotV)
{
    return _GGXSmith_G1_wo_numerator(a2, NdotL) * _GGXSmith_G1_wo_numerator(a2, NdotV);
}

float GGXSmithHeightCorrelated(in float a2, in float NdotL, in float NdotV)
{
    float denom = NdotV*sqrt(a2 + (1.0 - a2)*NdotL*NdotL) + NdotL*sqrt(a2 + (1.0 - a2)*NdotV*NdotV);
    return 2.0*NdotL*NdotV / denom;
}

float GGXSmithHeightCorrelated_wo_numerator(in float a2, in float NdotL, in float NdotV)
{
    float denom = NdotV*sqrt(a2 + (1.0 - a2)*NdotL*NdotL) + NdotL*sqrt(a2 + (1.0 - a2)*NdotV*NdotV);
    return 0.5 / denom;
}

// Note a, not a2!
float GGXSmithHeightCorrelated_approx(in float a, in float NdotL, in float NdotV)
{
    float num = 2.0*NdotL*NdotV;
    return num / mix(num, NdotL+NdotV, a);
}

// Note a, not a2!
float GGXSmithHeightCorrelated_approx_wo_numerator(in float a, in float NdotL, in float NdotV)
{
    return 0.5 / mix(2.0*NdotL*NdotV, NdotL+NdotV, a);
}

//Taken from https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel
float GGXSmithHeightCorrelated_aniso_wo_numerator(in float at, in float ab, in float TdotL, in float TdotV, in float BdotL, in float BdotV, in float NdotL, in float NdotV)
{
    float Vterm = NdotL * length(vec3(at*TdotV, ab*BdotV, NdotV));
    float Lterm = NdotV * length(vec3(at*TdotL, ab*BdotL, NdotL));
    return 0.5 / (Vterm + Lterm);
}

#endif
)";
    }
    static std::string getFresnel(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_
#define _BRDF_SPECULAR_FRESNEL_FRESNEL_INCLUDED_

vec3 FresnelSchlick(in vec3 F0, in float VdotH)
{
    float x = 1.0 - VdotH;
    return F0 + (1.0 - F0) * x*x*x*x*x;
}

// code from https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
vec3 Fresnel_conductor(vec3 Eta2, vec3 Etak2, float CosTheta)
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
float Fresnel_dielectric(in float Eta, in float CosTheta)
{
   float SinTheta2 = 1.0 - CosTheta * CosTheta;

   float t0 = sqrt(1.0 - (SinTheta2 / (Eta * Eta)));
   float t1 = Eta * t0;
   float t2 = Eta * CosTheta;

   float rs = (CosTheta - t1) / (CosTheta + t1);
   float rp = (t0 - t2) / (t0 + t2);

   return 0.5 * (rs * rs + rp * rp);
}

#endif
)";
    }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"brdf/diffuse/lambert\\.glsl"}, &getLambert },
            { std::regex{"brdf/diffuse/oren_nayar\\.glsl"}, &getOrenNayar },
            { std::regex{"brdf/specular/ndf/ggx_trowbridge_reitz\\.glsl"}, &getGGXTrowbridgeReitz },
            { std::regex{"brdf/specular/geom/ggx_smith\\.glsl"}, &getGGXSmith },
            { std::regex{"brdf/specular/fresnel/fresnel\\.glsl"}, &getFresnel },
            { std::regex{"common\\.glsl"}, &getCommons },
            { std::regex{"brdf/diffuse/fresnel_correction\\.glsl"}, &getDiffuseFresnelCorrectionFactor },
        };
    }
};

}}

#endif //__IRR_C_GLSL_BSDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
