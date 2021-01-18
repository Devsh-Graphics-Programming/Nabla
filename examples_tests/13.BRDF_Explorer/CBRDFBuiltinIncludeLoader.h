// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED
#define C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED

#include "nbl/asset/IBuiltinIncludeLoader.h"

class CBRDFBuiltinIncludeLoader : public nbl::asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/brdf/"; }

private:
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
    nbl::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"diffuse/oren_nayar\\.glsl"}, &getOrenNayar },
            { std::regex{"specular/ndf/ggx_trowbridge_reitz\\.glsl"}, &getGGXTrowbridgeReitz },
            { std::regex{"specular/geom/ggx_smith\\.glsl"}, &getGGXSmith },
            { std::regex{"specular/fresnel/fresnel\\.glsl"}, &getFresnel }
        };
    }
};

#endif //C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED
