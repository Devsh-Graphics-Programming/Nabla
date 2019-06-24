#ifndef C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED
#define C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED

#include "irr/asset/IBuiltinIncludeLoader.h"

class CBRDFBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
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
    vec2 cos_theta = vec2(NdotL, NdotV);
    vec2 cos_theta2 = cos_theta*cos_theta;
    float sin_theta = sqrt((1.0 - cos_theta2.x) * (1.0 - cos_theta2.y)); //this is actually equal to (sin(theta_i) * sin(theta_r))
    float C = sin_theta / max(cos_theta.x, cos_theta.y);
    
    vec3 light_plane = normalize(L - cos_theta.x*N);
    vec3 view_plane = normalize(V - cos_theta.y*N);
    float cos_phi = max(0.0, dot(light_plane, view_plane));//not sure about this

    return (AB.x + AB.y * cos_phi * C) / 3.14159265359;
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
vec3 Fresnel_conductor(vec3 Eta, vec3 Etak, float CosTheta)
{  
   float CosTheta2 = CosTheta * CosTheta;
   float SinTheta2 = 1.0 - CosTheta2;
   vec3 Eta2 = Eta * Eta;
   vec3 Etak2 = Etak * Etak;

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
            { std::regex{"diffuse/oren_nayar\\.glsl"}, &getOrenNayar },
            { std::regex{"specular/ndf/ggx_trowbridge_reitz\\.glsl"}, &getGGXTrowbridgeReitz },
            { std::regex{"specular/geom/ggx_smith\\.glsl"}, &getGGXSmith },
            { std::regex{"specular/fresnel/fresnel\\.glsl"}, &getFresnel }
        };
    }
};

#endif //C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED