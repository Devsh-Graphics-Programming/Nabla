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

    return cos_theta.x * (AB.x + AB.y * cos_phi * C) / 3.14159265359;
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

float GGXSmith(in float a2, in float NdotL, in float NdotV)
{
    return _GGXSmith_G1_(a2, NdotL) * _GGXSmith_G1_(a2, NdotV);
}

#endif
)";
    }
    static std::string getFresnelSchlick(const std::string&)
    {
        return
R"(#ifndef _BRDF_SPECULAR_FRESNEL_FRESNEL_SCHLICK_INCLUDED_
#define _BRDF_SPECULAR_FRESNEL_FRESNEL_SCHLICK_INCLUDED_

vec3 FresnelSchlick(in vec3 F0, in float NdotV)
{
    float x = 1.0 - NdotV;
    return F0 + (1.0 - F0) * x*x*x*x*x;
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
            { std::regex{"specular/fresnel/fresnel_schlick\\.glsl"}, &getFresnelSchlick }
        };
    }
};

#endif //C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED