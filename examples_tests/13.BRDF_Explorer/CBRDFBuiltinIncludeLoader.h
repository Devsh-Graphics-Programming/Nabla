#ifndef C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED
#define C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED

#include "irr/asset/IBuiltinIncludeLoader.h"

class CBRDFBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/brdf/"; }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        /* TODO */
        return {
            { std::regex{"diffuse/lambert\\.glsl"}, [this](const std::string&) { return nullptr; }},
            { std::regex{"diffuse/oren_nayar\\.glsl"}, [](const std::string&) { return nullptr; }},
            { std::regex{"specular/ndf/ggx_trowbridge_reitz\\.glsl"}, nullptr },
            { std::regex{"specular/geom/ggx_smith\\.glsl"}, nullptr },
            { std::regex{"specular/fresnel/fresnel_schlick\\.glsl"}, nullptr }
        };
    }
};

#endif //C_BRDF_BUILTIN_INCLUDE_LOADER_H_INCLUDED