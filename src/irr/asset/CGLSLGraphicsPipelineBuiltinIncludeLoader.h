#ifndef __IRR_C_GLSL_GRAPHICS_PIPELINE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_GRAPHICS_PIPELINE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr {
namespace asset
{    

class CGLSLGraphicsPipelineBuiltinIncludeLoader : public IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/graphicspipeline/"; }

private:
    static std::string getMTLloaderCommonDefinitions(const std::string&)
    {
        return
R"(#ifndef _IRR_MTL_LOADER_COMMON_INCLUDED_
#define _IRR_MTL_LOADER_COMMON_INCLUDED_

struct irr_glsl_MTLMaterialParameters 
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    vec4 Tf;//w component doesnt matter
    float Ns;
    float d;
    float bm;
    float Ni;
    float roughness;
    float metallic;
    float sheen;
    float clearcoatThickness;
    float clearcoatRoughness;
    float anisotropy;
    float anisoRotation;
    //extra info
    uint extra;
};

#endif
)";
    }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"loaders/mtl/common\\.glsl"}, &getMTLloaderCommonDefinitions },
        };
    }
};

}}

#endif//__IRR_C_GLSL_GRAPHICS_PIPELINE_BUILTIN_INCLUDE_LOADER_H_INCLUDED__