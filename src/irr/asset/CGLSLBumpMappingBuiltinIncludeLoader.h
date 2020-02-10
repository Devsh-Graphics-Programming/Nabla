#ifndef __IRR_C_GLSL_BUMP_MAPPING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_BUMP_MAPPING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr {
namespace asset
{    

class CGLSLBumpMappingBuiltinIncludeLoader : public IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/bump_mapping/"; }

private:
    static std::string getHeightMapHeader(const std::string&)
    {
        return
R"(#ifndef _IRR_BUMP_MAPPING_HEIGHT_MAP_INCLUDED_
#define _IRR_BUMP_MAPPING_HEIGHT_MAP_INCLUDED_

vec3 irr_glsl_perturbNormal_heightMap(in vec3 vtxN, in mat2x3 dPdScreen, in vec2 dHdScreen)
{
    vec3 r1 = cross(dPdScreen[1], vtxN);
    vec3 r2 = cross(vtxN, dPdScreen[0]);
    vec3 surfGrad = (r1*dHdScreen.x + r2*dHdScreen.y) / dot(dPdScreen[0],r1);
    return normalize(vtxN - surfGrad);
}

#endif
)";
    }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"height_mapping\\.glsl"}, &getHeightMapHeader },
        };
    }
};

}}

#endif//__IRR_C_GLSL_BUMP_MAPPING_BUILTIN_INCLUDE_LOADER_H_INCLUDED__