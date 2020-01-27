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

vec3 irr_glsl_perturbNormal_heightMap(in vec3 vtxN, in vec3 dpdx, in vec3 dpdy, in float dhdx, in float dhdy)
{
    vec3 r1 = cross(dpdy, vtxN);
    vec3 r2 = cross(vtxN, dpdx);
    vec3 surfGrad = (r1*dhdx + r2*dhdy) / dot(dpdx,r1);
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