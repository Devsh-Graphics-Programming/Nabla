#ifndef __IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr {
namespace asset
{    

class CGLSLBRDFBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/vertex_utils/"; }

private:
    static std::string getVertexUtils(const std::string&)
    {
        return
R"(#ifndef _IRR_VERTEX_UTILS_INCLUDED_
#define _IRR_VERTEX_UTILS_INCLUDED_

vec3 pseudoMul4x4with3x1(in mat4 m, in vec3 v)
{
    return (m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3]).xyz;
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

}}

#endif//__IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__