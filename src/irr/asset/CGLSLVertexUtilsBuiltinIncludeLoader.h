#ifndef __IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr {
namespace asset
{    

class CGLSLVertexUtilsBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
public:
    const char* getVirtualDirectoryName() const override { return "glsl/vertex_utils/"; }

private:
    static std::string getVertexUtils(const std::string&)
    {
        return
R"(#ifndef _IRR_VERTEX_UTILS_INCLUDED_
#define _IRR_VERTEX_UTILS_INCLUDED_

#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>


struct irr_glsl_SBasicViewParameters
{
    mat4 MVP;
    mat4x3 MV;
    mat4x3 NormalMatAndEyePos;
};
mat3 irr_glsl_SBasicViewParameters_GetNormalMat(in mat4x3 _NormalMatAndEyePos)
{
    return mat3(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x3(_NormalMatAndEyePos));
}
vec3 irr_glsl_SBasicViewParameters_GetEyePos(in mat4x3 _NormalMatAndEyePos)
{
    return irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x3(_NormalMatAndEyePos)[3];
}

vec4 irr_glsl_pseudoMul4x4with3x1(in mat4 m, in vec3 v)
{
    return m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3];
}
vec3 irr_glsl_pseudoMul3x4with3x1(in mat4x3 m, in vec3 v)
{
    return m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3];
}

#endif
)";
    }

protected:
    irr::core::vector<std::pair<std::regex, HandleFunc_t>> getBuiltinNamesToFunctionMapping() const override
    {
        return {
            { std::regex{"vertex_utils\\.glsl"}, &getVertexUtils },
        };
    }
};

}}

#endif//__IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__