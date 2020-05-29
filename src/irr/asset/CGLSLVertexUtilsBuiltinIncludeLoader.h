#ifndef __IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__
#define __IRR_C_GLSL_VERTEX_UTILS_BUILTIN_INCLUDE_LOADER_H_INCLUDED__

#include "irr/asset/IBuiltinIncludeLoader.h"

namespace irr {
namespace asset
{    

class CGLSLVertexUtilsBuiltinIncludeLoader : public irr::asset::IBuiltinIncludeLoader
{
public:
    // TODO: don't like that the include paht is `irr/builtin/glsl/vertex_utils/vertex_utils.glsl"
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
mat4x3 irr_glsl_pseudoMul4x3with4x3(in mat4x3 lhs, in mat4x3 rhs)
{
    mat4x3 result;
    for (int i=0; i<4; i++)
        result[i] = lhs[0]*rhs[i][0]+lhs[1]*rhs[i][1]+lhs[2]*rhs[i][2];
    result[3] += lhs[3];
    return result;
}
mat4 irr_glsl_pseudoMul4x4with4x3(in mat4 proj, in mat4x3 tform)
{
    mat4 result;
    for (int i=0; i<4; i++)
        result[i] = proj[0]*tform[i][0]+proj[1]*tform[i][1]+proj[2]*tform[i][2];
    result[3] += proj[3];
    return result;
}

float irr_glsl_lengthManhattan(float v)
{
    return abs(v);
}
float irr_glsl_lengthManhattan(vec2 v)
{
	v = abs(v);
    return v.x+v.y;
}
float irr_glsl_lengthManhattan(vec3 v)
{
    v = abs(v);
    return v.x+v.y+v.z;
}
float irr_glsl_lengthManhattan(vec4 v)
{
    v = abs(v);
    return v.x+v.y+v.z+v.w;
}

float irr_glsl_lengthSq(in float v)
{
    return v*v;
}
float irr_glsl_lengthSq(in vec2 v)
{
    return dot(v,v);
}
float irr_glsl_lengthSq(in vec3 v)
{
    return dot(v,v);
}
float irr_glsl_lengthSq(in vec4 v)
{
    return dot(v,v);
}

bool irr_glsl_couldBeVisible(in mat4 proj, in mat2x3 bbox)
{
    mat4 pTpose = transpose(proj);
    mat4 xyPlanes = mat4(pTpose[3]+pTpose[0],pTpose[3]+pTpose[1],pTpose[3]-pTpose[0],pTpose[3]-pTpose[1]);
    vec4 farPlane = pTpose[3]+pTpose[2];

#define getClosestDP(R) (dot(mix(bbox[1],bbox[0],lessThan(R.xyz,vec3(0.f)) ),R.xyz)+R.w>0.f)

    return  getClosestDP(xyPlanes[0])&&getClosestDP(xyPlanes[1])&&
            getClosestDP(xyPlanes[2])&&getClosestDP(xyPlanes[3])&&
            getClosestDP(pTpose[3])&&getClosestDP(farPlane);
#undef getClosestDP
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