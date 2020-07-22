#ifndef _IRR_VERT_INPUTS_DEFINED_
#define _IRR_VERT_INPUTS_DEFINED_
layout (location = 0) in vec3 vPos;
#ifndef _NO_UV
layout (location = 2) in vec2 vUV;
#endif
layout (location = 3) in vec3 vNormal;
#endif //_IRR_VERT_INPUTS_DEFINED_

#ifndef _IRR_VERT_OUTPUTS_DEFINED_
#define _IRR_VERT_OUTPUTS_DEFINED_
layout (location = 0) out vec3 LocalPos;
layout (location = 1) out vec3 ViewPos;
layout (location = 2) out vec3 Normal;
#ifndef _NO_UV
layout (location = 3) out vec2 UV;
#endif
#endif //_IRR_VERT_OUTPUTS_DEFINED_

#include <irr/builtin/glsl/utils/vertex.glsl>

#ifndef _IRR_VERT_SET1_BINDINGS_DEFINED_
#define _IRR_VERT_SET1_BINDINGS_DEFINED_
layout (set = 1, binding = 0, row_major, std140) uniform UBO
{
    irr_glsl_SBasicViewParameters params;
} CamData;
#endif //_IRR_VERT_SET1_BINDINGS_DEFINED_

#ifndef _IRR_VERT_MAIN_DEFINED_
#define _IRR_VERT_MAIN_DEFINED_
void main()
{
    LocalPos = vPos;
    gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CamData.params.MVP), vPos);
    ViewPos = irr_glsl_pseudoMul3x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CamData.params.MV), vPos);
    mat3 normalMat = irr_glsl_SBasicViewParameters_GetNormalMat(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CamData.params.NormalMatAndEyePos));
    Normal = normalMat*normalize(vNormal);
#ifndef _NO_UV
    UV = vUV;
#endif
}
#endif //_IRR_VERT_MAIN_DEFINED_