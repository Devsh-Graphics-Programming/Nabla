#include "common.glsl"

layout(local_size_x = _IRR_GLSL_WORKGROUP_SIZE_) in;

layout(set = 0, binding = 0, std430) writeonly buffer outBitCount
{
    uint bitcountOutput[];
};