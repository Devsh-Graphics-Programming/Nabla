#include "common.glsl"

layout(local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;

layout(set = 0, binding = 0, std430) readonly buffer inputBuff
{
    uint inputValue[];
};
layout(set = 0, binding = 1, std430) writeonly buffer outand
{
    uint andOutput[];
};
layout(set = 0, binding = 2, std430) writeonly buffer outxor
{
    uint xorOutput[];
};
layout(set = 0, binding = 3, std430) writeonly buffer outor
{
    uint orOutput[];
};
layout(set = 0, binding = 4, std430) writeonly buffer outadd
{
    uint addOutput[];
};
layout(set = 0, binding = 5, std430) writeonly buffer outmul
{
    uint multOutput[];
};
layout(set = 0, binding = 6, std430) writeonly buffer outmin
{
    uint minOutput[];
};
layout(set = 0, binding = 7, std430) writeonly buffer outmax
{
    uint maxOutput[];
};
layout(set = 0, binding = 8, std430) writeonly buffer outbitcount
{
    uint bitCountOutput[];
};