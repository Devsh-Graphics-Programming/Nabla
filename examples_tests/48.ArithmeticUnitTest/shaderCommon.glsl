#include "common.glsl"

#extension GL_EXT_control_flow_attributes : require

#include "irr/builtin/glsl/subgroup/arithmetic_portability.glsl"

layout(local_size_x = _IRR_GLSL_WORKGROUP_SIZE_) in;

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


#define CONDITIONAL_CLEAR_IMPL(IDENTITY_VALUE) if (((_IRR_GLSL_WORKGROUP_SIZE_)&(irr_glsl_SubgroupSize-1u))!=0u) \
    { \
		barrier(); \
		memoryBarrierShared(); \
        SUBGROUP_SCRATCH_CLEAR(_IRR_GLSL_WORKGROUP_SIZE_,IDENTITY_VALUE) \
		barrier(); \
		memoryBarrierShared(); \
    }

#define CONDITIONAL_CLEAR_AND CONDITIONAL_CLEAR_IMPL(~0u)
#define CONDITIONAL_CLEAR_OR_XOR_ADD CONDITIONAL_CLEAR_IMPL(0u)
#define CONDITIONAL_CLEAR_MUL CONDITIONAL_CLEAR_IMPL(1u)
#define CONDITIONAL_CLEAR_MIN CONDITIONAL_CLEAR_IMPL(~0u)
#define CONDITIONAL_CLEAR_MAX CONDITIONAL_CLEAR_IMPL(0u)