#include "shaderCommon.glsl"

#include "irr/builtin/glsl/subgroup/arithmetic_portability.glsl"

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