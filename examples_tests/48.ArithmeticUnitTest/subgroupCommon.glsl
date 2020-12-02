#include "shaderCommon.glsl"

#include "nbl/builtin/glsl/subgroup/arithmetic_portability.glsl"

#define CONDITIONAL_CLEAR_HEAD const bool automaticInitialize = ((_NBL_GLSL_WORKGROUP_SIZE_)&(nbl_glsl_SubgroupSize-1u))==0u; \
	const uint sourceVal = inputValue[gl_GlobalInvocationID.x];

#define CONDITIONAL_CLEAR_IMPL(IDENTITY_VALUE) if (!automaticInitialize) \
    { \
		barrier(); \
		memoryBarrierShared(); \
        SUBGROUP_SCRATCH_INITIALIZE(sourceVal,_NBL_GLSL_WORKGROUP_SIZE_,IDENTITY_VALUE,nbl_glsl_identityFunction) \
		barrier(); \
		memoryBarrierShared(); \
    }

#define CONDITIONAL_CLEAR_AND CONDITIONAL_CLEAR_IMPL(~0u)
#define CONDITIONAL_CLEAR_OR_XOR_ADD CONDITIONAL_CLEAR_IMPL(0u)
#define CONDITIONAL_CLEAR_MUL CONDITIONAL_CLEAR_IMPL(1u)
#define CONDITIONAL_CLEAR_MIN CONDITIONAL_CLEAR_IMPL(~0u)
#define CONDITIONAL_CLEAR_MAX CONDITIONAL_CLEAR_IMPL(0u)