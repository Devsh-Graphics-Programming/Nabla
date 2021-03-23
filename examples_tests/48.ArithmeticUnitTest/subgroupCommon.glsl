#include "shaderCommon.glsl"

#include "nbl/builtin/glsl/subgroup/arithmetic_portability.glsl"

#define CONDITIONAL_CLEAR_HEAD const bool automaticInitialize = ((_NBL_GLSL_WORKGROUP_SIZE_)&(nbl_glsl_SubgroupSize-1u))==0u; \
	const uint sourceVal = inputValue[gl_GlobalInvocationID.x]; \
    if (gl_GlobalInvocationID.x==0) \
    { \
        subgroupSizeAnd = nbl_glsl_SubgroupSize; \
        subgroupSizeXor = nbl_glsl_SubgroupSize; \
        subgroupSizeOr = nbl_glsl_SubgroupSize; \
        subgroupSizeAdd = nbl_glsl_SubgroupSize; \
        subgroupSizeMult = nbl_glsl_SubgroupSize; \
        subgroupSizeMin = nbl_glsl_SubgroupSize; \
        subgroupSizeMax = nbl_glsl_SubgroupSize; \
        subgroupSizeBitcount = nbl_glsl_SubgroupSize; \
    }

#define CONDITIONAL_CLEAR_IMPL(IDENTITY_VALUE) if (!automaticInitialize) \
    { \
		barrier(); \
        SUBGROUP_SCRATCH_INITIALIZE(sourceVal,_NBL_GLSL_WORKGROUP_SIZE_,IDENTITY_VALUE,nbl_glsl_identityFunction) \
    }

#define CONDITIONAL_CLEAR_AND CONDITIONAL_CLEAR_IMPL(~0u)
#define CONDITIONAL_CLEAR_OR_XOR_ADD CONDITIONAL_CLEAR_IMPL(0u)
#define CONDITIONAL_CLEAR_MUL CONDITIONAL_CLEAR_IMPL(1u)
#define CONDITIONAL_CLEAR_MIN CONDITIONAL_CLEAR_IMPL(~0u)
#define CONDITIONAL_CLEAR_MAX CONDITIONAL_CLEAR_IMPL(0u)