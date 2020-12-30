#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_SHARED_ARITHMETIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_SHARED_ARITHMETIC_PORTABILITY_INCLUDED_


#include <nbl/builtin/glsl/limits/numeric.glsl>
#include <nbl/builtin/glsl/math/typeless_arithmetic.glsl>
#include <nbl/builtin/glsl/subgroup/basic_portability.glsl>


/* TODO: @Hazardu or someone finish the definitions as soon as Nabla can report Vulkan GLSL equivalent caps
#ifdef GL_KHR_subgroup_basic


#define _NBL_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_ 0


#else
*/

#if defined(NBL_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR)
	#define NBL_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(LAST_ITEM)  ((((NBL_GLSL_EVAL(LAST_ITEM)&(-nbl_glsl_SubgroupSize))<<1)|(NBL_GLSL_EVAL(LAST_ITEM)&nbl_glsl_SubgroupSize))+nbl_glsl_HalfSubgroupSize+1)
#else
	#define NBL_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(LAST_ITEM)  ((NBL_GLSL_EVAL(LAST_ITEM)<<1)+(nbl_glsl_MaxSubgroupSize>>1)+1)
#endif


#define _NBL_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_	NBL_GLSL_EVAL(NBL_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(_NBL_GLSL_WORKGROUP_SIZE_-1))



//#endif //GL_KHR_subgroup_arithmetic


#endif
