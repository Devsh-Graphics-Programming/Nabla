#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_SHARED_ARITHMETIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_SHARED_ARITHMETIC_PORTABILITY_INCLUDED_


#include <irr/builtin/glsl/limits/numeric.glsl>
#include <irr/builtin/glsl/math/typeless_arithmetic.glsl>
#include <irr/builtin/glsl/subgroup/basic_portability.glsl>


/* TODO: @Hazardu or someone finish the definitions as soon as Nabla can report Vulkan GLSL equivalent caps
#ifdef GL_KHR_subgroup_basic


#define _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_ 0


#else
*/

#if defined(IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR)
	#define IRR_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(LAST_ITEM)  ((((IRR_GLSL_EVAL(LAST_ITEM)&(-irr_glsl_SubgroupSize))<<1)|(IRR_GLSL_EVAL(LAST_ITEM)&irr_glsl_SubgroupSize))+irr_glsl_HalfSubgroupSize+1)
#else
	#define IRR_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(LAST_ITEM)  ((IRR_GLSL_EVAL(LAST_ITEM)<<1)+(irr_glsl_MaxSubgroupSize>>1)+1)
#endif


#define _IRR_GLSL_SUBGROUP_ARITHMETIC_EMULATION_SHARED_SIZE_NEEDED_	IRR_GLSL_EVAL(IRR_GLSL_SUBGROUP_EMULATION_SCRATCH_BOUND(_IRR_GLSL_WORKGROUP_SIZE_-1))



//#endif //GL_KHR_subgroup_arithmetic


#endif
