#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

#include <irr/builtin/glsl/macros.glsl>

#ifndef _IRR_GLSL_WORKGROUP_SIZE_
	#error "User needs to let us know the size of the workgroup via _IRR_GLSL_WORKGROUP_SIZE_!"
#endif

#define irr_glsl_MaxWorkgroupSize 1024

#define irr_glsl_MinSubgroupSize 4
#define irr_glsl_MaxSubgroupSize 128
// TODO: define this properly from gl_SubgroupSize and available extensions
#define IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
#define irr_glsl_SubgroupSize 4

#if IRR_GLSL_EVAL(irr_glsl_SubgroupSize)<IRR_GLSL_EVAL(irr_glsl_MinSubgroupSize)
	#error "Something went very wrong when figuring out irr_glsl_SubgroupSize!"
#endif
#define irr_glsl_HalfSubgroupSize (irr_glsl_SubgroupSize>>1u)

#endif