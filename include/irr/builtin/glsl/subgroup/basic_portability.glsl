#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

#include <irr/builtin/glsl/macros.glsl>

#ifndef _IRR_GLSL_WORKGROUP_SIZE_
	#error "User needs to let us know the size of the workgroup via _IRR_GLSL_WORKGROUP_SIZE_!"
#endif


#define irr_glsl_MaxWorkgroupSizeLog2 11
#define irr_glsl_MaxWorkgroupSize (0x1<<irr_glsl_MaxWorkgroupSizeLog2)


#define irr_glsl_MinSubgroupSizeLog2 2
#define irr_glsl_MinSubgroupSize (0x1<<irr_glsl_MinSubgroupSizeLog2)

#define irr_glsl_MaxSubgroupSizeLog2 7
#define irr_glsl_MaxSubgroupSize (0x1<<irr_glsl_MaxSubgroupSizeLog2)


// TODO: define this properly from gl_SubgroupSize and available extensions
#define IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
#define irr_glsl_SubgroupSizeLog2 2
#define irr_glsl_SubgroupSize (0x1<<irr_glsl_SubgroupSizeLog2)


#if IRR_GLSL_EVAL(irr_glsl_SubgroupSizeLog2)<IRR_GLSL_EVAL(irr_glsl_MinSubgroupSizeLog2)
	#error "Something went very wrong when figuring out irr_glsl_SubgroupSize!"
#endif
#define irr_glsl_HalfSubgroupSizeLog2 (irr_glsl_SubgroupSizeLog2-1)
#define irr_glsl_HalfSubgroupSize (0x1<<irr_glsl_HalfSubgroupSizeLog2)


#endif