#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_

#include <nbl/builtin/glsl/macros.glsl>

#ifndef _NBL_GLSL_WORKGROUP_SIZE_
	#error "User needs to let us know the size of the workgroup via _NBL_GLSL_WORKGROUP_SIZE_!"
#endif


#define nbl_glsl_MaxWorkgroupSizeLog2 11
#define nbl_glsl_MaxWorkgroupSize (0x1<<nbl_glsl_MaxWorkgroupSizeLog2)


#define nbl_glsl_MinSubgroupSizeLog2 2
#define nbl_glsl_MinSubgroupSize (0x1<<nbl_glsl_MinSubgroupSizeLog2)

#define nbl_glsl_MaxSubgroupSizeLog2 7
#define nbl_glsl_MaxSubgroupSize (0x1<<nbl_glsl_MaxSubgroupSizeLog2)


// TODO: define this properly from gl_SubgroupSize and available extensions
#define NBL_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
#define nbl_glsl_SubgroupSizeLog2 2
#define nbl_glsl_SubgroupSize (0x1<<nbl_glsl_SubgroupSizeLog2)


#if NBL_GLSL_EVAL(nbl_glsl_SubgroupSizeLog2)<NBL_GLSL_EVAL(nbl_glsl_MinSubgroupSizeLog2)
	#error "Something went very wrong when figuring out nbl_glsl_SubgroupSize!"
#endif
#define nbl_glsl_HalfSubgroupSizeLog2 (nbl_glsl_SubgroupSizeLog2-1)
#define nbl_glsl_HalfSubgroupSize (0x1<<nbl_glsl_HalfSubgroupSizeLog2)


#endif