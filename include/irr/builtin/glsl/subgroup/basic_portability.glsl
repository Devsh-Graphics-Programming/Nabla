#ifndef _IRR_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_
#define _IRR_BUILTIN_GLSL_SUBGROUP_BASIC_PORTABILITY_INCLUDED_


#ifndef _IRR_GLSL_WORKGROUP_SIZE_
#error "User needs to let us know the size of the workgroup via _IRR_GLSL_WORKGROUP_SIZE_!"
#endif

#define irr_glsl_MaxWorkgroupSize 1024u

#define irr_glsl_MinSubgroupSize 4u
#define irr_glsl_MaxSubgroupSize 128u
// TODO: define this properly from gl_SubgroupSize and available extensions
#define IRR_GLSL_SUBGROUP_SIZE_IS_CONSTEXPR
#define irr_glsl_SubgroupSize 4u

#if irr_glsl_SubgroupSize<irr_glsl_MinSubgroupSize
	#error "Something went very wrong when figuring out irr_glsl_SubgroupSize!"
#endif
#define irr_glsl_HalfSubgroupSize (irr_glsl_SubgroupSize>>1u)

#endif