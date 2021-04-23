#ifndef _NBL_BUILTIN_GLSL_SUBGROUP_SHARED_SHUFFLE_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_GLSL_SUBGROUP_SHARED_SHUFFLE_PORTABILITY_INCLUDED_


#include <nbl/builtin/glsl/subgroup/basic_portability.glsl>


#ifdef NBL_GL_KHR_shader_subgroup_shuffle


#define _NBL_GLSL_SUBGROUP_SHUFFLE_EMULATION_SHARED_SIZE_NEEDED_ 0


#else


#ifndef _NBL_GLSL_WORKGROUP_SIZE_
#error "_NBL_GLSL_WORKGROUP_SIZE_ should be defined."
#endif

#define _NBL_GLSL_SUBGROUP_SHUFFLE_EMULATION_SHARED_SIZE_NEEDED_ ((_NBL_GLSL_WORKGROUP_SIZE_+nbl_glsl_MaxSubgroupSize-1)&(-nbl_glsl_MaxSubgroupSize))


#endif // NBL_GL_KHR_shader_subgroup_shuffle


#endif
