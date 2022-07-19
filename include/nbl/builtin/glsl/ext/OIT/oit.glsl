#ifndef _NBL_GLSL_OIT_GLSL_INCLUDED_
#define _NBL_GLSL_OIT_GLSL_INCLUDED_

#ifndef NBL_GLSL_OIT_SET_NUM
#define NBL_GLSL_OIT_SET_NUM 2
#endif
#ifndef NBL_GLSL_COLOR_IMAGE_BINDING
#define NBL_GLSL_COLOR_IMAGE_BINDING 0
#endif
#ifndef NBL_GLSL_DEPTH_IMAGE_BINDING
#define NBL_GLSL_DEPTH_IMAGE_BINDING 1
#endif
#ifndef NBL_GLSL_VIS_IMAGE_BINDING
#define NBL_GLSL_VIS_IMAGE_BINDING 2
#endif
#ifndef NBL_GLSL_SPINLOCK_IMAGE_BINDING
#define NBL_GLSL_SPINLOCK_IMAGE_BINDING 3
#endif

// TODO remove later, this should be inserted into GLSL automatically by engine (or not if shader interlock ext not present)
#define NBL_GL_ARB_fragment_shader_interlock

#define NBL_GLSL_OIT_NODE_COUNT 4

#if NBL_GLSL_OIT_NODE_COUNT==4

#ifndef __cplusplus
#define NBL_GLSL_OIT_IMG_FORMAT_COLOR    rgba32ui
#define NBL_GLSL_OIT_IMG_FORMAT_DEPTH    rgba16ui
#define NBL_GLSL_OIT_IMG_FORMAT_VIS      rgba8

#define nbl_glsl_oit_bvec_t          bvec4
#define nbl_glsl_oit_ivec_t          ivec4
#define nbl_glsl_oit_uvec_t          uvec4
#define nbl_glsl_oit_vec_t           vec4
#else
#define NBL_GLSL_OIT_IMG_FORMAT_COLOR    nbl::asset::EF_R32G32B32A32_UINT
#define NBL_GLSL_OIT_IMG_FORMAT_DEPTH    nbl::asset::EF_R16G16B16A16_UINT
#define NBL_GLSL_OIT_IMG_FORMAT_VIS      nbl::asset::EF_R8G8B8A8_UNORM
#endif

#elif NBL_GLSL_OIT_NODE_COUNT==2

#ifndef __cplusplus
#define NBL_GLSL_OIT_IMG_FORMAT_COLOR    rg32ui
#define NBL_GLSL_OIT_IMG_FORMAT_DEPTH    rg16ui
#define NBL_GLSL_OIT_IMG_FORMAT_VIS      rg8

#define nbl_glsl_oit_bvec_t          bvec2
#define nbl_glsl_oit_ivec_t          ivec2
#define nbl_glsl_oit_uvec_t          uvec2
#define nbl_glsl_oit_vec_t           vec2
#else
#define NBL_GLSL_OIT_IMG_FORMAT_COLOR    nbl::asset::EF_R32G32_UINT
#define NBL_GLSL_OIT_IMG_FORMAT_DEPTH    nbl::asset::EF_R16G16_UINT
#define NBL_GLSL_OIT_IMG_FORMAT_VIS      nbl::asset::EF_R8G8_UNORM
#endif

#else

#error "Allowed OIT node counts are 4 and 2!"

#endif

#ifndef __cplusplus

#define nbl_glsl_oit_color_nodes_t   nbl_glsl_oit_uvec_t
#define nbl_glsl_oit_depth_nodes_t   nbl_glsl_oit_uvec_t
#define nbl_glsl_oit_vis_nodes_t     nbl_glsl_oit_vec_t

#ifdef  _NBL_GLSL_OIT_GLSL_RESOLVE_FRAG_
#define IMAGE_QUALIFIERS uniform readonly
#define VIS_QUALIFIERS uniform
#else
#define IMAGE_QUALIFIERS uniform coherent
#define VIS_QUALIFIERS IMAGE_QUALIFIERS
#endif
layout(set = NBL_GLSL_OIT_SET_NUM, binding = NBL_GLSL_COLOR_IMAGE_BINDING, NBL_GLSL_OIT_IMG_FORMAT_COLOR) IMAGE_QUALIFIERS uimage2D g_color;
layout(set = NBL_GLSL_OIT_SET_NUM, binding = NBL_GLSL_DEPTH_IMAGE_BINDING, NBL_GLSL_OIT_IMG_FORMAT_DEPTH) IMAGE_QUALIFIERS uimage2D g_depth;
layout(set = NBL_GLSL_OIT_SET_NUM, binding = NBL_GLSL_VIS_IMAGE_BINDING,   NBL_GLSL_OIT_IMG_FORMAT_VIS) VIS_QUALIFIERS image2D g_vis;
#undef IMAGE_QUALIFIERS
#undef VIS_QUALIFIERS

#ifdef NBL_GL_ARB_fragment_shader_interlock
#define NBL_GLSL_OIT_CRITICAL_SECTION(FUNC) beginInvocationInterlockARB(); FUNC; endInvocationInterlockARB()
#else
layout(set = NBL_GLSL_OIT_SET_NUM, binding = NBL_GLSL_SPINLOCK_IMAGE_BINDING,   r32ui) uniform coherent uimage2D g_lock;
#define NBL_GLSL_OIT_CRITICAL_SECTION(FUNC) for (bool done=gl_HelperInvocation; !done;) {\
	if (imageAtomicExchange(g_lock,ivec2(gl_FragCoord.xy),1u)==0u) \
	{ \
		FUNC; \
		imageStore(g_lock,ivec2(gl_FragCoord.xy),uvec4(0u)); \
		done = true; \
	} \
}
#endif


float nbl_glsl_oit_get_rev_depth()
{
    float d = gl_FragCoord.z;
    return d;
}
uint nbl_glsl_oit_encode_depth(in float d)
{
    uint du = floatBitsToUint(d);
    // 9 bits of mantissa and 7 of exponent
    return bitfieldExtract(du, 14, 16);
}

#endif //!__cplusplus

#endif //_NBL_GLSL_OIT_GLSL_INCLUDED_
