#ifndef _NBL_GLSL_OIT_INSERT_NODE_GLSL_INCLUDED_
#define _NBL_GLSL_OIT_INSERT_NODE_GLSL_INCLUDED_

layout(early_fragment_tests) in;

#include <nbl/builtin/glsl/ext/OIT/oit.glsl>

void nbl_glsl_oit_swap_node(inout float a_vis, inout uint a_depth, inout uint a_col, inout float b_vis, inout uint b_depth, inout uint b_col)
{
	float t_vis = a_vis;
	uint t_depth = a_depth;
	uint t_col = a_col;

	a_vis = b_vis;
	a_depth = b_depth;
	a_col = b_col;

	b_vis = t_vis;
	b_depth = t_depth;
	b_col = t_col;
}

void nbl_glsl_oit_insert_node(in ivec2 coord, in uint mydepth, in float myvis, in uint mycolor)
{
	nbl_glsl_oit_color_nodes_t color;
	nbl_glsl_oit_depth_nodes_t depth;
	nbl_glsl_oit_vis_nodes_t vis;
#if NBL_GLSL_OIT_NODE_COUNT==4
    depth = imageLoad(g_depth, coord);
    vis = imageLoad(g_vis, coord);
#elif NBL_GLSL_OIT_NODE_COUNT==2
    depth = imageLoad(g_depth, coord).rg;
    vis = imageLoad(g_vis, coord).rg;
#endif

	nbl_glsl_oit_bvec_t notValidMask = equal(nbl_glsl_oit_vis_nodes_t(1.0),vis);
	// it requires GL_EXT_shader_integer_mix for mix() with integers XD
	depth = floatBitsToUint( mix(uintBitsToFloat(depth),nbl_glsl_oit_vec_t(0.0),notValidMask) );
	nbl_glsl_oit_bvec_t closerMask = greaterThanEqual(nbl_glsl_oit_uvec_t(mydepth),depth);
	nbl_glsl_oit_vis_nodes_t maskedVis = mix(vis, nbl_glsl_oit_vis_nodes_t(1.0), closerMask);

	//if (maskedVis.x*maskedVis.y*maskedVis.z*maskedVis.w < 0.1)
		//discard;

#if NBL_GLSL_OIT_NODE_COUNT==4
    color = imageLoad(g_color, coord);
#elif NBL_GLSL_OIT_NODE_COUNT==2
    color = imageLoad(g_color, coord).rg;
#endif
	// it requires GL_EXT_shader_integer_mix for mix() with integers XD
	color = floatBitsToUint( mix(uintBitsToFloat(color),nbl_glsl_oit_vec_t(0.0),notValidMask) );

	if (closerMask[0])
	{
		nbl_glsl_oit_swap_node(vis[0], depth[0], color[0], myvis, mydepth, mycolor);
	}
	if (closerMask[1])
	{
		nbl_glsl_oit_swap_node(vis[1], depth[1], color[1], myvis, mydepth, mycolor);
	}
#if NBL_GLSL_OIT_NODE_COUNT>2
	if (closerMask[2])
	{
		nbl_glsl_oit_swap_node(vis[2], depth[2], color[2], myvis, mydepth, mycolor);
	}
	if (closerMask[3])
	{
		nbl_glsl_oit_swap_node(vis[3], depth[3], color[3], myvis, mydepth, mycolor);
	}
#endif

	// all nodes were valid before insertion, means we have overflow
	if (!notValidMask[NBL_GLSL_OIT_NODE_COUNT-1])
	{
		//MLAB
		color[NBL_GLSL_OIT_NODE_COUNT-1] = packUnorm4x8(unpackUnorm4x8(color[NBL_GLSL_OIT_NODE_COUNT-1]) + unpackUnorm4x8(mycolor)*vis[NBL_GLSL_OIT_NODE_COUNT-1]);
		vis[NBL_GLSL_OIT_NODE_COUNT-1] *= myvis;
	}

#if NBL_GLSL_OIT_NODE_COUNT==4
	imageStore(g_color, coord, color);
	imageStore(g_depth, coord, depth);
	imageStore(g_vis, coord, vis);
#elif NBL_GLSL_OIT_NODE_COUNT==2
	imageStore(g_color, coord, uvec4(color, 0u, 0u));
	imageStore(g_depth, coord, uvec4(depth, 0u, 0u));
	imageStore(g_vis, coord, vec4(vis, 0.0, 0.0));
#endif
}

#endif //_NBL_GLSL_OIT_INSERT_NODE_GLSL_INCLUDED_
