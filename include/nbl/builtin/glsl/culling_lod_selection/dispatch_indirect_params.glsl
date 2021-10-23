#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_PARAMS_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_PARAMS_GLSL_INCLUDED_

struct nbl_glsl_culling_lod_selection_dispatch_indirect_params_t
{
	nbl_glsl_DispatchIndirectCommand_t instanceCullAndLoDSelect; // cleared to 1 by draw cull
	nbl_glsl_DispatchIndirectCommand_t instanceDrawCountPrefixSum; // cleared to 1 by scatter, filled out by instance Cull
	nbl_glsl_DispatchIndirectCommand_t instanceDrawCull; // set by the draw count prefix sum
	nbl_glsl_DispatchIndirectCommand_t instanceRefCountingSortScatter; // clear to 1 by draw count prefix sum, filled out by draw cull
	nbl_glsl_DispatchIndirectCommand_t drawCompact; // TODO
};

#endif