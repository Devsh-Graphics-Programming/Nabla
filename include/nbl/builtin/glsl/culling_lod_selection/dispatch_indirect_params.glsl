#ifndef _NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_PARAMS_GLSL_INCLUDED_
#define _NBL_GLSL_CULLING_LOD_SELECTION_DISPATCH_INDIRECT_PARAMS_GLSL_INCLUDED_

struct nbl_glsl_culling_lod_selection_dispatch_indirect_params_t
{
	nbl_glsl_DispatchIndirectCommand_t instanceCullAndLoDSelect; // cleared by indirect prefix sum TODO
	nbl_glsl_DispatchIndirectCommand_t instanceDrawCountPrefixSum; // cleared by instance draw cull
	nbl_glsl_DispatchIndirectCommand_t instanceDrawCull; // cleared by counting sort scatter
	nbl_glsl_DispatchIndirectCommand_t instanceRefCountingSortScatter; // cleared by LoD Cull and Select
	nbl_glsl_DispatchIndirectCommand_t drawCompact;
};

#endif