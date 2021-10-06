#ifndef _NBL_GLSL_LOD_LIBRARY_STRUCTS_GLSL_INCLUDED_
#define _NBL_GLSL_LOD_LIBRARY_STRUCTS_GLSL_INCLUDED_


struct nbl_glsl_lod_library_DrawcallInfo
{
	uint drawcallDWORDOffset;
	uint skinningAABBCountAndOffset; // TODO: review
};

struct nbl_glsl_lod_library_LoDInfoBase
{
	vec3 aabbMin;
	uint drawcallInfoCountAndTotalBoneCount;
	vec3 aabbMax;
};
#define NBL_GLSL_LOD_LIBRARY_LOD_INFO_BASE_SIZE 28

struct nbl_glsl_lod_library_DefaultLoDChoiceParams
{
	float distanceSqAtReferenceFoV;
};
#define NBL_GLSL_LOD_LIBRARY_DEFAULT_LOD_CHOICE_PARAMS_SIZE 4

/*
#ifdef nbl_glsl_lod_library_LoDChoiceParams_t
struct nbl_glsl_LoDInfo
{
	nbl_glsl_lod_library_LoDInfoBase base;
	nbl_glsl_lod_library_LoDChoiceParams_t choiceParams;
	nbl_glsl_lod_library_DrawcallInfo firstDrawcallInfo;
	// more drawcallInfos are stored past the end
};
#endif
*/

struct nbl_glsl_lod_library_LoDTable
{
	vec3 aabbMin;
	uint levelCount;
	vec3 aabbMax;
	uint firstLevelInfoUvec4Offset;
	// more leveInfoUvec4Offsets are stored past the end
};


#endif