#ifndef _NBL_GLSL_LOD_LIBRARY_STRUCTS_GLSL_INCLUDED_
#define _NBL_GLSL_LOD_LIBRARY_STRUCTS_GLSL_INCLUDED_


struct nbl_glsl_lod_library_DrawcallInfo
{
	uvec2 aabbMinRGB18E7S3;
	uvec2 aabbMaxRGB18E7S3;
	uint drawcallDWORDOffset;
	uint skinningAABBCountAndOffset; // TODO: review
};
#define NBL_GLSL_LOD_LIBRARY_DRAWCALL_INFO_SIZE 24
#define NBL_GLSL_LOD_LIBRARY_DRAWCALL_INFO_UVEC2_SIZE (NBL_GLSL_LOD_LIBRARY_DRAWCALL_INFO_SIZE>>3)

struct nbl_glsl_lod_library_LoDInfoBase
{
	uint drawcallInfoCountAndTotalBoneCount;
};
#define NBL_GLSL_LOD_LIBRARY_LOD_INFO_BASE_SIZE 4

struct nbl_glsl_lod_library_DefaultLoDChoiceParams
{
	float distanceSqAtReferenceFoV;
};
#define NBL_GLSL_LOD_LIBRARY_DEFAULT_LOD_CHOICE_PARAMS_SIZE 4
#define NBL_GLSL_CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_UVEC2_OFFSET (((NBL_GLSL_LOD_LIBRARY_LOD_INFO_BASE_SIZE+NBL_GLSL_LOD_LIBRARY_DEFAULT_LOD_CHOICE_PARAMS_SIZE-1)>>3)+1)

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
	uint firstLevelInfoUvec2Offset;
	// more leveInfoUvec2Offsets are stored past the end
};


#endif