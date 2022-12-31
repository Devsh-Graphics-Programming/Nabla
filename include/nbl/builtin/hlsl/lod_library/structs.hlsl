
#ifndef _NBL_GLSL_LOD_LIBRARY_STRUCTS_GLSL_INCLUDED_
#define _NBL_GLSL_LOD_LIBRARY_STRUCTS_GLSL_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace lod_library
{


struct DrawcallInfo
{
	uint2 aabbMinRGB18E7S3;
	uint2 aabbMaxRGB18E7S3;
	uint drawcallDWORDOffset;
	uint skinningAABBCountAndOffset; // TODO: review
};

#define DRAWCALL_INFO_SIZE 24
#define DRAWCALL_INFO_UVEC2_SIZE (DRAWCALL_INFO_SIZE>>3)

struct LoDInfoBase
{
	uint drawcallInfoCountAndTotalBoneCount;
};
#define LOD_INFO_BASE_SIZE 4

struct DefaultLoDChoiceParams
{
	float distanceSqAtReferenceFoV;
};
#define DEFAULT_LOD_CHOICE_PARAMS_SIZE 4
#define CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_UVEC2_OFFSET (((LOD_INFO_BASE_SIZE + DEFAULT_LOD_CHOICE_PARAMS_SIZE - 1) >> 3) + 1)

/*
#ifdef LoDChoiceParams_t
struct LoDInfo
{
	LoDInfoBase base;
	LoDChoiceParams_t choiceParams;
	DrawcallInfo firstDrawcallInfo;
	// more drawcallInfos are stored past the end
};
#endif
*/

struct LoDTable
{
	float3 aabbMin;
	uint levelCount;
	float3 aabbMax;
	uint firstLevelInfoUvec2Offset;
	// more leveInfoUvec2Offsets are stored past the end
};


}
}
}

#endif