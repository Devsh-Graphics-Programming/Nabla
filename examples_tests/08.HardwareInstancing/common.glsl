#define kOptimalWorkgroupSize 256u

#ifdef __cplusplus

#define uint	uint32_t

#define mat4x3	core::matrix3x4SIMD
#define mat4	core::matrix4SIMD

#else
#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

#include <irr/builtin/glsl/utils/indirect_commands.glsl>

#include <irr/builtin/glsl/utils/culling.glsl>
#endif

#define kMaxLoDLevels 2
struct ModelData_t
{
#ifdef __cplusplus
	union
	{
		struct
		{
			float MinEdge[3];
			uint uselessPadding0;
		};
		core::vectorSIMDf MinEdge4;
	};
	union
	{
		struct
		{
			float MaxEdge[3];
			uint uselessPadding1;
		};
		core::vectorSIMDf MaxEdge4;
	};
#endif
	float	LoDDistancesSq[kMaxLoDLevels];
	uint	LoDDMeshUUID[kMaxLoDLevels];
};

struct SceneNode_t
{
	mat4x3	worldTransform;
#ifdef __cplusplus
	core::matrix3x4SIMD worldNormalMatrix;
#else
	vec3	worldNormalMatrixRow0;
	vec3	worldNormalMatrixRow1;
	vec3	worldNormalMatrixRow2;
#endif
};