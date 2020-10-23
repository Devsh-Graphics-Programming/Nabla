#define kOptimalWorkgroupSize 256u
#define _IRR_GLSL_WORKGROUP_ARITHMETIC_WORKGROUP_SIZE_ kOptimalWorkgroupSize

#ifdef __cplusplus

#define uint	uint32_t
struct alignas(16) vec3
{
	float x, y, z;
};

#define mat4x3	core::matrix3x4SIMD
#define mat4	core::matrix4SIMD

#else
#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

#include <irr/builtin/glsl/utils/indirect_commands.glsl>

#include <irr/builtin/glsl/utils/culling.glsl>
#endif

struct Mesh_t
{
	vec3	MinAABBEdge;
	uint	meshBuffersOffset;
	vec3	MaxAABBEdge;
	uint	meshBuffersCount;
};

struct SceneNode_t
{
	mat4x3	worldTransform;
#ifdef __cplusplus
	union
	{
		core::matrix3x4SIMD worldNormalMatrix;
		struct
		{
#endif
			vec3	worldNormalMatrixRow0;
			vec3	worldNormalMatrixRow1;
			vec3	worldNormalMatrixRow2;
#ifdef __cplusplus
		};
	};
#endif
};

#define kMaxLoDLevels 2
struct ModelData_t
{
	vec3	MinAABBEdge;
	float	uselessPadding0;
	vec3	MaxAABBEdge;
	float	uselessPadding1;
	float	LoDDistancesSq[kMaxLoDLevels];
	uint	LoDDMeshUUID[kMaxLoDLevels];
};

struct VisibleObject_t
{
	mat4	modelViewProjectionMatrix;
	vec3	normalMatrixCol0;
	uint	cameraUUID;
	vec3	normalMatrixCol1;
	uint	objectUUID;
	vec3	normalMatrixCol2;
	uint	meshUUID;
};