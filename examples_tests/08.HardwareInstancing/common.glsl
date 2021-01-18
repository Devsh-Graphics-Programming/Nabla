#define _NBL_GLSL_WORKGROUP_SIZE_ 256u

#define _NBL_GLSL_MAX_INDIRECT_DRAW_COUNT_ _NBL_GLSL_WORKGROUP_SIZE_

#ifdef __cplusplus


#define uint	uint32_t
struct alignas(8) uvec2
{
	uint32_t x, y;
};
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

struct MeshBuffer_t
{
	uint primitiveCount;
	uint firstIndex;
	uint baseVertex;
	uint uselessPadding;
};

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
			uint	LoDLevelCount;
			vec3	worldNormalMatrixRow1;
			uint	LoDMeshesOffset;
			vec3	worldNormalMatrixRow2;
			uint	LoDDistancesSqOffset;
#ifdef __cplusplus
		};
	};
#endif
};

struct Camera_t
{
	mat4 viewProjMatrix;
	vec3 viewMatrixInverseRow0;
	float posX;
	vec3 viewMatrixInverseRow1;
	float posY;
	vec3 viewMatrixInverseRow2;
	float posZ;
	uint sourceMDIOffset;
	uint sourceMDIDWORDOffset;
	uint nextFrameMDIOffset;
	uint nextFrameMDIDWORDOffset;
};

struct VisibleMesh_t
{
	mat4	modelViewProjectionMatrix;
	vec3	normalMatrixCol0;
	uint	cameraUUID;
	vec3	normalMatrixCol1;
	uint	objectUUID;
	vec3	normalMatrixCol2;
	uint	meshBuffersOffset;
};

struct InstanceVisibleMeshRedirect_t
{
	uint drawUUID;
	uint instanceID;
	uint visibleMeshID;
};