// Todo(achal): Make this 512, made this only for test building the histogram
#define _NBL_GLSL_WORKGROUP_SIZE_ 256

#define LIGHT_CONTRIBUTION_THRESHOLD 2.f
// Todo(achal): This needs to be calculated by light intensity
#define LIGHT_RADIUS 25.f
#define CLIPMAP_EXTENT 11977.0674f

#define INVOCATIONS_PER_LIGHT 8
#define LIGHTS_PER_WORKGROUP (_NBL_GLSL_WORKGROUP_SIZE_/INVOCATIONS_PER_LIGHT)

layout(local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;

#include <nbl/builtin/glsl/shapes/aabb.glsl>

struct nbl_glsl_ext_ClusteredLighting_SpotLight
{
	vec3 position;
	float outerCosineOverCosineRange;
	uvec2 intensity;
	uvec2 direction;
};

struct intersection_record_t
{
	// Todo(achal): This is currently 7 bits per dim.
	// 1. For a 4x4x4 clipmap, 2 bits per dim would suffice
	// 2. For a 64^3 octree, 6 bits per dim would suffice
	uvec3 localClusterID;

	// Todo(achal): This is currently 4 bits per dim, because currently I have LOD_COUNT = 10
	// which is overkill for an octree and most likely for clipmap as well
	uint level;

	uint localLightIndex; // currently 12 bits, Todo(achal): Should be 22 bits because there could be a case where all lights intersect with a single cluster
	uint globalLightIndex; // currently 20 bits, Todo(achal): Make this 22
};

struct cone_t
{
	vec3 tip;
	float height;
	vec3 direction;
	float cosHalfAngle;
	float baseRadius;
};

bool isPointBehindPlane(in vec3 point, in vec4 plane)
{
	// As an optimization we can add an epsilon to 0, to ignore cones which have a
	// very very small intersecting region with the AABB, could help with FP precision
	// too when the point is on the plane
	return (dot(point, plane.xyz) + plane.w) <= 0.f /* + EPSILON*/;
}

cone_t getLightVolume(in nbl_glsl_ext_ClusteredLighting_SpotLight light)
{
	cone_t cone;

	// tip
	cone.tip = light.position;

	// height
	const float radiusSq = LIGHT_RADIUS * LIGHT_RADIUS;

	const vec3 intensity = nbl_glsl_decodeRGB19E7(light.intensity);
	const float maxIntensityComponent = max(max(intensity.r, intensity.g), intensity.b);
	const float determinant = clamp(1.f - ((2.f * LIGHT_CONTRIBUTION_THRESHOLD) / (maxIntensityComponent * radiusSq)), -1.f, 1.f);

	cone.height = LIGHT_RADIUS * inversesqrt(1.f / (determinant * determinant) - 1.f);

	// direction
	const vec2 dirXY = unpackSnorm2x16(light.direction[0]);
	const vec2 dirZW = unpackSnorm2x16(light.direction[1]);
	cone.direction = vec3(dirXY.xy, dirZW.x);

	// cosHalfAngle
	// Todo(achal): I cannot handle spotlights/cone intersection against AABB
	// if it has outerHalfAngle > 90.f, hence the `max`
	const float cosineRange = dirZW.y;
	cone.cosHalfAngle = max(light.outerCosineOverCosineRange * cosineRange, 1e-3f);

	// baseRadius
	const float tanOuterHalfAngle = sqrt(max(1.f - (cone.cosHalfAngle * cone.cosHalfAngle), 0.f)) / cone.cosHalfAngle;
	cone.baseRadius = cone.height * tanOuterHalfAngle;

	return cone;
}

// Todo(achal): Rename to getClusterAABB
nbl_glsl_shapes_AABB_t getCluster(in uvec3 localClusterID, in vec3 levelMinVertex, in float voxelSideLength)
{
	const vec3 camPos = pc.camPosClipmapExtent.xyz;

	nbl_glsl_shapes_AABB_t cluster;
	cluster.minVx = levelMinVertex + (localClusterID * voxelSideLength) + camPos;
	cluster.maxVx = levelMinVertex + (localClusterID + uvec3(1u)) * voxelSideLength + camPos;
	return cluster;
}

uvec2 packIntersectionRecord(in intersection_record_t record)
{
	uvec2 result = uvec2(0u);
	result.x |= record.localClusterID.x;
	result.x |= (record.localClusterID.y << 7);
	result.x |= (record.localClusterID.z << 14);
	result.x |= (record.level << 21);

	result.y |= record.localLightIndex;
	result.y |= (record.globalLightIndex << 12); // Todo(achal): Make this 10

	return result;
}

#define PLANE_COUNT 6
vec4[PLANE_COUNT] getAABBPlanes(in nbl_glsl_shapes_AABB_t aabb)
{
	vec4 planes[PLANE_COUNT];

	// 157
	vec3 p0 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z);
	vec3 p1 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z);
	vec3 p2 = vec3(aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
	planes[0].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[0].w = -dot(planes[0].xyz, p0);

	// 013
	p0 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z);
	p1 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z);
	p2 = vec3(aabb.maxVx.x, aabb.maxVx.y, aabb.minVx.z);
	planes[1].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[1].w = -dot(planes[1].xyz, p0);

	// 402
	p0 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z);
	p2 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.minVx.z);
	planes[2].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[2].w = -dot(planes[2].xyz, p0);

	// 546
	p0 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z);
	p2 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.maxVx.z);
	planes[3].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[3].w = -dot(planes[3].xyz, p0);

	// 451
	p0 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z);
	p2 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z);
	planes[4].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[4].w = -dot(planes[4].xyz, p0);

	// 762
	p0 = vec3(aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.maxVx.z);
	p2 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.minVx.z);
	planes[5].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[5].w = -dot(planes[5].xyz, p0);

	return planes;
}

bool coneIntersectAABB(in cone_t cone, in nbl_glsl_shapes_AABB_t aabb)
{
	vec4 planes[PLANE_COUNT] = getAABBPlanes(aabb);

	for (uint i = 0u; i < PLANE_COUNT; ++i)
	{
		const vec3 m = cross(cross(planes[i].xyz, cone.direction), cone.direction);
		const vec3 farthestBasePoint = cone.tip + (cone.direction * cone.height) - (m * cone.baseRadius); // farthest to plane's surface, away from positive half-space

		// There are two edge cases here:
		// 1. When cone's direction and plane's normal are anti-parallel
		//		There is no reason to check farthestBasePoint in this case, because cone's tip is the farthest point!
		//		But there is no harm in doing so.
		// 2. When cone's direction and plane's normal are parallel
		//		This edge case will get handled nicely by the farthestBasePoint coming as center of the base of the cone itself
		if (isPointBehindPlane(cone.tip, planes[i]) && isPointBehindPlane(farthestBasePoint, planes[i]))
			return false;
	}

	return true;
}