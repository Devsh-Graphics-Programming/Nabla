#ifndef CULL_COMMON_H
#define CULL_COMMON_H

#ifndef _NBL_GLSL_WORKGROUP_SIZE_
#error "_NBL_GLSL_WORKGROUP_SIZE_ must be defined"
#endif

#ifndef LIGHT_CONTRIBUTION_THRESHOLD
#error "LIGHT_CONTRIBUTION_THRESHOLD must be defined"
#endif

#ifndef LIGHT_RADIUS
#error "LIGHT_RADIUS must be defined"
#endif

#define INVOCATIONS_PER_LIGHT 8
#define LIGHTS_PER_WORKGROUP (_NBL_GLSL_WORKGROUP_SIZE_/INVOCATIONS_PER_LIGHT)

layout(local_size_x = _NBL_GLSL_WORKGROUP_SIZE_) in;

#include <nbl/builtin/glsl/shapes/aabb.glsl>
#include <nbl/builtin/glsl/math/quaternions.glsl>

#include <../intersection_record.glsl>

struct nbl_glsl_ext_ClusteredLighting_SpotLight
{
	vec3 position;
	float outerCosineOverCosineRange;
	uvec2 intensity;
	uvec2 direction;
};

struct cone_t
{
	vec3 tip;
	float height;
	vec3 direction;
	float cosHalfAngle;
	float baseRadius;
};

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
	const vec3 camPos = pc.camPosGenesisVoxelExtent.xyz;

	nbl_glsl_shapes_AABB_t cluster;
	cluster.minVx = levelMinVertex + (localClusterID * voxelSideLength) + camPos;
	cluster.maxVx = levelMinVertex + (localClusterID + uvec3(1u)) * voxelSideLength + camPos;
	return cluster;
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
	planes[0].w = dot(planes[0].xyz, p0);

	// 013
	p0 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z);
	p1 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z);
	p2 = vec3(aabb.maxVx.x, aabb.maxVx.y, aabb.minVx.z);
	planes[1].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[1].w = dot(planes[1].xyz, p0);

	// 402
	p0 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z);
	p2 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.minVx.z);
	planes[2].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[2].w = dot(planes[2].xyz, p0);

	// 546
	p0 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z);
	p2 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.maxVx.z);
	planes[3].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[3].w = dot(planes[3].xyz, p0);

	// 451
	p0 = vec3(aabb.minVx.x, aabb.minVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.maxVx.z);
	p2 = vec3(aabb.maxVx.x, aabb.minVx.y, aabb.minVx.z);
	planes[4].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[4].w = dot(planes[4].xyz, p0);

	// 762
	p0 = vec3(aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
	p1 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.maxVx.z);
	p2 = vec3(aabb.minVx.x, aabb.maxVx.y, aabb.minVx.z);
	planes[5].xyz = normalize(cross(p1 - p0, p2 - p0));
	planes[5].w = dot(planes[5].xyz, p0);

	return planes;
}

bvec3 and(in bvec3 a, in bvec3 b)
{
	return bvec3(a.x && b.x, a.y && b.y, a.z && b.z);
}

bvec3 or(in bvec3 a, in bvec3 b)
{
	return bvec3(a.x || b.x, a.y || b.y, a.z || b.z);
}

float projectedSphericalVertex(in vec3 origin, in vec3 planeNormal, in vec3 pos)
{
	return dot(normalize(pos - origin), planeNormal);
}

vec3 findFarthestPointOnConeInDirection(in vec3 planeNormal, in cone_t cone)
{
	const vec3 m = cross(cross(planeNormal, cone.direction), cone.direction);
	const vec3 farthestBasePoint = cone.tip + (cone.direction * cone.height) - (m * cone.baseRadius); // farthest to plane's surface, away from positive half-space
	return farthestBasePoint;
}

bool cullCone(in cone_t cone, in nbl_glsl_shapes_AABB_t aabb)
{
	float maxCosine = projectedSphericalVertex(cone.tip, cone.direction, vec3(aabb.minVx.x, aabb.minVx.y, aabb.minVx.z));

	for (uint i = 1u; i < 8u; ++i)
	{
		const uvec3 t = (uvec3(i) >> uvec3(0, 1, 2)) & 0x1u;
		const vec3 vertex = mix(aabb.minVx, aabb.maxVx, t);

		// assuming cone.direction is normalized
		maxCosine = max(projectedSphericalVertex(cone.tip, cone.direction, vertex), maxCosine);
	}

	const bool allVerticesOutsideCone = maxCosine < cone.cosHalfAngle;

	if (cone.cosHalfAngle <= 0.f) // obtuse
	{
		return allVerticesOutsideCone; // cull if whole AABB is inside complementary acute cone
	}
	else if (any(or(greaterThan(aabb.minVx, cone.tip), greaterThan(cone.tip, aabb.maxVx))) && allVerticesOutsideCone)
	{
		// step 1
		for (uint i = 0u; i < 8u; ++i)
		{
			const uvec3 t = (uvec3(i) >> uvec3(0, 1, 2)) & 0x1u;
			const vec3 vertex = mix(aabb.minVx, aabb.maxVx, t);

			const vec3 waypoint = normalize(vertex - cone.tip);
			
			const vec3 normal = nbl_glsl_slerp_impl_impl(cone.direction, normalize(waypoint), sqrt(1.f - cone.cosHalfAngle * cone.cosHalfAngle));
			if (dot(nbl_glsl_shapes_AABB_getFarthestPointInFront(aabb, normal) - cone.tip, normal) < 0.f)
				return true;
		}

		vec4 planes[PLANE_COUNT] = getAABBPlanes(aabb);

		for (uint i = 0u; i < PLANE_COUNT; ++i)
		{
			const vec3 normal = planes[i].xyz;

			float farthestPoint = dot(normal, cone.tip);
			if (dot(normal, cone.direction) < cone.cosHalfAngle)
				farthestPoint = max(dot(normal, findFarthestPointOnConeInDirection(normal, cone)), farthestPoint); // https://www.3dgep.com/forward-plus/#Frustum-Cone_Culling 
			else
				farthestPoint += (cone.height / cone.cosHalfAngle);

			if (farthestPoint < planes[i].w)
				return true;
		}
	}
	return false;
}

// Todo(achal): Can make this cone_t something like light_volume_t..
bool lightIntersectAABB(in cone_t cone, in nbl_glsl_shapes_AABB_t aabb)
{
#if 1
	const vec3 sphereMinPoint = cone.tip - cone.height;
	const vec3 sphereMaxPoint = cone.tip + cone.height;

	if (!any(and(lessThan(aabb.minVx, sphereMaxPoint),lessThan(sphereMinPoint, aabb.maxVx))))
		return false;

	const vec3 closestPoint = clamp(cone.tip, aabb.minVx, aabb.maxVx);

	if (dot(closestPoint - cone.tip, closestPoint - cone.tip) > (cone.height * cone.height))
		return false;

	if (cone.cosHalfAngle <= -(1.f - 1e-3f))
		return false;
#endif

	return !cullCone(cone, aabb);
}

float getLightImportanceMagnitude(in nbl_glsl_ext_ClusteredLighting_SpotLight light)
{
	const vec3 intensity = nbl_glsl_decodeRGB19E7(light.intensity);

	const vec3 lightToCamera = pc.camPosGenesisVoxelExtent.xyz - light.position;
	const float lenSq = dot(lightToCamera, lightToCamera);
	const float radiusSq = LIGHT_RADIUS * LIGHT_RADIUS;
	const float attenuation = 0.5f * radiusSq * (1.f - inversesqrt(1.f + radiusSq / lenSq));
	const vec3 importance = intensity * attenuation;
	return sqrt(dot(importance, importance));
}
#endif