#ifndef _INTERSECTION_RECORD_H_
#define _INTERSECTION_RECORD_H_

struct intersection_record_t
{
	// Todo(achal): This is currently 7 bits per dim.
	// 1. For a 4x4x4 clipmap, 2 bits per dim would suffice
	// 2. For a 64^3 octree, 6 bits per dim would suffice
	uvec3 localClusterID;

	// Todo(achal): This is currently 4 bits per dim, because currently I have LOD_COUNT = 10
	// which is overkill for an octree and most likely for clipmap as well
	// Todo(achal): Do we really need this anymore?
	uint level;

	uint localLightIndex; // currently 12 bits, Todo(achal): Should be 22 bits because there could be a case where all lights intersect with a single cluster
	uint globalLightIndex; // currently 20 bits, Todo(achal): Make this 22
};

uvec2 packIntersectionRecord(in intersection_record_t record)
{
	uvec2 result = uvec2(0u);
	result.x |= (record.localClusterID.x & 0x7Fu);
	result.x |= ((record.localClusterID.y & 0x7Fu) << 7);
	result.x |= ((record.localClusterID.z & 0x7Fu) << 14);
	result.x |= ((record.level & 0xFu) << 21);

	result.y |= (record.localLightIndex & 0xFFFu);
	result.y |= (record.globalLightIndex << 12); // Todo(achal): Do we want a limit on how many lights it is possible to have in a cluster, if so, what?

	return result;
}

uint getGlobalLightIndex(in uvec2 packedIntersectionRecord)
{
	return (packedIntersectionRecord.y >> 12) & 0xFFFFF;
}

uvec3 getLocalClusterID(in uvec2 packedIntersectionRecord)
{
	uvec3 result;
	result.x = packedIntersectionRecord.x & 0x7Fu;
	result.y = (packedIntersectionRecord.x >> 7) & 0x7Fu;
	result.z = (packedIntersectionRecord.x >> 14) & 0x7Fu;
	return result;
}

uint getLevel(in uvec2 packedIntersectionRecord)
{
	return (packedIntersectionRecord.x >> 21) & 0x7F;
}

uint getLocalLightIndex(in uvec2 packedIntersectionRecord)
{
	return packedIntersectionRecord.y & 0xFFF;
}

#endif