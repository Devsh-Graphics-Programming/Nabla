// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SHAPES_OBB_SILHOUETTE_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_OBB_SILHOUETTE_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>
#include <nbl/builtin/hlsl/shapes/obb.hlsl>

namespace nbl
{
namespace hlsl
{
namespace shapes
{

// Max vertices in an OBB silhouette after horizon clipping: a 6-vertex
// silhouette can gain at most one extra vertex from the partial clip endpoints.
NBL_CONSTEXPR uint32_t MaxOBBSilhouetteVertices = 7;

// ============================================================================
// 27-config silhouette table for an axis-aligned cube viewed from any of the
// 27 region cells (3 per axis). Each entry: {count, v0, v1, v2, v3, v4, v5}
// with vertices in CCW order relative to the viewer.
// ============================================================================
// Human-readable LUT kept for reference / debugging. The hot path uses the
// packed binSilhouettes form below.
static const uint32_t silhouettes[27][7] = {
	{6, 1, 3, 2, 6, 4, 5}, // 0: Black
	{6, 2, 6, 4, 5, 7, 3}, // 1: White
	{6, 0, 4, 5, 7, 3, 2}, // 2: Gray
	{6, 1, 3, 7, 6, 4, 5}, // 3: Red
	{4, 4, 5, 7, 6, 0, 0}, // 4: Green
	{6, 0, 4, 5, 7, 6, 2}, // 5: Blue
	{6, 0, 1, 3, 7, 6, 4}, // 6: Yellow
	{6, 0, 1, 5, 7, 6, 4}, // 7: Magenta
	{6, 0, 1, 5, 7, 6, 2}, // 8: Cyan
	{6, 1, 3, 2, 6, 7, 5}, // 9: Orange
	{4, 2, 6, 7, 3, 0, 0}, // 10: Light Orange
	{6, 0, 4, 6, 7, 3, 2}, // 11: Dark Orange
	{4, 1, 3, 7, 5, 0, 0}, // 12: Pink
	{4, 0, 4, 6, 7, 3, 2}, // 13: Light Pink
	{4, 0, 4, 6, 2, 0, 0}, // 14: Deep Rose
	{6, 0, 1, 3, 7, 5, 4}, // 15: Purple
	{4, 0, 1, 5, 4, 0, 0}, // 16: Light Purple
	{6, 0, 1, 5, 4, 6, 2}, // 17: Indigo
	{6, 0, 2, 6, 7, 5, 1}, // 18: Dark Green
	{6, 0, 2, 6, 7, 3, 1}, // 19: Lime
	{6, 0, 4, 6, 7, 3, 1}, // 20: Forest Green
	{6, 0, 2, 3, 7, 5, 1}, // 21: Navy
	{4, 0, 2, 3, 1, 0, 0}, // 22: Sky Blue
	{6, 0, 4, 6, 2, 3, 1}, // 23: Teal
	{6, 0, 2, 3, 7, 5, 4}, // 24: Brown
	{6, 0, 2, 3, 1, 5, 4}, // 25: Tan/Beige
	{6, 1, 5, 4, 6, 2, 3}, // 26: Dark Brown
};

// Binary packed silhouettes: bits 0-17 hold 6 corner indices (3 bits each),
// bits 29-31 hold the vertex count. Hot-path uses this; LUT above is reference.
static const uint32_t binSilhouettes[27] = {
	0b11000000000000101100110010011001,
	0b11000000000000011111101100110010,
	0b11000000000000010011111101100000,
	0b11000000000000101100110111011001,
	0b10000000000000000000110111101100,
	0b11000000000000010110111101100000,
	0b11000000000000100110111011001000,
	0b11000000000000100110111101001000,
	0b11000000000000010110111101001000,
	0b11000000000000101111110010011001,
	0b10000000000000000000011111110010,
	0b11000000000000010011111110100000,
	0b10000000000000000000101111011001,
	0b11000000000000010011111110100000,
	0b10000000000000000000010110100000,
	0b11000000000000100101111011001000,
	0b10000000000000000000100101001000,
	0b11000000000000010110100101001000,
	0b11000000000000001101111110010000,
	0b11000000000000001011111110010000,
	0b11000000000000001011111110100000,
	0b11000000000000001101111011010000,
	0b10000000000000000000001011010000,
	0b11000000000000001011010110100000,
	0b11000000000000100101111011010000,
	0b11000000000000100101001011010000,
	0b11000000000000011010110100101001,
};

struct BinSilhouette
{
	static BinSilhouette create(uint32_t configIndex)
	{
		BinSilhouette s;
		s.data = binSilhouettes[configIndex];
		return s;
	}

	uint32_t getVertexIndex(uint32_t index) NBL_CONST_MEMBER_FUNC
	{
		return (data >> (3u * index)) & 0x7u;
	}

	uint32_t getVertexCount() NBL_CONST_MEMBER_FUNC
	{
		return (data >> 29u) & 0x7u;
	}

	void rotr(uint32_t shift, uint32_t size)
	{
		data = nbl::hlsl::rotr(data, shift, size);
	}

	void rotl(uint32_t shift, uint32_t size)
	{
		data = nbl::hlsl::rotl(data, shift, size);
	}

	uint32_t data;
};

// Metadata-only descriptor of a clipped OBB silhouette (12 bytes). Vertex
// positions are NOT stored, consumers call materialize(view, verts) to
// fill a local array on demand, keeping vec3 storage out of struct-passing.
//
// silData: bits 0-17 rotated 3-bit corner indices (positive-z corners first
// in CCW order, then negative-z), bits 24-28 configIndex, bits 29-31 silhouette size.
// positiveCount: positive-z corners surviving the clip.
// count: emitted vertex count (positiveCount + 2 on partial clip, 0 if fully clipped).
struct ClippedSilhouette
{
	uint32_t   silData;       // rotated BinSilhouette data + size
	uint32_t   positiveCount; // # of positive-z OBB corners after rotation
	uint32_t   count;         // total emitted vertex count consumers cascade on
	float32_t3 shadingPoint;  // observer position; baked into clipping + materialize

	static ClippedSilhouette create(shapes::OBBView<float32_t> view, float32_t3 shadingPoint)
	{
		uint32_t3 region;
		uint32_t  configIndex, vertexCount;
		// OBB-local observer coord along axis i is dot(col_i, shadingPoint - minCorner);
		// compare against [0, |col_i|^2] for branchless 27-config classify.
		const float32_t3 toMin   = view.minCorner - shadingPoint;
		float32_t3 sqScales = float32_t3(dot(view.columns[0], view.columns[0]), dot(view.columns[1], view.columns[1]), dot(view.columns[2], view.columns[2]));
		float32_t3 proj     = -float32_t3(dot(view.columns[0], toMin), dot(view.columns[1], toMin), dot(view.columns[2], toMin));

		uint32_t3 below = uint32_t3(proj < float32_t3(0, 0, 0));
		uint32_t3 above = uint32_t3(proj > sqScales);
		region          = uint32_t3(uint32_t3(1u, 1u, 1u) + below - above);

		configIndex = region.x + region.y * 3u + region.z * 9u;

		BinSilhouette sil = BinSilhouette::create(configIndex);
		vertexCount       = sil.getVertexCount();

		// Always evaluate all 6 slots so the loop unrolls without a runtime
		// branch on vertexCount; high bits are masked off below.
		uint32_t validMask = (1u << vertexCount) - 1u;
		uint32_t clipMask  = 0u;
		NBL_UNROLL
		for (uint32_t i = 0; i < 6; i++)
			clipMask |= (hlsl::select(view.getVertexZ(sil.getVertexIndex(i)) < shadingPoint.z, 1u, 0u)) << i;
		clipMask &= validMask;

		uint32_t clipCount = countbits(clipMask);
		uint32_t invertedMask = ~clipMask & validMask;

		// clipMask is masked to validMask, so the shift can't pull garbage into bit 0.
		bool wrapAround = (clipMask & (clipMask >> (vertexCount - 1))) != 0u;

		uint32_t rotateAmount = nbl::hlsl::select(wrapAround, firstbitlow(invertedMask), // first positive
			firstbithigh(clipMask) + 1); // first vertex after last negative

		sil.rotr(rotateAmount * 3, vertexCount * 3);

		ClippedSilhouette self;
		// rotr wipes bits above width, so re-inject vertexCount and pack configIndex.
		self.silData       = sil.data | (configIndex << 24u) | (vertexCount << 29u);
		self.positiveCount = vertexCount - clipCount;
		const bool fullyClipped = (clipCount == vertexCount);
		const bool partialClip  = (clipCount > 0) && !fullyClipped;
		self.count              = nbl::hlsl::select(fullyClipped, 0u, self.positiveCount + (partialClip ? 2u : 0u));
		self.shadingPoint       = shadingPoint;

		return self;
	}

	uint32_t cornerIndex(uint32_t k) NBL_CONST_MEMBER_FUNC
	{
		return (silData >> (3u * k)) & 0x7u;
	}

	uint32_t  getVertexCount() NBL_CONST_MEMBER_FUNC { return (silData >> 29u) & 0x7u; }
	uint32_t  getConfigIndex() NBL_CONST_MEMBER_FUNC { return (silData >> 24u) & 0x1Fu; }
	uint32_t3 getRegion() NBL_CONST_MEMBER_FUNC
	{
		const uint32_t ci = getConfigIndex();
		return uint32_t3(ci % 3u, (ci / 3u) % 3u, ci / 9u);
	}
	BinSilhouette getOriginalBinSilhouette() NBL_CONST_MEMBER_FUNC { return BinSilhouette::create(getConfigIndex()); }

	// Fill `count` vertices into the caller's local array. Each vertex is
	// view.getVertex(cornerIndex(K)), columns[0/1/2] indexed by literal so
	// SROA keeps them in registers and the 3 conditional adds run in parallel.
	// Cascade on count rather than for+break so every vertices[K] write uses
	// a literal slot index, otherwise the array demotes to Function memory.
	// Vertices are returned in shading-point-relative coordinates (i.e.
	// view.getVertex(...) - shadingPoint), so direction-from-shading-point
	// reductions in consumers (cross/dot, gnomonic projection, horizon clip
	// to z=0) are correct.
	void materialize(shapes::OBBView<float32_t> view, out float32_t3 vertices[MaxOBBSilhouetteVertices]) NBL_CONST_MEMBER_FUNC
	{
		// Zero the unused tail; some consumers (DCE sinks) read
		// the full 7-wide array.
		NBL_UNROLL
		for (uint32_t init = 0; init < MaxOBBSilhouetteVertices; init++)
			vertices[init] = float32_t3(0.0f, 0.0f, 0.0f);
		if (count == 0)
			return;

		vertices[0] = view.getVertex(cornerIndex(0)) - shadingPoint;
		if (positiveCount > 1)
		{
			vertices[1] = view.getVertex(cornerIndex(1)) - shadingPoint;
			if (positiveCount > 2)
			{
				vertices[2] = view.getVertex(cornerIndex(2)) - shadingPoint;
				if (positiveCount > 3)
				{
					vertices[3] = view.getVertex(cornerIndex(3)) - shadingPoint;
					if (positiveCount > 4)
					{
						vertices[4] = view.getVertex(cornerIndex(4)) - shadingPoint;
						if (positiveCount > 5)
						{
							vertices[5] = view.getVertex(cornerIndex(5)) - shadingPoint;
							if (positiveCount > 6)
								vertices[6] = view.getVertex(cornerIndex(6)) - shadingPoint;
						}
					}
				}
			}
		}

		// Partial-clip: two extra getVertex calls for the negative-z endpoints
		// around the positive run, lerped to z=0 (in shading-point-relative
		// frame). Cascaded for literal slot indices.
		if (count > positiveCount)
		{
			const uint32_t   silSize   = (silData >> 29u) & 0x7u;
			const float32_t3 vFirstNeg = view.getVertex(cornerIndex(positiveCount)) - shadingPoint;
			const float32_t3 vLastNeg  = view.getVertex(cornerIndex(silSize - 1u)) - shadingPoint;
			const float32_t3 vFirstPos = vertices[0];

			if (positiveCount == 1)
			{
				const float32_t3 vLastPos = vertices[0];
				const float32_t  tA       = vLastPos.z / (vLastPos.z - vFirstNeg.z);
				vertices[1]               = lerp(vLastPos, vFirstNeg, tA);
				const float32_t tB        = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
				vertices[2]               = lerp(vLastNeg, vFirstPos, tB);
			}
			else if (positiveCount == 2)
			{
				const float32_t3 vLastPos = vertices[1];
				const float32_t  tA       = vLastPos.z / (vLastPos.z - vFirstNeg.z);
				vertices[2]               = lerp(vLastPos, vFirstNeg, tA);
				const float32_t tB        = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
				vertices[3]               = lerp(vLastNeg, vFirstPos, tB);
			}
			else if (positiveCount == 3)
			{
				const float32_t3 vLastPos = vertices[2];
				const float32_t  tA       = vLastPos.z / (vLastPos.z - vFirstNeg.z);
				vertices[3]               = lerp(vLastPos, vFirstNeg, tA);
				const float32_t tB        = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
				vertices[4]               = lerp(vLastNeg, vFirstPos, tB);
			}
			else if (positiveCount == 4)
			{
				const float32_t3 vLastPos = vertices[3];
				const float32_t  tA       = vLastPos.z / (vLastPos.z - vFirstNeg.z);
				vertices[4]               = lerp(vLastPos, vFirstNeg, tA);
				const float32_t tB        = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
				vertices[5]               = lerp(vLastNeg, vFirstPos, tB);
			}
			else // positiveCount == 5; positiveCount == 6 -> count == 8 > 7, impossible
			{
				const float32_t3 vLastPos = vertices[4];
				const float32_t  tA       = vLastPos.z / (vLastPos.z - vFirstNeg.z);
				vertices[5]               = lerp(vLastPos, vFirstNeg, tA);
				const float32_t tB        = vLastNeg.z / (vLastNeg.z - vFirstPos.z);
				vertices[6]               = lerp(vLastNeg, vFirstPos, tB);
			}
		}
	}

	// materialize + per-vertex normalize. Cascaded for literal slot indices.
	void materializeNormalized(shapes::OBBView<float32_t> view, out float32_t3 vertices[MaxOBBSilhouetteVertices]) NBL_CONST_MEMBER_FUNC
	{
		materialize(view, vertices);
		vertices[0] = nbl::hlsl::normalize(vertices[0]);
		if (count > 1)
		{
			vertices[1] = nbl::hlsl::normalize(vertices[1]);
			if (count > 2)
			{
				vertices[2] = nbl::hlsl::normalize(vertices[2]);
				if (count > 3)
				{
					vertices[3] = nbl::hlsl::normalize(vertices[3]);
					if (count > 4)
					{
						vertices[4] = nbl::hlsl::normalize(vertices[4]);
						if (count > 5)
						{
							vertices[5] = nbl::hlsl::normalize(vertices[5]);
							if (count > 6)
								vertices[6] = nbl::hlsl::normalize(vertices[6]);
						}
					}
				}
			}
		}
	}
};

struct SilEdgeNormals
{
	// Sentinel for unused edge slots: dot(dir, (0,0,-1)) = -dir.z. Callers
	// gate isInside on dir.z > 0, so this dot is always negative for them,
	// its asuint has the sign bit set, which makes the bitwise-AND
	// reduction in isInside() pass through the real sign bits unchanged.
	static SilEdgeNormals initSentinel()
	{
		SilEdgeNormals result;
		NBL_UNROLL
		for (uint32_t i = 0; i < MaxOBBSilhouetteVertices; i++)
			result.edgeNormals[i] = float32_t3(0.0f, 0.0f, -1.0f);
		return result;
	}

	// Build per-edge cross products from a materialized vertex array.
	static SilEdgeNormals create(float32_t3 vertices[MaxOBBSilhouetteVertices], uint32_t count)
	{
		SilEdgeNormals result = initSentinel();

		float32_t3 v0 = vertices[0];
		float32_t3 v1 = vertices[1];
		float32_t3 v2 = vertices[2];

		result.edgeNormals[0] = cross(v0, v1);
		result.edgeNormals[1] = cross(v1, v2);

		if (count > 3)
		{
			float32_t3 v3         = vertices[3];
			result.edgeNormals[2] = cross(v2, v3);

			if (count > 4)
			{
				float32_t3 v4         = vertices[4];
				result.edgeNormals[3] = cross(v3, v4);

				if (count > 5)
				{
					float32_t3 v5         = vertices[5];
					result.edgeNormals[4] = cross(v4, v5);

					if (count > 6)
					{
						float32_t3 v6         = vertices[6];
						result.edgeNormals[5] = cross(v5, v6);
						result.edgeNormals[6] = cross(v6, v0);
					}
					else
					{
						result.edgeNormals[5] = cross(v5, v0);
					}
				}
				else
				{
					result.edgeNormals[4] = cross(v4, v0);
				}
			}
			else
			{
				result.edgeNormals[3] = cross(v3, v0);
			}
		}
		else
		{
			result.edgeNormals[2] = cross(v2, v0);
		}

		return result;
	}

	// Sign-bit AND reduction: dot <= 0 iff asuint(dot) sign bit set (modulo +0.0
	// exact-boundary samples, which never hit in practice). 6 ANDs on the INT
	// pipe instead of 6 fmaxes on the FP pipe; lets the FP pipe stay busy with
	// the 7 dot products on Ampere's split FP/INT scheduler.
	bool isInside(float32_t3 dir)
	{
		const float32_t d0 = hlsl::dot(dir, edgeNormals[0]);
		const float32_t d1 = hlsl::dot(dir, edgeNormals[1]);
		const float32_t d2 = hlsl::dot(dir, edgeNormals[2]);
		const float32_t d3 = hlsl::dot(dir, edgeNormals[3]);
		const float32_t d4 = hlsl::dot(dir, edgeNormals[4]);
		const float32_t d5 = hlsl::dot(dir, edgeNormals[5]);
		const float32_t d6 = hlsl::dot(dir, edgeNormals[6]);
		const uint32_t allNeg = asuint(d0) & asuint(d1) & asuint(d2) & asuint(d3) & asuint(d4) & asuint(d5) & asuint(d6);
		return (allNeg & 0x80000000u) != 0u;
	}

	// Transform edge normals from world-space to the pyramid's local frame in-place.
	// After this, edgeNormals[i] = (dot(n, axis1), dot(n, axis2), dot(n, axis3))
	// and isInsideLocal() can do 2-FMA half-plane tests without extra storage.
	// NOTE: destroys world-space normals, isInside() will no longer work correctly.
	void transformToLocal(float32_t3 axis1, float32_t3 axis2, float32_t3 axis3)
	{
		NBL_UNROLL
		for (uint32_t i = 0; i < MaxOBBSilhouetteVertices; i++)
		{
			float32_t3 n   = edgeNormals[i];
			edgeNormals[i] = float32_t3(dot(n, axis1), dot(n, axis2), dot(n, axis3));
		}
	}

	// 2D gnomonic containment test after transformToLocal().
	//   dot(dir_unnorm, n_local) = localX * n.x + localY * n.y + n.z
	bool isInsideLocal(float32_t localX, float32_t localY)
	{
		float32_t maxDot = localX * edgeNormals[0].x + localY * edgeNormals[0].y + edgeNormals[0].z;
		maxDot           = hlsl::max(maxDot, localX * edgeNormals[1].x + localY * edgeNormals[1].y + edgeNormals[1].z);
		maxDot           = hlsl::max(maxDot, localX * edgeNormals[2].x + localY * edgeNormals[2].y + edgeNormals[2].z);
		maxDot           = hlsl::max(maxDot, localX * edgeNormals[3].x + localY * edgeNormals[3].y + edgeNormals[3].z);
		maxDot           = hlsl::max(maxDot, localX * edgeNormals[4].x + localY * edgeNormals[4].y + edgeNormals[4].z);
		maxDot           = hlsl::max(maxDot, localX * edgeNormals[5].x + localY * edgeNormals[5].y + edgeNormals[5].z);
		maxDot           = hlsl::max(maxDot, localX * edgeNormals[6].x + localY * edgeNormals[6].y + edgeNormals[6].z);
		return maxDot <= 0.0f;
	}

	float32_t3 edgeNormals[MaxOBBSilhouetteVertices];
};

}
}
}

#endif
