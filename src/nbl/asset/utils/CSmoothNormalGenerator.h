// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_
#define _NBL_ASSET_C_SMOOTH_NORMAL_GENERATOR_H_INCLUDED_

#include "nbl/asset/utils/CVertexHashGrid.h"
#include "nbl/builtin/hlsl/shapes/triangle.hlsl"

#include <array>
#include <concepts>
#include <limits>
#include <span>


namespace nbl::asset 
{

template<typename PositionT>
concept SmoothNormalPosition = std::same_as<PositionT, hlsl::float32_t3> || std::same_as<PositionT, hlsl::float64_t3>;

class CSmoothNormalGenerator final
{
	public:
		CSmoothNormalGenerator() = delete;
		~CSmoothNormalGenerator() = delete;

		enum class EAccumulationMode : uint8_t
		{
			AreaWeighted,
			AngleWeighted
		};

		template<SmoothNormalPosition PositionT = hlsl::float32_t3>
		struct SAccumulatedCorner
		{
			uint32_t vertexIx = 0u;
			uint32_t accumulationGroup = 0u;
			PositionT position = PositionT(0.f, 0.f, 0.f);
		};

		template<SmoothNormalPosition PositionT = hlsl::float32_t3>
		class CAccumulatedNormals final
		{
			public:
				using vector_t = PositionT;

				explicit CAccumulatedNormals(const EAccumulationMode mode = EAccumulationMode::AreaWeighted) : m_mode(mode) {}

				NBL_FORCE_INLINE void reserveVertices(const size_t count)
				{
					if (count > m_vertexCount)
						m_vertexCount = count;
					if (count > m_groupsByVertex.capacity() && !m_groupsByVertex.empty())
						m_groupsByVertex.reserve(growSize(count));
				}

				NBL_FORCE_INLINE void reserveGroups(const size_t count)
				{
					if (count > m_accumulatedNormals.capacity())
						m_accumulatedNormals.reserve(growSize(count));
				}

				NBL_FORCE_INLINE void prepareIdentityGroups(const size_t count)
				{
					if (!m_groupsByVertex.empty())
						return;
					ensureGroupStorage(count);
				}

				NBL_FORCE_INLINE bool addTriangle(const std::array<SAccumulatedCorner<PositionT>, 3>& corners)
				{
					if (canUseIdentityFastPath(corners))
						return addTriangle(corners[0].vertexIx, corners[0].position, corners[1].vertexIx, corners[1].position, corners[2].vertexIx, corners[2].position);
					for (const auto& corner : corners)
					{
						if (!registerCorner(corner))
							return false;
					}
					return accumulateTriangle(corners, [](const SAccumulatedCorner<PositionT>& corner) { return corner.accumulationGroup; });
				}

				NBL_FORCE_INLINE bool addTriangle(const uint32_t i0, const PositionT& p0, const uint32_t i1, const PositionT& p1, const uint32_t i2, const PositionT& p2)
				{
					const size_t maxIx = std::max(static_cast<size_t>(i0), std::max(static_cast<size_t>(i1), static_cast<size_t>(i2)));
					const size_t requiredCount = maxIx + 1ull;
					if (requiredCount > m_vertexCount)
						m_vertexCount = requiredCount;
					ensureGroupStorage(requiredCount);
					if (m_groupsByVertex.empty())
						return accumulateTriangle(p0, p1, p2, i0, i1, i2);
					return addTriangle({{
						{.vertexIx = i0, .accumulationGroup = i0, .position = p0},
						{.vertexIx = i1, .accumulationGroup = i1, .position = p1},
						{.vertexIx = i2, .accumulationGroup = i2, .position = p2}
					}});
				}

				NBL_FORCE_INLINE bool addPreparedIdentityTriangle(const uint32_t i0, const PositionT& p0, const uint32_t i1, const PositionT& p1, const uint32_t i2, const PositionT& p2)
				{
					if (!m_groupsByVertex.empty())
						return false;
					const size_t requiredCount = std::max(static_cast<size_t>(i0), std::max(static_cast<size_t>(i1), static_cast<size_t>(i2))) + 1ull;
					if (requiredCount > m_vertexCount)
						m_vertexCount = requiredCount;
					if (requiredCount > m_accumulatedNormals.size())
						return false;
					return accumulateTriangle(p0, p1, p2, i0, i1, i2);
				}

				template<typename NormalT = hlsl::float32_t3>
				NBL_FORCE_INLINE bool finalize(const std::span<NormalT> normals, const std::span<const uint8_t> normalNeedsGeneration = {}, const NormalT& fallback = NormalT(0.f, 0.f, 1.f)) const
				{
					if (!normalNeedsGeneration.empty() && normalNeedsGeneration.size() != normals.size())
						return false;
					if (normals.size() < m_vertexCount)
						return false;

					if (m_groupsByVertex.empty())
					{
						for (size_t vertexIx = 0ull; vertexIx < m_vertexCount; ++vertexIx)
						{
							if (!normalNeedsGeneration.empty() && normalNeedsGeneration[vertexIx] == 0u)
								continue;
							const auto normal = vertexIx < m_accumulatedNormals.size() ? m_accumulatedNormals[vertexIx] : vector_t(0.f, 0.f, 0.f);
							const auto lenSq = hlsl::dot(normal, normal);
							normals[vertexIx] = (lenSq > 1e-20f) ? (normal * hlsl::rsqrt(lenSq)) : fallback;
						}
						return true;
					}

					for (size_t vertexIx = 0ull; vertexIx < m_vertexCount; ++vertexIx)
					{
						if (!normalNeedsGeneration.empty() && normalNeedsGeneration[vertexIx] == 0u)
							continue;
						const uint32_t group = resolveGroup(static_cast<uint32_t>(vertexIx));
						if (group == InvalidGroup)
							return false;

						const auto normal = group < m_accumulatedNormals.size() ? m_accumulatedNormals[group] : vector_t(0.f, 0.f, 0.f);
						const auto lenSq = hlsl::dot(normal, normal);
						normals[vertexIx] = (lenSq > 1e-20f) ? (normal * hlsl::rsqrt(lenSq)) : fallback;
					}
					return true;
				}

			private:
				static inline constexpr uint32_t InvalidGroup = std::numeric_limits<uint32_t>::max();

				static NBL_FORCE_INLINE size_t growSize(const size_t required)
				{
					return required > 1ull ? std::bit_ceil(required) : 1ull;
				}

				template<typename GroupFn>
				NBL_FORCE_INLINE bool accumulateTriangle(const std::array<SAccumulatedCorner<PositionT>, 3>& corners, GroupFn&& groupFn)
				{
					return accumulateTriangle(
						corners[0].position, corners[1].position, corners[2].position,
						groupFn(corners[0]), groupFn(corners[1]), groupFn(corners[2])
					);
				}

				NBL_FORCE_INLINE void ensureGroupStorage(const size_t requiredCount)
				{
					if (requiredCount <= m_accumulatedNormals.size())
						return;
					const size_t grownCount = growSize(requiredCount);
					if (requiredCount > m_accumulatedNormals.capacity())
						m_accumulatedNormals.reserve(grownCount);
					m_accumulatedNormals.resize(grownCount, vector_t(0.f, 0.f, 0.f));
				}

				NBL_FORCE_INLINE bool accumulateTriangle(const PositionT& p0, const PositionT& p1, const PositionT& p2, const uint32_t g0, const uint32_t g1, const uint32_t g2)
				{
					const auto edge10 = p1 - p0;
					const auto edge20 = p2 - p0;
					const auto faceNormal = hlsl::cross(edge10, edge20);
					const auto faceLenSq = hlsl::dot(faceNormal, faceNormal);
					if (faceLenSq <= 1e-20f)
						return true;

					if (m_mode == EAccumulationMode::AreaWeighted)
					{
						m_accumulatedNormals[g0] += faceNormal;
						m_accumulatedNormals[g1] += faceNormal;
						m_accumulatedNormals[g2] += faceNormal;
						return true;
					}

					const auto weights = hlsl::shapes::util::anglesFromTriangleEdges(p2 - p1, p0 - p2, p1 - p0);
					const auto unitNormal = faceNormal * hlsl::rsqrt(faceLenSq);
					m_accumulatedNormals[g0] += unitNormal * weights.x;
					m_accumulatedNormals[g1] += unitNormal * weights.y;
					m_accumulatedNormals[g2] += unitNormal * weights.z;
					return true;
				}

				NBL_FORCE_INLINE bool canUseIdentityFastPath(const std::array<SAccumulatedCorner<PositionT>, 3>& corners) const
				{
					if (!m_groupsByVertex.empty())
						return false;
					for (const auto& corner : corners)
					{
						if (corner.vertexIx != corner.accumulationGroup)
							return false;
					}
					return true;
				}

				NBL_FORCE_INLINE uint32_t resolveGroup(const uint32_t vertexIx) const
				{
					if (vertexIx >= m_vertexCount)
						return InvalidGroup;
					if (m_groupsByVertex.empty())
						return vertexIx;
					if (vertexIx >= m_groupsByVertex.size())
						return vertexIx;
					const uint32_t mapped = m_groupsByVertex[vertexIx];
					return mapped == InvalidGroup ? vertexIx : mapped;
				}

				NBL_FORCE_INLINE bool registerCorner(const SAccumulatedCorner<PositionT>& corner)
				{
					if ((static_cast<size_t>(corner.vertexIx) + 1ull) > m_vertexCount)
						m_vertexCount = static_cast<size_t>(corner.vertexIx) + 1ull;
					ensureGroupStorage(static_cast<size_t>(corner.accumulationGroup) + 1ull);
					if (m_groupsByVertex.empty())
					{
						if (corner.vertexIx == corner.accumulationGroup)
							return true;
						m_groupsByVertex.reserve(growSize(m_vertexCount));
					}
					else if (corner.vertexIx >= m_groupsByVertex.size())
						m_groupsByVertex.reserve(growSize(m_vertexCount));
					if (corner.vertexIx >= m_groupsByVertex.size())
						m_groupsByVertex.resize(growSize(static_cast<size_t>(corner.vertexIx) + 1ull), InvalidGroup);
					auto& group = m_groupsByVertex[corner.vertexIx];
					if (group == InvalidGroup)
					{
						if (corner.vertexIx == corner.accumulationGroup)
							return true;
						group = corner.accumulationGroup;
						return true;
					}
					return group == corner.accumulationGroup;
				}

				EAccumulationMode m_mode;
				size_t m_vertexCount = 0ull;
				core::vector<uint32_t> m_groupsByVertex;
				core::vector<vector_t> m_accumulatedNormals;
		};

		struct VertexData
		{
			//offset of the vertex into index buffer
			uint32_t index;
			uint32_t hash;
			hlsl::float32_t3 weightedNormal;
			//position of the vertex in 3D space
			hlsl::float32_t3 position;

			hlsl::float32_t3 getPosition() const
			{
				return position;
			}

			void setHash(uint32_t newHash)
			{
				hash = newHash;
			}

			uint32_t getHash() const
			{
				return hash;
			};

		};

		using VxCmpFunction = std::function<bool(const VertexData&, const VertexData&, const ICPUPolygonGeometry*)>;

		using VertexHashMap = CVertexHashGrid<VertexData>;

		struct Result
		{
			VertexHashMap vertexHashGrid;
			core::smart_refctd_ptr<ICPUPolygonGeometry> geom;
		};
		static Result calculateNormals(const ICPUPolygonGeometry* polygon, float epsilon, VxCmpFunction function, const bool recomputeHash=true);

	private:
		static VertexHashMap setupData(const ICPUPolygonGeometry* polygon, float epsilon);
		static core::smart_refctd_ptr<ICPUPolygonGeometry> processConnectedVertices(const ICPUPolygonGeometry* polygon, VertexHashMap& vertices, float epsilon, VxCmpFunction vxcmp);
};

}
#endif
