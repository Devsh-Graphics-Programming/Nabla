// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IAccelerationStructure.h"
#include "nbl/asset/ICPUBuffer.h"

#include "nbl/builtin/hlsl/acceleration_structures.hlsl"

namespace nbl::asset
{

class ICPUBottomLevelAccelerationStructure final : public IPreHashed, public IBottomLevelAccelerationStructure
{
	public:
		static inline bool validBuildFlags(const core::bitflag<BUILD_FLAGS> flags) {return validBuildFlags(flags);}

		//
		inline ICPUBottomLevelAccelerationStructure() = default;

		//
		inline core::bitflag<BUILD_FLAGS> getBuildFlags() const { return m_buildFlags; }
		// you will not be able to set the `GEOMETRY_TYPE_IS_AABB_BIT` flag this way
		inline void setBuildFlags(const core::bitflag<BUILD_FLAGS> buildFlags)
		{ 
			if(!isMutable())
				return;
			m_buildFlags &= BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
			constexpr auto everyBitButAABB = ~BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
			m_buildFlags |= buildFlags&everyBitButAABB;
		}

		//
		inline std::span<uint32_t> getGeometryPrimitiveCounts()
		{
			if (isMutable() && m_geometryPrimitiveCount)
				return {m_geometryPrimitiveCount->begin(),m_geometryPrimitiveCount->end()};
			return {};
		}
		inline std::span<const uint32_t> getGeometryPrimitiveCounts() const
		{
			if (m_geometryPrimitiveCount)
				return {m_geometryPrimitiveCount->begin(),m_geometryPrimitiveCount->end()};
			return {};
		}

		//
		inline uint32_t getGeometryCount() const
		{
			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
				return m_AABBGeoms ? m_AABBGeoms->size():0u;
			return m_triangleGeoms ? m_triangleGeoms->size():0u;
		}

		//
		inline std::span<Triangles<asset::ICPUBuffer>> getTriangleGeometries()
		{
			if (!isMutable() || !m_triangleGeoms)
				return {};
			return {m_triangleGeoms->begin(),m_triangleGeoms->end()};
		}
		inline std::span<const Triangles<asset::ICPUBuffer>> getTriangleGeometries() const
		{
			if (!m_triangleGeoms)
				return {};
			return {m_triangleGeoms->begin(),m_triangleGeoms->end()};
		}
		inline bool setGeometries(core::smart_refctd_dynamic_array<Triangles<ICPUBuffer>>&& geometries, core::smart_refctd_dynamic_array<uint32_t>&& ranges)
		{
			if (!isMutable())
				return false;
			m_buildFlags &= ~BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
			m_geometryPrimitiveCount = std::move(ranges);
			m_triangleGeoms = std::move(geometries);
			m_AABBGeoms = nullptr;
			return true;
		}

		//
		inline std::span<AABBs<asset::ICPUBuffer>> getAABBGeometries()
		{
			if (!isMutable() || !m_AABBGeoms)
				return {};
			return {m_AABBGeoms->data(),m_AABBGeoms->size()};
		}
		inline std::span<const AABBs<asset::ICPUBuffer>> getAABBGeometries() const
		{
			if (!m_AABBGeoms)
				return {};
			return {m_AABBGeoms->data(),m_AABBGeoms->size()};
		}
		inline bool setGeometries(core::smart_refctd_dynamic_array<AABBs<ICPUBuffer>>&& geometries, core::smart_refctd_dynamic_array<uint32_t>&& ranges)
		{
			if (!isMutable())
				return false;
			m_buildFlags |= BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
			m_geometryPrimitiveCount = std::move(ranges);
			m_triangleGeoms = nullptr;
			m_AABBGeoms = std::move(geometries);
			return true;
		}

		// WARNING: This call is expensive
		inline bool usesMotion() const override
		{
			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
				return false;
			for (const auto& triangles : getTriangleGeometries())
			if (triangles.vertexData[1].buffer)
				return true;
			return false;
		}

		//!
		constexpr static inline auto AssetType = ET_BOTOM_LEVEL_ACCELERATION_STRUCTURE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cp = core::make_smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>();

			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
			{
				if (m_AABBGeoms && !m_AABBGeoms->empty())
					cp->m_AABBGeoms = core::make_refctd_dynamic_array<decltype(m_AABBGeoms)>(*m_AABBGeoms);
			}
			else if (m_triangleGeoms && !m_triangleGeoms->empty())
				cp->m_triangleGeoms = core::make_refctd_dynamic_array<decltype(m_triangleGeoms)>(*m_triangleGeoms);

			cp->m_buildFlags = m_buildFlags;
			cp->m_geometryPrimitiveCount = core::make_refctd_dynamic_array<decltype(m_geometryPrimitiveCount)>(*m_geometryPrimitiveCount);
			return cp;
		}

		// Do not report anything as a dependant, we'll simply drop the data instead of discarding its contents
		inline core::unordered_set<const IAsset*> computeDependants() const override
		{
			return {};
		}

		inline core::unordered_set<IAsset*> computeDependants() override
		{
			return {};
		}

		inline core::blake3_hash_t computeContentHash() const override
		{
			if (!missingContent())
				return INVALID_HASH;
			const bool isAABB = m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT);
			core::blake3_hasher hasher;
			hasher << isAABB;
			auto countIt = m_geometryPrimitiveCount->begin();
			if (isAABB)
			{
				for (const auto& aabb : *m_AABBGeoms)
				{
					const void* data = aabb.data.isValid() ? aabb.data.buffer->getPointer():nullptr;
					if (!data)
						return INVALID_HASH;
					const auto count = *(countIt++);
					hasher << count;
					hasher << aabb.geometryFlags;
					hasher << aabb.stride;
					const auto begin = reinterpret_cast<const uint8_t*>(data)+aabb.data.offset;
					const auto end = begin+aabb.stride*count;
					for (auto it=begin; it<end; it+=aabb.stride)
						hasher.update(it,sizeof(AABB_t));
				}
			}
			else
			{
				for (const auto& triangles : *m_triangleGeoms)
				{
					//
					const size_t vertexSize = getTexelOrBlockBytesize(triangles.vertexFormat);
					if (vertexSize==0)
						return INVALID_HASH;
					//
					const uint8_t* verticesA = triangles.vertexData[0].isValid() ? reinterpret_cast<const uint8_t*>(triangles.vertexData[0].buffer->getPointer()):nullptr;
					if (!verticesA)
						return INVALID_HASH;
					verticesA += triangles.vertexData[0].offset;
					//
					const uint8_t* indices = nullptr;
					if (triangles.indexType!=E_INDEX_TYPE::EIT_UNKNOWN)
					{
						indices = triangles.indexData.isValid() ? reinterpret_cast<const uint8_t*>(triangles.indexData.buffer->getPointer()):nullptr;
						if (!indices)
							return INVALID_HASH;
						indices += triangles.indexData.offset;
					}
					const auto count = *(countIt++);
					const auto indexCount = count*3;
					hasher << count;
					hasher << triangles.transform;
					hasher << triangles.maxVertex;
					hasher << triangles.vertexStride;
					hasher << triangles.vertexFormat;
					hasher << triangles.indexType;
					hasher << triangles.geometryFlags;
					// now hash the triangle data
					const bool usesMotion = triangles.vertexData[1].isValid();
					const uint8_t* verticesB = nullptr;
					if (usesMotion)
						verticesB = reinterpret_cast<const uint8_t*>(triangles.vertexData[1].buffer->getPointer())+triangles.vertexData[1].offset;
					auto hashIndices = [&]<typename IndexType>() -> void
					{
						for (auto i=0; i<indexCount; i++)
						{
							uint32_t vertexIndex = i;
							if constexpr (std::is_integral_v<IndexType>)
								vertexIndex = reinterpret_cast<const IndexType*>(indices)[i];
							hasher.update(verticesA+vertexIndex*triangles.vertexStride,vertexSize);
							if (usesMotion)
								hasher.update(verticesB+vertexIndex*triangles.vertexStride,vertexSize);
						}
					};
					switch (triangles.indexType)
					{
						case E_INDEX_TYPE::EIT_16BIT:
							hashIndices.operator()<uint16_t>();
							break;
						case E_INDEX_TYPE::EIT_32BIT:
							hashIndices.operator()<uint32_t>();
							break;
						default:
							hashIndices.operator()<void>();
							break;
					}
				}
			}
			return static_cast<core::blake3_hash_t>(hasher);
		}

		inline bool missingContent() const override
		{
			return !m_geometryPrimitiveCount || !m_triangleGeoms && !m_AABBGeoms;
		}

	protected:
		virtual ~ICPUBottomLevelAccelerationStructure() = default;

		inline void discardContent_impl() override
		{
			m_triangleGeoms = nullptr;
			m_AABBGeoms = nullptr;
			m_geometryPrimitiveCount = nullptr;
		}

	private:
		// more wasteful than a union but easier on the refcounting
		core::smart_refctd_dynamic_array<Triangles<ICPUBuffer>> m_triangleGeoms = nullptr;
		core::smart_refctd_dynamic_array<AABBs<ICPUBuffer>> m_AABBGeoms = nullptr;
		core::smart_refctd_dynamic_array<uint32_t> m_geometryPrimitiveCount = nullptr;
		core::bitflag<BUILD_FLAGS> m_buildFlags = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
};

class ICPUTopLevelAccelerationStructure final : public IAsset, public ITopLevelAccelerationStructure
{
		using blas_ref_t = core::smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>;

	public:
		static inline bool validBuildFlags(const core::bitflag<BUILD_FLAGS> flags) {return validBuildFlags(flags);}

		//
		ICPUTopLevelAccelerationStructure() = default;

    inline core::unordered_set<const IAsset*> computeDependants() const override
		{
			core::unordered_set<const IAsset*> dependants;
			for (const auto& instance : *m_instances)
				dependants.insert(instance.getBase().blas.get());
			return dependants;
		}

		//
		inline auto& getBuildRangeInfo()
		{
			assert(isMutable());
			return m_buildRangeInfo;
		}
		inline auto& getBuildRangeInfo() const {return m_buildRangeInfo;}
		
		//
		inline core::bitflag<BUILD_FLAGS> getBuildFlags() const {return m_buildFlags;}
		inline void setBuildFlags(const core::bitflag<BUILD_FLAGS> buildFlags)
		{ 
			if(!isMutable())
				return;
			m_buildFlags = buildFlags;
		}

		//
		using Instance = Instance<blas_ref_t>;
		using StaticInstance = StaticInstance<blas_ref_t>;
		using MatrixMotionInstance = MatrixMotionInstance<blas_ref_t>;
		using SRTMotionInstance = SRTMotionInstance<blas_ref_t>;
		struct PolymorphicInstance final
		{
			inline INSTANCE_TYPE getType() const
			{
				return static_cast<INSTANCE_TYPE>(instance.index());
			}

			inline Instance& getBase()
			{
				return std::visit([](auto& typedInstance)->Instance&{return typedInstance.base;},instance);
			}
			inline const Instance& getBase() const {return const_cast<PolymorphicInstance*>(this)->getBase();}


			std::variant<StaticInstance,MatrixMotionInstance,SRTMotionInstance> instance = StaticInstance{};
		};

		std::span<PolymorphicInstance> getInstances()
		{
			if (!isMutable() || !m_instances)
				return {};
			return {m_instances->data(),m_instances->size()};
		}
		std::span<const PolymorphicInstance> getInstances() const
		{
			if (!m_instances)
				return {};
			return {m_instances->data(),m_instances->size()};
		}
		bool setInstances(core::smart_refctd_dynamic_array<PolymorphicInstance>&& _instances)
		{
			if (!isMutable())
				return false;
			m_instances = std::move(_instances);
			return true;
		}

		// WARNING: This call is expensive, much more than for BLASes!
		inline bool usesMotion() const override
		{
			for (const auto& instance : *m_instances)
			if (instance.getType()!=INSTANCE_TYPE::STATIC || instance.getBase().blas && instance.getBase().blas->usesMotion())
				return true;
			return false;
		}

		//!
		constexpr static inline auto AssetType = ET_TOP_LEVEL_ACCELERATION_STRUCTURE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cp = core::make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();

			cp->m_instances = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<PolymorphicInstance>>(*m_instances);
			cp->m_buildRangeInfo = m_buildRangeInfo;
			cp->m_buildFlags = m_buildFlags;

			if (_depth--)
			for (auto& instance : *cp->m_instances)
			{
				auto* blas = instance.getBase().blas.get();
				if (blas)
					instance.getBase().blas = core::move_and_static_cast<ICPUBottomLevelAccelerationStructure>(blas->clone(_depth));
			}

			return cp;
		}

	protected:
		virtual ~ICPUTopLevelAccelerationStructure() = default;

	private:
		core::smart_refctd_dynamic_array<PolymorphicInstance> m_instances = nullptr;
		hlsl::acceleration_structures::top_level::BuildRangeInfo m_buildRangeInfo;
		core::bitflag<BUILD_FLAGS> m_buildFlags = BUILD_FLAGS::PREFER_FAST_BUILD_BIT;
};

}

#endif