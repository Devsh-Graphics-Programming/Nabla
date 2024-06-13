// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IAccelerationStructure.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl::asset
{

class ICPUAccelerationStructure : public IAsset, public IAccelerationStructure
{
	public:
		// WARNING: This call is expensive, especially for TLASes!
		virtual bool usesMotion() const = 0;

	protected:
		using IAccelerationStructure::IAccelerationStructure;
};
NBL_ENUM_ADD_BITWISE_OPERATORS(IBottomLevelAccelerationStructure<ICPUAccelerationStructure>::BUILD_FLAGS);
NBL_ENUM_ADD_BITWISE_OPERATORS(ITopLevelAccelerationStructure<ICPUAccelerationStructure>::BUILD_FLAGS);

class ICPUBottomLevelAccelerationStructure final : public IBottomLevelAccelerationStructure<ICPUAccelerationStructure>
{
	public:
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
		inline uint32_t getGeometryCount() const
		{
			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
				return m_AABBGeoms->size();
			return m_triangleGeoms ? m_triangleGeoms->size():0u;
		}

		//
		inline core::SRange<Triangles<asset::ICPUBuffer>> getTriangleGeometries()
		{
			if (!isMutable() || !m_triangleGeoms)
				return {nullptr,nullptr};
			return {m_triangleGeoms->begin(),m_triangleGeoms->end()};
		}
		inline core::SRange<const Triangles<asset::ICPUBuffer>> getTriangleGeometries() const
		{
			if (!m_triangleGeoms)
				return {nullptr,nullptr};
			return {m_triangleGeoms->begin(),m_triangleGeoms->end()};
		}
		inline bool setGeometries(core::smart_refctd_dynamic_array<Triangles<ICPUBuffer>>&& geometries)
		{
			if (!isMutable())
				return false;
			m_buildFlags &= BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
			m_triangleGeoms = std::move(geometries);
			m_AABBGeoms = nullptr;
			return true;
		}

		//
		inline core::SRange<AABBs<asset::ICPUBuffer>> getAABBGeometries()
		{
			if (!isMutable() || !m_AABBGeoms)
				return {nullptr,nullptr};
			return {m_AABBGeoms->begin(),m_AABBGeoms->end()};
		}
		inline core::SRange<const AABBs<asset::ICPUBuffer>> getAABBGeometries() const
		{
			if (!m_AABBGeoms)
				return {nullptr,nullptr};
			return {m_AABBGeoms->begin(),m_AABBGeoms->end()};
		}
		inline bool setGeometries(core::smart_refctd_dynamic_array<AABBs<ICPUBuffer>>&& geometries)
		{
			if (!isMutable())
				return false;
			m_buildFlags |= BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT;
			m_triangleGeoms = nullptr;
			m_AABBGeoms = std::move(geometries);
			return true;
		}

		//
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

		//!
		inline size_t conservativeSizeEstimate() const override
		{
			return sizeof(ICPUBottomLevelAccelerationStructure)+(m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT) ? sizeof(AABBs<ICPUBuffer>):sizeof(Triangles<ICPUBuffer>))*getGeometryCount();
		}

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cp = core::make_smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>();
			clone_common(cp.get());

			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
			{
				if (m_AABBGeoms && !m_AABBGeoms->empty())
					cp->m_AABBGeoms = core::make_refctd_dynamic_array<decltype(m_AABBGeoms)>(*m_AABBGeoms);
			}
			else if (m_triangleGeoms && !m_triangleGeoms->empty())
				cp->m_triangleGeoms = core::make_refctd_dynamic_array<decltype(m_triangleGeoms)>(*m_triangleGeoms);

			cp->m_buildFlags = m_buildFlags;
			return cp;
		}

		
		//!
		bool compatible(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUBottomLevelAccelerationStructure*>(_other);
			if (other->m_buildFlags!=m_buildFlags)
				return false;
			if (other->getGeometryCount()!=getGeometryCount())
				return false;
			const uint32_t geometryCount = getGeometryCount();
			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
			{
				for (auto i=0u; i<geometryCount; i++)
				{
					const auto& src = other->m_AABBGeoms->operator[](i);
					const auto& dst = m_AABBGeoms->operator[](i);
					if (dst.stride!=src.stride || dst.geometryFlags!=src.geometryFlags)
						return false;
				}
			}
			else
			{
				for (auto i=0u; i<geometryCount; i++)
				{
					const auto& src = other->m_triangleGeoms->operator[](i);
					const auto& dst = m_triangleGeoms->operator[](i);
					if (dst.maxVertex!=src.maxVertex || dst.vertexStride!=src.vertexStride || dst.vertexFormat!=src.vertexFormat || dst.indexType!=src.indexType || dst.geometryFlags!=src.geometryFlags)
						return false;
					if (dst.vertexData[1].isValid())
						return false;
					if (dst.indexType!=EIT_UNKNOWN && dst.indexData.isValid())
						return false;
				}
			}
			return true;
		}

	protected:
		virtual ~ICPUBottomLevelAccelerationStructure() = default;

		virtual uint32_t getDependencyCount() const override 
		{
			return m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT) ?  m_AABBGeoms->size() : m_triangleGeoms->size() * 3; 
		}

		virtual core::smart_refctd_ptr<IAsset> getDependency(uint32_t index) const override {
			if (index < getDependencyCount())
			{
				SBufferBinding<const ICPUBuffer> item;
				if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT)) {
					if (index < m_AABBGeoms->size()) {
						item = (*m_AABBGeoms)[index].data;
					}
				}
				else {
					uint32_t idiv3 = index / 3;
					uint32_t imod3 = index % 3;

					switch (imod3)
					{
					case 0:
					case 1:
						item =(*m_triangleGeoms)[idiv3].vertexData[imod3];
					case 2:
						item = (*m_triangleGeoms)[idiv3].indexData;
					default:
						return nullptr;
					}
				}
				if (item.isValid())
					return item.buffer;
			}
			return nullptr;
		}

	private:
		// more wasteful than a union but easier on the refcounting
		core::smart_refctd_dynamic_array<Triangles<ICPUBuffer>> m_triangleGeoms = nullptr;
		core::smart_refctd_dynamic_array<AABBs<ICPUBuffer>> m_AABBGeoms = nullptr;
		core::bitflag<BUILD_FLAGS> m_buildFlags = BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
};

class ICPUTopLevelAccelerationStructure final : public ITopLevelAccelerationStructure<ICPUAccelerationStructure>
{
		using blas_ref_t = core::smart_refctd_ptr<ICPUBottomLevelAccelerationStructure>;

	public:
		ICPUTopLevelAccelerationStructure() = default;
		
		//
		inline core::bitflag<BUILD_FLAGS> getBuildFlags() const {return m_buildFlags;}
		inline void setBuildFlags(const core::bitflag<BUILD_FLAGS> buildFlags)
		{ 
			if(!isMutable())
				return;
			m_buildFlags = buildFlags;
			// we always clear this flag as we always store instances as polymorphic for ICPUTopLevelAccelerationStructure
			m_buildFlags &= ~BUILD_FLAGS::INSTANCE_DATA_IS_POINTERS_TYPE_ENCODED_LSB;
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

		core::SRange<PolymorphicInstance> getInstances()
		{
			if (!isMutable() || !m_instances)
				return {nullptr,nullptr};
			return {m_instances->begin(),m_instances->end()};
		}
		core::SRange<const PolymorphicInstance> getInstances() const
		{
			if (!m_instances)
				return {nullptr,nullptr};
			return {m_instances->begin(),m_instances->end()};
		}
		bool setInstances(core::smart_refctd_dynamic_array<PolymorphicInstance>&& _instances)
		{
			if (!isMutable())
				return false;
			m_instances = std::move(_instances);
			return true;
		}

		//
		inline bool usesMotion() const override
		{
			for (const auto& instance : *m_instances)
			if (instance.getType()!=INSTANCE_TYPE::STATIC)
				return true;
			return false;
		}

		//!
		constexpr static inline auto AssetType = ET_BOTOM_LEVEL_ACCELERATION_STRUCTURE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }

		//!
		inline size_t conservativeSizeEstimate() const override
		{
			return sizeof(ICPUBottomLevelAccelerationStructure)+sizeof(PolymorphicInstance)*m_instances->size();
		}

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cp = core::make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();
			clone_common(cp.get());

			cp->m_instances = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<PolymorphicInstance>>(*m_instances);
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
		bool compatible(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUTopLevelAccelerationStructure*>(_other);
			if (other->m_buildFlags!=m_buildFlags)
				return false;
			if (!other->m_instances || other->m_instances->size()!=m_instances->size())
				return false;
			const auto instanceCount = m_instances->size();
			for (auto i=0u; i<instanceCount; i++)
			{
				const auto& dstInstance = m_instances->operator[](i);
				const auto& srcInstance = other->m_instances->operator[](i);
				if (dstInstance.getType()!=srcInstance.getType())
					return false;
			}
			return true;
		}

		virtual ~ICPUTopLevelAccelerationStructure() = default;

		virtual uint32_t getDependencyCount() const override { return m_instances->size(); }

		virtual core::smart_refctd_ptr<IAsset> getDependency(uint32_t index) const override {
			return index >= getDependencyCount() ? nullptr : (*m_instances)[index].getBase().blas;
		}

		void hash_impl(size_t& seed) const override {
			core::hash_combine(seed, m_buildFlags.value);
			const auto instanceCount = m_instances->size();
			for (auto i = 0u; i < instanceCount; i++)
			{
				const auto& instance = m_instances->operator[](i);
				core::hash_combine(seed, instance.getType());
			}
		}
	private:
		core::smart_refctd_dynamic_array<PolymorphicInstance> m_instances = nullptr;
		core::bitflag<BUILD_FLAGS> m_buildFlags = BUILD_FLAGS::PREFER_FAST_BUILD_BIT;
};

}

#endif