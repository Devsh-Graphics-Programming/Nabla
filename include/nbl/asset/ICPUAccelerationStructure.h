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

class ICPUBottomLevelAccelerationStructure final : public IBottomLevelAccelerationStructure<ICPUAccelerationStructure>//TODO: sort this out later, public IPreHashed
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
			return cp;
		}

	protected:
		virtual ~ICPUBottomLevelAccelerationStructure() = default;

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

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cp = core::make_smart_refctd_ptr<ICPUTopLevelAccelerationStructure>();

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
		virtual ~ICPUTopLevelAccelerationStructure() = default;

	private:
		core::smart_refctd_dynamic_array<PolymorphicInstance> m_instances = nullptr;
		core::bitflag<BUILD_FLAGS> m_buildFlags = BUILD_FLAGS::PREFER_FAST_BUILD_BIT;
};

}

#endif