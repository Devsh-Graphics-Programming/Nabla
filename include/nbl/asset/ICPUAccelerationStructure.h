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
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			convertToDummyObject_common(referenceLevelsBelowToConvert);

			if (referenceLevelsBelowToConvert--)
			{
				if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
				{
					for (auto& geometry : *m_AABBGeoms)
					if (geometry.data.buffer)
						const_cast<ICPUBuffer*>(geometry.data.buffer.get())->convertToDummyObject(referenceLevelsBelowToConvert);
				}
				else
				{
					for (auto& geometry : *m_triangleGeoms)
					{
						for (auto j=0u; j<2u; j++)
						if (geometry.vertexData[j].buffer)
							const_cast<ICPUBuffer*>(geometry.vertexData[j].buffer.get())->convertToDummyObject(referenceLevelsBelowToConvert);
						if (geometry.indexData.buffer)
							const_cast<ICPUBuffer*>(geometry.indexData.buffer.get())->convertToDummyObject(referenceLevelsBelowToConvert);
					}
				}
			}
		}
		
		//!
		inline bool canBeRestoredFrom(const IAsset* _other) const override
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
					if (!asset::canBeRestoredFrom(dst.data,src.data))
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
					if (!asset::canBeRestoredFrom(dst.vertexData[0],src.vertexData[0]))
						return false;
					if (dst.vertexData[1].isValid() && !asset::canBeRestoredFrom(dst.vertexData[1], src.vertexData[1]))
						return false;
					if (dst.indexType!=EIT_UNKNOWN && dst.indexData.isValid() && !asset::canBeRestoredFrom(dst.indexData,src.indexData))
						return false;
				}
			}
			return true;
		}

	protected:
		virtual ~ICPUBottomLevelAccelerationStructure() = default;

		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			_levelsBelow--;
			if (_levelsBelow==0u)
				return;

			auto condRestoreBufferInBinding = [_levelsBelow](SBufferBinding<const ICPUBuffer>& dst, const SBufferBinding<const ICPUBuffer>& src)
			{
				if (dst.isValid())
					return;
				return restoreFromDummy_impl_call(const_cast<ICPUBuffer*>(dst.buffer.get()),const_cast<ICPUBuffer*>(src.buffer.get()),_levelsBelow);
			};

			const uint32_t geometryCount = getGeometryCount();
			auto* other = static_cast<ICPUBottomLevelAccelerationStructure*>(_other);
			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
			{
				for (auto i=0u; i<geometryCount; i++)
					condRestoreBufferInBinding(m_AABBGeoms->operator[](i).data,other->m_AABBGeoms->operator[](i).data);
			}
			else
			{
				for (auto i=0u; i<geometryCount; i++)
				{
					auto& srcGeom = m_triangleGeoms->operator[](i);
					const auto& dstGeom = other->m_triangleGeoms->operator[](i);
					condRestoreBufferInBinding(srcGeom.vertexData[0],dstGeom.vertexData[0]);
					condRestoreBufferInBinding(srcGeom.vertexData[1],dstGeom.vertexData[1]);
					condRestoreBufferInBinding(srcGeom.indexData,dstGeom.indexData);
				}
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			_levelsBelow--;
			if (_levelsBelow==0u)
				return false;

			if (m_buildFlags.hasFlags(BUILD_FLAGS::GEOMETRY_TYPE_IS_AABB_BIT))
			{
				for (const auto& aabbGeom : *m_AABBGeoms)
				if (aabbGeom.data.buffer && aabbGeom.data.buffer->isAnyDependencyDummy(_levelsBelow))
					return true;
			}
			else
			{
				for (const auto& triGeom : *m_triangleGeoms)
				{
					if (triGeom.vertexData[0].buffer && triGeom.vertexData[0].buffer->isAnyDependencyDummy(_levelsBelow))
						return true;
					if (triGeom.vertexData[1].buffer && triGeom.vertexData[1].buffer->isAnyDependencyDummy(_levelsBelow))
						return true;
					if (triGeom.indexData.buffer && triGeom.indexData.buffer->isAnyDependencyDummy(_levelsBelow))
						return true;
				}
			}
			return false;
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

		//!
		inline void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			convertToDummyObject_common(referenceLevelsBelowToConvert);
			
			if (--referenceLevelsBelowToConvert)
			for (auto& instance : *m_instances)
			{
				auto* blas = instance.getBase().blas.get();
				if (blas)
					blas->convertToDummyObject(referenceLevelsBelowToConvert);
			}
		}
		
		//!
		inline bool canBeRestoredFrom(const IAsset* _other) const override
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
				auto* blas = dstInstance.getBase().blas.get();
				if (blas && !blas->canBeRestoredFrom(srcInstance.getBase().blas.get()))
					return false;
			}
			return true;
		}

	protected:
		virtual ~ICPUTopLevelAccelerationStructure() = default;

		inline void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			_levelsBelow--;
			if (_levelsBelow==0u)
				return;

			auto* other = static_cast<ICPUTopLevelAccelerationStructure*>(_other);
			const auto srcInstances = other->getInstances();
			const auto instanceCount = srcInstances.size();
			for (auto i=0u; i<instanceCount; i++)
			{
				auto* blas = m_instances->operator[](i).getBase().blas.get();
				if (blas)
					restoreFromDummy_impl_call(blas,srcInstances[i].getBase().blas.get(),_levelsBelow);
			}
		}

		inline bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			if (_levelsBelow--)
			for (const auto& instance : *m_instances)
			{
				const auto* blas = instance.getBase().blas.get();
				if (blas && blas->isAnyDependencyDummy())
					return true;
			}
			return false;
		}

	private:
		core::smart_refctd_dynamic_array<PolymorphicInstance> m_instances = nullptr;
		core::bitflag<BUILD_FLAGS> m_buildFlags = BUILD_FLAGS::PREFER_FAST_BUILD_BIT;
};

}

#endif