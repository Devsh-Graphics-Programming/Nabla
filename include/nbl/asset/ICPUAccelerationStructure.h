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

class ICPUAccelerationStructure : public IAsset
{
		using PseudoBase = IAccelerationStructure;

	public:
		// a few aliases to make usage simpler
		using CREATE_FLAGS = PseudoBase::CREATE_FLAGS;
		using BUILD_FLAGS = PseudoBase::BUILD_FLAGS;
		using GEOMETRY_FLAGS = PseudoBase::GEOMETRY_FLAGS;

		//
		inline void setAccelerationStructureSize(const uint64_t accelerationStructureSize)
		{ 
			if(!isMutable())
				return;
			m_accelerationStructureSize = accelerationStructureSize;
		}
		inline uint64_t getAccelerationStructureSize() const { return m_accelerationStructureSize; }

	private:
		core::smart_refctd_dynamic_array<BuildRangeInfo> m_buildRangeInfos;
		size_t m_accelerationStructureSize = 0ull;
		core::bitflag<PseudoBase::BUILD_FLAGS> m_buildFlags = PseudoBase::BUILD_FLAGS::PREFER_FAST_TRACE_BIT;
};

class ICPUBottomLevelAccelerationStructure final : public IAsset, public IBottomLevelAccelerationStructure
{
	public:
		//
		ICPUAccelerationStructure(SCreationParams&& _params)
			: params(std::move(_params))
			, m_hasBuildInfo(false)
			, m_accelerationStructureSize(0)
			, m_buildRangeInfos(nullptr)
		{}

		inline void setBuildInfoAndRanges(HostBuildGeometryInfo&& buildInfo, const core::smart_refctd_dynamic_array<BuildRangeInfo>& buildRangeInfos)
		{
			if(!isMutable())
				return;

			assert(validateBuildInfoAndRanges(std::move(buildInfo), buildRangeInfos));
			m_buildInfo = std::move(buildInfo);
			m_buildRangeInfos = buildRangeInfos;
			m_hasBuildInfo = true;
		}

		inline const core::SRange<const BuildRangeInfo> getBuildRanges() const 
		{ 
			if (m_buildRangeInfos)
				return {m_buildRangeInfos->begin(),m_buildRangeInfos->end()};
			return {nullptr,nullptr};
		}
		
		inline const uint32_t getBuildRangesCount() const { return getBuildRanges().size(); }

		//!
		size_t conservativeSizeEstimate() const override
		{
			size_t buildInfoSize = m_buildInfo.getGeometries().size() * sizeof(HostBuildGeometryInfo::Geom); 
			size_t buildRangesSize = getBuildRanges().size() * sizeof(BuildRangeInfo); 
			return sizeof(ICPUAccelerationStructure)+buildInfoSize+buildRangesSize;
		}

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto par = params;
			auto cp = core::smart_refctd_ptr<ICPUAccelerationStructure>(new ICPUAccelerationStructure(std::move(par)), core::dont_grab);
			clone_common(cp.get());
			
			cp->m_accelerationStructureSize = this->m_accelerationStructureSize;
			if(this->hasBuildInfo())
			{
				cp->m_hasBuildInfo = true;
				cp->m_buildInfo.type = this->m_buildInfo.type;
				cp->m_buildInfo.buildFlags = this->m_buildInfo.buildFlags;
				cp->m_buildInfo.buildMode = this->m_buildInfo.buildMode;
				
				auto geoms = m_buildInfo.getGeometries().begin();
				const auto geomsCount = m_buildInfo.getGeometries().size();

				cp->m_buildInfo.geometries = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<HostBuildGeometryInfo::Geom>>(geomsCount);
				auto outGeoms = cp->m_buildInfo.getGeometries().begin();

				for(uint32_t i = 0; i < geomsCount; ++i)
				{
					auto geom = geoms[i];
					auto & outGeom = outGeoms[i];
					if(geom.type == EGT_TRIANGLES)
					{
						outGeom.data.triangles.indexType = geom.data.triangles.indexType;
						outGeom.data.triangles.maxVertex = geom.data.triangles.maxVertex;
						outGeom.data.triangles.vertexFormat = geom.data.triangles.vertexFormat;
						outGeom.data.triangles.vertexStride = geom.data.triangles.vertexStride;

						outGeom.data.triangles.indexData.offset = geom.data.triangles.indexData.offset;
						outGeom.data.triangles.indexData.buffer = (_depth > 0u && geom.data.triangles.indexData.buffer) ?
							core::smart_refctd_ptr_static_cast<ICPUBuffer>(geom.data.triangles.indexData.buffer->clone(_depth - 1u)) :
							geom.data.triangles.indexData.buffer;

						outGeom.data.triangles.vertexData.offset = geom.data.triangles.vertexData.offset;
						outGeom.data.triangles.vertexData.buffer = (_depth > 0u && geom.data.triangles.vertexData.buffer) ?
							core::smart_refctd_ptr_static_cast<ICPUBuffer>(geom.data.triangles.vertexData.buffer->clone(_depth - 1u)) :
							geom.data.triangles.vertexData.buffer;
						
						outGeom.data.triangles.transformData.offset = geom.data.triangles.transformData.offset;
						outGeom.data.triangles.transformData.buffer = (_depth > 0u && geom.data.triangles.transformData.buffer) ?
							core::smart_refctd_ptr_static_cast<ICPUBuffer>(geom.data.triangles.transformData.buffer->clone(_depth - 1u)) :
							geom.data.triangles.transformData.buffer;
					}
					else if(geom.type == EGT_AABBS)
					{
						outGeom.data.aabbs.stride = geom.data.aabbs.stride;
						outGeom.data.aabbs.data.offset = geom.data.aabbs.data.offset;
						outGeom.data.aabbs.data.buffer = (_depth > 0u && geom.data.aabbs.data.buffer) ?
							core::smart_refctd_ptr_static_cast<ICPUBuffer>(geom.data.aabbs.data.buffer->clone(_depth - 1u)) :
							geom.data.aabbs.data.buffer;
					}
					else if(geom.type == EGT_INSTANCES)
					{
						outGeom.data.instances.data.offset = geom.data.instances.data.offset;
						outGeom.data.instances.data.buffer = (_depth > 0u && geom.data.instances.data.buffer) ?
							core::smart_refctd_ptr_static_cast<ICPUBuffer>(geom.data.instances.data.buffer->clone(_depth - 1u)) :
							geom.data.instances.data.buffer;
					}
				}

				auto buildRangesCount = this->m_buildRangeInfos->size();
				cp->m_buildRangeInfos = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<BuildRangeInfo>>(buildRangesCount);
				for(uint32_t i = 0; i < buildRangesCount; ++i)
					cp->m_buildRangeInfos->begin()[i] = this->m_buildRangeInfos->begin()[i];
			}

			return cp;
		}

		//!
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			convertToDummyObject_common(referenceLevelsBelowToConvert);
			
			if (referenceLevelsBelowToConvert)
			{
				--referenceLevelsBelowToConvert;
				auto geoms = m_buildInfo.getGeometries().begin();
				const auto geomsCount = m_buildInfo.getGeometries().size();
				for(uint32_t i = 0; i < geomsCount; ++i)
				{
					auto geom = geoms[i];
					if(geom.type == EGT_TRIANGLES)
					{
						if(geom.data.triangles.indexData.buffer)
							geom.data.triangles.indexData.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
						if(geom.data.triangles.vertexData.buffer)
							geom.data.triangles.vertexData.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
						if(geom.data.triangles.transformData.buffer)
							geom.data.triangles.transformData.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
					}
					else if(geom.type == EGT_AABBS)
					{
						if(geom.data.aabbs.data.buffer)
							geom.data.aabbs.data.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
					}
					else if(geom.type == EGT_INSTANCES)
					{
						if(geom.data.instances.data.buffer)
							geom.data.instances.data.buffer->convertToDummyObject(referenceLevelsBelowToConvert);
					}
				}
			}
			if (canBeConvertedToDummy()) {
				m_buildRangeInfos = nullptr;
				m_hasBuildInfo = false;
				m_accelerationStructureSize = 0ull;
			}
		}
		
		//!
		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUAccelerationStructure*>(_other);
			_NBL_TODO();
			return false;
		}

		//!
		constexpr static inline auto AssetType = ET_ACCELERATION_STRUCTURE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }


	protected:
		virtual ~ICPUAccelerationStructure() = default;

		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUAccelerationStructure*>(_other);
			_NBL_TODO();
			return;
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			--_levelsBelow;
			auto geoms = m_buildInfo.getGeometries().begin();
			const auto geomsCount = m_buildInfo.getGeometries().size();
			for(uint32_t i = 0; i < geomsCount; ++i)
			{
				auto geom = geoms[i];
				if(geom.type == EGT_TRIANGLES)
				{
					if(geom.data.triangles.indexData.buffer)
						return geom.data.triangles.indexData.buffer->isAnyDependencyDummy(_levelsBelow);
					if(geom.data.triangles.vertexData.buffer)
						return geom.data.triangles.vertexData.buffer->isAnyDependencyDummy(_levelsBelow);
					if(geom.data.triangles.transformData.buffer)
						return geom.data.triangles.transformData.buffer->isAnyDependencyDummy(_levelsBelow);
				}
				else if(geom.type == EGT_AABBS)
				{
					if(geom.data.aabbs.data.buffer)
						return geom.data.aabbs.data.buffer->isAnyDependencyDummy(_levelsBelow);
				}
				else if(geom.type == EGT_INSTANCES)
				{
					if(geom.data.instances.data.buffer)
						return geom.data.instances.data.buffer->isAnyDependencyDummy(_levelsBelow);
				}
			}
			return false;
		}
		
		inline bool validateBuildInfoAndRanges(HostBuildGeometryInfo&& buildInfo, const core::smart_refctd_dynamic_array<BuildRangeInfo>& buildRangeInfos)
		{
			// Validate
			return 
				(buildRangeInfos->empty() == false) &&
				(buildInfo.getGeometries().size() > 0) &&
				(buildInfo.getGeometries().size() == buildRangeInfos->size()) &&
				(buildInfo.type == params.type) &&
				(buildInfo.buildMode == IAccelerationStructure::EBM_BUILD);
		}

	private:
		core::smart_refctd_dynamic_array<Geometry<HostAddressType>> m_geometries;
		core::smart_refctd_dynamic_array<BuildRangeInfo> m_buildRangeInfos;
};

}

#endif