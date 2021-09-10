// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_ASSET_I_CPU_ACCELERATION_STRUCTURE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_ACCELERATION_STRUCTURE_H_INCLUDED_

#include "nbl/asset/IAsset.h"
#include "nbl/asset/IAccelerationStructure.h"
#include "nbl/asset/ICPUBuffer.h"

namespace nbl
{
namespace asset
{

class ICPUAccelerationStructure final : public IAccelerationStructure, public IAsset
{
	using Base = IAccelerationStructure;

	public:
		struct SCreationParams
		{
			E_CREATE_FLAGS	flags;
			Base::E_TYPE	type;
			bool operator==(const SCreationParams& rhs) const
			{
				return flags == rhs.flags && type == rhs.type;
			}
			bool operator!=(const SCreationParams& rhs) const
			{
				return !operator==(rhs);
			}
		};
		
		using HostAddressType = asset::SBufferBinding<asset::ICPUBuffer>;
		template<typename AddressType>
		struct BuildGeometryInfo
		{
			BuildGeometryInfo() 
				: type(static_cast<Base::E_TYPE>(0u))
				, buildFlags(static_cast<E_BUILD_FLAGS>(0u))
				, buildMode(static_cast<E_BUILD_MODE>(0u))
				, geometries(core::SRange<Geometry<AddressType>>(nullptr, nullptr))
			{}
			~BuildGeometryInfo() = default;
			Base::E_TYPE	type; // TODO: Can deduce from creationParams.type?
			E_BUILD_FLAGS	buildFlags;
			E_BUILD_MODE	buildMode;
			core::SRange<Geometry<AddressType>> geometries;
		};
		using HostBuildGeometryInfo = BuildGeometryInfo<HostAddressType>;

		inline const auto& getCreationParameters() const
		{
			return params;
		}

		//!
		inline static bool validateCreationParameters(const SCreationParams& _params)
		{
			return true;
		}

	public:
		static core::smart_refctd_ptr<ICPUAccelerationStructure> create(SCreationParams&& params)
		{
			if (!validateCreationParameters(params))
				return nullptr;

			return core::make_smart_refctd_ptr<ICPUAccelerationStructure>(std::move(params));
		}

		ICPUAccelerationStructure(SCreationParams&& _params) 
			: params(std::move(_params))
		{}

		inline void setAccelerationStructureSize(uint64_t accelerationStructureSize) { m_accelerationStructureSize = accelerationStructureSize; }
		inline uint64_t getAccelerationStructureSize() const { return m_accelerationStructureSize; }

		inline void setBuildInfoAndRanges(HostBuildGeometryInfo* buildInfo, const core::SRange<BuildRangeInfo>& buildRangeInfos)
		{
			m_buildInfo = buildInfo;
			m_buildRangeInfo = buildRangeInfos.begin();
			m_buildRangeInfoCount = buildRangeInfos.size(); // is m_buildRangeInfoCount needed? just for validation
			assert(buildInfo->geometries.size() == m_buildRangeInfoCount);
		}

		inline const BuildRangeInfo* getBuildRangesPtr() const { return m_buildRangeInfo; }
		inline const HostBuildGeometryInfo* getBuildInfo() const { return m_buildInfo; }
		inline const uint32_t getBuildRangesCount() const { return m_buildRangeInfoCount; }
		inline const core::SRange<const BuildRangeInfo> getBuildRanges() const { return core::SRange<const BuildRangeInfo>(m_buildRangeInfo, m_buildRangeInfo + m_buildRangeInfoCount); }

		//!
		size_t conservativeSizeEstimate() const override
		{
			return sizeof(SCreationParams);
		}

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			return nullptr; //TODO
		}

		//!
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert=0u) override
		{
			return; //TODO
		}
		
		//!
		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUAccelerationStructure*>(_other);
			const auto& rhs = other->params;

			if (params.flags != rhs.flags)
				return false;
			if (params.type != rhs.type)
				return false;
			 //TODO
			return true;
		}

		//!
		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_ACCELERATION_STRUCTURE;
		inline IAsset::E_TYPE getAssetType() const override { return AssetType; }


	protected:

		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUAccelerationStructure*>(_other);

			if (_levelsBelow)
			{
				//TODO
			}
		}

		bool isAnyDependencyDummy_impl(uint32_t _levelsBelow) const override
		{
			// TODO
			return false;
		}

		virtual ~ICPUAccelerationStructure() = default;

	private:
		SCreationParams params;
		uint64_t m_accelerationStructureSize;

		HostBuildGeometryInfo* m_buildInfo;
		BuildRangeInfo* m_buildRangeInfo = nullptr;
		uint32_t m_buildRangeInfoCount = 0;
};

}
}

#endif