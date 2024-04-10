// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_CPU_PIPELINE_CACHE_H_INCLUDED_
#define _NBL_ASSET_I_CPU_PIPELINE_CACHE_H_INCLUDED_

#include "nbl/core/decl/Types.h"

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUDescriptorSetLayout.h"

namespace nbl::asset
{

class ICPUPipelineCache final : public IAsset
{
	public:
		struct SCacheKey
		{
			std::string deviceAndDriverUUID = {};
			core::smart_refctd_dynamic_array<uint8_t> meta = {};

			bool operator<(const SCacheKey& _rhs) const
			{
				if (deviceAndDriverUUID==_rhs.deviceAndDriverUUID)
				{
					if (!meta || !_rhs.meta)
						return bool(_rhs.meta);
					if (meta->size()==_rhs.meta->size())
						return memcmp(meta->data(), _rhs.meta->data(), meta->size()) < 0;
					return meta->size()<_rhs.meta->size();
				}
				return deviceAndDriverUUID < _rhs.deviceAndDriverUUID;
			}
		};
		struct SCacheVal
		{
			core::smart_refctd_dynamic_array<uint8_t> bin;
		};

		using entries_map_t = core::map<SCacheKey,SCacheVal>;

		// ctor
		explicit ICPUPipelineCache(entries_map_t&& _entries) : m_cache(std::move(_entries)) {}

		//
		const auto& getEntries() const {return m_cache;}

		// `IAsset` methods
		size_t conservativeSizeEstimate() const override { return 0ull; /*TODO*/ }
		void convertToDummyObject(uint32_t referenceLevelsBelowToConvert = 0u) override
		{
			if (canBeConvertedToDummy())
				m_cache.clear();
		}

		_NBL_STATIC_INLINE_CONSTEXPR auto AssetType = ET_PIPELINE_CACHE;
		inline E_TYPE getAssetType() const override { return AssetType; }

		core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cache_cp = m_cache;
		
			return core::make_smart_refctd_ptr<ICPUPipelineCache>(std::move(cache_cp));
		}

		bool canBeRestoredFrom(const IAsset* _other) const override
		{
			auto* other = static_cast<const ICPUPipelineCache*>(_other);
			if (m_cache.size() != other->m_cache.size())
				return false;

			return true;
		}

	protected:
		void restoreFromDummy_impl(IAsset* _other, uint32_t _levelsBelow) override
		{
			auto* other = static_cast<ICPUPipelineCache*>(_other);
			const bool restorable = willBeRestoredFrom(_other);
			if (restorable)
				std::swap(m_cache, other->m_cache);
		}

	private:
		entries_map_t m_cache;
};

}

#endif