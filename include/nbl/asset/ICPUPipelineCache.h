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

class ICPUPipelineCache final : public IAsset // TODO: PreHashed ?
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
		
		constexpr static inline auto AssetType = ET_PIPELINE_CACHE;
		inline E_TYPE getAssetType() const override { return AssetType; }

		inline core::smart_refctd_ptr<IAsset> clone(uint32_t _depth = ~0u) const override
		{
			auto cache_cp = m_cache;
			return core::make_smart_refctd_ptr<ICPUPipelineCache>(std::move(cache_cp));
		}

		inline size_t getDependantCount() const override {return 0;}

	protected:
		inline IAsset* getDependant_impl(const size_t ix) override {return nullptr;}

	private:
		entries_map_t m_cache;
};

}

#endif