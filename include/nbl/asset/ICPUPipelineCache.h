// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_PIPELINE_CACHE_H_INCLUDED__
#define __NBL_ASSET_I_CPU_PIPELINE_CACHE_H_INCLUDED__

#include "nbl/core/decl/Types.h"

#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPUDescriptorSetLayout.h"
#include "nbl/asset/ISpecializedShader.h"

namespace nbl::asset
{

class ICPUPipelineCache final : public IAsset
{
public:
	enum E_BACKEND : uint8_t
	{
		EB_OPENGL,
		EB_VULKAN
	};
	struct GPUID
	{
		E_BACKEND backend;
		std::string UUID;

		bool operator<(const GPUID& _rhs) const
		{
			if (backend==_rhs.backend)
			{
				return UUID < _rhs.UUID;
			}
			return backend < _rhs.backend;
		}
		bool operator==(const GPUID& _rhs) const
		{
			return backend==_rhs.backend && UUID==_rhs.UUID;
		}
	};

#include "nbl/nblpack.h"
	//needed to create entries of COpenGLPipelineCache
	struct SGLKeyMeta
	{
		struct SBinding
		{
			uint32_t binding;
			E_DESCRIPTOR_TYPE type;
			uint32_t count;
			asset::ISpecializedShader::E_SHADER_STAGE stageFlags;
			//TODO currently IDescriptorSetLayout::isIdentificallyDefined() compares just pointers of immutable samplers
			//which is not really of any use when COpenGLPipelineCache is created (and so are created pipeline layouts needed for comparing compatibility) from CPU pipeline cache
			//so how i see this, is that the only way it's gonna work, is to rework isIdentificallyDefined() to compare sampler params instead of pointers
			//Thinking about how vulkan's pipeline cache is working, i think vulkan while checking layout compatibility also compares just sampler params (and not checking for exact same objects)
			//const core::smart_refctd_ptr<sampler_type>* samplers;
		} PACK_STRUCT;
		struct SSpecInfo
		{
			struct SEntry
			{
				uint32_t id;
				uint32_t value;
			} PACK_STRUCT;

			char entryPoint[128];
			asset::ISpecializedShader::E_SHADER_STAGE shaderStage;
			uint32_t entryCnt;
		} PACK_STRUCT;

		uint64_t spirvHash[4];
		uint32_t bindingsPerSet[4];
		SBinding bindings[1];
		//entries array goes just after bindings array
		//SEntry entries[1];

		static size_t calcMetaSize(uint32_t _bndCnt, uint32_t _scCnt)
		{
			return sizeof(SSpecInfo)-sizeof(SSpecInfo::SEntry) + _scCnt*sizeof(SSpecInfo::SEntry) + sizeof(spirvHash) + sizeof(bindingsPerSet) + _bndCnt*sizeof(SBinding);
		}
	} PACK_STRUCT;
#include "nbl/nblunpack.h"
	struct SCacheKey
	{
		GPUID gpuid;
		core::smart_refctd_dynamic_array<uint8_t> meta;

		bool operator<(const SCacheKey& _rhs) const
		{
			if (gpuid==_rhs.gpuid)
			{
				if (gpuid.backend==EB_VULKAN)//in vulkan `meta` is always nullptr
					return false;
				if (meta->size()==_rhs.meta->size())
				{
					return memcmp(meta->data(), _rhs.meta->data(), meta->size()) < 0;
				}
				return meta->size()<_rhs.meta->size();
			}
			return gpuid < _rhs.gpuid;
		}
	};
	struct SCacheVal
	{
		uint32_t extra;
		core::smart_refctd_dynamic_array<uint8_t> bin;
	};

	using entries_map_t = core::map<SCacheKey, SCacheVal>;

	explicit ICPUPipelineCache(entries_map_t&& _entries) : m_cache(std::move(_entries)) {}

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