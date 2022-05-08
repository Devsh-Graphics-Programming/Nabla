// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_S_ASSET_BUNDLE_H_INCLUDED__
#define __NBL_ASSET_S_ASSET_BUNDLE_H_INCLUDED__

#include <string>
#include "nbl/asset/IAsset.h"
#include "nbl/asset/metadata/IAssetMetadata.h"

namespace nbl
{
namespace asset
{

//! A class storing Assets with the same type
class NBL_API SAssetBundle
{
		inline bool allSameTypeAndNotNull()
		{
			if (m_contents->size() == 0ull)
				return true;
			if (!*m_contents->begin())
				return false;
			IAsset::E_TYPE t = (*m_contents->begin())->getAssetType();
			for (auto it=m_contents->cbegin(); it!=m_contents->cend(); it++)
				if (!(*it) || (*it)->getAssetType()!=t)
					return false;
			return true;
		}
	public:
		using contents_container_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IAsset> >;
    
		SAssetBundle(const size_t assetCount=0ull) : m_metadata(nullptr), m_contents(core::make_refctd_dynamic_array<contents_container_t>(assetCount)), m_cacheKey("")
		{
		}
		SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>&& _metadata, contents_container_t&& _contents) : m_metadata(std::move(_metadata)), m_contents(std::move(_contents)), m_cacheKey{}
		{
			assert(allSameTypeAndNotNull());
		}
		SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>&& _metadata, std::initializer_list<core::smart_refctd_ptr<IAsset> > _contents) : 
			SAssetBundle(std::move(_metadata),core::make_refctd_dynamic_array<contents_container_t>(_contents))
		{
		}
		template<typename ContainerT>
		SAssetBundle(core::smart_refctd_ptr<IAssetMetadata>&& _metadata, ContainerT&& _contents) :
			SAssetBundle(std::move(_metadata),core::make_refctd_dynamic_array<contents_container_t>(std::forward<ContainerT>(_contents)))
		{
		}

		//! Returning a type associated with current stored Assets
		/**
			An Asset type is specified in E_TYPE enum.
			@see E_TYPE
		*/
		inline IAsset::E_TYPE getAssetType() const { return m_contents->front()->getAssetType(); }

		//! Getting beginning and end of an Asset stored by m_contents
		inline core::SRange<const core::smart_refctd_ptr<IAsset>> getContents() const
		{
			return core::SRange<const core::smart_refctd_ptr<IAsset>>(m_contents->begin(),m_contents->end());
		}

		//! Whether this asset bundle is in a cache and should be removed from cache to destroy
		inline bool isInAResourceCache() const { return m_isCached; }

		//! Only valid if isInAResourceCache() returns true
		inline const std::string& getCacheKey() const { return m_cacheKey; }

		//! Returns SAssetBundle's metadata. @see IAssetMetadata
		//inline IAssetMetadata* getMetadata() { return m_metadata.get(); } // shouldn't be allowed

		//! Returns SAssetBundle's metadata. @see IAssetMetadata
		inline const IAssetMetadata* getMetadata() const { return m_metadata.get(); }

		//!
		inline void setMetadata(core::smart_refctd_ptr<IAssetMetadata>&& _metadata)	{ m_metadata = std::move(_metadata); }

		//! Overloaded operator checking if both collections of Assets\b are\b the same arrays in memory
		inline bool operator==(const SAssetBundle& _other) const
		{
			if (m_metadata != _other.m_metadata)
				return false;
            if (m_contents->size() != _other.m_contents->size())
                return false;
            for (size_t i = 0ull; i < m_contents->size(); ++i)
                if ((*m_contents)[i] != (*_other.m_contents)[i])
                    return false;
            return true;
		}

		//! Overloaded operator checking if both collections of Assets\b aren't\b the same arrays in memory
		inline bool operator!=(const SAssetBundle& _other) const
		{
			return !((*this) != _other);
		}

	private:
		friend class IAssetManager;

		inline void setNewCacheKey(const std::string& newKey) { m_cacheKey = newKey; }
		inline void setNewCacheKey(std::string&& newKey) { m_cacheKey = std::move(newKey); }
		inline void setCached(bool val) { m_isCached = val; }

		friend class IAssetLoader;
		inline void setAsset(const uint32_t offset, core::smart_refctd_ptr<IAsset>&& _asset)
		{
			m_contents->operator[](offset) = std::move(_asset);
			assert(allSameTypeAndNotNull());
		}


		core::smart_refctd_ptr<IAssetMetadata> m_metadata;
		contents_container_t m_contents;

		std::string m_cacheKey;
		bool m_isCached = false;
};

}
}

#endif
