// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/IAssetLoader.h"

#include "nbl/asset/IAssetManager.h"

using namespace nbl;
using namespace asset;

// todo NEED DOCS
IAssetLoader::IAssetLoaderOverride::IAssetLoaderOverride(IAssetManager* _manager) : m_manager(_manager), m_system(m_manager->getSystem())
{
}

SAssetBundle IAssetLoader::IAssetLoaderOverride::findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if ((levelFlag & ECF_DUPLICATE_TOP_LEVEL) == ECF_DUPLICATE_TOP_LEVEL)
        return {};

    auto found = m_manager->findAssets(inSearchKey, inAssetTypes);
    if (!found->size())
        return handleSearchFail(inSearchKey, ctx, hierarchyLevel);
    return chooseRelevantFromFound(found->begin(), found->end(), ctx, hierarchyLevel);
}

void IAssetLoader::IAssetLoaderOverride::insertAssetIntoCache(SAssetBundle& asset, const std::string& supposedKey, const SAssetLoadContext& ctx, const uint32_t hierarchyLevel)
{
    m_manager->changeAssetKey(asset, supposedKey);

    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if (!(levelFlag&ECF_DONT_CACHE_TOP_LEVEL))
        m_manager->insertAssetIntoCache(asset,ASSET_MUTABILITY_ON_CACHE_INSERT);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchy(_filename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchyWithAllContent(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
	auto firstLoad = interm_getAssetInHierarchy(_mgr,_filename,_params,_hierarchyLevel,_override);
	auto bundleHasAllContent = [](const SAssetBundle& retval)->bool
	{
		for (const auto& asset : retval.getContents())
		if (IPreHashed::anyDependantDiscardedContents(asset.get()))
			return false;
		return true;
	};
	if (bundleHasAllContent(firstLoad))
		return firstLoad;

	IAssetLoader::SAssetLoadParams paramCopy = _params;
	paramCopy.cacheFlags = ECF_DUPLICATE_REFERENCES;
	auto secondLoad = interm_getAssetInHierarchy(_mgr,_filename,paramCopy,_hierarchyLevel,_override);
	if (bundleHasAllContent(secondLoad))
	{
		_mgr->removeAssetFromCache(firstLoad);
		return secondLoad;
	}
	else
		return {};
}

void IAssetLoader::interm_setAssetMutability(const IAssetManager* _mgr, IAsset* _asset, const bool _val)
{
    _mgr->setAssetMutability(_asset, _val);
}

bool IAssetLoader::insertBuiltinAssetIntoCache(IAssetManager* _mgr, SAssetBundle& _asset, const std::string _path)
{
    _mgr->changeAssetKey(_asset, _path);
    return _mgr->insertBuiltinAssetIntoCache(_asset);
}

bool IAssetLoader::insertBuiltinAssetIntoCache(IAssetManager* _mgr,  core::smart_refctd_ptr<IAsset>& _asset, core::smart_refctd_ptr<IAssetMetadata>&& metadata, const std::string _path)
{
    asset::SAssetBundle bundle(std::move(metadata), { _asset });
    return insertBuiltinAssetIntoCache(_mgr, bundle, _path);
}

bool IAssetLoader::insertBuiltinAssetIntoCache(IAssetManager* _mgr, core::smart_refctd_ptr<IAsset>&& _asset, core::smart_refctd_ptr<IAssetMetadata>&& metadata, const std::string _path)
{
    asset::SAssetBundle bundle(std::move(metadata), { std::move(_asset) });
    return insertBuiltinAssetIntoCache(_mgr, bundle, _path);
}
