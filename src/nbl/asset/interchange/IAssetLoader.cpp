// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/asset/interchange/IAssetLoader.h"

#include "nbl/asset/IAssetManager.h"

using namespace nbl;
using namespace asset;

// todo NEED DOCS
IAssetLoader::IAssetLoaderOverride::IAssetLoaderOverride(IAssetManager* _manager)
    : m_manager(_manager), m_system(m_manager->getSystem())
{
}

SAssetBundle IAssetLoader::IAssetLoaderOverride::findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if((levelFlag & ECF_DUPLICATE_TOP_LEVEL) == ECF_DUPLICATE_TOP_LEVEL)
        return {};

    auto found = m_manager->findAssets(inSearchKey, inAssetTypes);
    if(!found->size())
        return handleSearchFail(inSearchKey, ctx, hierarchyLevel);
    return chooseRelevantFromFound(found->begin(), found->end(), ctx, hierarchyLevel);
}

void IAssetLoader::IAssetLoaderOverride::insertAssetIntoCache(SAssetBundle& asset, const std::string& supposedKey, const SAssetLoadContext& ctx, const uint32_t hierarchyLevel)
{
    m_manager->changeAssetKey(asset, supposedKey);

    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if(!(levelFlag & ECF_DONT_CACHE_TOP_LEVEL))
        m_manager->insertAssetIntoCache(asset, ASSET_MUTABILITY_ON_CACHE_INSERT);
}

core::smart_refctd_ptr<IAsset> IAssetLoader::IAssetLoaderOverride::handleRestore(core::smart_refctd_ptr<IAsset>&& _chosenAsset, SAssetBundle& _bundle, SAssetBundle& _reloadedBundle, uint32_t _restoreLevels)
{
    if(_bundle.getContents().size() != _reloadedBundle.getContents().size())
        return nullptr;

    auto dummies = _bundle.getContents();
    auto found_it = std::find(dummies.begin(), dummies.end(), _chosenAsset);
    if(found_it == dummies.end())
        return nullptr;
    const uint32_t ix = found_it - dummies.begin();

    auto reloaded = _reloadedBundle.getContents();
    if(_chosenAsset->isADummyObjectForCache() && !_chosenAsset->canBeRestoredFrom(reloaded.begin()[ix].get()))
        return nullptr;

    _chosenAsset->restoreFromDummy(reloaded.begin()[ix].get(), _restoreLevels);

    return _chosenAsset;
}

void IAssetLoader::IAssetLoaderOverride::handleRestore(SAssetBundle& _bundle, SAssetBundle& _reloadedBundle, uint32_t _restoreLevels)
{
    if(_bundle.getContents().size() != _reloadedBundle.getContents().size())
        return;

    auto dummies = _bundle.getContents();
    auto reloaded = _reloadedBundle.getContents();
    const uint32_t cnt = dummies.size();

    for(uint32_t i = 0u; i < cnt; ++i)
        if(dummies.begin()[i]->isADummyObjectForCache() && dummies.begin()[i]->canBeRestoredFrom(reloaded.begin()[i].get()))
            dummies.begin()[i].get()->restoreFromDummy(reloaded.begin()[i].get(), _restoreLevels);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchy(_filename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr->getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr->getAssetInHierarchy(_filename, _params, _hierarchyLevel);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchyWholeBundleRestore(IAssetManager* _mgr, system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchyWholeBundleRestore(_file, _supposedFilename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchyWholeBundleRestore(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchyWholeBundleRestore(_filename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchyWholeBundleRestore(IAssetManager* _mgr, system::IFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr->getAssetInHierarchyWholeBundleRestore(_file, _supposedFilename, _params, _hierarchyLevel);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchyWholeBundleRestore(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr->getAssetInHierarchyWholeBundleRestore(_filename, _params, _hierarchyLevel);
}

void IAssetLoader::interm_setAssetMutability(const IAssetManager* _mgr, IAsset* _asset, IAsset::E_MUTABILITY _val)
{
    _mgr->setAssetMutability(_asset, _val);
}

/*
void IAssetLoader::interm_restoreDummyAsset(IAssetManager* _mgr, SAssetBundle& _bundle)
{
    _mgr->restoreDummyAsset(_bundle);
}

void IAssetLoader::interm_restoreDummyAsset(IAssetManager* _mgr, IAsset* _asset, const std::string _path)
{
    SAssetBundle bundle({core::smart_refctd_ptr<IAsset>(_asset)}, _path);
    interm_restoreDummyAsset(_mgr, bundle);
}
*/

bool IAssetLoader::insertBuiltinAssetIntoCache(IAssetManager* _mgr, SAssetBundle& _asset, const std::string _path)
{
    _mgr->changeAssetKey(_asset, _path);
    return _mgr->insertBuiltinAssetIntoCache(_asset);
}

bool IAssetLoader::insertBuiltinAssetIntoCache(IAssetManager* _mgr, core::smart_refctd_ptr<IAsset>& _asset, core::smart_refctd_ptr<IAssetMetadata>&& metadata, const std::string _path)
{
    asset::SAssetBundle bundle(std::move(metadata), {_asset});
    return insertBuiltinAssetIntoCache(_mgr, bundle, _path);
}

bool IAssetLoader::insertBuiltinAssetIntoCache(IAssetManager* _mgr, core::smart_refctd_ptr<IAsset>&& _asset, core::smart_refctd_ptr<IAssetMetadata>&& metadata, const std::string _path)
{
    asset::SAssetBundle bundle(std::move(metadata), {std::move(_asset)});
    return insertBuiltinAssetIntoCache(_mgr, bundle, _path);
}
