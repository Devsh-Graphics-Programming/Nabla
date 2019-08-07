#include "irr/asset/IAssetLoader.h"

#include "irr/asset/IAssetManager.h"

using namespace irr;
using namespace asset;

// todo NEED DOCS
IAssetCachableBundle* IAssetLoader::IAssetLoaderOverride::findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if ((levelFlag & ECF_DUPLICATE_TOP_LEVEL) == ECF_DUPLICATE_TOP_LEVEL)
        return nullptr;

    core::vector<IAssetCachableBundle*> found = m_manager->findAssets(inSearchKey, inAssetTypes);
    if (!found.size())
        return handleSearchFail(inSearchKey, ctx, hierarchyLevel);
    return chooseRelevantFromFound(found, ctx, hierarchyLevel);
}

IAssetCachableBundle* IAssetLoader::IAssetLoaderOverride::insertAssetIntoCache(IAssetBundle* asset, const std::string& supposedKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel)*2ull);
    IAssetCachableBundle* newAssetBundle = IAssetCachableBundle::fromAssetBundle(asset);
    m_manager->changeAssetKey(newAssetBundle, supposedKey);
    if (!(levelFlag&ECF_DONT_CACHE_TOP_LEVEL))
    {
        m_manager->insertAssetIntoCache(newAssetBundle);
    }

    return newAssetBundle;
}

IAssetCachableBundle * IAssetLoader::interm_getAssetInHierarchy(IAssetManager& _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr.getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel, _override);
}

IAssetCachableBundle* IAssetLoader::interm_getAssetInHierarchy(IAssetManager& _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr.getAssetInHierarchy(_filename, _params, _hierarchyLevel, _override);
}

IAssetCachableBundle* IAssetLoader::interm_getAssetInHierarchy(IAssetManager& _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr.getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel);
}

IAssetCachableBundle* IAssetLoader::interm_getAssetInHierarchy(IAssetManager& _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr.getAssetInHierarchy(_filename, _params, _hierarchyLevel);
}
