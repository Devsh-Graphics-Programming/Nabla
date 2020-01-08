#include "irr/asset/IAssetLoader.h"

#include "irr/asset/IAssetManager.h"

using namespace irr;
using namespace asset;

// todo NEED DOCS
IAssetLoader::IAssetLoaderOverride::IAssetLoaderOverride(IAssetManager* _manager) : m_manager(_manager), m_filesystem(m_manager->getFileSystem())
{
}

SAssetBundle IAssetLoader::IAssetLoaderOverride::findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if ((levelFlag & ECF_DUPLICATE_TOP_LEVEL) == ECF_DUPLICATE_TOP_LEVEL)
        return {};

    auto found = m_manager->findAssets(inSearchKey, inAssetTypes);
    if (!found->size())
        return handleSearchFail(inSearchKey, ctx, hierarchyLevel);
    return chooseRelevantFromFound(found->begin(), found->end(), ctx, hierarchyLevel);
}

void IAssetLoader::IAssetLoaderOverride::insertAssetIntoCache(SAssetBundle& asset, const std::string& supposedKey, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    m_manager->changeAssetKey(asset, supposedKey);

    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if (!(levelFlag&ECF_DONT_CACHE_TOP_LEVEL))
        m_manager->insertAssetIntoCache(asset);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel, IAssetLoader::IAssetLoaderOverride* _override)
{
    return _mgr->getAssetInHierarchy(_filename, _params, _hierarchyLevel, _override);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, io::IReadFile* _file, const std::string& _supposedFilename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr->getAssetInHierarchy(_file, _supposedFilename, _params, _hierarchyLevel);
}

SAssetBundle IAssetLoader::interm_getAssetInHierarchy(IAssetManager* _mgr, const std::string& _filename, const IAssetLoader::SAssetLoadParams& _params, uint32_t _hierarchyLevel)
{
    return _mgr->getAssetInHierarchy(_filename, _params, _hierarchyLevel);
}

void IAssetLoader::interm_setAssetMutable(const IAssetManager* _mgr, IAsset* _asset, bool _val)
{
    _mgr->setAssetMutable(_asset, _val);
}
