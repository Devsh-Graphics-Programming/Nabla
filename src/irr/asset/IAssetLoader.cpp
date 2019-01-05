#include "irr/asset/IAssetLoader.h"

#include "irr/asset/IAssetManager.h"

using namespace irr;
using namespace asset;

// todo NEED DOCS
IAsset* IAssetLoader::IAssetLoaderOverride::findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if ((levelFlag & ECF_DUPLICATE_TOP_LEVEL) == ECF_DUPLICATE_TOP_LEVEL)
        return nullptr;

    core::vector<IAsset*> found = m_manager->findAssets(inSearchKey, inAssetTypes);
    if (!found.size())
        return handleSearchFail(inSearchKey, ctx, hierarchyLevel);
    return found.front();
}

void IAssetLoader::IAssetLoaderOverride::setAssetCacheKey(IAsset* asset, const std::string& supposedKey, const SAssetLoadContext& ctx, uint32_t hierarchyLevel)
{
    m_manager->changeAssetKey(asset, supposedKey);
}

void IAssetLoader::IAssetLoaderOverride::insertAssetIntoCache(IAsset* asset, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if (levelFlag&ECF_DONT_CACHE_TOP_LEVEL)
        asset->grab(); // because loader will call drop() straight after insertAssetIntoCache
    else
        m_manager->insertAssetIntoCache(asset);
}
