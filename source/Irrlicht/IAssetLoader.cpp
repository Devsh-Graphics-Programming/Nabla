#include "IAssetLoader.h"

#include "IAssetManager.h"

using namespace irr;
using namespace asset;

IAsset* IAssetLoader::IAssetLoaderOverride::findCachedAsset(const std::string& inSearchKey, const IAsset::E_TYPE* inAssetTypes, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if ((levelFlag & ECF_DUPLICATE_TOP_LEVEL) == ECF_DUPLICATE_TOP_LEVEL)
        return nullptr;

    auto found = m_manager->findAssets(inSearchKey, inAssetTypes);
    uint32_t i = 0u;
    while (IAsset::E_TYPE type = inAssetTypes[i++])
    {
        const uint32_t ix = IAsset::typeFlagToIndex(type);
        if (IAssetManager::AssetCacheType::isNonZeroRange(found[ix]))
            return found[ix].first->second;
    }
    return nullptr;
}

void IAssetLoader::IAssetLoaderOverride::insertAssetIntoCache(IAsset* asset, const SAssetLoadContext& ctx, const uint32_t& hierarchyLevel)
{
    auto levelFlag = ctx.params.cacheFlags >> (uint64_t(hierarchyLevel) * 2ull);
    if (levelFlag&ECF_DONT_CACHE_TOP_LEVEL)
        asset->grab(); // because loader will call drop() straight after insertAssetIntoCache
    else
        m_manager->insertAssetIntoCache(asset);
}
