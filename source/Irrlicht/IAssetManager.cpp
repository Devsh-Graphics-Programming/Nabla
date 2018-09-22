#include "IAssetManager.h"

using namespace irr;
using namespace asset;

std::function<void(IAsset*)> irr::asset::makeAssetGreetFunc(const IAssetManager* const _mgr)
{
    return [_mgr](IAsset* _asset) { _asset->grab(); _mgr->setAssetCached(_asset, true); };
}
std::function<void(IAsset*)> irr::asset::makeAssetDisposeFunc(const IAssetManager* const _mgr)
{
    return [_mgr](IAsset* _asset) { _mgr->setAssetCached(_asset, false); _asset->drop(); };
}