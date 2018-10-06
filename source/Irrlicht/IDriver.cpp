#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "IAssetManager.h"

namespace irr { namespace video
{

template<typename AssetType>
core::vector<typename asset_traits<AssetType>::GPUObjectType*> IDriver::getGPUObjectsFromAssets(AssetType** _assets)
{
    core::vector<typename asset_traits<AssetType>::GPUObjectType*> res;
    while (_assets)
    {
        res.push_back(getGPUObjectFromAsset<AssetType>(_assets[0]));
        ++_assets;
    }
    return res;
}
template core::vector<typename IDriver::asset_traits<core::ICPUBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<core::ICPUBuffer>(core::ICPUBuffer** _assets);
template core::vector<typename IDriver::asset_traits<scene::ICPUMeshBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<scene::ICPUMeshBuffer>(scene::ICPUMeshBuffer** _assets);
template core::vector<typename IDriver::asset_traits<scene::ICPUMesh>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<scene::ICPUMesh>(scene::ICPUMesh** _assets);
template core::vector<typename IDriver::asset_traits<asset::ICPUTexture>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(asset::ICPUTexture** _assets);

template<typename AssetType>
typename asset_traits<AssetType>::GPUObjectType* IDriver::getGPUObjectFromAsset(AssetType* _asset)
{
    core::IReferenceCounted* gpu = m_device->getAssetManager().getGPUObject(_asset);
    if (gpu)
        return dynamic_cast<typename asset_traits<AssetType>::GPUObjectType*>(gpu);
    return createGPUObjectFromAsset(_asset);
}
template typename IDriver::asset_traits<core::ICPUBuffer>::GPUObjectType* IDriver::getGPUObjectFromAsset<core::ICPUBuffer>(core::ICPUBuffer* _asset);
template typename IDriver::asset_traits<scene::ICPUMeshBuffer>::GPUObjectType* IDriver::getGPUObjectFromAsset<scene::ICPUMeshBuffer>(scene::ICPUMeshBuffer* _asset);
template typename IDriver::asset_traits<scene::ICPUMesh>::GPUObjectType* IDriver::getGPUObjectFromAsset<scene::ICPUMesh>(scene::ICPUMesh* _asset);
template typename IDriver::asset_traits<asset::ICPUTexture>::GPUObjectType* IDriver::getGPUObjectFromAsset<asset::ICPUTexture>(asset::ICPUTexture* _asset);

}}

