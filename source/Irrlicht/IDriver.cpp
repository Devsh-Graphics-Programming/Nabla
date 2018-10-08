#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "IAssetManager.h"

namespace irr { namespace video
{

template<typename AssetType>
core::vector<typename IDriver::asset_traits<AssetType>::GPUObjectType*> IDriver::getGPUObjectsFromAssets(AssetType** const _begin, AssetType** const _end)
{
    core::vector<AssetType*> notFound;
    core::vector<size_t> pos;
    core::vector<typename asset_traits<AssetType>::GPUObjectType*> res;
    AssetType** it = _begin;
    while (it != _end)
    {
        core::IReferenceCounted* gpu = m_device->getAssetManager().getGPUObject(*it);
        if (!gpu)
        {
            notFound.push_back(*it);
            pos.push_back(it - _begin);
        }
        else res.push_back(dynamic_cast<typename asset_traits<AssetType>::GPUObjectType*>(gpu));
        ++it;
    }
    core::vector<typename asset_traits<AssetType>::GPUObjectType*> created = createGPUObjectFromAsset(notFound.data(), notFound.data()+notFound.size());
    for (size_t i = 0u; i < created.size(); ++i)
    {
        m_device->getAssetManager().convertAssetToEmptyCacheHandle(notFound[i], created[i]);
        res.insert(res.begin() + pos[i], created[i]);
    }
    
    return res;
}
template core::vector<typename IDriver::asset_traits<core::ICPUBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<core::ICPUBuffer>(core::ICPUBuffer** const, core::ICPUBuffer** const);
template core::vector<typename IDriver::asset_traits<scene::ICPUMeshBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<scene::ICPUMeshBuffer>(scene::ICPUMeshBuffer** const, scene::ICPUMeshBuffer** const);
template core::vector<typename IDriver::asset_traits<scene::ICPUMesh>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<scene::ICPUMesh>(scene::ICPUMesh** const, scene::ICPUMesh** const);
template core::vector<typename IDriver::asset_traits<asset::ICPUTexture>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(asset::ICPUTexture** const, asset::ICPUTexture** const);

}}

