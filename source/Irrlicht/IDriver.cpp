#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "irr/asset/IAssetManager.h"
#include "irr/video/IGPUObjectFromAssetConverter.h"

namespace irr { namespace video
{

template<typename AssetType>
core::vector<typename video::asset_traits<AssetType>::GPUObjectType*> IDriver::getGPUObjectsFromAssets(AssetType** const _begin, AssetType** const _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(&m_device->getAssetManager(), this);
    if (!_converter)
        _converter = &def;
    auto res = _converter->getGPUObjectsFromAssets(_begin, _end);

    return res;
}
template core::vector<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(asset::ICPUBuffer** const, asset::ICPUBuffer** const, IGPUObjectFromAssetConverter* _converter);
template core::vector<typename video::asset_traits<asset::ICPUMeshBuffer>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(asset::ICPUMeshBuffer** const, asset::ICPUMeshBuffer** const, IGPUObjectFromAssetConverter* _converter);
template core::vector<typename video::asset_traits<asset::ICPUMesh>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(asset::ICPUMesh** const, asset::ICPUMesh** const, IGPUObjectFromAssetConverter* _converter);
template core::vector<typename video::asset_traits<asset::ICPUTexture>::GPUObjectType*> IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(asset::ICPUTexture** const, asset::ICPUTexture** const, IGPUObjectFromAssetConverter* _converter);

}}

