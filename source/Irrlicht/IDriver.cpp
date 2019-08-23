#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "irr/asset/IAssetManager.h"
#include "irr/video/IGPUObjectFromAssetConverter.h"

namespace irr { namespace video
{

template<typename AssetType>
core::vector<core::smart_refctd_ptr<typename video::asset_traits<AssetType>::GPUObjectType> > IDriver::getGPUObjectsFromAssets(AssetType* const* const _begin, AssetType* const* const _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if (!_converter)
        _converter = &def;
    return _converter->getGPUObjectsFromAssets(_begin, _end);
}

template core::vector<core::smart_refctd_ptr<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType> > IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(asset::ICPUBuffer* const* const, asset::ICPUBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template core::vector<core::smart_refctd_ptr<typename video::asset_traits<asset::ICPUMeshBuffer>::GPUObjectType> > IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(asset::ICPUMeshBuffer* const* const, asset::ICPUMeshBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template core::vector<core::smart_refctd_ptr<typename video::asset_traits<asset::ICPUMesh>::GPUObjectType> > IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(asset::ICPUMesh* const* const, asset::ICPUMesh* const* const, IGPUObjectFromAssetConverter* _converter);
template core::vector<core::smart_refctd_ptr<typename video::asset_traits<asset::ICPUTexture>::GPUObjectType> > IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(asset::ICPUTexture* const* const, asset::ICPUTexture* const* const, IGPUObjectFromAssetConverter* _converter);

}}

