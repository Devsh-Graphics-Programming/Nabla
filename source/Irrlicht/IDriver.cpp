#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "irr/asset/IAssetManager.h"
#include "irr/video/IGPUObjectFromAssetConverter.h"

namespace irr { namespace video
{

//! Maybe we can reduce code duplication here some day
template<typename AssetType>
created_gpu_object_array<AssetType> IDriver::getGPUObjectsFromAssets(AssetType* const* const _begin, AssetType* const* const _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if (!_converter)
        _converter = &def;
    return _converter->getGPUObjectsFromAssets<AssetType>(_begin, _end);
}

template created_gpu_object_array<asset::ICPUBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(asset::ICPUBuffer* const* const, asset::ICPUBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(asset::ICPUMeshBuffer* const* const, asset::ICPUMeshBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(asset::ICPUMesh* const* const, asset::ICPUMesh* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUTexture> IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(asset::ICPUTexture* const* const, asset::ICPUTexture* const* const, IGPUObjectFromAssetConverter* _converter);

template<typename AssetType>
created_gpu_object_array<AssetType> IDriver::getGPUObjectsFromAssets(const core::smart_refctd_ptr<asset::IAsset>* _begin, const core::smart_refctd_ptr<asset::IAsset>* _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if (!_converter)
        _converter = &def;
    return _converter->getGPUObjectsFromAssets<AssetType>(_begin, _end);
}

template created_gpu_object_array<asset::ICPUBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(const core::smart_refctd_ptr<asset::IAsset>*, const core::smart_refctd_ptr<asset::IAsset>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(const core::smart_refctd_ptr<asset::IAsset>*, const core::smart_refctd_ptr<asset::IAsset>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(const core::smart_refctd_ptr<asset::IAsset>*, const core::smart_refctd_ptr<asset::IAsset>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUTexture> IDriver::getGPUObjectsFromAssets<asset::ICPUTexture>(const core::smart_refctd_ptr<asset::IAsset>*, const core::smart_refctd_ptr<asset::IAsset>*, IGPUObjectFromAssetConverter* _converter);

}}

