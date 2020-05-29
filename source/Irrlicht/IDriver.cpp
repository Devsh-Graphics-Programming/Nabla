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
template created_gpu_object_array<asset::ICPUImage> IDriver::getGPUObjectsFromAssets<asset::ICPUImage>(asset::ICPUImage* const* const, asset::ICPUImage* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImageView> IDriver::getGPUObjectsFromAssets<asset::ICPUImageView>(asset::ICPUImageView* const* const, asset::ICPUImageView* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUShader> IDriver::getGPUObjectsFromAssets<asset::ICPUShader>(asset::ICPUShader* const* const, asset::ICPUShader* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSpecializedShader> IDriver::getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(asset::ICPUSpecializedShader* const* const, asset::ICPUSpecializedShader* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(asset::ICPUMeshBuffer* const* const, asset::ICPUMeshBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(asset::ICPUMesh* const* const, asset::ICPUMesh* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUPipelineLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(asset::ICPUPipelineLayout* const* const, asset::ICPUPipelineLayout* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IDriver::getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(asset::ICPURenderpassIndependentPipeline* const* const, asset::ICPURenderpassIndependentPipeline* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUComputePipeline> IDriver::getGPUObjectsFromAssets<asset::ICPUComputePipeline>(asset::ICPUComputePipeline* const* const, asset::ICPUComputePipeline* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSetLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(asset::ICPUDescriptorSetLayout* const* const, asset::ICPUDescriptorSetLayout* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSampler> IDriver::getGPUObjectsFromAssets<asset::ICPUSampler>(asset::ICPUSampler* const* const, asset::ICPUSampler* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSet> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(asset::ICPUDescriptorSet* const* const, asset::ICPUDescriptorSet* const* const, IGPUObjectFromAssetConverter* _converter);

template<typename AssetType>
created_gpu_object_array<AssetType> IDriver::getGPUObjectsFromAssets(const core::smart_refctd_ptr<AssetType>* _begin, const core::smart_refctd_ptr<AssetType>* _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if (!_converter)
        _converter = &def;
    return _converter->getGPUObjectsFromAssets<AssetType>(_begin, _end);
}

template created_gpu_object_array<asset::ICPUBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(const core::smart_refctd_ptr<asset::ICPUBuffer>*, const core::smart_refctd_ptr<asset::ICPUBuffer>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImage> IDriver::getGPUObjectsFromAssets<asset::ICPUImage>(const core::smart_refctd_ptr<asset::ICPUImage>*, const core::smart_refctd_ptr<asset::ICPUImage>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImageView> IDriver::getGPUObjectsFromAssets<asset::ICPUImageView>(const core::smart_refctd_ptr<asset::ICPUImageView>*, const core::smart_refctd_ptr<asset::ICPUImageView>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUShader> IDriver::getGPUObjectsFromAssets<asset::ICPUShader>(const core::smart_refctd_ptr<asset::ICPUShader>*, const core::smart_refctd_ptr<asset::ICPUShader>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSpecializedShader> IDriver::getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(const core::smart_refctd_ptr<asset::ICPUSpecializedShader>*, const core::smart_refctd_ptr<asset::ICPUSpecializedShader>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(const core::smart_refctd_ptr<asset::ICPUMeshBuffer>*, const core::smart_refctd_ptr<asset::ICPUMeshBuffer>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(const core::smart_refctd_ptr<asset::ICPUMesh>*, const core::smart_refctd_ptr<asset::ICPUMesh>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUPipelineLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(const core::smart_refctd_ptr<asset::ICPUPipelineLayout>*, const core::smart_refctd_ptr<asset::ICPUPipelineLayout>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IDriver::getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(const core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>*, const core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUComputePipeline> IDriver::getGPUObjectsFromAssets<asset::ICPUComputePipeline>(const core::smart_refctd_ptr<asset::ICPUComputePipeline>*, const core::smart_refctd_ptr<asset::ICPUComputePipeline>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSetLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(const core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>*, const core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSampler> IDriver::getGPUObjectsFromAssets<asset::ICPUSampler>(const core::smart_refctd_ptr<asset::ICPUSampler>*, const core::smart_refctd_ptr<asset::ICPUSampler>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSet> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(const core::smart_refctd_ptr<asset::ICPUDescriptorSet>*, const core::smart_refctd_ptr<asset::ICPUDescriptorSet>*, IGPUObjectFromAssetConverter* _converter);

}}

