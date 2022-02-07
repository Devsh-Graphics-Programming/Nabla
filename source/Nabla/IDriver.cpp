// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "IDriver.h"

#include "IrrlichtDevice.h"
#include "nbl/asset/IAssetManager.h"
#include "nbl/video/IGPUObjectFromAssetConverter.h"

namespace nbl
{
namespace video
{
template<class AssetType>
struct AssetBundleIterator
{
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;

    AssetBundleIterator(const core::smart_refctd_ptr<asset::IAsset>* _ptr)
        : ptr(_ptr) {}

    // general operators
    inline AssetType* operator*()
    {
        return static_cast<AssetType*>(ptr->get());
    }
    inline const AssetType* operator*() const
    {
        return static_cast<const AssetType*>(ptr->get());
    }
    inline const core::smart_refctd_ptr<asset::IAsset>* operator->() const
    {
        return ptr;
    }

    // arithmetic operators
    inline AssetBundleIterator<AssetType>& operator++()
    {
        ++ptr;
        return *this;
    }
    inline AssetBundleIterator<AssetType> operator++(int)
    {
        return AssetBundleIterator<AssetType>(ptr++);
    }
    inline difference_type operator-(const AssetBundleIterator<AssetType>& other) const
    {
        return ptr - other.ptr;
    }

    // comparison operators
    inline bool operator!=(const AssetBundleIterator<AssetType>& other) const
    {
        return ptr != other.ptr;
    }

private:
    const core::smart_refctd_ptr<asset::IAsset>* ptr;
};

//! Maybe we can reduce code duplication here some day
template<typename AssetType>
created_gpu_object_array<AssetType> IDriver::getGPUObjectsFromAssets(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>& _range, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if(!_converter)
        _converter = &def;
    AssetBundleIterator<AssetType> begin(_range.begin());
    AssetBundleIterator<AssetType> end(_range.end());
    return _converter->getGPUObjectsFromAssets<AssetType, const AssetBundleIterator<AssetType>>(begin, end);
}

template created_gpu_object_array<asset::ICPUBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImage> IDriver::getGPUObjectsFromAssets<asset::ICPUImage>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImageView> IDriver::getGPUObjectsFromAssets<asset::ICPUImageView>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUShader> IDriver::getGPUObjectsFromAssets<asset::ICPUShader>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSpecializedShader> IDriver::getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSkeleton> IDriver::getGPUObjectsFromAssets<asset::ICPUSkeleton>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUPipelineLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IDriver::getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUComputePipeline> IDriver::getGPUObjectsFromAssets<asset::ICPUComputePipeline>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSetLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSampler> IDriver::getGPUObjectsFromAssets<asset::ICPUSampler>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSet> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUAnimationLibrary> IDriver::getGPUObjectsFromAssets<asset::ICPUAnimationLibrary>(const core::SRange<const core::smart_refctd_ptr<asset::IAsset>>&, IGPUObjectFromAssetConverter* _converter);

template<typename AssetType>
created_gpu_object_array<AssetType> IDriver::getGPUObjectsFromAssets(const AssetType* const* const _begin, const AssetType* const* const _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if(!_converter)
        _converter = &def;
    return _converter->getGPUObjectsFromAssets<AssetType>(_begin, _end);
}

template created_gpu_object_array<asset::ICPUBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(const asset::ICPUBuffer* const* const, const asset::ICPUBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImage> IDriver::getGPUObjectsFromAssets<asset::ICPUImage>(const asset::ICPUImage* const* const, const asset::ICPUImage* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImageView> IDriver::getGPUObjectsFromAssets<asset::ICPUImageView>(const asset::ICPUImageView* const* const, const asset::ICPUImageView* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUShader> IDriver::getGPUObjectsFromAssets<asset::ICPUShader>(const asset::ICPUShader* const* const, const asset::ICPUShader* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSpecializedShader> IDriver::getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(const asset::ICPUSpecializedShader* const* const, const asset::ICPUSpecializedShader* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSkeleton> IDriver::getGPUObjectsFromAssets<asset::ICPUSkeleton>(const asset::ICPUSkeleton* const* const, const asset::ICPUSkeleton* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(const asset::ICPUMeshBuffer* const* const, const asset::ICPUMeshBuffer* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(const asset::ICPUMesh* const* const, const asset::ICPUMesh* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUPipelineLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(const asset::ICPUPipelineLayout* const* const, const asset::ICPUPipelineLayout* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IDriver::getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(const asset::ICPURenderpassIndependentPipeline* const* const, const asset::ICPURenderpassIndependentPipeline* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUComputePipeline> IDriver::getGPUObjectsFromAssets<asset::ICPUComputePipeline>(const asset::ICPUComputePipeline* const* const, const asset::ICPUComputePipeline* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSetLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(const asset::ICPUDescriptorSetLayout* const* const, const asset::ICPUDescriptorSetLayout* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSampler> IDriver::getGPUObjectsFromAssets<asset::ICPUSampler>(const asset::ICPUSampler* const* const, const asset::ICPUSampler* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSet> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(const asset::ICPUDescriptorSet* const* const, const asset::ICPUDescriptorSet* const* const, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUAnimationLibrary> IDriver::getGPUObjectsFromAssets<asset::ICPUAnimationLibrary>(const asset::ICPUAnimationLibrary* const* const, const asset::ICPUAnimationLibrary* const* const, IGPUObjectFromAssetConverter* _converter);

template<typename AssetType>
created_gpu_object_array<AssetType> IDriver::getGPUObjectsFromAssets(const core::smart_refctd_ptr<AssetType>* _begin, const core::smart_refctd_ptr<AssetType>* _end, IGPUObjectFromAssetConverter* _converter)
{
    IGPUObjectFromAssetConverter def(m_device->getAssetManager(), this);
    if(!_converter)
        _converter = &def;
    return _converter->getGPUObjectsFromAssets<AssetType>(_begin, _end);
}

template created_gpu_object_array<asset::ICPUBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUBuffer>(const core::smart_refctd_ptr<asset::ICPUBuffer>*, const core::smart_refctd_ptr<asset::ICPUBuffer>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImage> IDriver::getGPUObjectsFromAssets<asset::ICPUImage>(const core::smart_refctd_ptr<asset::ICPUImage>*, const core::smart_refctd_ptr<asset::ICPUImage>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUImageView> IDriver::getGPUObjectsFromAssets<asset::ICPUImageView>(const core::smart_refctd_ptr<asset::ICPUImageView>*, const core::smart_refctd_ptr<asset::ICPUImageView>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUShader> IDriver::getGPUObjectsFromAssets<asset::ICPUShader>(const core::smart_refctd_ptr<asset::ICPUShader>*, const core::smart_refctd_ptr<asset::ICPUShader>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSpecializedShader> IDriver::getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(const core::smart_refctd_ptr<asset::ICPUSpecializedShader>*, const core::smart_refctd_ptr<asset::ICPUSpecializedShader>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSkeleton> IDriver::getGPUObjectsFromAssets<asset::ICPUSkeleton>(const core::smart_refctd_ptr<asset::ICPUSkeleton>*, const core::smart_refctd_ptr<asset::ICPUSkeleton>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMeshBuffer> IDriver::getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(const core::smart_refctd_ptr<asset::ICPUMeshBuffer>*, const core::smart_refctd_ptr<asset::ICPUMeshBuffer>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUMesh> IDriver::getGPUObjectsFromAssets<asset::ICPUMesh>(const core::smart_refctd_ptr<asset::ICPUMesh>*, const core::smart_refctd_ptr<asset::ICPUMesh>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUPipelineLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(const core::smart_refctd_ptr<asset::ICPUPipelineLayout>*, const core::smart_refctd_ptr<asset::ICPUPipelineLayout>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IDriver::getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(const core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>*, const core::smart_refctd_ptr<asset::ICPURenderpassIndependentPipeline>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUComputePipeline> IDriver::getGPUObjectsFromAssets<asset::ICPUComputePipeline>(const core::smart_refctd_ptr<asset::ICPUComputePipeline>*, const core::smart_refctd_ptr<asset::ICPUComputePipeline>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSetLayout> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(const core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>*, const core::smart_refctd_ptr<asset::ICPUDescriptorSetLayout>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUSampler> IDriver::getGPUObjectsFromAssets<asset::ICPUSampler>(const core::smart_refctd_ptr<asset::ICPUSampler>*, const core::smart_refctd_ptr<asset::ICPUSampler>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUDescriptorSet> IDriver::getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(const core::smart_refctd_ptr<asset::ICPUDescriptorSet>*, const core::smart_refctd_ptr<asset::ICPUDescriptorSet>*, IGPUObjectFromAssetConverter* _converter);
template created_gpu_object_array<asset::ICPUAnimationLibrary> IDriver::getGPUObjectsFromAssets<asset::ICPUAnimationLibrary>(const core::smart_refctd_ptr<asset::ICPUAnimationLibrary>*, const core::smart_refctd_ptr<asset::ICPUAnimationLibrary>*, IGPUObjectFromAssetConverter* _converter);

// TODO: would be nice if something like this worked... eh might have to resort to a Macro
#if 0
template<typename AssetType>
struct SingleFunctionInstantiator
{
protected:
    static auto instantiate(IDriver* driver)
    {
        core::smart_refctd_ptr<AssetType> smartptr;

        created_gpu_object_array<AssetType> retval = driver->getGPUObjectsFromAssets<AssetType>(core::SRange<const core::smart_refctd_ptr<asset::IAsset>>(nullptr, nullptr), nullptr);
        retval = driver->getGPUObjectsFromAssets<AssetType>(&smartptr.get(), &smartptr.get(), nullptr);
        retval = driver->getGPUObjectsFromAssets<AssetType>(&smartptr, &smartptr, nullptr);
        return retval;
    }
};
template<typename... AssetTypes>
struct MultiFunctionInstantiator : public SingleFunctionInstantiator<AssetTypes>...
{
};

template MultiFunctionInstantiator<asset::ICPUBuffer,
    asset::ICPUShader, asset::ICPUImage, asset::ICPUSkeleton,
    asset::ICPUSpecializedShader, asset::ICPUImageView, asset::ICPUSampler,
    asset::ICPUDescriptorSetLayout,
    asset::ICPUDescriptorSet, asset::ICPUPipelineLayout,
    asset::ICPUComputePipeline, asset::ICPURenderpassIndependentPipeline,
    asset::ICPUMeshBuffer,
    asset::ICPUMesh
>;
#endif
}
}
