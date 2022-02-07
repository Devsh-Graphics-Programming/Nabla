// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// Do not include this in headers, please
#ifndef __NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED__

#include <iterator>

#include "nbl/core/core.h"
#include "nbl/asset/asset.h"

#include "IDriver.h"
#include "IDriverMemoryBacked.h"
#include "nbl/video/IGPUMesh.h"
#include "CLogger.h"
#include "nbl/video/asset_traits.h"
#include "nbl/core/alloc/LinearAddressAllocator.h"
#include "nbl/video/IGPUPipelineCache.h"

namespace nbl
{
namespace video
{
namespace impl
{
// non-pointer iterator type is AssetBundleIterator<> (see IDriver.cpp)
template<typename iterator_type>
inline constexpr bool is_const_iterator_v =
    (std::is_pointer_v<iterator_type> && is_pointer_to_const_object_v<iterator_type>) ||
    (!std::is_pointer_v<iterator_type> && std::is_const_v<iterator_type>);
}

class IGPUObjectFromAssetConverter
{
public:
    struct SParams
    {
        IGPUPipelineCache* pipelineCache = nullptr;
    };

protected:
    asset::IAssetManager* m_assetManager;
    video::IDriver* m_driver;

    template<typename AssetType, typename iterator_type>
    struct get_asset_raw_ptr
    {
        static inline AssetType* value(iterator_type it) { return static_cast<AssetType*>(it->get()); }
    };

    template<typename AssetType>
    struct get_asset_raw_ptr<AssetType, AssetType**>
    {
        static inline AssetType* value(AssetType** it) { return *it; }
    };
    template<typename AssetType>
    struct get_asset_raw_ptr<AssetType, AssetType* const*>
    {
        static inline AssetType* value(AssetType* const* it) { return *it; }
    };

    template<typename AssetType>
    struct Hash
    {
        inline std::size_t operator()(AssetType* asset) const
        {
            return std::hash<AssetType*>{}(asset);
        }
    };

    template<typename AssetType>
    struct KeyEqual
    {
        bool operator()(AssetType* lhs, AssetType* rhs) const { return lhs == rhs; }
    };

public:
    IGPUObjectFromAssetConverter(asset::IAssetManager* _assetMgr, video::IDriver* _drv)
        : m_assetManager{_assetMgr}, m_driver{_drv} {}

    virtual ~IGPUObjectFromAssetConverter() = default;

    inline virtual created_gpu_object_array<asset::ICPUBuffer> create(const asset::ICPUBuffer** const _begin, const asset::ICPUBuffer** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUSkeleton> create(const asset::ICPUSkeleton** const _begin, const asset::ICPUSkeleton** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUMeshBuffer> create(const asset::ICPUMeshBuffer** const _begin, const asset::ICPUMeshBuffer** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUMesh> create(const asset::ICPUMesh** const _begin, const asset::ICPUMesh** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUImage> create(const asset::ICPUImage** const _begin, const asset::ICPUImage** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUShader> create(const asset::ICPUShader** const _begin, const asset::ICPUShader** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUSpecializedShader> create(const asset::ICPUSpecializedShader** const _begin, const asset::ICPUSpecializedShader** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUBufferView> create(const asset::ICPUBufferView** const _begin, const asset::ICPUBufferView** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUDescriptorSetLayout> create(const asset::ICPUDescriptorSetLayout** const _begin, const asset::ICPUDescriptorSetLayout** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUSampler> create(const asset::ICPUSampler** const _begin, const asset::ICPUSampler** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUPipelineLayout> create(const asset::ICPUPipelineLayout** const _begin, const asset::ICPUPipelineLayout** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> create(const asset::ICPURenderpassIndependentPipeline** const _begin, const asset::ICPURenderpassIndependentPipeline** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUImageView> create(const asset::ICPUImageView** const _begin, const asset::ICPUImageView** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUDescriptorSet> create(const asset::ICPUDescriptorSet** const _begin, const asset::ICPUDescriptorSet** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUComputePipeline> create(const asset::ICPUComputePipeline** const _begin, const asset::ICPUComputePipeline** const _end, const SParams& _params);
    inline virtual created_gpu_object_array<asset::ICPUAnimationLibrary> create(const asset::ICPUAnimationLibrary** const _begin, const asset::ICPUAnimationLibrary** const _end, const SParams& _params);

    //! iterator_type is always either `[const] core::smart_refctd_ptr<AssetType>*[const]*` or `[const] AssetType*[const]*`
    template<typename AssetType, typename iterator_type>
    std::enable_if_t<!impl::is_const_iterator_v<iterator_type>, created_gpu_object_array<AssetType>>
    getGPUObjectsFromAssets(iterator_type _begin, iterator_type _end, const SParams& _params = {})
    {
        const auto assetCount = _end - _begin;
        auto res = core::make_refctd_dynamic_array<created_gpu_object_array<AssetType>>(assetCount);

        core::vector<AssetType*> notFound;
        notFound.reserve(assetCount);
        core::vector<size_t> pos;
        pos.reserve(assetCount);

        for(iterator_type it = _begin; it != _end; it++)
        {
            const auto index = it - _begin;

            //if (*it)
            //{
            auto gpu = m_assetManager->findGPUObject(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
            if(!gpu)
            {
                if((*it)->isADummyObjectForCache())
                    notFound.push_back(nullptr);
                else
                    notFound.push_back(get_asset_raw_ptr<AssetType, iterator_type>::value(it));
                pos.push_back(index);
            }
            else
                res->operator[](index) = core::move_and_dynamic_cast<typename video::asset_traits<AssetType>::GPUObjectType>(gpu);
            //}
            //res->operator[](index) = nullptr;
        }

        if(notFound.size())
        {
            decltype(res) created = create(const_cast<const AssetType**>(notFound.data()), const_cast<const AssetType**>(notFound.data() + notFound.size()), _params);
            for(size_t i = 0u; i < created->size(); ++i)
            {
                auto& input = created->operator[](i);
                handleGPUObjCaching(notFound[i], input);
                res->operator[](pos[i]) = std::move(input);  // ok to move because the `created` array will die after the next scope
            }
        }

        return res;
    }
    template<typename AssetType, typename const_iterator_type>
    std::enable_if_t<impl::is_const_iterator_v<const_iterator_type>, created_gpu_object_array<AssetType>>
    getGPUObjectsFromAssets(const_iterator_type _begin, const_iterator_type _end, const SParams& _params = {})
    {
        if constexpr(std::is_pointer_v<const_iterator_type>)
        {
            using iterator_type = pointer_to_nonconst_object_t<const_iterator_type>;

            auto begin = const_cast<iterator_type>(_begin);
            auto end = const_cast<iterator_type>(_end);

            return getGPUObjectsFromAssets<AssetType, iterator_type>(begin, end, _params);
        }
        else
        {
            using iterator_type = std::remove_const_t<const_iterator_type>;

            iterator_type& begin = const_cast<iterator_type&>(_begin);
            iterator_type& end = const_cast<iterator_type&>(_end);

            return getGPUObjectsFromAssets<AssetType, iterator_type>(begin, end, _params);
        }
    }

protected:
    virtual inline void handleGPUObjCaching(asset::IAsset* _asset, const core::smart_refctd_ptr<core::IReferenceCounted>& _gpuobj)
    {
        if(_asset && _gpuobj)
            m_assetManager->convertAssetToEmptyCacheHandle(_asset, core::smart_refctd_ptr(_gpuobj));
    }

    //! TODO: Make this faster and not call any allocator
    template<typename T>
    static inline core::vector<size_t> eliminateDuplicatesAndGenRedirs(core::vector<T*>& _input)
    {
        core::vector<size_t> redirs;

        core::unordered_map<T*, size_t, Hash<T>, KeyEqual<T>> firstOccur;
        size_t i = 0u;
        for(T* el : _input)
        {
            if(!el)
            {
                redirs.push_back(0xdeadbeefu);
                continue;
            }

            auto r = firstOccur.insert({el, i});
            redirs.push_back(r.first->second);
            if(r.second)
                ++i;
        }
        for(const auto& p : firstOccur)
            _input.push_back(p.first);
        _input.erase(_input.begin(), _input.begin() + (_input.size() - firstOccur.size()));
        std::sort(_input.begin(), _input.end(), [&firstOccur](T* a, T* b) { return firstOccur[a] < firstOccur[b]; });

        return redirs;
    }
};

class CAssetPreservingGPUObjectFromAssetConverter : public IGPUObjectFromAssetConverter
{
public:
    using IGPUObjectFromAssetConverter::IGPUObjectFromAssetConverter;

protected:
    virtual inline void handleGPUObjCaching(asset::IAsset* _asset, const core::smart_refctd_ptr<core::IReferenceCounted>& _gpuobj) override
    {
        if(_asset && _gpuobj)
            m_assetManager->insertGPUObjectIntoCache(_asset, core::smart_refctd_ptr(_gpuobj));
    }
};

// need to specialize outside because of GCC
template<>
struct IGPUObjectFromAssetConverter::Hash<const asset::ICPURenderpassIndependentPipeline>
{
    inline std::size_t operator()(const asset::ICPURenderpassIndependentPipeline* _ppln) const
    {
        constexpr size_t bytesToHash =
            asset::SVertexInputParams::serializedSize() +
            asset::SBlendParams::serializedSize() +
            asset::SRasterizationParams::serializedSize() +
            asset::SPrimitiveAssemblyParams::serializedSize() +
            sizeof(void*) * asset::ICPURenderpassIndependentPipeline::SHADER_STAGE_COUNT +  //shaders
            sizeof(void*);  //layout
        uint8_t mem[bytesToHash]{};
        uint32_t offset = 0u;
        _ppln->getVertexInputParams().serialize(mem + offset);
        offset += asset::SVertexInputParams::serializedSize();
        _ppln->getBlendParams().serialize(mem + offset);
        offset += asset::SBlendParams::serializedSize();
        _ppln->getRasterizationParams().serialize(mem + offset);
        offset += sizeof(asset::SRasterizationParams);
        _ppln->getPrimitiveAssemblyParams().serialize(mem + offset);
        offset += sizeof(asset::SPrimitiveAssemblyParams);
        const asset::ICPUSpecializedShader** shaders = reinterpret_cast<const asset::ICPUSpecializedShader**>(mem + offset);
        for(uint32_t i = 0u; i < asset::ICPURenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++i)
            shaders[i] = _ppln->getShaderAtIndex(i);
        offset += asset::ICPURenderpassIndependentPipeline::SHADER_STAGE_COUNT * sizeof(void*);
        reinterpret_cast<const asset::ICPUPipelineLayout**>(mem + offset)[0] = _ppln->getLayout();

        const std::size_t hs = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(mem), bytesToHash));

        return hs;
    }
};
template<>
struct IGPUObjectFromAssetConverter::Hash<const asset::ICPUComputePipeline>
{
    inline std::size_t operator()(const asset::ICPUComputePipeline* _ppln) const
    {
        constexpr size_t bytesToHash =
            sizeof(void*) +  //shader
            sizeof(void*);  //layout
        uint8_t mem[bytesToHash]{};

        reinterpret_cast<const asset::ICPUSpecializedShader**>(mem)[0] = _ppln->getShader();
        reinterpret_cast<const asset::ICPUPipelineLayout**>(mem + sizeof(void*))[0] = _ppln->getLayout();

        const std::size_t hs = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(mem), bytesToHash));

        return hs;
    }
};

template<>
struct IGPUObjectFromAssetConverter::KeyEqual<const asset::ICPURenderpassIndependentPipeline>
{
    //equality depends on hash only
    bool operator()(const asset::ICPURenderpassIndependentPipeline* lhs, const asset::ICPURenderpassIndependentPipeline* rhs) const { return true; }
};
template<>
struct IGPUObjectFromAssetConverter::KeyEqual<const asset::ICPUComputePipeline>
{
    //equality depends on hash only
    bool operator()(const asset::ICPUComputePipeline* lhs, const asset::ICPUComputePipeline* rhs) const { return true; }
};

auto IGPUObjectFromAssetConverter::create(const asset::ICPUBuffer** const _begin, const asset::ICPUBuffer** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUBuffer>  // TODO: improve for caches of very large buffers!!!
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBuffer>>(assetCount);

    const uint64_t alignment =
        std::max<uint64_t>(
            std::max<uint64_t>(m_driver->getRequiredTBOAlignment(), m_driver->getRequiredUBOAlignment()),
            std::max<uint64_t>(m_driver->getRequiredSSBOAlignment(), _NBL_SIMD_ALIGNMENT));

    auto reqs = m_driver->getDeviceLocalGPUMemoryReqs();
    reqs.vulkanReqs.alignment = alignment;
    const uint64_t maxBufferSize = m_driver->getMaxBufferSize();
    auto out = res->begin();
    auto firstInBlock = out;
    auto newBlock = [&]() -> auto
    {
        return core::LinearAddressAllocator<uint64_t>(nullptr, 0u, 0u, alignment, maxBufferSize);
    };
    auto addrAllctr = newBlock();
    auto finalizeBlock = [&]() -> void {
        reqs.vulkanReqs.size = addrAllctr.get_allocated_size();
        if(reqs.vulkanReqs.size == 0u)
            return;

        auto gpubuffer = m_driver->createGPUBufferOnDedMem(reqs);
        for(auto it = firstInBlock; it != out; it++)
            if(auto output = *it)
            {
                auto cpubuffer = _begin[std::distance(res->begin(), it)];
                m_driver->updateBufferRangeViaStagingBuffer(gpubuffer.get(), output->getOffset(), cpubuffer->getSize(), cpubuffer->getPointer());
                output->setBuffer(core::smart_refctd_ptr(gpubuffer));
            }
    };
    for(auto it = _begin; it != _end; it++, out++)
    {
        auto cpubuffer = *it;
        if(cpubuffer->getSize() > maxBufferSize)
            continue;

        uint64_t addr = addrAllctr.alloc_addr(cpubuffer->getSize(), alignment);
        if(addr == decltype(addrAllctr)::invalid_address)
        {
            finalizeBlock();
            firstInBlock = out;
            addrAllctr = newBlock();
            addr = addrAllctr.alloc_addr(cpubuffer->getSize(), alignment);
        }
        assert(addr != decltype(addrAllctr)::invalid_address);
        *out = core::make_smart_refctd_ptr<typename video::asset_traits<asset::ICPUBuffer>::GPUObjectType>(addr);
    }
    finalizeBlock();

    return res;
}
namespace impl
{
template<typename MapIterator>
struct CustomBoneNameIterator
{
    inline CustomBoneNameIterator(const MapIterator& it)
        : m_it(it) {}
    inline CustomBoneNameIterator(MapIterator&& it)
        : m_it(std::move(it)) {}

    inline bool operator!=(const CustomBoneNameIterator<MapIterator>& other) const
    {
        return m_it != other.m_it;
    }

    inline CustomBoneNameIterator<MapIterator>& operator++()
    {
        ++m_it;
        return *this;
    }
    inline CustomBoneNameIterator<MapIterator> operator++(int)
    {
        return m_it++;
    }

    inline const auto& operator*() const
    {
        return m_it->first;
    }
    inline auto& operator*()
    {
        return m_it->first;
    }

    using iterator_category = typename std::iterator_traits<MapIterator>::iterator_category;
    using difference_type = typename std::iterator_traits<MapIterator>::difference_type;
    using value_type = const char*;
    using reference = std::add_lvalue_reference_t<value_type>;
    using pointer = std::add_pointer_t<value_type>;

private:
    MapIterator m_it;
};
}
auto IGPUObjectFromAssetConverter::create(const asset::ICPUSkeleton** _begin, const asset::ICPUSkeleton** _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUSkeleton>
{
    const size_t assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSkeleton>>(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(assetCount * 2u);

    for(ptrdiff_t i = 0u; i < assetCount; i++)
    {
        const asset::ICPUSkeleton* cpusk = _begin[i];
        cpuBuffers.push_back(cpusk->getParentJointIDBinding().buffer.get());
        if(const asset::ICPUBuffer* buf = cpusk->getDefaultTransformBinding().buffer.get())
            cpuBuffers.push_back(buf);
    }

    using redirs_t = core::vector<size_t>;
    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);

    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data() + cpuBuffers.size(), _params);

    size_t bufIter = 0ull;
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUSkeleton* cpusk = _begin[i];

        asset::SBufferBinding<IGPUBuffer> parentJointIDBinding;
        {
            parentJointIDBinding.offset = cpusk->getParentJointIDBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            parentJointIDBinding.offset += gpubuf->getOffset();
            parentJointIDBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
        asset::SBufferBinding<IGPUBuffer> defaultTransformBinding;
        if(cpusk->getDefaultTransformBinding().buffer)
        {
            defaultTransformBinding.offset = cpusk->getDefaultTransformBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            defaultTransformBinding.offset += gpubuf->getOffset();
            defaultTransformBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        (*res)[i] = core::make_smart_refctd_ptr<IGPUSkeleton>(std::move(parentJointIDBinding), std::move(defaultTransformBinding), cpusk->getJointNameToIDMap());
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(const asset::ICPUMeshBuffer** _begin, const asset::ICPUMeshBuffer** _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUMeshBuffer>
{
    const size_t assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMeshBuffer>>(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(assetCount * (asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT + 1u));
    core::vector<const asset::ICPUSkeleton*> cpuSkeletons;
    cpuSkeletons.reserve(assetCount);
    core::vector<const asset::ICPUDescriptorSet*> cpuDescSets;
    cpuDescSets.reserve(assetCount);
    core::vector<const asset::ICPURenderpassIndependentPipeline*> cpuPipelines;
    cpuPipelines.reserve(assetCount);

    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUMeshBuffer* cpumb = _begin[i];

        if(cpumb->getPipeline())
            cpuPipelines.push_back(cpumb->getPipeline());
        if(cpumb->getAttachedDescriptorSet())
            cpuDescSets.push_back(cpumb->getAttachedDescriptorSet());

        if(const asset::ICPUBuffer* buf = cpumb->getInverseBindPoseBufferBinding().buffer.get())
            cpuBuffers.push_back(buf);
        if(const asset::ICPUBuffer* buf = cpumb->getJointAABBBufferBinding().buffer.get())
            cpuBuffers.push_back(buf);
        if(cpumb->getSkeleton())
            cpuSkeletons.push_back(cpumb->getSkeleton());

        for(size_t b = 0ull; b < asset::ICPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            if(const asset::ICPUBuffer* buf = cpumb->getVertexBufferBindings()[b].buffer.get())
                cpuBuffers.push_back(buf);
        }
        if(const asset::ICPUBuffer* buf = cpumb->getIndexBufferBinding().buffer.get())
            cpuBuffers.push_back(buf);
    }

    using redirs_t = core::vector<size_t>;

    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
    redirs_t skelRedirs = eliminateDuplicatesAndGenRedirs(cpuSkeletons);
    redirs_t dsRedirs = eliminateDuplicatesAndGenRedirs(cpuDescSets);
    redirs_t pplnRedirs = eliminateDuplicatesAndGenRedirs(cpuPipelines);

    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data() + cpuBuffers.size(), _params);
    auto gpuSkeletons = getGPUObjectsFromAssets<asset::ICPUSkeleton>(cpuSkeletons.data(), cpuSkeletons.data() + cpuSkeletons.size(), _params);
    auto gpuDescSets = getGPUObjectsFromAssets<asset::ICPUDescriptorSet>(cpuDescSets.data(), cpuDescSets.data() + cpuDescSets.size(), _params);
    auto gpuPipelines = getGPUObjectsFromAssets<asset::ICPURenderpassIndependentPipeline>(cpuPipelines.data(), cpuPipelines.data() + cpuPipelines.size(), _params);

    size_t pplnIter = 0ull, dsIter = 0ull, skelIter = 0ull, bufIter = 0ull;
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUMeshBuffer* cpumb = _begin[i];

        IGPURenderpassIndependentPipeline* gpuppln = nullptr;
        if(cpumb->getPipeline())
            gpuppln = (*gpuPipelines)[pplnRedirs[pplnIter++]].get();
        IGPUDescriptorSet* gpuds = nullptr;
        if(cpumb->getAttachedDescriptorSet())
            gpuds = (*gpuDescSets)[dsRedirs[dsIter++]].get();

        asset::SBufferBinding<IGPUBuffer> invBindPoseBinding;
        if(cpumb->getInverseBindPoseBufferBinding().buffer)
        {
            invBindPoseBinding.offset = cpumb->getInverseBindPoseBufferBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            invBindPoseBinding.offset += gpubuf->getOffset();
            invBindPoseBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
        asset::SBufferBinding<IGPUBuffer> jointAABBBinding;
        if(cpumb->getJointAABBBufferBinding().buffer)
        {
            jointAABBBinding.offset = cpumb->getJointAABBBufferBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            jointAABBBinding.offset += gpubuf->getOffset();
            jointAABBBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
        IGPUSkeleton* gpuskel = nullptr;
        if(cpumb->getSkeleton())
            gpuskel = (*gpuSkeletons)[skelRedirs[skelIter++]].get();

        asset::SBufferBinding<IGPUBuffer> vtxBindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
        for(size_t b = 0ull; b < IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT; ++b)
        {
            const auto& cpubnd = cpumb->getVertexBufferBindings()[b];
            if(cpubnd.buffer)
            {
                vtxBindings[b].offset = cpubnd.offset;
                auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
                vtxBindings[b].offset += gpubuf->getOffset();
                vtxBindings[b].buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
            }
        }

        asset::SBufferBinding<IGPUBuffer> idxBinding;
        if(cpumb->getIndexBufferBinding().buffer)
        {
            idxBinding.offset = cpumb->getIndexBufferBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            idxBinding.offset += gpubuf->getOffset();
            idxBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> gpuppln_(gpuppln);
        core::smart_refctd_ptr<IGPUDescriptorSet> gpuds_(gpuds);
        core::smart_refctd_ptr<IGPUSkeleton> gpuskel_(gpuskel);
        (*res)[i] = core::make_smart_refctd_ptr<IGPUMeshBuffer>(std::move(gpuppln_), std::move(gpuds_), vtxBindings, std::move(idxBinding));
        (*res)[i]->setBoundingBox(cpumb->getBoundingBox());
        memcpy((*res)[i]->getPushConstantsDataPtr(), _begin[i]->getPushConstantsDataPtr(), IGPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE);
        (*res)[i]->setSkin(std::move(invBindPoseBinding), std::move(jointAABBBinding), std::move(gpuskel_), core::min(cpumb->getMaxJointsPerVertex(), cpumb->deduceMaxJointsPerVertex()));
        (*res)[i]->setBaseInstance(_begin[i]->getBaseInstance());
        (*res)[i]->setBaseVertex(_begin[i]->getBaseVertex());
        (*res)[i]->setIndexCount(_begin[i]->getIndexCount());
        (*res)[i]->setInstanceCount(_begin[i]->getInstanceCount());
        (*res)[i]->setIndexType(_begin[i]->getIndexType());
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(const asset::ICPUMesh** const _begin, const asset::ICPUMesh** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUMesh>
{
    const size_t assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUMesh>>(assetCount);

    core::vector<const asset::ICPUMeshBuffer*> cpuDeps;
    cpuDeps.reserve(assetCount);
    for(auto i = 0u; i < assetCount; i++)
    {
        auto* _asset = _begin[i];
        for(auto mesh : _asset->getMeshBuffers())
            cpuDeps.emplace_back(mesh);
    }

    core::vector<size_t> redir = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUMeshBuffer>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);
    for(size_t i = 0u, j = 0u; i < assetCount; ++i)
    {
        auto* _asset = _begin[i];
        auto cpuMeshBuffers = _asset->getMeshBuffers();

        auto& output = res->operator[](i);
        output = core::make_smart_refctd_ptr<video::IGPUMesh>(cpuMeshBuffers.size());
        output->setBoundingBox(_asset->getBoundingBox());

        auto gpuMeshBuffersIt = output->getMeshBufferIterator();
        for(auto mesh : cpuMeshBuffers)
        {
            *(gpuMeshBuffersIt++) = core::smart_refctd_ptr(gpuDeps->operator[](redir[j]));
            ++j;
        }
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUImage** const _begin, const asset::ICPUImage** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUImage>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImage>>(assetCount);

    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUImage* cpuimg = _begin[i];
        asset::IImage::SCreationParams params = cpuimg->getCreationParameters();
        const bool integerFmt = asset::isIntegerFormat(params.format);
        if(!integerFmt)
            params.mipLevels = 1u + static_cast<uint32_t>(std::log2(static_cast<float>(core::max<uint32_t>(core::max<uint32_t>(params.extent.width, params.extent.height), params.extent.depth))));
        auto gpuimg = m_driver->createDeviceLocalGPUImageOnDedMem(std::move(params));

        auto regions = cpuimg->getRegions();
        auto count = regions.size();
        if(count)
        {
            // TODO: @Criss why isn't this buffer caches and why are we not going through recursive asset creation and getting ICPUBuffer equivalents? (we can always discard/not cache the GPU Buffers created only for image data upload)
            auto tmpBuff = m_driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuimg->getBuffer()->getSize(), cpuimg->getBuffer()->getPointer());
            m_driver->copyBufferToImage(tmpBuff.get(), gpuimg.get(), count, regions.begin());
            if(!integerFmt)
            {
                uint32_t lowestPresentMip = 1u;
                while(cpuimg->getRegions(lowestPresentMip).size())
                    lowestPresentMip++;
                // generate temporary image view to make sure we don't screw up any explicit mip levels
                IGPUImageView::SCreationParams tmpViewParams;
                tmpViewParams.subresourceRange.levelCount = params.mipLevels + 1u - lowestPresentMip;
                // if not all mip levels have been manually specified
                if(tmpViewParams.subresourceRange.levelCount > 1u)
                {
                    tmpViewParams.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
                    tmpViewParams.image = core::smart_refctd_ptr(gpuimg);
                    switch(params.type)
                    {
                        case asset::IImage::ET_1D:
                            tmpViewParams.viewType = IGPUImageView::ET_1D_ARRAY;
                            break;
                        case asset::IImage::ET_2D:
                            if(params.flags & asset::IImage::ECF_CUBE_COMPATIBLE_BIT)
                                tmpViewParams.viewType = IGPUImageView::ET_CUBE_MAP_ARRAY;
                            else
                                tmpViewParams.viewType = IGPUImageView::ET_2D_ARRAY;
                            break;
                        case asset::IImage::ET_3D:
                            tmpViewParams.viewType = IGPUImageView::ET_3D;
                            break;
                        default:
                            assert(false);
                            break;
                    }
                    tmpViewParams.format = params.format;
                    //tmpViewParams.subresourceRange.aspectMask
                    tmpViewParams.subresourceRange.baseMipLevel = lowestPresentMip - 1u;
                    tmpViewParams.subresourceRange.layerCount = params.arrayLayers;
                    auto tmpView = m_driver->createGPUImageView(std::move(tmpViewParams));
                    // deprecated OpenGL path (do with compute shader in the future)
                    tmpView->regenerateMipMapLevels();
                }
            }
        }

        res->operator[](i) = std::move(gpuimg);
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUShader** const _begin, const asset::ICPUShader** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUShader>>(assetCount);

    for(ptrdiff_t i = 0u; i < assetCount; ++i)
        res->operator[](i) = m_driver->createGPUShader(core::smart_refctd_ptr<const asset::ICPUShader>(_begin[i]));

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUSpecializedShader** const _begin, const asset::ICPUSpecializedShader** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUSpecializedShader>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSpecializedShader>>(assetCount);

    core::vector<const asset::ICPUShader*> cpuDeps;
    cpuDeps.reserve(res->size());

    const asset::ICPUSpecializedShader** it = _begin;
    while(it != _end)
    {
        cpuDeps.push_back((*it)->getUnspecialized());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUShader>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);

    for(ptrdiff_t i = 0; i < assetCount; ++i)
    {
        res->operator[](i) = m_driver->createGPUSpecializedShader(gpuDeps->operator[](redirs[i]).get(), _begin[i]->getSpecializationInfo());
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUBufferView** const _begin, const asset::ICPUBufferView** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUBufferView>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUBufferView>>(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBufs(assetCount, nullptr);
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
        cpuBufs[i] = _begin[i]->getUnderlyingBuffer();

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufs);
    auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufs.data(), cpuBufs.data() + cpuBufs.size(), _params);

    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUBufferView* cpubufview = _begin[i];
        IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[i]].get();
        (*res)[i] = m_driver->createGPUBufferView(gpubuf->getBuffer(), cpubufview->getFormat(), gpubuf->getOffset() + cpubufview->getOffsetInBuffer(), cpubufview->getByteSize());
        ;
    }

    return res;
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUDescriptorSetLayout** const _begin, const asset::ICPUDescriptorSetLayout** const _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUDescriptorSetLayout>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSetLayout>>(assetCount);

    core::vector<asset::ICPUSampler*> cpuSamplers;  //immutable samplers
    size_t maxSamplers = 0ull;
    size_t maxBindingsPerDescSet = 0ull;
    size_t maxSamplersPerDescSet = 0u;
    for(auto dsl : core::SRange<const asset::ICPUDescriptorSetLayout*>(_begin, _end))
    {
        size_t samplersInDS = 0u;
        for(const auto& bnd : dsl->getBindings())
        {
            const uint32_t samplerCnt = bnd.samplers ? bnd.count : 0u;
            maxSamplers += samplerCnt;
            samplersInDS += samplerCnt;
        }
        maxBindingsPerDescSet = core::max<size_t>(maxBindingsPerDescSet, dsl->getBindings().size());
        maxSamplersPerDescSet = core::max<size_t>(maxSamplersPerDescSet, samplersInDS);
    }
    cpuSamplers.reserve(maxSamplers);

    for(auto dsl : core::SRange<const asset::ICPUDescriptorSetLayout*>(_begin, _end))
    {
        for(const auto& bnd : dsl->getBindings())
        {
            if(bnd.samplers)
            {
                for(uint32_t i = 0u; i < bnd.count; ++i)
                    cpuSamplers.push_back(bnd.samplers[i].get());
            }
        }
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data() + cpuSamplers.size(), _params);
    size_t gpuSmplrIter = 0ull;

    using gpu_bindings_array_t = core::smart_refctd_dynamic_array<IGPUDescriptorSetLayout::SBinding>;
    auto tmpBindings = core::make_refctd_dynamic_array<gpu_bindings_array_t>(maxBindingsPerDescSet);
    using samplers_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUSampler>>;
    auto tmpSamplers = core::make_refctd_dynamic_array<samplers_array_t>(maxSamplersPerDescSet * maxBindingsPerDescSet);
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        core::smart_refctd_ptr<IGPUSampler>* smplr_ptr = tmpSamplers->data();
        const asset::ICPUDescriptorSetLayout* cpudsl = _begin[i];
        size_t bndIter = 0ull;
        for(const auto& bnd : cpudsl->getBindings())
        {
            IGPUDescriptorSetLayout::SBinding gpubnd;
            gpubnd.binding = bnd.binding;
            gpubnd.type = bnd.type;
            gpubnd.count = bnd.count;
            gpubnd.stageFlags = bnd.stageFlags;
            gpubnd.samplers = nullptr;

            if(bnd.samplers)
            {
                for(uint32_t s = 0u; s < gpubnd.count; ++s)
                    smplr_ptr[s] = (*gpuSamplers)[redirs[gpuSmplrIter++]];
                gpubnd.samplers = smplr_ptr;
                smplr_ptr += gpubnd.count;
            }
            (*tmpBindings)[bndIter++] = gpubnd;
        }
        (*res)[i] = m_driver->createGPUDescriptorSetLayout((*tmpBindings).data(), (*tmpBindings).data() + bndIter);
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUSampler> IGPUObjectFromAssetConverter::create(const asset::ICPUSampler** const _begin, const asset::ICPUSampler** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUSampler>>(assetCount);

    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUSampler* cpusmplr = _begin[i];
        res->operator[](i) = m_driver->createGPUSampler(cpusmplr->getParams());
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUPipelineLayout> IGPUObjectFromAssetConverter::create(const asset::ICPUPipelineLayout** const _begin, const asset::ICPUPipelineLayout** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUPipelineLayout>>(assetCount);

    // TODO: Deal with duplication of layouts and any other resource that can be present at different resource tree levels
    // SOLUTION: a `creationCache` object as the last parameter to the `create` function
    core::vector<const asset::ICPUDescriptorSetLayout*> cpuDSLayouts;
    cpuDSLayouts.reserve(assetCount * asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT);

    for(const asset::ICPUPipelineLayout* pl : core::SRange<const asset::ICPUPipelineLayout*>(_begin, _end))
    {
        for(uint32_t ds = 0u; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
            if(pl->getDescriptorSetLayout(ds))
                cpuDSLayouts.push_back(pl->getDescriptorSetLayout(ds));
    }
    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDSLayouts);

    auto gpuDSLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuDSLayouts.data(), cpuDSLayouts.data() + cpuDSLayouts.size(), _params);

    size_t dslIter = 0ull;
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUPipelineLayout* cpupl = _begin[i];
        IGPUDescriptorSetLayout* dsLayouts[asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT]{};
        for(size_t ds = 0ull; ds < asset::ICPUPipelineLayout::DESCRIPTOR_SET_COUNT; ++ds)
        {
            if(cpupl->getDescriptorSetLayout(ds))
                dsLayouts[ds] = (*gpuDSLayouts)[redirs[dslIter++]].get();
        }
        res->operator[](i) = m_driver->createGPUPipelineLayout(
            cpupl->getPushConstantRanges().begin(), cpupl->getPushConstantRanges().end(),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[0]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[1]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[2]),
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>(dsLayouts[3]));
    }

    return res;
}

inline created_gpu_object_array<asset::ICPURenderpassIndependentPipeline> IGPUObjectFromAssetConverter::create(const asset::ICPURenderpassIndependentPipeline** const _begin, const asset::ICPURenderpassIndependentPipeline** const _end, const SParams& _params)
{
    constexpr size_t SHADER_STAGE_COUNT = asset::ICPURenderpassIndependentPipeline::SHADER_STAGE_COUNT;

    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPURenderpassIndependentPipeline>>(assetCount);

    core::vector<const asset::ICPUPipelineLayout*> cpuLayouts;
    cpuLayouts.reserve(assetCount);
    core::vector<const asset::ICPUSpecializedShader*> cpuShaders;
    cpuShaders.reserve(assetCount * SHADER_STAGE_COUNT);

    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];
        cpuLayouts.push_back(cpuppln->getLayout());

        for(size_t s = 0ull; s < SHADER_STAGE_COUNT; ++s)
            if(const asset::ICPUSpecializedShader* shdr = cpuppln->getShaderAtIndex(static_cast<asset::ICPURenderpassIndependentPipeline::E_SHADER_STAGE_IX>(s)))
                cpuShaders.push_back(shdr);
    }

    core::vector<size_t> layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    core::vector<size_t> shdrRedirs = eliminateDuplicatesAndGenRedirs(cpuShaders);

    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data() + cpuLayouts.size(), _params);
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data() + cpuShaders.size(), _params);

    size_t shdrIter = 0ull;
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPURenderpassIndependentPipeline* cpuppln = _begin[i];

        IGPUPipelineLayout* layout = (*gpuLayouts)[layoutRedirs[i]].get();

        IGPUSpecializedShader* shaders[SHADER_STAGE_COUNT]{};
        size_t local_shdr_count = 0ull;
        for(size_t s = 0ull; s < SHADER_STAGE_COUNT; ++s)
            if(cpuppln->getShaderAtIndex(static_cast<asset::ICPURenderpassIndependentPipeline::E_SHADER_STAGE_IX>(s)))
                shaders[local_shdr_count++] = (*gpuShaders)[shdrRedirs[shdrIter++]].get();

        (*res)[i] = m_driver->createGPURenderpassIndependentPipeline(
            _params.pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>(layout),
            shaders, shaders + local_shdr_count,
            cpuppln->getVertexInputParams(),
            cpuppln->getBlendParams(),
            cpuppln->getPrimitiveAssemblyParams(),
            cpuppln->getRasterizationParams());
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUImageView> IGPUObjectFromAssetConverter::create(const asset::ICPUImageView** const _begin, const asset::ICPUImageView** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImageView>>(assetCount);

    core::vector<asset::ICPUImage*> cpuDeps;
    cpuDeps.reserve(res->size());

    const asset::ICPUImageView** it = _begin;
    while(it != _end)
    {
        cpuDeps.push_back((*it)->getCreationParameters().image.get());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);
    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUImage>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);

    for(ptrdiff_t i = 0; i < assetCount; ++i)
    {
        const asset::ICPUImageView::SCreationParams& cpuparams = _begin[i]->getCreationParameters();
        IGPUImageView::SCreationParams params;
        memcpy(&params.components, &cpuparams.components, sizeof(params.components));
        params.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(cpuparams.flags);
        params.format = cpuparams.format;
        params.subresourceRange = cpuparams.subresourceRange;
        params.subresourceRange.levelCount = (*gpuDeps)[redirs[i]]->getCreationParameters().mipLevels - params.subresourceRange.baseMipLevel;
        params.viewType = static_cast<IGPUImageView::E_TYPE>(cpuparams.viewType);
        params.image = (*gpuDeps)[redirs[i]];
        (*res)[i] = m_driver->createGPUImageView(std::move(params));
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUDescriptorSet> IGPUObjectFromAssetConverter::create(const asset::ICPUDescriptorSet** const _begin, const asset::ICPUDescriptorSet** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUDescriptorSet>>(assetCount);

    struct BindingDescTypePair_t
    {
        uint32_t binding;
        asset::E_DESCRIPTOR_TYPE descType;
        size_t count;
    };
    auto isBufferDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        using namespace asset;
        switch(t)
        {
            case EDT_UNIFORM_BUFFER: [[fallthrough]];
            case EDT_STORAGE_BUFFER: [[fallthrough]];
            case EDT_UNIFORM_BUFFER_DYNAMIC: [[fallthrough]];
            case EDT_STORAGE_BUFFER_DYNAMIC:
                return true;
                break;
            default:
                return false;
                break;
        }
    };
    auto isBufviewDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        using namespace asset;
        return t == EDT_STORAGE_TEXEL_BUFFER || t == EDT_STORAGE_TEXEL_BUFFER;
    };
    auto isSampledImgViewDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        return t == asset::EDT_COMBINED_IMAGE_SAMPLER;
    };
    auto isStorageImgDesc = [](asset::E_DESCRIPTOR_TYPE t) {
        return t == asset::EDT_STORAGE_IMAGE;
    };

    // TODO: Deal with duplication of layouts and any other resource that can be present at different resource tree levels
    core::vector<const asset::ICPUDescriptorSetLayout*> cpuLayouts;
    cpuLayouts.reserve(assetCount);
    uint32_t maxWriteCount = 0ull;
    uint32_t descCount = 0ull;
    uint32_t bufCount = 0ull;
    uint32_t bufviewCount = 0ull;
    uint32_t sampledImgViewCount = 0ull;
    uint32_t storageImgViewCount = 0ull;
    for(ptrdiff_t i = 0u; i < assetCount; i++)
    {
        const asset::ICPUDescriptorSet* cpuds = _begin[i];
        cpuLayouts.push_back(cpuds->getLayout());

        for(auto j = 0u; j <= cpuds->getMaxDescriptorBindingIndex(); j++)
        {
            const uint32_t cnt = cpuds->getDescriptors(j).size();
            if(cnt)
                maxWriteCount++;
            descCount += cnt;

            const auto type = cpuds->getDescriptorsType(j);
            if(isBufferDesc(type))
                bufCount += cnt;
            else if(isBufviewDesc(type))
                bufviewCount += cnt;
            else if(isSampledImgViewDesc(type))
                sampledImgViewCount += cnt;
            else if(isStorageImgDesc(type))
                storageImgViewCount += cnt;
        }
    }

    core::vector<asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(bufCount);
    core::vector<asset::ICPUBufferView*> cpuBufviews;
    cpuBufviews.reserve(bufviewCount);
    core::vector<asset::ICPUImageView*> cpuImgViews;
    cpuImgViews.reserve(storageImgViewCount + sampledImgViewCount);
    core::vector<asset::ICPUSampler*> cpuSamplers;
    cpuSamplers.reserve(sampledImgViewCount);
    for(ptrdiff_t i = 0u; i < assetCount; i++)
    {
        const asset::ICPUDescriptorSet* cpuds = _begin[i];
        for(auto j = 0u; j <= cpuds->getMaxDescriptorBindingIndex(); j++)
        {
            const auto type = cpuds->getDescriptorsType(j);
            for(const auto& info : cpuds->getDescriptors(j))
            {
                if(isBufferDesc(type))
                    cpuBuffers.push_back(static_cast<asset::ICPUBuffer*>(info.desc.get()));
                else if(isBufviewDesc(type))
                    cpuBufviews.push_back(static_cast<asset::ICPUBufferView*>(info.desc.get()));
                else if(isSampledImgViewDesc(type))
                {
                    cpuImgViews.push_back(static_cast<asset::ICPUImageView*>(info.desc.get()));
                    if(info.image.sampler)
                        cpuSamplers.push_back(info.image.sampler.get());
                }
                else if(isStorageImgDesc(type))
                    cpuImgViews.push_back(static_cast<asset::ICPUImageView*>(info.desc.get()));
            }
        }
    }

    using redirs_t = core::vector<size_t>;
    redirs_t layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);
    redirs_t bufviewRedirs = eliminateDuplicatesAndGenRedirs(cpuBufviews);
    redirs_t imgViewRedirs = eliminateDuplicatesAndGenRedirs(cpuImgViews);
    redirs_t smplrRedirs = eliminateDuplicatesAndGenRedirs(cpuSamplers);

    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUDescriptorSetLayout>(cpuLayouts.data(), cpuLayouts.data() + cpuLayouts.size(), _params);
    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data() + cpuBuffers.size(), _params);
    auto gpuBufviews = getGPUObjectsFromAssets<asset::ICPUBufferView>(cpuBufviews.data(), cpuBufviews.data() + cpuBufviews.size(), _params);
    auto gpuImgViews = getGPUObjectsFromAssets<asset::ICPUImageView>(cpuImgViews.data(), cpuImgViews.data() + cpuImgViews.size(), _params);
    auto gpuSamplers = getGPUObjectsFromAssets<asset::ICPUSampler>(cpuSamplers.data(), cpuSamplers.data() + cpuSamplers.size(), _params);

    core::vector<IGPUDescriptorSet::SWriteDescriptorSet> writes(maxWriteCount);
    auto write_it = writes.begin();
    core::vector<IGPUDescriptorSet::SDescriptorInfo> descInfos(descCount);
    {
        auto info = descInfos.begin();
        //iterators
        uint32_t bi = 0u, bvi = 0u, ivi = 0u, si = 0u;
        for(ptrdiff_t i = 0u; i < assetCount; i++)
        {
            IGPUDescriptorSetLayout* gpulayout = gpuLayouts->operator[](layoutRedirs[i]).get();
            res->operator[](i) = m_driver->createGPUDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(gpulayout));
            auto gpuds = res->operator[](i).get();

            const asset::ICPUDescriptorSet* cpuds = _begin[i];
            for(uint32_t j = 0u; j <= cpuds->getMaxDescriptorBindingIndex(); j++)
            {
                auto descriptors = cpuds->getDescriptors(j);
                if(descriptors.size() == 0u)
                    continue;

                const auto type = cpuds->getDescriptorsType(j);
                write_it->dstSet = gpuds;
                write_it->binding = j;
                write_it->arrayElement = 0;
                write_it->count = descriptors.size();
                write_it->descriptorType = type;
                write_it->info = &(*info);
                bool allDescriptorsPresent = true;
                for(const auto& desc : descriptors)
                {
                    if(isBufferDesc(type))
                    {
                        core::smart_refctd_ptr<video::IGPUOffsetBufferPair> buffer = bufRedirs[bi] >= gpuBuffers->size() ? nullptr : gpuBuffers->operator[](bufRedirs[bi]);
                        if(buffer)
                        {
                            info->desc = core::smart_refctd_ptr<video::IGPUBuffer>(buffer->getBuffer());
                            info->buffer.offset = desc.buffer.offset + buffer->getOffset();
                            info->buffer.size = desc.buffer.size;
                        }
                        else
                        {
                            info->desc = nullptr;
                            info->buffer.offset = 0u;
                            info->buffer.size = 0u;
                        }
                        ++bi;
                    }
                    else if(isBufviewDesc(type))
                    {
                        info->desc = bufviewRedirs[bvi] >= gpuBufviews->size() ? nullptr : gpuBufviews->operator[](bufviewRedirs[bvi]);
                        ++bvi;
                    }
                    else if(isSampledImgViewDesc(type) || isStorageImgDesc(type))
                    {
                        info->desc = imgViewRedirs[ivi] >= gpuImgViews->size() ? nullptr : gpuImgViews->operator[](imgViewRedirs[ivi]);
                        ++ivi;
                        info->image.imageLayout = desc.image.imageLayout;
                        if(isSampledImgViewDesc(type) && desc.image.sampler)
                            info->image.sampler = gpuSamplers->operator[](smplrRedirs[si++]);
                    }
                    allDescriptorsPresent = allDescriptorsPresent && info->desc;
                    info++;
                }
                if(allDescriptorsPresent)
                    write_it++;
            }
        }
    }

    m_driver->updateDescriptorSets(write_it - writes.begin(), writes.data(), 0u, nullptr);

    return res;
}

inline created_gpu_object_array<asset::ICPUComputePipeline> IGPUObjectFromAssetConverter::create(const asset::ICPUComputePipeline** const _begin, const asset::ICPUComputePipeline** const _end, const SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUComputePipeline>>(assetCount);

    core::vector<const asset::ICPUPipelineLayout*> cpuLayouts;
    core::vector<const asset::ICPUSpecializedShader*> cpuShaders;
    cpuLayouts.reserve(res->size());
    cpuShaders.reserve(res->size());

    const asset::ICPUComputePipeline** it = _begin;
    while(it != _end)
    {
        cpuShaders.push_back((*it)->getShader());
        cpuLayouts.push_back((*it)->getLayout());
        ++it;
    }

    core::vector<size_t> shdrRedirs = eliminateDuplicatesAndGenRedirs(cpuShaders);
    core::vector<size_t> layoutRedirs = eliminateDuplicatesAndGenRedirs(cpuLayouts);
    auto gpuShaders = getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(cpuShaders.data(), cpuShaders.data() + cpuShaders.size(), _params);
    auto gpuLayouts = getGPUObjectsFromAssets<asset::ICPUPipelineLayout>(cpuLayouts.data(), cpuLayouts.data() + cpuLayouts.size(), _params);

    for(ptrdiff_t i = 0; i < assetCount; ++i)
    {
        auto layout = (*gpuLayouts)[layoutRedirs[i]];
        auto shdr = (*gpuShaders)[shdrRedirs[i]];
        (*res)[i] = m_driver->createGPUComputePipeline(_params.pipelineCache, std::move(layout), std::move(shdr));
    }

    return res;
}
auto IGPUObjectFromAssetConverter::create(const asset::ICPUAnimationLibrary** _begin, const asset::ICPUAnimationLibrary** _end, const SParams& _params) -> created_gpu_object_array<asset::ICPUAnimationLibrary>
{
    const size_t assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUAnimationLibrary>>(assetCount);

    core::vector<const asset::ICPUBuffer*> cpuBuffers;
    cpuBuffers.reserve(assetCount * 3u);

    for(ptrdiff_t i = 0u; i < assetCount; i++)
    {
        const asset::ICPUAnimationLibrary* cpuanim = _begin[i];
        cpuBuffers.push_back(cpuanim->getKeyframeStorageBinding().buffer.get());
        cpuBuffers.push_back(cpuanim->getTimestampStorageBinding().buffer.get());
        if(const asset::ICPUBuffer* buf = cpuanim->getAnimationStorageRange().buffer.get())
            cpuBuffers.push_back(buf);
    }

    using redirs_t = core::vector<size_t>;
    redirs_t bufRedirs = eliminateDuplicatesAndGenRedirs(cpuBuffers);

    auto gpuBuffers = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBuffers.data(), cpuBuffers.data() + cpuBuffers.size(), _params);

    size_t bufIter = 0ull;
    for(ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::IAnimationLibrary<asset::ICPUBuffer>* cpuanim = _begin[i];

        asset::SBufferBinding<IGPUBuffer> keyframeBinding, timestampBinding;
        {
            keyframeBinding.offset = cpuanim->getKeyframeStorageBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            keyframeBinding.offset += gpubuf->getOffset();
            keyframeBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
        {
            timestampBinding.offset = cpuanim->getTimestampStorageBinding().offset;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            timestampBinding.offset += gpubuf->getOffset();
            timestampBinding.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }
        asset::SBufferRange<IGPUBuffer> animationRange;
        if(cpuanim->getAnimationStorageRange().buffer)
        {
            animationRange.offset = cpuanim->getAnimationStorageRange().offset;
            animationRange.size = cpuanim->getAnimationStorageRange().size;
            auto& gpubuf = (*gpuBuffers)[bufRedirs[bufIter++]];
            animationRange.offset += gpubuf->getOffset();
            animationRange.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
        }

        (*res)[i] = core::make_smart_refctd_ptr<IGPUAnimationLibrary>(std::move(keyframeBinding), std::move(timestampBinding), std::move(animationRange), cpuanim);
    }

    return res;
}

}
}  //nbl::video

#endif
