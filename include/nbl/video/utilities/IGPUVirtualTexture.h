// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
#define __NBL_VIDEO_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

class IGPUVirtualTexture final : public asset::IVirtualTexture<IGPUImageView, IGPUSampler>
{
    using base_t = asset::IVirtualTexture<IGPUImageView, IGPUSampler>;

    core::smart_refctd_ptr<ILogicalDevice> m_logicalDevice;

    static inline core::smart_refctd_ptr<IGPUCommandBuffer> createTransferCommandBuffer(ILogicalDevice* logicalDevice, IQueue* queue)
    {
        const auto queueFamilyIndex = queue->getFamilyIndex();
        assert((logicalDevice->getPhysicalDevice()->getQueueFamilyProperties().begin()[queueFamilyIndex].queueFlags&IQueue::FAMILY_FLAGS::TRANSFER_BIT).value);
        //now copy from CPU counterpart resources that can be shared (i.e. just copy state) between CPU and GPU
        //and convert to GPU those which can't be "shared": page table and VT resident storages along with their images and views
        core::smart_refctd_ptr<IGPUCommandBuffer> gpuCommandBuffer;
        {
            auto gpuCommandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
            logicalDevice->createCommandBuffers(gpuCommandPool.get(),IGPUCommandBuffer::LEVEL::PRIMARY,1u,&gpuCommandBuffer);
            assert(gpuCommandBuffer);
            // buffer should hold onto pool with refcounted backlink
        }
        gpuCommandBuffer->begin(IGPUCommandBuffer::USAGE::NONE);
        return gpuCommandBuffer;
    }
    static inline core::smart_refctd_ptr<IGPUImage> createGPUImageFromCPU(
        ILogicalDevice* logicalDevice, IGPUCommandBuffer* cmdbuf, IGPUFence* fence, IQueue* queue, const asset::ICPUImage* _cpuimg,
        uint32_t& waitSemaphoreCount, IGPUSemaphore* const* &semaphoresToWaitBeforeExecution, const asset::E_PIPELINE_STAGE_FLAGS* &stagesToWaitForPerSemaphore,
        video::IUtilities* utilities
    )
    {
        core::smart_refctd_ptr<IGPUImage> gpuImage;
        {
            IGPUImage::SCreationParams cpuImageParams = {};
            cpuImageParams = _cpuimg->getCreationParameters();

            auto regions = _cpuimg->getRegions();
            assert(regions.size());
            assert(false); // TODO: redo completely with streaming update
            //gpuImage = utilities->createFilledDeviceLocalImageOnDedMem(cmdbuf,std::move(cpuImageParams),gpuTexelBuffer.get(),regions.size(),regions.begin(),IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL);
        }

        return gpuImage;
    }
    void from_CPU_ctor_impl(
        IGPUCommandBuffer* cmdbuf, IGPUFence* fenceToSignal, IQueue* queue, asset::ICPUVirtualTexture* _cpuvt,
        uint32_t &waitSemaphoreCount, IGPUSemaphore* const* &semaphoresToWaitBeforeExecution, const asset::E_PIPELINE_STAGE_FLAGS* &stagesToWaitForPerSemaphore,
        video::IUtilities* utilities
    )
    {
        auto* cpuPgt = _cpuvt->getPageTable();
        m_pageTable = createGPUImageFromCPU(m_logicalDevice.get(),cmdbuf,fenceToSignal,queue,cpuPgt,waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore, utilities);

        m_precomputed = _cpuvt->getPrecomputedData();

        m_pgTabAddrAlctr_reservedSpc = _cpuvt->copyVirtualSpaceAllocatorsState(m_pageTable->getCreationParameters().arrayLayers, m_pageTableLayerAllocators.data());
        
        m_viewFormatToLayer = _cpuvt->getViewFormatToLayerMapping();

        const auto& cpuStorages = _cpuvt->getResidentStorages();
        for (const auto& pair : cpuStorages)
        {
            auto* cpuStorage = static_cast<asset::ICPUVirtualTexture::ICPUVTResidentStorage*>(pair.second.get());
            const asset::E_FORMAT_CLASS fmtClass = pair.first;

            m_storage.insert({fmtClass,core::make_smart_refctd_ptr<IGPUVTResidentStorage>(m_logicalDevice.get(),cmdbuf,fenceToSignal,queue,cpuStorage,waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore, utilities)});
        }

        auto createViewsFromCPU = [this](core::vector<SamplerArray::Sampler>& _dst, decltype(_cpuvt->getFloatViews()) _src) -> void
        {
            for (const auto& v : _src)
            {
                const asset::E_FORMAT format = v.view->getCreationParameters().format;

                const auto& storage = m_storage.find(asset::getFormatClass(format))->second;
                auto view = storage->createView(format);
                SamplerArray::Sampler s{ format, std::move(view) };
                _dst.push_back(std::move(s));
            }
        };
        m_fsamplers.views.reserve(_cpuvt->getFloatViews().size());
        createViewsFromCPU(m_fsamplers.views, _cpuvt->getFloatViews());
        m_isamplers.views.reserve(_cpuvt->getIntViews().size());
        createViewsFromCPU(m_isamplers.views, _cpuvt->getIntViews());
        m_usamplers.views.reserve(_cpuvt->getUintViews().size());
        createViewsFromCPU(m_usamplers.views, _cpuvt->getUintViews());
    }

protected:
    class IGPUVTResidentStorage final : public base_t::IVTResidentStorage
    {
        using storage_base_t = base_t::IVTResidentStorage;
        using cpu_counterpart_t = asset::ICPUVirtualTexture::ICPUVTResidentStorage;

    public:
        IGPUVTResidentStorage(ILogicalDevice* _logicalDevice, IGPUCommandBuffer* cmdbuf, IGPUFence* fenceToSignal, IQueue* queue, const cpu_counterpart_t* _cpuStorage,
            uint32_t& waitSemaphoreCount, IGPUSemaphore* const*& semaphoresToWaitBeforeExecution, const asset::E_PIPELINE_STAGE_FLAGS*& stagesToWaitForPerSemaphore, video::IUtilities* utilities
        ) :
            storage_base_t(
                createGPUImageFromCPU(_logicalDevice,cmdbuf,fenceToSignal,queue,_cpuStorage->image.get(),waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore, utilities),
                _cpuStorage->tileAlctr,
                _cpuStorage->m_alctrReservedSpace,
                _cpuStorage->m_decodeAddr_layerShift,
                _cpuStorage->m_decodeAddr_xMask
            ),
            m_logicalDevice(_logicalDevice)
        {

        }

        IGPUVTResidentStorage(ILogicalDevice* _logicalDevice, asset::E_FORMAT _format, uint32_t _tilesPerDim) :
            storage_base_t(_format, _tilesPerDim),
            m_logicalDevice(_logicalDevice)
        {

        }

        IGPUVTResidentStorage(ILogicalDevice* _logicalDevice, asset::E_FORMAT _format, uint32_t _tileExtent, uint32_t _layers, uint32_t _tilesPerDim) :
            storage_base_t(_format, _layers, _tilesPerDim),
            m_logicalDevice(_logicalDevice)
        {
            deferredInitialization(_tileExtent, _layers);
        }

        void deferredInitialization(uint32_t tileExtent, uint32_t _layers) override
        {
            storage_base_t::deferredInitialization(tileExtent, _layers);

            if (image)
                return;

            const uint32_t tilesPerDim = getTilesPerDim();
            const uint32_t extent = tileExtent * tilesPerDim;

            IGPUImage::SCreationParams params = {};
            params.extent = { extent, extent, 1u };
            params.format = imageFormat;
            params.arrayLayers = _layers;
            params.mipLevels = 1u;
            params.type = asset::IImage::ET_2D;
            params.samples = asset::IImage::ESCF_1_BIT;
            params.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0);
            // TODO: final layout should be readonly (if there's transfer necessary, then we start in transfer dst)

            image = m_logicalDevice->createImage(std::move(params));
            m_logicalDevice->allocate(image->getMemoryReqs(), image.get());
        }

    private:
        core::smart_refctd_ptr<IGPUImageView> createView_internal(IGPUImageView::SCreationParams&& _params) const override
        {
            return m_logicalDevice->createImageView(std::move(_params));
        }

        ILogicalDevice* m_logicalDevice;
    };

public:
    IGPUVirtualTexture(
        ILogicalDevice* _logicalDevice,
        physical_tiles_per_dim_log2_callback_t&& _callback,
        const base_t::IVTResidentStorage::SCreationParams* _residentStorageParams,
        uint32_t _residentStorageCount,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _pgTabLayers = 32u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) :
        base_t(
            std::move(_callback),
            _maxAllocatableTexSz_log2-_pgSzxy_log2,
            _pgTabLayers,
            _pgSzxy_log2,
            _tilePadding
        ),
        m_logicalDevice(_logicalDevice)
    {
        m_pageTable = createPageTable(m_pgtabSzxy_log2, _pgTabLayers, _pgSzxy_log2, _maxAllocatableTexSz_log2);
        initResidentStorage(_residentStorageParams, _residentStorageCount);
    }

    IGPUVirtualTexture(
        ILogicalDevice* _logicalDevice,
        physical_tiles_per_dim_log2_callback_t&& _callback,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) : base_t(
            std::move(_callback),
            _maxAllocatableTexSz_log2-_pgSzxy_log2,
            MAX_PAGE_TABLE_LAYERS,
            _pgSzxy_log2,
            _tilePadding
        ),
        m_logicalDevice(_logicalDevice)
    {
    }

    /*
        The queue must have TRANSFER capability
    */
    IGPUVirtualTexture(
        ILogicalDevice* _logicalDevice, IGPUCommandBuffer* cmdbuf, IGPUFence* fenceToSignal, IQueue* queue, asset::ICPUVirtualTexture* _cpuvt,
        uint32_t &waitSemaphoreCount, IGPUSemaphore* const* &semaphoresToWaitBeforeExecution, const asset::PIPELINE_STAGE_FLAGS* &stagesToWaitForPerSemaphore,
        video::IUtilities* utilities
    ) :
        base_t(
            _cpuvt->getPhysicalStorageExtentCallback(),
            _cpuvt->getPageTableExtent_log2(),
            _cpuvt->getPageTable()->getCreationParameters().arrayLayers,
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getTilePadding(),
            false
            ),
        m_logicalDevice(_logicalDevice)
    {
        from_CPU_ctor_impl(cmdbuf,fenceToSignal,queue,_cpuvt,waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore, utilities);
    }

    /*
        This function will submit
        The queue must have TRANSFER capability
    */
    IGPUVirtualTexture(
        ILogicalDevice* _logicalDevice, IGPUFence* fenceToSignal, IQueue* queue, asset::ICPUVirtualTexture* _cpuvt, video::IUtilities* utilities,
        uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
        const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
    ) :
        base_t(
            _cpuvt->getPhysicalStorageExtentCallback(),
            _cpuvt->getPageTableExtent_log2(),
            _cpuvt->getPageTable()->getCreationParameters().arrayLayers,
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getTilePadding(),
            false
        ),
        m_logicalDevice(_logicalDevice)
    {
        auto cmdbuf = createTransferCommandBuffer(_logicalDevice,queue);
        from_CPU_ctor_impl(cmdbuf.get(),fenceToSignal,queue,_cpuvt,waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore, utilities);
        cmdbuf->end();

        IQueue::SSubmitInfo info;
        info.commandBufferCount = 1u;
        info.commandBuffers = &cmdbuf.get();
        info.waitSemaphoreCount = waitSemaphoreCount;
        info.pWaitSemaphores = semaphoresToWaitBeforeExecution;
        info.pWaitDstStageMask = stagesToWaitForPerSemaphore;
        info.signalSemaphoreCount = signalSemaphoreCount;
        info.pSignalSemaphores = semaphoresToSignal;
        queue->submit(1u,&info,fenceToSignal);
    }

    bool commit(const SMasterTextureData& _addr, const IGPUImage* _img, const asset::IImage::SSubresourceRange& _subres, asset::ISampler::E_TEXTURE_CLAMP _uwrap, asset::ISampler::E_TEXTURE_CLAMP _vwrap, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor) override
    {
        assert(0);
        return false;
    }

    SViewAliasTextureData createAlias(const SMasterTextureData& _addr, asset::E_FORMAT _viewingFormat, const asset::IImage::SSubresourceRange& _subresRelativeToMaster) override
    {
        assert(0);
        return SViewAliasTextureData::invalid();
    }

    bool free(const SMasterTextureData& _addr) override
    {
        assert(0);
        return false;
    }

    auto getDSlayoutBindings(IGPUDescriptorSetLayout::SBinding* _outBindings, core::smart_refctd_ptr<IGPUSampler>* _outSamplers, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return getDSlayoutBindings_internal<IGPUDescriptorSetLayout>(_outBindings, _outSamplers, _pgtBinding, _fsamplersBinding, _isamplersBinding, _usamplersBinding);
    }

    auto getDescriptorSetWrites(IGPUDescriptorSet::SWriteDescriptorSet* _outWrites, IGPUDescriptorSet::SDescriptorInfo* _outInfo, IGPUDescriptorSet* _dstSet, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return getDescriptorSetWrites_internal<IGPUDescriptorSet>(_outWrites, _outInfo, _dstSet, _pgtBinding, _fsamplersBinding, _isamplersBinding, _usamplersBinding);
    }

protected:
    core::smart_refctd_ptr<IGPUImageView> createPageTableView() const override
    {
        return m_logicalDevice->createImageView(createPageTableViewCreationParams());
    }
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(asset::E_FORMAT _format, uint32_t _tileExtent, uint32_t _layers, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<IGPUVTResidentStorage>(m_logicalDevice.get(), _format, _tileExtent, _layers, _tilesPerDim);
    }
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(asset::E_FORMAT _format, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<IGPUVTResidentStorage>(m_logicalDevice.get(), _format, _tilesPerDim);
    }
    core::smart_refctd_ptr<IGPUImage> createPageTableImage(IGPUImage::SCreationParams&& _params) const override
    {
        auto img = m_logicalDevice->createImage(std::move(_params));
        m_logicalDevice->allocate(img->getMemoryReqs(), img.get());
        return img;
    }
    core::smart_refctd_ptr<IGPUSampler> createSampler(const asset::ISampler::SParams& _params) const override
    {
        return m_logicalDevice->createSampler(_params);
    }
};

}

#endif
