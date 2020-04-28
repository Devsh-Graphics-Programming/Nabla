#ifndef __IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
#define __IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__

//TODO change to ICPUVirtualTexture
#include <irr/asset/ITexturePacker.h>
#include <irr/video/IGPUImageView.h>
#include "IVideoDriver.h"
#include <irr/asset/IAssetManager.h>

namespace irr {
namespace video
{

class IGPUVirtualTexture final : public asset::IVirtualTexture<video::IGPUImageView>
{
    using base_t = asset::IVirtualTexture<video::IGPUImageView>;

    IVideoDriver* m_driver;

protected:
    class IGPUVTResidentStorage final : public base_t::IVTResidentStorage
    {
        using base_t = base_t::IVTResidentStorage;
        using cpu_counterpart_t = asset::ICPUVirtualTexture::ICPUVTResidentStorage;

        static core::smart_refctd_ptr<IGPUImage> createGPUImageFromCPU(IVideoDriver* _driver, asset::IAssetManager* _am, asset::ICPUImage* _cpuimg)
        {
            auto params = _cpuimg->getCreationParameters();
            auto img = _driver->createDeviceLocalGPUImageOnDedMem(std::move(params));

            auto regions = _cpuimg->getRegions();
            if (regions.size())
            {
                auto texelBuf = _driver->createFilledDeviceLocalGPUBufferOnDedMem(_cpuimg->getBuffer()->getSize(), _cpuimg->getBuffer()->getPointer());
                _driver->copyBufferToImage(texelBuf.get(), img.get(), regions.size(), regions.begin());
            }
            _am->convertAssetToEmptyCacheHandle(_cpuimg, core::smart_refctd_ptr(img));

            return img;
        }

    public:
        IGPUVTResidentStorage(IVideoDriver* _driver, asset::IAssetManager* _am, const cpu_counterpart_t* _cpuStorage) :
            //TODO awaiting fix: base_t ctor should copy addr alctr state from cpu counterpart
            base_t(
                createGPUImageFromCPU(_driver, _am, _cpuStorage->image.get()),
                _cpuStorage->m_assignedPageTableLayers,
                _cpuStorage->m_addr_layerShift,
                _cpuStorage->m_addr_xMask
            ),
            m_driver(_driver)
        {

        }

        IGPUVTResidentStorage(IVideoDriver* _driver, asset::E_FORMAT _format, uint32_t _extent, uint32_t _layers, uint32_t _tilesPerDim) :
            base_t(_layers, _tilesPerDim),
            m_driver(_driver)
        {
            IGPUImage::SCreationParams params;
            params.extent = { _extent,_extent,1u };
            params.format = _format;
            params.arrayLayers = _layers;
            params.mipLevels = 1u;
            params.type = asset::IImage::ET_2D;
            params.samples = asset::IImage::ESCF_1_BIT;
            params.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0);

            image = m_driver->createDeviceLocalGPUImageOnDedMem(std::move(params));
        }

    private:
        core::smart_refctd_ptr<IGPUImageView> createView_internal(IGPUImageView::SCreationParams&& _params) const override
        {
            return m_driver->createGPUImageView(std::move(_params));
        }

        IVideoDriver* m_driver = nullptr;
    };

public:
    IGPUVirtualTexture(
        IVideoDriver* _driver,
        const base_t::IVTResidentStorage::SCreationParams* _residentStorageParams,
        uint32_t _residentStorageCount,
        uint32_t _pgTabSzxy_log2 = 8u,
        uint32_t _pgTabLayers = 256u,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) :
        base_t(
            _pgTabSzxy_log2,
            _pgTabLayers,
            _pgSzxy_log2,
            _tilePadding
        ),
        m_driver(_driver)
    {
        m_pageTable = createPageTable(_pgTabSzxy_log2, _pgTabLayers, _pgSzxy_log2, _maxAllocatableTexSz_log2);
        initResidentStorage(_residentStorageParams, _residentStorageCount);
    }
    IGPUVirtualTexture(IVideoDriver* _driver, asset::IAssetManager* _am, asset::ICPUVirtualTexture* _cpuvt) :
        base_t(
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getPageTable()->getCreationParameters().arrayLayers,
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getTilePadding(),
            false
            ),
        m_driver(_driver)
    {
        //now copy from CPU counterpart resources that can be shared (i.e. just copy state) between CPU and GPU
        //and convert to GPU those which can't be "shared": page table and VT resident storages along with their images and views

        m_pageTable = std::move(_driver->getGPUObjectsFromAssets(&_cpuvt->getPageTable(), &_cpuvt->getPageTable()+1)->front());

        m_freePageTableLayerIDs = _cpuvt->getFreePageTableLayersStack();

        memcpy(m_layerToViewIndexMapping->data(), _cpuvt->getLayerToViewIndexMapping().begin(), 
            _cpuvt->getLayerToViewIndexMapping().size()*sizeof(decltype(m_layerToViewIndexMapping)::pointee::value_type));

        const auto& cpuStorages = _cpuvt->getResidentStorages();
        for (const auto& pair : cpuStorages)
        {
            auto* cpuStorage = static_cast<asset::ICPUVirtualTexture::ICPUVTResidentStorage*>(pair.second.get());
            const asset::E_FORMAT_CLASS fmtClass = pair.first;

            m_storage.insert({fmtClass, core::make_smart_refctd_ptr<IGPUVTResidentStorage>(_driver, _am, cpuStorage)});
        }

        m_fsamplers.views = _cpuvt->getFloatViews().size() ? _driver->getGPUObjectsFromAssets(_cpuvt->getFloatViews().begin(), _cpuvt->getFloatViews().end()) : nullptr;
        m_isamplers.views = _cpuvt->getIntViews().size() ? _driver->getGPUObjectsFromAssets(_cpuvt->getIntViews().begin(), _cpuvt->getIntViews().end()) : nullptr;
        m_usamplers.views = _cpuvt->getUintViews().size() ? _driver->getGPUObjectsFromAssets(_cpuvt->getUintViews().begin(), _cpuvt->getUintViews().end()) : nullptr;
    }

    STextureData pack(const image_t* _img, const asset::IImage::SSubresourceRange& _subres, asset::ISampler::E_TEXTURE_CLAMP _wrapu, asset::ISampler::E_TEXTURE_CLAMP _wrapv, asset::ISampler::E_TEXTURE_BORDER_COLOR _borderColor) override
    {
        assert(0);
        return STextureData::invalid();
    }

    bool free(const STextureData& _addr, const asset::IImage* _img, const asset::IImage::SSubresourceRange& _subres) override
    {
        assert(0);
        return false;
    }

    core::smart_refctd_ptr<IGPUImageView> createPageTableView() const override
    {
        IGPUImageView::SCreationParams params;
        params.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0);
        params.format = m_pageTable->getCreationParameters().format;
        params.subresourceRange.aspectMask = static_cast<asset::IImage::E_ASPECT_FLAGS>(0);
        params.subresourceRange.baseArrayLayer = 0u;
        params.subresourceRange.layerCount = m_pageTable->getCreationParameters().arrayLayers;
        params.subresourceRange.baseMipLevel = 0u;
        params.subresourceRange.levelCount = m_pageTable->getCreationParameters().mipLevels;
        params.viewType = asset::IImageView<IGPUImage>::ET_2D_ARRAY;
        params.image = m_pageTable;

        return m_driver->createGPUImageView(std::move(params));
    }

protected:
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(asset::E_FORMAT _format, uint32_t _extent, uint32_t _layers, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<IGPUVTResidentStorage>(m_driver, _format, _extent, _layers, _tilesPerDim);//TODO driver
    }
    core::smart_refctd_ptr<IGPUImage> createImage(IGPUImage::SCreationParams&& _params) const override
    {
        return m_driver->createDeviceLocalGPUImageOnDedMem(std::move(_params));
    }
};

}}

#endif // !__IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
