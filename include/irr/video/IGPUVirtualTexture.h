#ifndef __IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
#define __IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__

#include <irr/asset/ICPUVirtualTexture.h>
#include <irr/video/IGPUImageView.h>
#include "IVideoDriver.h"
#include <irr/asset/IAssetManager.h>
#include <irr/video/IGPUDescriptorSet.h>

namespace irr {
namespace video
{

namespace impl
{
inline core::smart_refctd_ptr<IGPUImage> createGPUImageFromCPU(IVideoDriver* _driver, asset::ICPUImage* _cpuimg)
{
    auto params = _cpuimg->getCreationParameters();
    auto img = _driver->createDeviceLocalGPUImageOnDedMem(std::move(params));

    auto regions = _cpuimg->getRegions();
    assert(regions.size());
    auto texelBuf = _driver->createFilledDeviceLocalGPUBufferOnDedMem(_cpuimg->getBuffer()->getSize(), _cpuimg->getBuffer()->getPointer());
    _driver->copyBufferToImage(texelBuf.get(), img.get(), regions.size(), regions.begin());

    return img;
}
}

class IGPUVirtualTexture final : public asset::IVirtualTexture<video::IGPUImageView, video::IGPUSampler>
{
    using base_t = asset::IVirtualTexture<video::IGPUImageView, video::IGPUSampler>;

    IVideoDriver* m_driver;

protected:
    class IGPUVTResidentStorage final : public base_t::IVTResidentStorage
    {
        using base_t = base_t::IVTResidentStorage;
        using cpu_counterpart_t = asset::ICPUVirtualTexture::ICPUVTResidentStorage;

    public:
        IGPUVTResidentStorage(IVideoDriver* _driver, const cpu_counterpart_t* _cpuStorage) :
            //TODO awaiting fix: base_t ctor should copy addr alctr state from cpu counterpart
            base_t(
                impl::createGPUImageFromCPU(_driver, _cpuStorage->image.get()),
                _cpuStorage->m_decodeAddr_layerShift,
                _cpuStorage->m_decodeAddr_xMask
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
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _pgTabLayers = 32u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) :
        base_t(
            _maxAllocatableTexSz_log2-_pgSzxy_log2,
            _pgTabLayers,
            _pgSzxy_log2,
            _tilePadding
        ),
        m_driver(_driver)
    {
        m_pageTable = createPageTable(m_pgtabSzxy_log2, _pgTabLayers, _pgSzxy_log2, _maxAllocatableTexSz_log2);
        initResidentStorage(_residentStorageParams, _residentStorageCount);
    }
    IGPUVirtualTexture(IVideoDriver* _driver, asset::ICPUVirtualTexture* _cpuvt) :
        base_t(
            _cpuvt->getPageTableExtent_log2(),
            _cpuvt->getPageTable()->getCreationParameters().arrayLayers,
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getTilePadding(),
            false
            ),
        m_driver(_driver)
    {
        //now copy from CPU counterpart resources that can be shared (i.e. just copy state) between CPU and GPU
        //and convert to GPU those which can't be "shared": page table and VT resident storages along with their images and views

        auto* cpuPgt = _cpuvt->getPageTable();
        m_pageTable = impl::createGPUImageFromCPU(m_driver, cpuPgt);

        m_precomputed = _cpuvt->getPrecomputedData();

        //TODO copy m_viewFormatToLayer from cpuvt
        //TODO copy m_layerToSamplerType

        const auto& cpuStorages = _cpuvt->getResidentStorages();
        for (const auto& pair : cpuStorages)
        {
            auto* cpuStorage = static_cast<asset::ICPUVirtualTexture::ICPUVTResidentStorage*>(pair.second.get());
            const asset::E_FORMAT_CLASS fmtClass = pair.first;

            m_storage.insert({fmtClass, core::make_smart_refctd_ptr<IGPUVTResidentStorage>(_driver, cpuStorage)});
        }

        auto createViewsFromCPU = [this](core::smart_refctd_ptr<IGPUImageView>* _dst, core::SRange<const core::smart_refctd_ptr<asset::ICPUImageView>> _src) -> void
        {
            for (uint32_t i = 0u; i < _src.size(); ++i)
            {
                const auto& v = _src.begin()[i];
                const asset::E_FORMAT format = v->getCreationParameters().format;

                const auto& storage = m_storage.find(asset::getFormatClass(format))->second;
                _dst[i] = storage->createView(format);
            }
        };
        m_fsamplers.views = _cpuvt->getFloatViews().size() ? core::make_refctd_dynamic_array<decltype(m_fsamplers.views)>(_cpuvt->getFloatViews().size()) : nullptr;
        createViewsFromCPU(m_fsamplers.views->data(), _cpuvt->getFloatViews());
        m_isamplers.views = _cpuvt->getIntViews().size() ? core::make_refctd_dynamic_array<decltype(m_isamplers.views)>(_cpuvt->getIntViews().size()) : nullptr;
        createViewsFromCPU(m_isamplers.views->data(), _cpuvt->getIntViews());
        m_usamplers.views = _cpuvt->getUintViews().size() ? core::make_refctd_dynamic_array<decltype(m_usamplers.views)>(_cpuvt->getUintViews().size()) : nullptr;
        createViewsFromCPU(m_usamplers.views->data(), _cpuvt->getUintViews());
    }

    bool commit(const SMasterTextureData& _addr, const IGPUImage* _img, const asset::IImage::SSubresourceRange& _subres) override
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
        return m_driver->createGPUImageView(createPageTableViewCreationParams());
    }
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(asset::E_FORMAT _format, uint32_t _extent, uint32_t _layers, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<IGPUVTResidentStorage>(m_driver, _format, _extent, _layers, _tilesPerDim);
    }
    core::smart_refctd_ptr<IGPUImage> createPageTableImage(IGPUImage::SCreationParams&& _params) const override
    {
        return m_driver->createDeviceLocalGPUImageOnDedMem(std::move(_params));
    }
    core::smart_refctd_ptr<IGPUSampler> createSampler(const asset::ISampler::SParams& _params) const override
    {
        return m_driver->createGPUSampler(_params);
    }
};

}}

#endif // !__IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
