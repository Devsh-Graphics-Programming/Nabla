#ifndef __IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
#define __IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__

//TODO change to ICPUVirtualTexture
#include <irr/asset/ITexturePacker.h>
#include <irr/video/IGPUImageView.h>
#include "IVideoDriver.h"

namespace irr {
namespace video
{

class IGPUVirtualTexture final : public asset::IVirtualTexture<video::IGPUImageView>
{
    using base_t = asset::IVirtualTexture<video::IGPUImageView>;

protected:
    class IGPUVTResidentStorage final : public base_t::IVTResidentStorage
    {
        using base_t = base_t::IVTResidentStorage;
        using cpu_counterpart_t = asset::ICPUVirtualTexture::ICPUVTResidentStorage;

    public:
        IGPUVTResidentStorage(IVideoDriver* _driver, const cpu_counterpart_t* _cpuStorage) :
            //TODO awaiting fix: base_t ctor should copy addr alctr state from cpu counterpart
            base_t(
                std::move(_driver->getGPUObjectsFromAssets(&_cpuStorage->image.get(),&_cpuStorage->image.get()+1)->front()),
                _cpuStorage->m_assignedPageTableLayers
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
        core::smart_refctd_ptr<IGPUImageView> createView_internal(IGPUImageView::SCreationParams&& _params) override
        {
            return m_driver->createGPUImageView(std::move(_params));
        }

        IVideoDriver* m_driver = nullptr;
    };

public:
    IGPUVirtualTexture(IVideoDriver* _driver, asset::ICPUVirtualTexture* _cpuvt) :
        base_t(
            nullptr, 0u, 
            std::move(_driver->getGPUObjectsFromAssets(&_cpuvt->getPageTable(), &_cpuvt->getPageTable()+1)->front()),
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getPageTable()->getCreationParameters().arrayLayers,
            _cpuvt->getPageExtent_log2(),
            _cpuvt->getTilePadding(),
            false
            )
    {
        //now copy from CPU counterpart resources that can be shared (i.e. just copy state) between CPU and GPU
        //and convert to GPU those which can't be "shared" (page table already got converted): VT resident storages along with their images and views

        //TODO copy addr allocators state

        m_freePageTableLayerIDs = _cpuvt->getFreePageTableLayersStack();

        memcpy(m_layerToViewIndexMapping->data(), _cpuvt->getLayerToViewIndexMapping().begin(), 
            _cpuvt->getLayerToViewIndexMapping().size()*sizeof(decltype(m_layerToViewIndexMapping)::pointee::value_type));

        const auto& cpuStorages = _cpuvt->getResidentStorages();
        for (const auto& pair : cpuStorages)
        {
            auto* cpuStorage = static_cast<asset::ICPUVirtualTexture::ICPUVTResidentStorage*>(pair.second.get());
            const asset::E_FORMAT_CLASS fmtClass = pair.first;

            m_storage.insert({fmtClass, core::make_smart_refctd_ptr<IGPUVTResidentStorage>(_driver, cpuStorage)});
        }
        //TODO fill samplers arrays
    }
};

}}

#endif // !__IRR_I_GPU_VIRTUAL_TEXTURE_H_INCLUDED__
