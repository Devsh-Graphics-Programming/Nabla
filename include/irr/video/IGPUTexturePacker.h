#ifndef __IRR_I_GPU_TEXTURE_PACKER_H_INCLUDED__
#define __IRR_I_GPU_TEXTURE_PACKER_H_INCLUDED__

#include "irr/asset/ITexturePacker.h"
#include "IVideoDriver.h"

namespace irr {
namespace video 
{

class IGPUTexturePacker : public asset::ITexturePacker
{
    static core::smart_refctd_ptr<IGPUImage> createGPUImageFromCPU(IVideoDriver* _driver, const asset::ICPUImage* _cpuimg)
    {
        auto params = _cpuimg->getCreationParameters();
        auto img = _driver->createDeviceLocalGPUImageOnDedMem(std::move(params));

        auto regions = _cpuimg->getRegions();
        if (regions.size())
        {
            auto texelBuf = _driver->createFilledDeviceLocalGPUBufferOnDedMem(_cpuimg->getBuffer()->getSize(), _cpuimg->getBuffer()->getPointer());
            _driver->copyBufferToImage(texelBuf.get(), img.get(), regions.size(), regions.begin());
        }

        return img;
    }
    static core::smart_refctd_ptr<IGPUImageView> createView_common(IVideoDriver* _driver, IGPUImage* _img)
    {
        IGPUImageView::SCreationParams params;
        params.flags = static_cast<asset::IImageView<IGPUImage>::E_CREATE_FLAGS>(0);
        params.format = _img->getCreationParameters().format;
        params.subresourceRange.aspectMask = static_cast<asset::IImage::E_ASPECT_FLAGS>(0);
        params.subresourceRange.baseArrayLayer = 0u;
        params.subresourceRange.layerCount = _img->getCreationParameters().arrayLayers;
        params.subresourceRange.baseMipLevel = 0u;
        params.subresourceRange.levelCount = _img->getCreationParameters().mipLevels;
        params.image = core::smart_refctd_ptr<IGPUImage>(_img);
        params.viewType = asset::IImageView<IGPUImage>::ET_2D_ARRAY;

        return _driver->createGPUImageView(std::move(params));
    }

public:
    IGPUTexturePacker(IVideoDriver* _driver, asset::ICPUTexturePacker* _cpupacker) : 
        m_pageTable(createGPUImageFromCPU(_driver, _cpupacker->getPageTable())),
        m_physAddrTex(createGPUImageFromCPU(_driver, _cpupacker->getPhysicalAddressTexture()))
    {
    }

    page_tab_offset_t pack(const IGPUImage* _img, const IGPUImage::SSubresourceRange& _subres, asset::ISampler::E_TEXTURE_CLAMP _wrapu, asset::ISampler::E_TEXTURE_CLAMP _wrapv)
    {
        assert(false);//not implemented
    }

    void free(page_tab_offset_t _addr, const asset::IImage* _img, const asset::IImage::SSubresourceRange& _subres)
    {
        assert(false);//not implented
    }

    IGPUImage* getPhysicalAddressTexture() const { return m_physAddrTex.get(); }
    core::smart_refctd_ptr<IGPUImageView> createPhysicalAddressTextureView(IVideoDriver* _driver) const
    {
        return createView_common(_driver, m_physAddrTex.get());
    }
    IGPUImage* getPageTable() const { return m_pageTable.get(); }
    core::smart_refctd_ptr<IGPUImageView> createPageTableView(IVideoDriver* _driver) const
    {
        return createView_common(_driver, m_pageTable.get());
    }

private:
    core::smart_refctd_ptr<IGPUImage> m_pageTable;
    core::smart_refctd_ptr<IGPUImage> m_physAddrTex;
};

}}

#endif