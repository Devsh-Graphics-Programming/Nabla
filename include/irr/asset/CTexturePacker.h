#ifndef __IRR_C_TEXTURE_PACKER_H_INCLUDED__
#define __IRR_C_TEXTURE_PACKER_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/asset/ICPUImage.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"

namespace irr {
namespace asset
{

class CTexturePacker
{
    using addr_alctr_t = core::GeneralpurposeAddressAllocator<uint32_t>;

public:
    struct STexOffset
    {
        uint32_t x, y;
    };

    CTexturePacker(E_FORMAT _format, uint32_t _szxy = 1u<<14, uint32_t _pgSzxy = 256u) :
        m_pgSzxy(_pgSzxy),
        m_alctrReservedSpace(core::make_refctd_dynamic_array<decltype(m_alctrReservedSpace)>(
            addr_alctr_t::reserved_size(_szxy / _pgSzxy, _szxy / _pgSzxy, 1ull))
        ),
        m_addrAlctr(reinterpret_cast<void*>(m_alctrReservedSpace->data()), 0u, 0u, _szxy/_pgSzxy, _szxy,_pgSzxy, 1u)
    {
        assert(core::isPoT(_szxy));
        assert(core::isPoT(_pgSzxy));

        ICPUImage::SCreationParams params;
        params.extent = {_szxy,_szxy,1u};
        params.format = _format;
        params.arrayLayers = 1u;
        params.mipLevels = 1u;
        params.type = IImage::ET_2D;
        params.samples = IImage::ESCF_1_BIT;
        params.flags = static_cast<IImage::E_CREATE_FLAGS>(0);

        m_megaimg = ICPUImage::create(std::move(params));
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull);
        auto& region = regions->front();
        region.imageSubresource.mipLevel = 0u;
        region.imageSubresource.baseArrayLayer = 0u;
        region.imageSubresource.layerCount = 1u;
        region.bufferOffset = 0u;
        region.bufferRowLength = _szxy;
        region.bufferImageHeight = 0u; //tightly packed
        region.imageOffset = {0u, 0u, 0u};
        region.imageExtent = {_szxy,_szxy,1u};
        auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(_format)*_szxy*_szxy);
        m_megaimg->setBufferAndRegions(std::move(buffer), regions);
    }

    STexOffset pack(const ICPUImage* _img)
    {
        assert(m_megaimg->getCreationParameters().format==_img->getCreationParameters().format);

        auto extent = _img->getCreationParameters().extent;
        uint32_t rnd = core::roundUpToPoT(std::max(extent.width,extent.height))/m_pgSzxy;
        uint32_t pgCnt = rnd*rnd;
        uint32_t morton_addr = m_addrAlctr.alloc_addr(pgCnt, pgCnt);
        assert(morton_addr!=addr_alctr_t::invalid_address);

        auto dec_morton = [](uint32_t x) -> uint32_t {
            x = x & 0x55555555u;
            x = (x | (x >> 1)) & 0x33333333u;
            x = (x | (x >> 2)) & 0x0F0F0F0Fu;
            x = (x | (x >> 4)) & 0x00FF00FFu;
            x = (x | (x >> 8)) & 0x0000FFFFu;
            return x;
        };
        core::vector2du32_SIMD texOffset(dec_morton(morton_addr), dec_morton(morton_addr >> 1));
        texOffset *= m_pgSzxy;
        const uint32_t texelSz = getTexelOrBlockBytesize(m_megaimg->getCreationParameters().format);
        const uint64_t bufOffset = static_cast<uint64_t>(texOffset.y * m_megaimg->getCreationParameters().extent.width + texOffset.x) * texelSz;

        for (const auto& reg : _img->getRegions())
        {
            if (reg.imageSubresource.mipLevel!=0u)
                continue;

            const uint8_t* src = reinterpret_cast<const uint8_t*>(_img->getBuffer()->getPointer())+reg.bufferOffset;
            uint8_t* dst = reinterpret_cast<uint8_t*>(m_megaimg->getBuffer()->getPointer()) + bufOffset;
            const uint32_t pitch = reg.bufferRowLength*texelSz;
            for (uint32_t i = 0u; i < reg.imageExtent.height; ++i)
            {
                memcpy(dst + ((reg.imageOffset.y + i)*m_megaimg->getCreationParameters().extent.width + reg.imageOffset.x)*texelSz, src+i*pitch, reg.imageExtent.width*texelSz);
            }
        }

        return {texOffset.x,texOffset.y};
    }

    ICPUImage* getImage() { return m_megaimg.get(); }

private:
    const uint32_t m_pgSzxy;
    core::smart_refctd_dynamic_array<uint8_t> m_alctrReservedSpace;
    addr_alctr_t m_addrAlctr;
    core::smart_refctd_ptr<ICPUImage> m_megaimg;
};

}}

#endif