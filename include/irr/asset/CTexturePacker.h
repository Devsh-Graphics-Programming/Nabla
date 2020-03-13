#ifndef __IRR_C_TEXTURE_PACKER_H_INCLUDED__
#define __IRR_C_TEXTURE_PACKER_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/asset/ICPUImage.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/math/morton.h"

namespace irr {
namespace asset
{

//TODO move this to different file
template<typename addr_type>
class CSquareAddressAllocator
{
protected:
    using lin_addr_alctr_t = core::GeneralpurposeAddressAllocator<addr_type>;

    const uint32_t ADDR_LAYER_SHIFT;
    const uint32_t ADDR_MORTON_MASK;

    core::smart_refctd_dynamic_array<uint8_t> m_alctrReservedSpace;
    lin_addr_alctr_t m_addrAlctr;

public:
    CSquareAllocator(uint32_t _addrBitCntPerDim, uint32_t _squareSz) : 
        ADDR_LAYER_SHIFT(_addrBitCntPerDim*2u),
        ADDR_MORTON_MASK((1u<<ADDR_LAYER_SHIFT)-1u),
        m_alctrReservedSpace(core::make_refctd_dynamic_array<decltype(m_alctrReservedSpace)>(
            lin_addr_alctr_t::reserved_size(_squareSz*_squareSz, _squareSz*_squareSz, 1ull))
        ),
        m_addrAlctr(reinterpret_cast<void*>(m_alctrReservedSpace->data()), 0u, 0u, _squareSz*_squareSz, _squareSz*_squareSz, 1u)
    {}

    inline addr_type alloc_addr(size_t _subsquareSz)
    {
        const size_t cnt = _subsquareSz*_subsquareSz;
        return m_addrAlctr.alloc_addr(cnt*cnt, cnt*cnt);
    }

    inline addr_type unpackAddress_x(addr_type _addr) const
    {
        return core::morton2d_decode_x(_addr&ADDR_MORTON_MASK);
    }
    inline addr_type unpackAddress_y(addr_type _addr) const
    {
        return core::morton2d_decode_y(_addr&ADDR_MORTON_MASK);
    }
    inline addr_type unpackAddress_layer(addr_type _addr) const
    {
        return (_addr>>ADDR_LAYER_SHIFT);
    }
};

class CTexturePacker
{
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t TILE_PADDING = 8u;

    _IRR_STATIC_INLINE_CONSTEXPR uint32_t ADDR_LAYER_SHIFT = 28u;//14bit 2d page address and 4bit for 3rd address dimension (layer)
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t ADDR_MORTON_MASK = (1u<<ADDR_LAYER_SHIFT)-1u;

    using addr_alctr_t = core::GeneralpurposeAddressAllocator<uint32_t>;

    static uint32_t megaimgLayerSize(uint32_t _pgSz, uint32_t _tilesPerDim)
    {
        return _tilesPerDim*(TILE_PADDING + _pgSz) + TILE_PADDING;
    }
    static core::vector2du32_SIMD pageCoords(uint32_t _x, uint32_t _y, uint32_t _pgSz)
    {
        core::vector2du32_SIMD coords(_x,_y);
        coords *= (_pgSz + TILE_PADDING);
        coords += TILE_PADDING;
        return coords;
    }

public:
    struct STexOffset
    {
        uint16_t x, y;
    };

    CTexturePacker(E_FORMAT _format, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u) :
        m_pgSzxy(_pgSzxy),
        m_tilesPerDim(_tilesPerDim),
        m_addrAlctr(14u, _tilesPerDim)
    {
        assert(core::isPoT(_tilesPerDim));
        assert(core::isPoT(_pgSzxy));

        const uint32_t megaimgSz = megaimgLayerSize(_pgSzxy, _tilesPerDim);

        ICPUImage::SCreationParams params;
        params.extent = {megaimgSz,megaimgSz,1u};
        params.format = _format;
        params.arrayLayers = _numLayers;
        params.mipLevels = 1u;
        params.type = IImage::ET_2D;
        params.samples = IImage::ESCF_1_BIT;
        params.flags = static_cast<IImage::E_CREATE_FLAGS>(0);

        m_megaimg = ICPUImage::create(std::move(params));
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull);
        auto& region = regions->front();
        region.imageSubresource.mipLevel = 0u;
        region.imageSubresource.baseArrayLayer = 0u;
        region.imageSubresource.layerCount = _numLayers;
        region.bufferOffset = 0u;
        region.bufferRowLength = megaimgSz;
        region.bufferImageHeight = 0u; //tightly packed
        region.imageOffset = {0u, 0u, 0u};
        region.imageExtent = {megaimgSz,megaimgSz,1u};
        auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(_format)*megaimgSz*megaimgSz*_numLayers);
        m_megaimg->setBufferAndRegions(std::move(buffer), regions);
    }

    STexOffset pack(const ICPUImage* _img)
    {
        assert(m_megaimg->getCreationParameters().format==_img->getCreationParameters().format);

        auto extent = _img->getCreationParameters().extent;
        uint32_t rnd = core::roundUpToPoT(std::max(extent.width,extent.height))/m_pgSzxy;
        uint32_t pgCnt = rnd*rnd;
        uint32_t addr = m_addrAlctr.alloc_addr(pgCnt, pgCnt);
        uint32_t addr_layer = m_addrAlctr.unpackAddress_layer(addr);
        assert(addr!=addr_alctr_t::invalid_address);

        const core::vector2du32_SIMD texOffset = pageCoords(m_addrAlctr.unpackAddress_x(addr), m_addrAlctr.unpackAddress_y(addr), m_pgSzxy);
        const uint32_t texelSz = getTexelOrBlockBytesize(m_megaimg->getCreationParameters().format);
        const uint64_t bufOffset = static_cast<uint64_t>(m_megaimg->getCreationParameters().extent.width*m_megaimg->getCreationParameters().extent.height*addr_layer + texOffset.y*m_megaimg->getCreationParameters().extent.width + texOffset.x) * texelSz;

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
    const uint32_t m_tilesPerDim;
    CSquareAddressAllocator<uint32_t> m_addrAlctr;//TODO this allocator will be used by GPU packer later to pack mip levels of stored later (when they actually exist)
    core::smart_refctd_ptr<ICPUImage> m_megaimg;
};

}}

#endif