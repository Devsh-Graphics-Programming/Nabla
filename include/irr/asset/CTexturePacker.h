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

    uint32_t ADDR_LAYER_SHIFT;
    uint32_t ADDR_MORTON_MASK;

    core::smart_refctd_dynamic_array<uint8_t> m_alctrReservedSpace;
    lin_addr_alctr_t m_addrAlctr;

public:
    _IRR_STATIC_INLINE_CONSTEXPR addr_type invalid_address = lin_addr_alctr_t::invalid_address;

    CSquareAddressAllocator(uint32_t _addrBitCntPerDim, uint32_t _squareSz, uint32_t _layers) :
        ADDR_LAYER_SHIFT(_addrBitCntPerDim*2u),
        ADDR_MORTON_MASK((1u<<ADDR_LAYER_SHIFT)-1u),
        m_alctrReservedSpace(core::make_refctd_dynamic_array<decltype(m_alctrReservedSpace)>(
            lin_addr_alctr_t::reserved_size(_squareSz*_squareSz, _squareSz*_squareSz, 1ull))
        ),
        m_addrAlctr(reinterpret_cast<void*>(m_alctrReservedSpace->data()), 0u, 0u, _squareSz*_squareSz, _squareSz*_squareSz*_layers, 1u)
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

class ITexturePacker
{
protected:
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t ADDR_LAYER_SHIFT = 28u;//14bit 2d page address and 4bit for 3rd address dimension (layer)
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t ADDR_MORTON_MASK = (1u<<ADDR_LAYER_SHIFT)-1u;

    const uint32_t m_pgSzxy;
    const uint32_t m_tilesPerDim;
    CSquareAddressAllocator<uint32_t> m_addrAlctr;

    core::smart_refctd_ptr<ICPUImage> m_pageTable;
public:
    struct STexOffset
    {
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t X_MASK = ((1u<<(ADDR_LAYER_SHIFT>>1))-1u);
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_addr = (~0u);

        STexOffset(uint32_t _addr) : addr(_addr) {}
        STexOffset(uint32_t _x, uint32_t _y, uint32_t _layer)
        {
            addr = (_x&X_MASK);
            addr |= (_y&X_MASK)<<(ADDR_LAYER_SHIFT>>1);
            addr |= (_layer<<ADDR_LAYER_SHIFT);
        }
        uint32_t x() const { return addr&X_MASK; }
        uint32_t y() const { return (addr>>(ADDR_LAYER_SHIFT>>1))&X_MASK; }
        uint32_t layer() const { return addr>>ADDR_LAYER_SHIFT; }

        uint32_t addr;
    };

    ITexturePacker(uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u) :
        m_pgSzxy(_pgSzxy),
        m_tilesPerDim(_tilesPerDim),
        m_addrAlctr(14u, _tilesPerDim, _numLayers)
    {
        ICPUImage::SCreationParams params;
        params.arrayLayers = _numLayers;
        params.extent = {_tilesPerDim,_tilesPerDim,1u};
        params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0);
        params.format = EF_R32_UINT;
        params.mipLevels = core::findMSB(_tilesPerDim)+1;
        params.samples = ICPUImage::ESCF_1_BIT;
        params.type = IImage::ET_2D;
        m_pageTable = ICPUImage::create(std::move(params));
        
        auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(m_pageTable->getCreationParameters().mipLevels);

        uint32_t bufOffset = 0u;
        for (uint32_t i = 0u; i < m_pageTable->getCreationParameters().mipLevels; ++i)
        {
            const uint32_t tilesPerLodDim = _tilesPerDim>>i;
            const uint32_t regionSz = tilesPerLodDim*tilesPerLodDim*_numLayers*4u;
            auto& region = (*regions)[i];
            region.bufferOffset = bufOffset;
            region.bufferImageHeight = 0u;
            region.bufferRowLength = tilesPerLodDim*4u;
            region.imageExtent = {tilesPerLodDim,tilesPerLodDim,1u};
            region.imageOffset = {0,0,0};
            region.imageSubresource.baseArrayLayer = 0u;
            region.imageSubresource.layerCount = _numLayers;
            region.imageSubresource.mipLevel = i;
            region.imageSubresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);

            bufOffset += regionSz;
        }
        auto buf = core::make_smart_refctd_ptr<ICPUBuffer>(bufOffset);

        m_pageTable->setBufferAndRegions(std::move(buf), regions);
    }

    core::smart_refctd_dynamic_array<STexOffset> alloc(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        auto extent = _img->getCreationParameters().extent;
        uint32_t levelCount = 1u;
        for (uint32_t i = 0u; (extent.width>>(_subres.baseMipLevel+i) > m_pgSzxy) || (extent.height>>(_subres.baseMipLevel+i) > m_pgSzxy); ++i)
            levelCount = i;
        levelCount = std::min(_subres.levelCount, levelCount);

        auto offsets = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<STexOffset>>(levelCount);
        for (uint32_t i = 0u; i < levelCount; ++i)
        {
            const uint32_t w = std::max(extent.width>>(_subres.baseMipLevel+i),1u);
            const uint32_t h = std::max(extent.height>>(_subres.baseMipLevel+i),1u);
            const uint32_t pgSqrSz = core::roundUpToPoT((std::max(w,h)+m_pgSzxy-1u)/m_pgSzxy);

            const uint32_t addr = m_addrAlctr.alloc_addr(pgSqrSz);
            (*offsets)[i] = (addr==CSquareAddressAllocator<uint32_t>::invalid_address) ? 
                STexOffset::invalid_addr :
                STexOffset(m_addrAlctr.unpackAddress_x(addr), m_addrAlctr.unpackAddress_y(addr), m_addrAlctr.unpackAddress_layer(addr));

            //i didnt know where to put page table page table and filling code so it is here for now
            uint32_t* const pgTab = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset
                );
            const uint32_t pitch = m_pageTable->getRegions().begin()[i].bufferRowLength;
            const uint32_t offset = (pitch*(m_tilesPerDim>>i))*(*offsets)[i].layer() + (*offsets)[i].y()*pitch + (*offsets)[i].x();
            for (uint32_t x = 0u; x < pgSqrSz; ++x)
                for (uint32_t y = 0u; y < pgSqrSz; ++y)
                    pgTab[offset + y*pitch + x] = (*offsets)[i].addr;
        }
        
        return offsets;
    }
};

class ICPUTexturePacker : protected ITexturePacker
{
    uint32_t megaimgLayerSize(uint32_t _pgSz, uint32_t _tilesPerDim) const
    {
        return _tilesPerDim*(TILE_PADDING + _pgSz) + TILE_PADDING;
    }
    core::vector2du32_SIMD pageCoords(uint32_t _x, uint32_t _y, uint32_t _pgSz) const
    {
        core::vector2du32_SIMD coords(_x,_y);
        coords *= (_pgSz + TILE_PADDING);
        coords += TILE_PADDING;
        return coords;
    }

public:
    ICPUTexturePacker(E_FORMAT _format, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u) :
        ITexturePacker(_pgSzxy, _tilesPerDim, _numLayers)
    {
        assert(core::isPoT(_tilesPerDim));

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

    STexOffset pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        //assert(m_megaimg->getCreationParameters().format==_img->getCreationParameters().format);

        const uint32_t texelSz = getTexelOrBlockBytesize(m_megaimg->getCreationParameters().format);
        auto offsets = alloc(_img, _subres);
        for (const auto& reg : _img->getRegions())
        {
            if (reg.imageSubresource.mipLevel>=_subres.baseMipLevel && reg.imageSubresource.mipLevel<(_subres.baseMipLevel+offsets->size()))
                continue;

            const uint32_t j = reg.imageSubresource.mipLevel-_subres.baseMipLevel;

            const core::vector2du32_SIMD texOffset = pageCoords((*offsets)[j].x(), (*offsets)[j].y(), m_pgSzxy);
            const uint64_t bufOffset = static_cast<uint64_t>(m_megaimg->getCreationParameters().extent.width * m_megaimg->getCreationParameters().extent.height * (*offsets)[j].layer() + texOffset.y * m_megaimg->getCreationParameters().extent.width + texOffset.x)* texelSz;

            const uint8_t* src = reinterpret_cast<const uint8_t*>(_img->getBuffer()->getPointer())+reg.bufferOffset;
            uint8_t* dst = reinterpret_cast<uint8_t*>(m_megaimg->getBuffer()->getPointer()) + bufOffset;
            const uint32_t pitch = reg.bufferRowLength*texelSz;
            for (uint32_t i = 0u; i < reg.imageExtent.height; ++i)
            {
                memcpy(dst + ((reg.imageOffset.y + i)*m_megaimg->getCreationParameters().extent.width + reg.imageOffset.x)*texelSz, src+i*pitch, reg.imageExtent.width*texelSz);
            }
        }

        return offsets->front();
    }

    ICPUImage* getImage() { return m_megaimg.get(); }

private:
    core::smart_refctd_ptr<ICPUImage> m_megaimg;
    const uint32_t TILE_PADDING = 8u;
};

}}

#endif