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
    _IRR_STATIC_INLINE_CONSTEXPR addr_type invalid_address = lin_addr_alctr_t::invalid_address;

    CSquareAddressAllocator(uint32_t _addrBitCntPerDim, uint32_t _squareSz, uint32_t _layers) :
        ADDR_LAYER_SHIFT(_addrBitCntPerDim<<1),
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
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t ADDR_LAYER_SHIFT = 10u;//10bit 2d page address ( 32=2^(10/2) pages per dimension ) and up to 6bit for 3rd address dimension (layer)
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t ADDR_MORTON_MASK = (1u<<ADDR_LAYER_SHIFT)-1u;

    const uint32_t m_pgSzxy;
    const uint32_t m_tilesPerDim;
    CSquareAddressAllocator<uint32_t> m_addrAlctr;

public:
    struct STexOffset
    {
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t X_MASK = ((1u<<(ADDR_LAYER_SHIFT>>1))-1u);
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_addr = 0xffffu;

        STexOffset(uint32_t _addr) : addr(_addr) {}
        STexOffset(uint32_t _x, uint32_t _y, uint32_t _layer)
        {
            addr = (_x&X_MASK);
            addr |= (_y&X_MASK)<<(ADDR_LAYER_SHIFT>>1);
            addr |= (_layer<<ADDR_LAYER_SHIFT);
        }
        uint32_t x() const { return addr&X_MASK; }
        uint32_t y() const { return (addr>>(ADDR_LAYER_SHIFT>>1))&X_MASK; }
        uint32_t layer() const { return (addr>>ADDR_LAYER_SHIFT)&0xffffu; }

        bool isValid() const { return (addr&0xffffu)==invalid_addr; }
        bool hasMipTailAddr() const { return addr&(0xffffu<<16); }
        STexOffset getMipTailAddr() const { return addr>>16; }

        //upper 16 bits are used for address of mip-tail page
        uint32_t addr;
    };

    //TODO will work only with _tilesPerDim=32, for different values ADDR_LAYER_SHIFT must be adjusted as well to correctly retrieve 2d+layer addr from morton+layer
    ITexturePacker(uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u) :
        m_pgSzxy(_pgSzxy),
        m_tilesPerDim(_tilesPerDim),
        m_addrAlctr(ADDR_LAYER_SHIFT>>1, _tilesPerDim, _numLayers)
    {
        assert(core::isPoT(_tilesPerDim));
        assert(core::isPoT(_pgSzxy));//because of packing mip-tail
    }

    core::smart_refctd_dynamic_array<STexOffset> alloc(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        auto extent = _img->getCreationParameters().extent;
        uint32_t levelCount = 1u;
        for (uint32_t i = 0u; (extent.width>>(_subres.baseMipLevel+i) >= m_pgSzxy) || (extent.height>>(_subres.baseMipLevel+i) >= m_pgSzxy); ++i)
            levelCount = i+1u;
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
        }
        if (levelCount > _subres.levelCount)
        {
            const uint32_t addr = m_addrAlctr.alloc_addr(1u);//1 page for mip-tail (square 1x1)
            if (addr!=CSquareAddressAllocator<uint32_t>::invalid_address)
                (*offsets).back().addr |= STexOffset(m_addrAlctr.unpackAddress_x(addr), m_addrAlctr.unpackAddress_y(addr), m_addrAlctr.unpackAddress_layer(addr)).addr<<16;
        }
        
        return offsets;
    }
};

class ICPUTexturePacker : protected ITexturePacker
{
    uint32_t physAddrTexLayerSz(uint32_t _pgSz, uint32_t _tilesPerDim) const
    {
        return _tilesPerDim*(TILE_PADDING + _pgSz) + TILE_PADDING;
    }
    core::vector3du32_SIMD pageCoords(STexOffset _txoffset, uint32_t _pgSz) const
    {
        core::vector3du32_SIMD coords(_txoffset.x(),_txoffset.y(), 0u);
        coords *= (_pgSz + TILE_PADDING);
        coords += TILE_PADDING;
        coords.z = _txoffset.layer();
        return coords;
    }

public:
    ICPUTexturePacker(E_FORMAT _format, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u, uint32_t _tilePad = 9u/*max_aniso/2+1*/) :
        ITexturePacker(_pgSzxy, _tilesPerDim, _numLayers), TILE_PADDING(_tilePad)
    {
        {
            const uint32_t SZ = physAddrTexLayerSz(_pgSzxy, _tilesPerDim);

            ICPUImage::SCreationParams params;
            params.extent = { SZ,SZ,1u };
            params.format = _format;
            params.arrayLayers = _numLayers;
            params.mipLevels = 1u;
            params.type = IImage::ET_2D;
            params.samples = IImage::ESCF_1_BIT;
            params.flags = static_cast<IImage::E_CREATE_FLAGS>(0);

            m_physAddrTex = ICPUImage::create(std::move(params));
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull);
            auto& region = regions->front();
            region.imageSubresource.mipLevel = 0u;
            region.imageSubresource.baseArrayLayer = 0u;
            region.imageSubresource.layerCount = _numLayers;
            region.bufferOffset = 0u;
            region.bufferRowLength = SZ;
            region.bufferImageHeight = 0u; //tightly packed
            region.imageOffset = { 0u, 0u, 0u };
            region.imageExtent = { SZ,SZ,1u };
            auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(_format) * SZ * SZ * _numLayers);
            m_physAddrTex->setBufferAndRegions(std::move(buffer), regions);
        }
        {
            ICPUImage::SCreationParams params;
            params.arrayLayers = _numLayers;
            params.extent = {_tilesPerDim,_tilesPerDim,1u};
            params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0);
            params.format = EF_R16G16_UINT;
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
    }

    STexOffset pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        //assert(m_megaimg->getCreationParameters().format==_img->getCreationParameters().format);

        const uint32_t texelSz = getTexelOrBlockBytesize(m_physAddrTex->getCreationParameters().format);
        auto offsets = alloc(_img, _subres);
        for (const auto& reg : _img->getRegions())
        {
            if (reg.imageSubresource.mipLevel<_subres.baseMipLevel || reg.imageSubresource.mipLevel>=(_subres.baseMipLevel+offsets->size()))
                continue;

            const uint32_t j = reg.imageSubresource.mipLevel-_subres.baseMipLevel;//"relative LoD"

            const core::vector3du32_SIMD texOffset = pageCoords((*offsets)[j], m_pgSzxy);
            const uint64_t bufOffset = static_cast<uint64_t>(m_physAddrTex->getCreationParameters().extent.width*m_physAddrTex->getCreationParameters().extent.height*texOffset.z + texOffset.y*m_physAddrTex->getCreationParameters().extent.width + texOffset.x) * texelSz;

            const uint8_t* src = reinterpret_cast<const uint8_t*>(_img->getBuffer()->getPointer())+reg.bufferOffset;
            uint8_t* dst = reinterpret_cast<uint8_t*>(m_physAddrTex->getBuffer()->getPointer()) + bufOffset;
            const uint32_t pitch = reg.bufferRowLength*texelSz;
            for (uint32_t i = 0u; i < reg.imageExtent.height; ++i)
            {
                memcpy(dst + ((reg.imageOffset.y + i)*m_physAddrTex->getCreationParameters().extent.width + reg.imageOffset.x)*texelSz, src+i*pitch, reg.imageExtent.width*texelSz);
            }
        }
        //pack tail mips into 1 page (if needed)
        if (offsets->back().hasMipTailAddr())
        {
            const core::vector3du32_SIMD texOffset = pageCoords((*offsets).back().getMipTailAddr(), m_pgSzxy);
            //TODO
        }
        //fill page table
        const VkExtent3D extent = _img->getCreationParameters().extent;
        for (uint32_t i = 0u; i < offsets->size(); ++i)
        {
            const uint32_t w = std::max(extent.width>>(_subres.baseMipLevel+i),1u);
            const uint32_t h = std::max(extent.height>>(_subres.baseMipLevel+i),1u);
            const uint32_t pgSqrSz = core::roundUpToPoT((std::max(w,h)+m_pgSzxy-1u)/m_pgSzxy);

            uint32_t* const pgTab = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset
                );
            const uint32_t pitch = m_pageTable->getRegions().begin()[i].bufferRowLength;
            const uint32_t pgtH = m_pageTable->getRegions().begin()[i].imageExtent.height;
            const uint32_t offset = (pitch*pgtH)*(*offsets)[i].layer() + (*offsets)[i].y()*pitch + (*offsets)[i].x();
            for (uint32_t x = 0u; x < pgSqrSz; ++x)
                for (uint32_t y = 0u; y < pgSqrSz; ++y)
                    pgTab[offset + y*pitch + x] = (*offsets)[i].addr;
        }

        return offsets->front();
    }

    ICPUImage* getImage() { return m_physAddrTex.get(); }

private:
    core::smart_refctd_ptr<ICPUImage> m_physAddrTex;
    core::smart_refctd_ptr<ICPUImage> m_pageTable;
    const uint32_t TILE_PADDING;
};

}}

#endif