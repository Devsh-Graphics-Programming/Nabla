#ifndef __IRR_C_TEXTURE_PACKER_H_INCLUDED__
#define __IRR_C_TEXTURE_PACKER_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/asset/ICPUImage.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/PoolAddressAllocator.h"
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

    const uint32_t ADDR_MORTON_MASK;

    core::smart_refctd_dynamic_array<uint8_t> m_alctrReservedSpace;
    lin_addr_alctr_t m_addrAlctr;

public:
    _IRR_STATIC_INLINE_CONSTEXPR addr_type invalid_address = lin_addr_alctr_t::invalid_address;

    CSquareAddressAllocator(uint32_t _squareSz) :
        ADDR_MORTON_MASK((_squareSz*_squareSz)-1u),
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
};

class ITexturePacker
{
protected:
    const uint32_t ADDR_LAYER_SHIFT;
    const uint32_t ADDR_MORTON_MASK;
    const uint32_t TEX_OFFSET_X_MASK;

    const uint32_t m_pgSzxy;
    const uint32_t m_tilesPerDim;
    CSquareAddressAllocator<uint32_t> m_pgtAddrAlctr;

public:
    using page_tab_offset_t = core::vector2du32_SIMD;
    static page_tab_offset_t page_tab_offset_invalid() { return page_tab_offset_t(~0u,~0u); }

    //! STexOffset is what is stored in texels of page table!
    struct STexOffset
    {
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_addr = 0xffffu;

        STexOffset(uint32_t _addr) : addr(_addr) {}

        //upper 16 bits are used for address of mip-tail page
        uint32_t addr;
    };
    STexOffset makeTexOffset(uint32_t _x, uint32_t _y, uint32_t _layer)
    {
        uint32_t addr = (_x&TEX_OFFSET_X_MASK);
        addr |= (_y&TEX_OFFSET_X_MASK)<<(ADDR_LAYER_SHIFT>>1);
        addr |= (_layer<<ADDR_LAYER_SHIFT);
        addr |= (STexOffset::invalid_addr<<16);
        return addr;
    }
    uint32_t texOffset_x(STexOffset _offset) const { return _offset.addr&TEX_OFFSET_X_MASK; }
    uint32_t texOffset_y(STexOffset _offset) const { return (_offset.addr>>(ADDR_LAYER_SHIFT>>1))&TEX_OFFSET_X_MASK; }
    uint32_t texOffset_layer(STexOffset _offset) const { return (_offset.addr&0xffffu)>>ADDR_LAYER_SHIFT; }
    bool texOffset_valid(STexOffset _offset) const { return (_offset.addr&0xffffu)!=STexOffset::invalid_addr; }
    bool texOffset_hasMipTailAddr(STexOffset _offset) const { return texOffset_valid(texOffset_mipTailAddr(_offset)); }
    STexOffset texOffset_mipTailAddr(STexOffset _offset) const { return _offset.addr>>16; }

    ITexturePacker(uint32_t _pgTabSzxy, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u) :
        ADDR_LAYER_SHIFT(core::findMSB(_tilesPerDim)<<1),
        ADDR_MORTON_MASK((1u<<ADDR_LAYER_SHIFT)-1u),
        TEX_OFFSET_X_MASK((1u<<(ADDR_LAYER_SHIFT>>1))-1u),
        m_pgSzxy(_pgSzxy),
        m_tilesPerDim(_tilesPerDim),
        m_pgtAddrAlctr(_pgTabSzxy)
    {
        assert(core::isPoT(_pgTabSzxy));//because of allocation using morton codes
        assert(core::isPoT(_tilesPerDim)); //actually physical addr tex doesnt even have to be square nor PoT, but it makes addr/offset encoding easier
        assert(core::isPoT(_pgSzxy));//because of packing mip-tail
    }

    page_tab_offset_t alloc(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        auto extent = _img->getCreationParameters().extent;

        const uint32_t w = std::max(extent.width>>_subres.baseMipLevel,1u);
        const uint32_t h = std::max(extent.height>>_subres.baseMipLevel,1u);
        const uint32_t pgSqrSz = core::roundUpToPoT((std::max(w,h)+m_pgSzxy-1u)/m_pgSzxy);

        const uint32_t addr = m_pgtAddrAlctr.alloc_addr(pgSqrSz);
        return (addr==CSquareAddressAllocator<uint32_t>::invalid_address) ? 
            page_tab_offset_invalid() :
            page_tab_offset_t(m_pgtAddrAlctr.unpackAddress_x(addr), m_pgtAddrAlctr.unpackAddress_y(addr));
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
        core::vector3du32_SIMD coords(texOffset_x(_txoffset),texOffset_y(_txoffset), 0u);
        coords *= (_pgSz + TILE_PADDING);
        coords += TILE_PADDING;
        coords.z = texOffset_layer(_txoffset);
        return coords;
    }

public:
    ICPUTexturePacker(E_FORMAT _format, uint32_t _pgTabSzxy, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u, uint32_t _tilePad = 9u/*max_aniso/2+1*/) :
        ITexturePacker(_pgTabSzxy, _pgSzxy, _tilesPerDim), TILE_PADDING(_tilePad),
        m_physPgAddrAlctr_reservedSpc(core::make_refctd_dynamic_array<decltype(m_physPgAddrAlctr_reservedSpc)>(phys_pg_addr_alctr_t::reserved_size(1u, _tilesPerDim*_tilesPerDim, 1u))),
        m_physPgAddrAlctr(m_physPgAddrAlctr_reservedSpc->data(), 0u, 0u, 1u, _numLayers*_tilesPerDim*_tilesPerDim, 1u)
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
            params.arrayLayers = 1u;
            params.extent = {_pgTabSzxy,_pgTabSzxy,1u};
            params.flags = static_cast<ICPUImage::E_CREATE_FLAGS>(0);
            params.format = EF_R16G16_UINT;
            params.mipLevels = core::findMSB(_pgTabSzxy)+1;
            params.samples = ICPUImage::ESCF_1_BIT;
            params.type = IImage::ET_2D;
            m_pageTable = ICPUImage::create(std::move(params));

            const uint32_t texelSz = getTexelOrBlockBytesize(m_pageTable->getCreationParameters().format);
        
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(m_pageTable->getCreationParameters().mipLevels);

            uint32_t bufOffset = 0u;
            for (uint32_t i = 0u; i < m_pageTable->getCreationParameters().mipLevels; ++i)
            {
                const uint32_t tilesPerLodDim = _pgTabSzxy>>i;
                const uint32_t regionSz = tilesPerLodDim*tilesPerLodDim*texelSz;
                auto& region = (*regions)[i];
                region.bufferOffset = bufOffset;
                region.bufferImageHeight = 0u;
                region.bufferRowLength = tilesPerLodDim*texelSz;
                region.imageExtent = {tilesPerLodDim,tilesPerLodDim,1u};
                region.imageOffset = {0,0,0};
                region.imageSubresource.baseArrayLayer = 0u;
                region.imageSubresource.layerCount = 1u;
                region.imageSubresource.mipLevel = i;
                region.imageSubresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);

                bufOffset += regionSz;
            }
            auto buf = core::make_smart_refctd_ptr<ICPUBuffer>(bufOffset);

            m_pageTable->setBufferAndRegions(std::move(buf), regions);
        }
    }

    page_tab_offset_t pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        //assert(m_megaimg->getCreationParameters().format==_img->getCreationParameters().format);
        const page_tab_offset_t pgtOffset = alloc(_img, _subres);
        if ((pgtOffset==page_tab_offset_invalid()).all())
            return pgtOffset;

        auto extent = _img->getCreationParameters().extent;
        uint32_t levelCount = 1u;
        for (uint32_t i = 0u; (extent.width>>(_subres.baseMipLevel+i) >= m_pgSzxy) || (extent.height>>(_subres.baseMipLevel+i) >= m_pgSzxy); ++i)
            levelCount = i+1u;
        levelCount = std::min(_subres.levelCount, levelCount);

        uint32_t miptailPgAddr = (~0u);
        const uint32_t texelSz = getTexelOrBlockBytesize(m_physAddrTex->getCreationParameters().format);
        //fill page table and pack present mips into physical addr texture
        for (uint32_t i = 0u; i < levelCount; ++i)
        {
            const uint32_t w = (std::max(extent.width>>(_subres.baseMipLevel+i),1u) + m_pgSzxy-1u) / m_pgSzxy;
            const uint32_t h = (std::max(extent.height>>(_subres.baseMipLevel+i),1u) + m_pgSzxy-1u) / m_pgSzxy;

            uint32_t* const pgTab = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset
                );
            const uint32_t pgtPitch = m_pageTable->getRegions().begin()[i].bufferRowLength;
            const uint32_t pgtH = m_pageTable->getRegions().begin()[i].imageExtent.height;
            const uint32_t offset = (pgtOffset.y>>i)*pgtPitch + (pgtOffset.x>>i);

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t physPgAddr = m_physPgAddrAlctr.alloc_addr(1u, 1u);
                    assert(physPgAddr<0xffffu);
                    if (physPgAddr==phys_pg_addr_alctr_t::invalid_address)
                        return pgtOffset;//TODO report an error of allocating physical pages somehow

                    if (i==(levelCount-1u) && levelCount>_subres.levelCount)
                    {
                        assert(w==1u && h==1u);
                        //TODO alloc another address and store on upper 16 bits of `addr` (for mip-tail)
                        const uint32_t physMiptailPgAddr = m_physPgAddrAlctr.alloc_addr(1u, 1u);
                        miptailPgAddr = physMiptailPgAddr;
                        assert(physMiptailPgAddr<0xffffu);
                        physPgAddr |= (physMiptailPgAddr<<16);
                    }
                    pgTab[offset + y*pgtPitch + x] = physPgAddr;

                    core::vector3du32_SIMD physPg = pageCoords(physPgAddr, m_pgSzxy);
                    for (const auto& reg : _img->getRegions())
                    {
                        if (reg.imageSubresource.mipLevel != (_subres.baseMipLevel+i))
                            continue;

                        auto src_txOffset = core::vector2du32_SIMD(x,y)*m_pgSzxy;

                        const uint32_t a_left = reg.imageOffset.x;
                        const uint32_t b_right = src_txOffset.x + m_pgSzxy;
                        const uint32_t a_right = a_left + reg.imageExtent.width;
                        const uint32_t b_left = src_txOffset.x;
                        const uint32_t a_bot = reg.imageOffset.y;
                        const uint32_t b_top = src_txOffset.y + m_pgSzxy;
                        const uint32_t a_top = a_bot + reg.imageExtent.height;
                        const uint32_t b_bot = src_txOffset.y;
                        if (a_left>b_right || a_right<b_left || a_top<b_bot || a_bot>b_top)
                            continue;

                        const core::vector2du32_SIMD regOffset = core::vector2du32_SIMD(a_left, a_bot);
                        const core::vector2du32_SIMD withinRegTxOffset = core::max(src_txOffset, regOffset)-regOffset;
                        const core::vector2du32_SIMD withinPgTxOffset = (withinRegTxOffset & core::vector2du32_SIMD(m_pgSzxy-1u));
                        src_txOffset += withinPgTxOffset;
                        const core::vector2du32_SIMD src_txOffsetEnd = core::min(core::vector2du32_SIMD(a_right,a_top), core::vector2du32_SIMD(b_right,b_top));

                        physPg += withinPgTxOffset;

                        const core::vector2du32_SIMD cpExtent = src_txOffsetEnd - src_txOffset;
                        const uint32_t src_offset_lin = (withinRegTxOffset.y*reg.bufferRowLength + withinRegTxOffset.x) * texelSz;
                        const uint32_t dst_offset_lin = (m_physAddrTex->getCreationParameters().extent.width*m_physAddrTex->getCreationParameters().extent.height*physPg.z + physPg.y*m_physAddrTex->getCreationParameters().extent.width + physPg.x) * texelSz;
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(_img->getBuffer()->getPointer()) + reg.bufferOffset + src_offset_lin;
                        uint8_t* dst = reinterpret_cast<uint8_t*>(m_physAddrTex->getBuffer()->getPointer()) + dst_offset_lin;
                        for (uint32_t j = 0u; j < cpExtent.y; ++j)
                        {
                            memcpy(dst + j*m_physAddrTex->getCreationParameters().extent.width*texelSz, src + j*reg.bufferRowLength*texelSz, cpExtent.x*texelSz);
                        }
                    }
                }
        }

        return pgtOffset;
    }

    ICPUImage* getImage() { return m_physAddrTex.get(); }

private:
    core::smart_refctd_ptr<ICPUImage> m_physAddrTex;
    core::smart_refctd_ptr<ICPUImage> m_pageTable;
    const uint32_t TILE_PADDING;

    using phys_pg_addr_alctr_t = core::PoolAddressAllocator<uint32_t>;
    core::smart_refctd_dynamic_array<uint8_t> m_physPgAddrAlctr_reservedSpc;
    phys_pg_addr_alctr_t m_physPgAddrAlctr;
};

}}

#endif