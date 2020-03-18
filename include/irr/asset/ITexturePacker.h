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
        const size_t cnt = _subsquareSz;
        return m_addrAlctr.alloc_addr(cnt*cnt, cnt*cnt);
    }
    inline void free_addr(addr_type _addr, size_t _subsquareSz)
    {
        m_addrAlctr.free_addr(_addr, _subsquareSz*_subsquareSz);
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

class ITexturePacker : public core::IReferenceCounted
{
protected:
    const uint32_t ADDR_LAYER_SHIFT;
    const uint32_t ADDR_MORTON_MASK;
    const uint32_t PHYS_PG_OFFSET_X_MASK;

    const uint32_t m_pgSzxy;
    const uint32_t m_tilesPerDim;
    CSquareAddressAllocator<uint32_t> m_pgtAddrAlctr;

public:
    using page_tab_offset_t = core::vector2du32_SIMD;
    static page_tab_offset_t page_tab_offset_invalid() { return page_tab_offset_t(~0u,~0u); }

    ITexturePacker(uint32_t _pgTabSzxy, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u) :
        ADDR_LAYER_SHIFT(core::findMSB(_tilesPerDim)<<1),
        ADDR_MORTON_MASK((1u<<ADDR_LAYER_SHIFT)-1u),
        PHYS_PG_OFFSET_X_MASK((1u<<(ADDR_LAYER_SHIFT>>1))-1u),
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
        const uint32_t pgSqrSz = computeSquareSz(_img, _subres);

        const uint32_t addr = m_pgtAddrAlctr.alloc_addr(pgSqrSz);
        return (addr==CSquareAddressAllocator<uint32_t>::invalid_address) ? 
            page_tab_offset_invalid() :
            page_tab_offset_t(m_pgtAddrAlctr.unpackAddress_x(addr), m_pgtAddrAlctr.unpackAddress_y(addr));
    }
    virtual void free(page_tab_offset_t _addr, const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        if ((_addr==page_tab_offset_invalid()).all())
            return;

        const uint32_t pgSqrSz = computeSquareSz(_img, _subres);

        m_pgtAddrAlctr.free_addr(core::morton2d_encode(_addr.x,_addr.y), pgSqrSz);
    }

private:
    uint32_t computeSquareSz(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        auto extent = _img->getCreationParameters().extent;

        const uint32_t w = std::max(extent.width>>_subres.baseMipLevel,1u);
        const uint32_t h = std::max(extent.height>>_subres.baseMipLevel,1u);

        return core::roundUpToPoT((std::max(w,h)+m_pgSzxy-1u)/m_pgSzxy);
    }
};

class ICPUTexturePacker : protected ITexturePacker
{
public:
    //! SPhysPgOffset is what is stored in texels of page table!
    struct SPhysPgOffset
    {
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_addr = 0xffffu;

        SPhysPgOffset(uint32_t _addr) : addr(_addr) {}

        //upper 16 bits are used for address of mip-tail page
        uint32_t addr;
    };
    SPhysPgOffset makeTexOffset(uint32_t _x, uint32_t _y, uint32_t _layer)
    {
        uint32_t addr = (_x & PHYS_PG_OFFSET_X_MASK);
        addr |= (_y & PHYS_PG_OFFSET_X_MASK) << (ADDR_LAYER_SHIFT >> 1);
        addr |= (_layer << ADDR_LAYER_SHIFT);
        addr |= (SPhysPgOffset::invalid_addr << 16);
        return addr;
    }
    uint32_t physPgOffset_x(SPhysPgOffset _offset) const { return _offset.addr & PHYS_PG_OFFSET_X_MASK; }
    uint32_t physPgOffset_y(SPhysPgOffset _offset) const { return (_offset.addr >> (ADDR_LAYER_SHIFT>>1)) & PHYS_PG_OFFSET_X_MASK; }
    uint32_t physPgOffset_layer(SPhysPgOffset _offset) const { return (_offset.addr & 0xffffu)>>ADDR_LAYER_SHIFT; }
    bool physPgOffset_valid(SPhysPgOffset _offset) const { return (_offset.addr&0xffffu) != SPhysPgOffset::invalid_addr; }
    bool physPgOffset_hasMipTailAddr(SPhysPgOffset _offset) const { return physPgOffset_valid(physPgOffset_mipTailAddr(_offset)); }
    SPhysPgOffset physPgOffset_mipTailAddr(SPhysPgOffset _offset) const { return _offset.addr>>16; }

    uint32_t physAddrTexLayerSz() const
    {
        return m_tilesPerDim*(TILE_PADDING+m_pgSzxy) + TILE_PADDING;
    }
    //! @returns texel-wise offset of physical page
    core::vector3du32_SIMD pageCoords(SPhysPgOffset _txoffset, uint32_t _pgSz) const
    {
        core::vector3du32_SIMD coords(physPgOffset_x(_txoffset), physPgOffset_y(_txoffset), 0u);
        coords *= (_pgSz + TILE_PADDING);
        coords += TILE_PADDING;
        coords.z = physPgOffset_layer(_txoffset);
        return coords;
    }
    struct SMiptailPacker
    {
        struct rect
        {
            int x, y, mx, my;
        };
        static bool computeMiptailOffsets(rect* res, int log2SIZE, int padding);
    };

    ICPUTexturePacker(E_FORMAT _format, uint32_t _pgTabSzxy, uint32_t _pgSzxy = 256u, uint32_t _tilesPerDim = 32u, uint32_t _numLayers = 4u, uint32_t _tilePad = 9u/*max_aniso/2+1*/) :
        ITexturePacker(_pgTabSzxy, _pgSzxy, _tilesPerDim), TILE_PADDING(_tilePad),
        m_physPgAddrAlctr_reservedSpc(core::make_refctd_dynamic_array<decltype(m_physPgAddrAlctr_reservedSpc)>(phys_pg_addr_alctr_t::reserved_size(1u, _tilesPerDim*_tilesPerDim, 1u))),
        m_physPgAddrAlctr(m_physPgAddrAlctr_reservedSpc->data(), 0u, 0u, 1u, _numLayers*_tilesPerDim*_tilesPerDim, 1u)
    {
        {
            const uint32_t SZ = physAddrTexLayerSz();

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
            uint32_t* bufptr = reinterpret_cast<uint32_t*>(buf->getPointer());
            std::fill(bufptr, bufptr+bufOffset/4u, SPhysPgOffset::invalid_addr);

            m_pageTable->setBufferAndRegions(std::move(buf), regions);
        }
        const uint32_t pgSzLog2 = core::findMSB(m_pgSzxy);
        bool ok = SMiptailPacker::computeMiptailOffsets(m_miptailOffsets, pgSzLog2, TILE_PADDING);
        assert(ok);
    }

    void free(page_tab_offset_t _addr, const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres) override
    {
        //free physical pages
        auto extent = _img->getCreationParameters().extent;
        const uint32_t levelCount = countLevelsTakingAtLeastOnePage(extent, _subres);

        for (uint32_t i = 0u; i < levelCount; ++i)
        {
            const uint32_t w = (std::max(extent.width>>(_subres.baseMipLevel+i),1u) + m_pgSzxy-1u) / m_pgSzxy;
            const uint32_t h = (std::max(extent.height>>(_subres.baseMipLevel+i),1u) + m_pgSzxy-1u) / m_pgSzxy;

            uint32_t* const pgTab = reinterpret_cast<uint32_t*>(
                reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + m_pageTable->getRegions().begin()[i].bufferOffset
                );
            const uint32_t pgtPitch = m_pageTable->getRegions().begin()[i].bufferRowLength;
            const uint32_t offset = (_addr.y>>i)*pgtPitch + (_addr.x>>i);

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    const uint32_t linTexelAddr = offset + y*pgtPitch + x;

                    SPhysPgOffset physPgOffset = pgTab[linTexelAddr];
                    if (physPgOffset_valid(physPgOffset))
                    {
                        pgTab[linTexelAddr] = SPhysPgOffset::invalid_addr;

                        m_physPgAddrAlctr.free_addr(physPgOffset.addr&0xffffu, 1u);
                        if (physPgOffset_hasMipTailAddr(physPgOffset))
                            m_physPgAddrAlctr.free_addr(physPgOffset_mipTailAddr(physPgOffset).addr&0xffffu, 1u);
                    }
                }
        }

        //free entries in page table
        ITexturePacker::free(_addr, _img, _subres);
    }

    page_tab_offset_t pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        //assert(m_megaimg->getCreationParameters().format==_img->getCreationParameters().format);
        const page_tab_offset_t pgtOffset = alloc(_img, _subres);
        if ((pgtOffset==page_tab_offset_invalid()).all())
            return pgtOffset;

        auto extent = _img->getCreationParameters().extent;
        const uint32_t levelCount = countLevelsTakingAtLeastOnePage(extent, _subres);

        uint32_t miptailPgAddr = SPhysPgOffset::invalid_addr;

        const uint32_t texelSz = getTexelOrBlockBytesize(m_physAddrTex->getCreationParameters().format);
        //fill page table and pack present mips into physical addr texture
        for (uint32_t i = 0u; i < _subres.levelCount; ++i)
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
                    uint32_t physPgAddr = (i>=levelCount) ? miptailPgAddr : m_physPgAddrAlctr.alloc_addr(1u, 1u);
                    assert(physPgAddr<SPhysPgOffset::invalid_addr);
                    if (physPgAddr==phys_pg_addr_alctr_t::invalid_address)
                    {
                        free(pgtOffset, _img, _subres);
                        return pgtOffset;
                    }

                    if (i==(levelCount-1u) && levelCount<_subres.levelCount)
                    {
                        assert(w==1u && h==1u);
                        uint32_t physMiptailPgAddr = m_physPgAddrAlctr.alloc_addr(1u, 1u);
                        assert(physMiptailPgAddr<SPhysPgOffset::invalid_addr);
                        miptailPgAddr = physMiptailPgAddr = (physMiptailPgAddr==phys_pg_addr_alctr_t::invalid_address) ? SPhysPgOffset::invalid_addr : physMiptailPgAddr;
                        physPgAddr |= (physMiptailPgAddr<<16);
                    }
                    else 
                        physPgAddr |= (SPhysPgOffset::invalid_addr<<16);
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
                        //special offset for packing tail mip levels into single page
                        const core::vector2du32_SIMD miptailOffset = (i>=levelCount) ? core::vector2du32_SIMD(m_miptailOffsets[i-levelCount].x,m_miptailOffsets[i-levelCount].y)+core::vector2du32_SIMD(TILE_PADDING,TILE_PADDING) : core::vector2du32_SIMD(0u,0u);

                        physPg += (withinPgTxOffset + miptailOffset);

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

    ICPUImage* getPhysicalAddressTexture() { return m_physAddrTex.get(); }
    ICPUImage* getPageTable() { return m_pageTable.get(); }

private:
    uint32_t countLevelsTakingAtLeastOnePage(const VkExtent3D& _extent, const ICPUImage::SSubresourceRange& _subres)
    {
        uint32_t levelCount = 1u;
        for (uint32_t i = 0u; (_extent.width>>(_subres.baseMipLevel+i) >= m_pgSzxy) || (_extent.height>>(_subres.baseMipLevel+i) >= m_pgSzxy); ++i)
            levelCount = i+1u;
        return std::min(_subres.levelCount, levelCount);
    }

    core::smart_refctd_ptr<ICPUImage> m_physAddrTex;
    core::smart_refctd_ptr<ICPUImage> m_pageTable;
    const uint32_t TILE_PADDING;

    using phys_pg_addr_alctr_t = core::PoolAddressAllocator<uint32_t>;
    core::smart_refctd_dynamic_array<uint8_t> m_physPgAddrAlctr_reservedSpc;
    phys_pg_addr_alctr_t m_physPgAddrAlctr;

    SMiptailPacker::rect m_miptailOffsets[9];
};

bool ICPUTexturePacker::SMiptailPacker::computeMiptailOffsets(rect* res, int log2SIZE, int padding=9)
{
    if (log2SIZE<7 || log2SIZE>10)
        return false;
    
    int SIZE = 0x1u<<log2SIZE;
	if ((SIZE>>1)+(SIZE>>2) + padding * 4 > SIZE)
		return false;
	

	int x1 = 0, y1 = 0;
	res[0].x = 0;
	res[0].mx = (SIZE >> 1) + padding * 2 -1;
	res[0].y = 0;
	res[0].my = (SIZE >> 1) + padding * 2 -1;
	y1 = res[0].my + 1;
	x1 = res[0].mx + 1;

	bool ofx1 = false, ofx2 = false;
	int i = 1;
	while (i < log2SIZE)
	{

		int s = (SIZE >> (i + 1)) + padding * 2;
		int x, y;
		if (i % 2 == 0) {	//on right
			if (i == 2)
			{
				x = x1;
				y = 0;
			}
			else
			{
				if (res[i - 2].my + s > y1)
				{
					if (res[i - 6].mx + s > SIZE || ofx1)
					{
						x = res[i - 4].mx + 1;
						y = res[i - 4].y;
						ofx1 = true;
					}
					else {

					x = res[i - 6].mx + 1;
					y = res[i - 6].y;
					}
				}
				else
				{
					y = res[i - 2].my + 1;
					x = x1;
				}
			}
		}
		else //those placed below the first square and going right are ~x2 larger than those above
		{
			if (i == 1)
			{
				x = 0;
				y = y1;
			}
			else {
				if (res[i - 2].mx + s > SIZE)
				{
					if (res[i - 6].my + s > SIZE ||ofx2)
					{
						x = res[i - 4].x;
						y = res[i - 4].my + 1;
						ofx2 = true;
					}
					else {
						x = res[i - 6].x;
						y = res[i - 6].my + 1;
					}
				}
				else 
				{
				    x = res[i - 2].mx + 1;
				    y = y1;
				}
			}
		}
		res[i].x = x;
		res[i].y = y;
		res[i].mx = x + s -1;
		res[i].my = y + s -1 ;

		i++;
	}
    return true;
}

}}

#endif