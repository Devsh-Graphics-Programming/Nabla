#ifndef __IRR_C_TEXTURE_PACKER_H_INCLUDED__
#define __IRR_C_TEXTURE_PACKER_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/asset/ICPUImage.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/core/math/morton.h"
#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/memory/memory.h"
//#include "irr/asset/filters/CCopyImageFilter.h"
#include "irr/asset/filters/CPaddedCopyImageFilter.h"
#include "irr/asset/filters/CFillImageFilter.h"

namespace irr {
namespace asset
{

class ITexturePacker : public core::IReferenceCounted
{
protected:
    const uint32_t m_addr_layerShift;
    const uint32_t m_physPgOffset_xMask;

    const uint32_t m_pgSzxy;
    const uint32_t m_pgSzxy_log2;
    const uint32_t m_tilesPerDim;
    const uint32_t m_pgtabSzxy_log2;

    using pg_tab_addr_alctr_t = core::GeneralpurposeAddressAllocator<uint32_t>;
    //core::smart_refctd_dynamic_array<uint8_t> m_pgTabAddrAlctr_reservedSpc;
    uint8_t* m_pgTabAddrAlctr_reservedSpc;
    pg_tab_addr_alctr_t m_pgTabAddrAlctr;

public:
#include "irr/irrpack.h"
    //must be 64bit
    struct STextureData
    {
        enum E_WRAP_MODE
        {
            EWM_REPEAT = 0b00,
            EWM_CLAMP = 0b01,
            EWM_MIRROR = 0b10,
            EWM_INVALID = 0b11
        };

        //1st dword
        uint64_t origsize_x : 16;
        uint64_t origsize_y : 16;

        //2nd dword
        uint64_t pgTab_x : 8;
        uint64_t pgTab_y : 8;
        uint64_t pgTab_layer : 8;
        uint64_t maxMip : 4;
        uint64_t wrap_x : 2;
        uint64_t wrap_y : 2;
    } PACK_STRUCT;
#include "irr/irrunpack.h"

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
        uint32_t addr = (_x & m_physPgOffset_xMask);
        addr |= (_y & m_physPgOffset_xMask) << (m_addr_layerShift >> 1);
        addr |= (_layer << m_addr_layerShift);
        addr |= (SPhysPgOffset::invalid_addr << 16);
        return addr;
    }
    uint32_t physPgOffset_x(SPhysPgOffset _offset) const { return _offset.addr & m_physPgOffset_xMask; }
    uint32_t physPgOffset_y(SPhysPgOffset _offset) const { return (_offset.addr >> (m_addr_layerShift>>1)) & m_physPgOffset_xMask; }
    uint32_t physPgOffset_layer(SPhysPgOffset _offset) const { return (_offset.addr & 0xffffu)>>m_addr_layerShift; }
    bool physPgOffset_valid(SPhysPgOffset _offset) const { return (_offset.addr&0xffffu) != SPhysPgOffset::invalid_addr; }
    bool physPgOffset_hasMipTailAddr(SPhysPgOffset _offset) const { return physPgOffset_valid(physPgOffset_mipTailAddr(_offset)); }
    SPhysPgOffset physPgOffset_mipTailAddr(SPhysPgOffset _offset) const { return _offset.addr>>16; }


    using page_tab_offset_t = core::vector3du32_SIMD;
    static page_tab_offset_t page_tab_offset_invalid() { return page_tab_offset_t(~0u,~0u,~0u); }
    
    //m_pgTabSzxy is PoT because of allocation using morton codes
    //m_tilesPerDim is PoT actually physical addr tex doesnt even have to be square nor PoT, but it makes addr/offset encoding easier
    //m_pgSzxy is PoT because of packing mip-tail
    ITexturePacker(uint32_t _pgTabSzxy_log2 = 8u, uint32_t _pgTabLayers = 4u, uint32_t _pgSzxy_log2 = 7u, uint32_t _tilesPerDim_log2 = 5u) :
        m_addr_layerShift(_tilesPerDim_log2<<1),
        m_physPgOffset_xMask((1u<<(m_addr_layerShift>>1))-1u),
        m_pgSzxy(1u<<_pgSzxy_log2),
        m_pgSzxy_log2(_pgSzxy_log2),
        m_tilesPerDim(1u<<_tilesPerDim_log2),
        m_pgtabSzxy_log2(_pgTabSzxy_log2),
        //m_pgTabAddrAlctr_reservedSpc(core::make_refctd_dynamic_array<decltype(m_pgTabAddrAlctr_reservedSpc)>(pg_tab_addr_alctr_t::reserved_size((1u<<_pgTabSzxy_log2)*(1u<<_pgTabSzxy_log2), (1u<<_pgTabSzxy_log2)*(1u<<_pgTabSzxy_log2)*_pgTabLayers, 1u))),
        m_pgTabAddrAlctr_reservedSpc(reinterpret_cast<uint8_t*>( _IRR_ALIGNED_MALLOC(pg_tab_addr_alctr_t::reserved_size((1u<<_pgTabSzxy_log2)*(1u<<_pgTabSzxy_log2), (1u<<_pgTabSzxy_log2)*(1u<<_pgTabSzxy_log2)*_pgTabLayers, 1u), _IRR_SIMD_ALIGNMENT) )),
        m_pgTabAddrAlctr(m_pgTabAddrAlctr_reservedSpc, 0u, 0u, (1u<<_pgTabSzxy_log2)*(1u<<_pgTabSzxy_log2), (1u<<_pgTabSzxy_log2)*(1u<<_pgTabSzxy_log2)*_pgTabLayers, 1u)
    {
    }

    virtual ~ITexturePacker()
    {
        _IRR_ALIGNED_FREE(m_pgTabAddrAlctr_reservedSpc);
    }

    uint32_t getPageSize() const { return m_pgSzxy; }

protected:
    page_tab_offset_t alloc(const IImage* _img, const IImage::SSubresourceRange& _subres)
    {
        const uint32_t pgtAddrLayerShift = m_pgtabSzxy_log2<<1;
        const uint32_t pgtAddr2dMask = (1u<<pgtAddrLayerShift)-1u;

        uint32_t szAndAlignment = computeSquareSz(_img, _subres);
        szAndAlignment *= szAndAlignment;

        uint32_t addr = pg_tab_addr_alctr_t::invalid_address;
        core::address_allocator_traits<pg_tab_addr_alctr_t>::multi_alloc_addr(m_pgTabAddrAlctr, 1u, &addr, &szAndAlignment, &szAndAlignment, nullptr);
        return (addr==pg_tab_addr_alctr_t::invalid_address) ? 
            page_tab_offset_invalid() :
            page_tab_offset_t(core::morton2d_decode_x(addr&pgtAddr2dMask), core::morton2d_decode_y(addr&pgtAddr2dMask), addr>>pgtAddrLayerShift);
    }
    virtual void free(page_tab_offset_t _addr, const IImage* _img, const IImage::SSubresourceRange& _subres)
    {
        if ((_addr==page_tab_offset_invalid()).all())
            return;

        uint32_t sz = computeSquareSz(_img, _subres);
        sz *= sz;
        const uint32_t addr = core::morton2d_encode(_addr.x, _addr.y);

        core::address_allocator_traits<pg_tab_addr_alctr_t>::multi_free_addr(m_pgTabAddrAlctr, 1u, &addr, &sz);
    }

    uint32_t neededPageCountForSide(uint32_t _sideExtent, uint32_t _level)
    {
        return (std::max(_sideExtent>>_level,1u)+m_pgSzxy-1u) / m_pgSzxy;
    }

private:
    uint32_t computeSquareSz(const IImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        auto extent = _img->getCreationParameters().extent;

        const uint32_t w = neededPageCountForSide(extent.width, _subres.baseMipLevel);
        const uint32_t h = neededPageCountForSide(extent.height, _subres.baseMipLevel);

        return core::roundUpToPoT(std::max(w,h));
    }
};

template<typename image_view_t>
class IVirtualTexture : public core::IReferenceCounted
{
protected:
    //! SPhysPgOffset is what is stored in texels of page table!
    struct SPhysPgOffset
    {
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t invalid_addr = 0xffffu;

        SPhysPgOffset(uint32_t _addr) : addr(_addr) {}

        //upper 16 bits are used for address of mip-tail page
        uint32_t addr;
    };

    uint32_t neededPageCountForSide(uint32_t _sideExtent, uint32_t _level)
    {
        return (std::max(_sideExtent>>_level, 1u)+m_pgSzxy-1u) / m_pgSzxy;
    }

private:
    uint32_t computeSquareSz(const IImage* _img, const ICPUImage::SSubresourceRange& _subres)
    {
        auto extent = _img->getCreationParameters().extent;

        const uint32_t w = neededPageCountForSide(extent.width, _subres.baseMipLevel);
        const uint32_t h = neededPageCountForSide(extent.height, _subres.baseMipLevel);

        return core::roundUpToPoT(std::max(w, h));
    }

public:
    struct SMiptailPacker
    {
        struct rect
        {
            int x, y, mx, my;

            inline core::vector2du32_SIMD extent() const { return core::vector2du32_SIMD(mx, my)+core::vector2du32_SIMD(1u)-core::vector2du32_SIMD(x,y); }
        };
        static inline bool computeMiptailOffsets(rect* res, int log2SIZE, int padding);
    };

#include "irr/irrpack.h"
    //must be 64bit
    struct STextureData
    {
        enum E_WRAP_MODE
        {
            EWM_REPEAT = 0b00,
            EWM_CLAMP = 0b01,
            EWM_MIRROR = 0b10,
            EWM_INVALID = 0b11
        };

        //1st dword
        uint64_t origsize_x : 16;
        uint64_t origsize_y : 16;

        //2nd dword
        uint64_t pgTab_x : 8;
        uint64_t pgTab_y : 8;
        uint64_t pgTab_layer : 8;
        uint64_t maxMip : 4;
        uint64_t wrap_x : 2;
        uint64_t wrap_y : 2;

        static STextureData invalid()
        {
            STextureData inv;
            memset(&inv,0,sizeof(inv));
            inv.wrap_x = EWM_INVALID;
            inv.wrap_y = EWM_INVALID;
            return inv;
        }
        static bool is_invalid(const STextureData& _td)
        {
            return _td.wrap_x==EWM_INVALID||_td.wrap_y==EWM_INVALID;
        }
    } PACK_STRUCT;
#include "irr/irrunpack.h"
    static_assert(sizeof(STextureData)==sizeof(uint64_t), "STextureData is not 64bit!");

protected:
    using image_t = typename decltype(image_view_t::SCreationParams::image)::pointee;

    using page_tab_offset_t = core::vector3du32_SIMD;
    static page_tab_offset_t page_tab_offset_invalid() { return page_tab_offset_t(~0u,~0u,~0u); }

    STextureData offsetToTextureData(const page_tab_offset_t& _offset, const ICPUImage* _img, ISampler::E_TEXTURE_CLAMP _wrapu, ISampler::E_TEXTURE_CLAMP _wrapv)
    {
        STextureData texData;
        texData.origsize_x = _img->getCreationParameters().extent.width;
        texData.origsize_y = _img->getCreationParameters().extent.height;

		texData.pgTab_x = _offset.x;
		texData.pgTab_y = _offset.y;
        texData.pgTab_layer = _offset.z;

        //getCreationParameters().mipLevels doesnt necesarilly mean that there wasnt allocated space for higher non-existent mip levels
        texData.maxMip = _img->getCreationParameters().mipLevels-1u-m_pgSzxy_log2;

        auto ETC_to_int = [](ISampler::E_TEXTURE_CLAMP _etc) -> uint32_t {
            switch (_etc)
            {
            case ISampler::ETC_REPEAT:
                return STextureData::EWM_REPEAT;
            case ISampler::ETC_CLAMP_TO_EDGE:
            case ISampler::ETC_CLAMP_TO_BORDER:
                return STextureData::EWM_CLAMP;
            case ISampler::ETC_MIRROR:
            case ISampler::ETC_MIRROR_CLAMP_TO_EDGE:
            case ISampler::ETC_MIRROR_CLAMP_TO_BORDER:
                return STextureData::EWM_MIRROR;
            default:
                return STextureData::EWM_INVALID;
            }
        };

        texData.wrap_x = ETC_to_int(_wrapu);
        texData.wrap_y = ETC_to_int(_wrapv);

        return texData;
    }

    page_tab_offset_t alloc(const IImage* _img, const IImage::SSubresourceRange& _subres, uint32_t _pgtLayer)
    {
        const uint32_t pgtAddr2dMask = (1u<<(m_pgtabSzxy_log2*2u))-1u;

        uint32_t szAndAlignment = computeSquareSz(_img, _subres);
        szAndAlignment *= szAndAlignment;

        uint32_t addr = pg_tab_addr_alctr_t::invalid_address;
        core::address_allocator_traits<pg_tab_addr_alctr_t>::multi_alloc_addr((*m_pageTableLayerAllocators)[_pgtLayer], 1u, &addr, &szAndAlignment, &szAndAlignment, nullptr);
        return (addr==pg_tab_addr_alctr_t::invalid_address) ? 
            page_tab_offset_invalid() :
            page_tab_offset_t(core::morton2d_decode_x(addr&pgtAddr2dMask), core::morton2d_decode_y(addr&pgtAddr2dMask), _pgtLayer);
    }
    virtual bool free(const STextureData& _addr, const IImage* _img, const IImage::SSubresourceRange& _subres)
    {
        uint32_t sz = computeSquareSz(_img, _subres);
        sz *= sz;
        const uint32_t addr = core::morton2d_encode(_addr.pgTab_x, _addr.pgTab_y);

        core::address_allocator_traits<pg_tab_addr_alctr_t>::multi_free_addr((*m_pageTableLayerAllocators)[_addr.pgTab_layer], 1u, &addr, &sz);

        return true;
    }

    uint32_t countLevelsTakingAtLeastOnePage(const VkExtent3D& _extent, const ICPUImage::SSubresourceRange& _subres)
    {
        const uint32_t baseMaxDim = core::roundUpToPoT(core::max(_extent.width, _extent.height))>>_subres.baseMipLevel;
        const int32_t lastFullMip = core::findMSB(baseMaxDim) - static_cast<int32_t>(m_pgSzxy_log2);

        return core::clamp(lastFullMip+1, 0, static_cast<int32_t>(m_pageTable->getCreationParameters().mipLevels));
    }

    //this is not static only because it has to call virtual member function
    core::smart_refctd_ptr<image_t> createPageTable(uint32_t _pgTabSzxy_log2, uint32_t _pgTabLayers, uint32_t _pgSzxy_log2, uint32_t _maxAllocatableTexSz_log2)
    {
        assert(_pgTabSzxy_log2<=8u);//otherwise STextureData encoding falls apart
        assert(_pgTabLayers<=256u);

        const uint32_t pgTabSzxy = 1u<<_pgTabSzxy_log2;
        image_t::SCreationParams params;
        params.arrayLayers = _pgTabLayers;
        params.extent = {pgTabSzxy,pgTabSzxy,1u};
        params.format = EF_R16G16_UINT;
        params.mipLevels = std::max(static_cast<int32_t>(_maxAllocatableTexSz_log2-_pgSzxy_log2+1u), 1);
        params.samples = IImage::ESCF_1_BIT;
        params.type = IImage::ET_2D;
        params.flags = static_cast<IImage::E_CREATE_FLAGS>(0);

        auto pgtab = createImage(std::move(params));
        IRR_PSEUDO_IF_CONSTEXPR_BEGIN(std::is_same<image_t,ICPUImage>::value)
        {
            const uint32_t texelSz = getTexelOrBlockBytesize(pgtab->getCreationParameters().format);
        
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(pgtab->getCreationParameters().mipLevels);

            uint32_t bufOffset = 0u;
            for (uint32_t i = 0u; i < pgtab->getCreationParameters().mipLevels; ++i)
            {
                const uint32_t tilesPerLodDim = pgTabSzxy>>i;
                const uint32_t regionSz = _pgTabLayers*tilesPerLodDim*tilesPerLodDim*texelSz;
                auto& region = (*regions)[i];
                region.bufferOffset = bufOffset;
                region.bufferImageHeight = 0u;
                region.bufferRowLength = tilesPerLodDim;
                region.imageExtent = {tilesPerLodDim,tilesPerLodDim,1u};
                region.imageOffset = {0,0,0};
                region.imageSubresource.baseArrayLayer = 0u;
                region.imageSubresource.layerCount = _pgTabLayers;
                region.imageSubresource.mipLevel = i;
                region.imageSubresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);

                bufOffset += regionSz;
            }
            auto buf = core::make_smart_refctd_ptr<ICPUBuffer>(bufOffset);
            uint32_t* bufptr = reinterpret_cast<uint32_t*>(buf->getPointer());
            std::fill(bufptr, bufptr+bufOffset/sizeof(uint32_t), SPhysPgOffset::invalid_addr);

            pgtab->setBufferAndRegions(std::move(buf), regions);
        } IRR_PSEUDO_IF_CONSTEXPR_END
        return pgtab;
    }

    const uint32_t m_pgSzxy;
    const uint32_t m_pgSzxy_log2;
    const uint32_t m_pgtabSzxy_log2;
    const uint32_t m_tilePadding;
    core::smart_refctd_ptr<image_t> m_pageTable;

    using pg_tab_addr_alctr_t = core::GeneralpurposeAddressAllocator<uint32_t>;
    core::smart_refctd_dynamic_array<pg_tab_addr_alctr_t> m_pageTableLayerAllocators;
    uint8_t* m_pgTabAddrAlctr_reservedSpc = nullptr;

    core::stack<uint32_t> m_freePageTableLayerIDs;
    core::smart_refctd_dynamic_array<uint32_t> m_layerToViewIndexMapping;

    struct SamplerArray
    {
        using range_t = core::SRange<const core::smart_refctd_ptr<image_view_t>>;
        // constant length, specified during construction
        core::smart_refctd_dynamic_array<core::smart_refctd_ptr<image_view_t>> views;

        range_t getViews() const {
            return views ? range_t(views->begin(),views->end()) : range_t(nullptr,nullptr);
        }
    };
    SamplerArray m_fsamplers, m_isamplers, m_usamplers;

    class IVTResidentStorage : public core::IReferenceCounted
    {
    protected:
        virtual ~IVTResidentStorage()
        {
            if (m_alctrReservedSpace)
                _IRR_ALIGNED_FREE(m_alctrReservedSpace);
        }

    public:
        struct SCreationParams
        {
            E_FORMAT_CLASS formatClass;
            const E_FORMAT* formats;
            uint32_t formatCount;
            uint32_t tilesPerDim_log2;
            uint32_t layerCount;
        };

        //_format implies format class and also is the format image is created with
        IVTResidentStorage(uint32_t _layers, uint32_t _tilesPerDim) :
            image(nullptr),//initialized in derived class's constructor
            m_alctrReservedSpace(reinterpret_cast<uint8_t*>(_IRR_ALIGNED_MALLOC(phys_pg_addr_alctr_t::reserved_size(1u, _layers*_tilesPerDim*_tilesPerDim, 1u), _IRR_SIMD_ALIGNMENT))),
            tileAlctr(m_alctrReservedSpace, 0u, 0u, 1u, _layers*_tilesPerDim*_tilesPerDim, 1u),
            m_addr_layerShift(core::findLSB(_tilesPerDim)<<1),
            m_physPgOffset_xMask((1u<<(m_addr_layerShift>>1))-1u)
        {
        }
        //TODO: this should also copy address allocator state, but from what i see our address allocators doesnt have copy ctors
        IVTResidentStorage(core::smart_refctd_ptr<image_t>&& _image, const core::vector<uint16_t>& _assignedLayers, uint32_t _layerShift, uint32_t _xmask) :
            image(std::move(_image)),
            m_assignedPageTableLayers(_assignedLayers),
            m_addr_layerShift(_layerShift),
            m_physPgOffset_xMask(_xmask)
        {

        }

        core::smart_refctd_ptr<image_view_t> createView(E_FORMAT _format) const
        {
            typename image_view_t::SCreationParams params;
            params.flags = static_cast<IImageView<image_t>::E_CREATE_FLAGS>(0);
            params.format = _format;
            params.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
            params.subresourceRange.baseArrayLayer = 0u;
            params.subresourceRange.layerCount = image->getCreationParameters().arrayLayers;
            params.subresourceRange.baseMipLevel = 0u;
            params.subresourceRange.levelCount = image->getCreationParameters().mipLevels;
            params.image = core::smart_refctd_ptr<image_t>(image);
            params.viewType = asset::IImageView<image_t>::ET_2D_ARRAY;

            return createView_internal(std::move(params));
        }
        auto getPageTableLayers() const
        {
            return std::make_pair(m_assignedPageTableLayers.begin(),m_assignedPageTableLayers.end());
        }
        void addPageTableLayer(uint16_t _layer)
        {
            auto rng = getPageTableLayers();
            auto it = std::lower_bound(rng.first,rng.second,_layer);
            m_assignedPageTableLayers.insert(it, _layer);
        }
        void removeLayerAssignment(uint16_t _layer)
        {
            auto rng = getPageTableLayers();
            auto it = std::lower_bound(rng.first, rng.second, _layer);
            if (it!=rng.second && *it==_layer)
                m_assignedPageTableLayers.erase(it);
        }

        inline uint32_t physPgOffset_x(SPhysPgOffset _offset) const { return _offset.addr & m_physPgOffset_xMask; }
        inline uint32_t physPgOffset_y(SPhysPgOffset _offset) const { return (_offset.addr >> (m_addr_layerShift>>1)) & m_physPgOffset_xMask; }
        inline uint32_t physPgOffset_layer(SPhysPgOffset _offset) const { return (_offset.addr & 0xffffu)>>m_addr_layerShift; }
        inline bool physPgOffset_valid(SPhysPgOffset _offset) const { return (_offset.addr&0xffffu) != SPhysPgOffset::invalid_addr; }
        inline bool physPgOffset_hasMipTailAddr(SPhysPgOffset _offset) const { return physPgOffset_valid(physPgOffset_mipTailAddr(_offset)); }
        inline SPhysPgOffset physPgOffset_mipTailAddr(SPhysPgOffset _offset) const { return _offset.addr>>16; }
        //! @returns texel-wise offset of physical page
        core::vector3du32_SIMD pageCoords(SPhysPgOffset _txoffset, uint32_t _pageSz, uint32_t _padding) const
        {
            core::vector3du32_SIMD coords(physPgOffset_x(_txoffset), physPgOffset_y(_txoffset), 0u);
            coords *= (_pageSz + 2u*_padding);
            coords += _padding;
            coords.z = physPgOffset_layer(_txoffset);
            return coords;
        }

        core::smart_refctd_ptr<image_t> image;
        using phys_pg_addr_alctr_t = core::PoolAddressAllocator<uint32_t>;
        uint8_t* m_alctrReservedSpace = nullptr;
        phys_pg_addr_alctr_t tileAlctr;
        core::vector<uint16_t> m_assignedPageTableLayers;
        const uint32_t m_addr_layerShift;
        const uint32_t m_physPgOffset_xMask;

    protected:
        virtual core::smart_refctd_ptr<image_view_t> createView_internal(typename image_view_t::SCreationParams&& _params) const = 0;
    };
    //since c++14 std::hash specialization for all enum types are given by standard
    core::unordered_map<E_FORMAT_CLASS, core::smart_refctd_ptr<IVTResidentStorage>> m_storage;

    _IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_PHYSICAL_PAGE_SIZE_LOG2 = 9u;
    typename SMiptailPacker::rect m_miptailOffsets[MAX_PHYSICAL_PAGE_SIZE_LOG2];

    virtual core::smart_refctd_ptr<image_t> createImage(typename image_t::SCreationParams&& _params) const = 0;
    virtual core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(E_FORMAT _format, uint32_t _extent, uint32_t _layers, uint32_t _tilesPerDim) = 0;

    virtual ~IVirtualTexture()
    {
        if (m_pgTabAddrAlctr_reservedSpc)
            _IRR_ALIGNED_FREE(m_pgTabAddrAlctr_reservedSpc);
    }

    // Delegated to separate method, because is strongly dependent on derived class and has to be called once derived class's object exist
    void initResidentStorage(
        const typename IVTResidentStorage::SCreationParams* _residentStorageParams,
        uint32_t _residentStorageCount
    )
    {
        {
            const uint32_t tileSz = m_pgSzxy+2u*m_tilePadding;
            for (uint32_t i = 0u; i < _residentStorageCount; ++i)
            {
                const auto& params = _residentStorageParams[i];
                const uint32_t tilesPerDim = (1u<<params.tilesPerDim_log2);
                const uint32_t extent = tilesPerDim*tileSz;
                assert(params.formatCount>0u);
                const E_FORMAT fmt = params.formats[0];
                const uint32_t layers = params.layerCount;
                m_storage.insert({params.formatClass, createVTResidentStorage(fmt, extent, layers, tilesPerDim)});
            }
        }
        {
            auto execPerFormat = [_residentStorageCount, _residentStorageParams,this] (auto f_fmtf, auto f_fmti, auto f_fmtu)
            {
                for (uint32_t i = 0u; i < _residentStorageCount; ++i)
                {
                    const auto& params = _residentStorageParams[i];
                    for (uint32_t j = 0u; j < params.formatCount; ++j)
                    {
                        const E_FORMAT fmt = params.formats[j];
                        auto* storage = m_storage[params.formatClass].get();
                        if (isNormalizedFormat(fmt)||isFloatingPointFormat(fmt)||isScaledFormat(fmt))
                            f_fmtf(fmt,storage);
                        else { //integer formats
                            if (isSignedFormat(fmt))
                                f_fmti(fmt,storage);
                            else
                                f_fmtu(fmt,storage);
                        }
                    }
                }
            };
            {
                uint32_t fcount = 0u, icount = 0u, ucount = 0u;
                execPerFormat([&](auto,auto) {++fcount; }, [&](auto,auto) {++icount; }, [&](auto,auto) {++ucount; });
                m_fsamplers.views = fcount ? core::make_refctd_dynamic_array<decltype(SamplerArray::views)>(fcount) : nullptr;
                m_isamplers.views = icount ? core::make_refctd_dynamic_array<decltype(SamplerArray::views)>(icount) : nullptr;
                m_usamplers.views = ucount ? core::make_refctd_dynamic_array<decltype(SamplerArray::views)>(ucount) : nullptr;
            }
            {
                uint32_t fi = 0u, ii = 0u, ui = 0u;

                execPerFormat(
                    [&fi, this](E_FORMAT _fmt, IVTResidentStorage* _storage) { (*m_fsamplers.views)[fi++] = _storage->createView(_fmt); },
                    [&ii, this](E_FORMAT _fmt, IVTResidentStorage* _storage) { (*m_isamplers.views)[ii++] = _storage->createView(_fmt); },
                    [&ui, this](E_FORMAT _fmt, IVTResidentStorage* _storage) { (*m_usamplers.views)[ui++] = _storage->createView(_fmt); }
                );
            }
        }
    }

public:
    IVirtualTexture(
        uint32_t _pgTabSzxy_log2 = 8u,
        uint32_t _pgTabLayers = 256u,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _tilePadding = 9u,
        bool _initSharedResources = true
    ) :
        m_pgSzxy(1u<<_pgSzxy_log2), m_pgSzxy_log2(_pgSzxy_log2), m_pgtabSzxy_log2(_pgTabSzxy_log2), m_tilePadding(_tilePadding),
        m_pageTableLayerAllocators(core::make_refctd_dynamic_array<decltype(m_pageTableLayerAllocators)>(_pgTabLayers)),
        m_layerToViewIndexMapping(core::make_refctd_dynamic_array<decltype(m_layerToViewIndexMapping)>(_pgTabLayers,~0u))
    {
        if (_initSharedResources)
        {
            uint32_t pgtabSzSqr = (1u<<_pgTabSzxy_log2);
            pgtabSzSqr *= pgtabSzSqr;
            const size_t spacePerAllocator = pg_tab_addr_alctr_t::reserved_size(pgtabSzSqr, pgtabSzSqr, 1u);
            m_pgTabAddrAlctr_reservedSpc = reinterpret_cast<uint8_t*>( _IRR_ALIGNED_MALLOC(spacePerAllocator*_pgTabLayers, _IRR_SIMD_ALIGNMENT) );
            for (uint32_t i = 0u; i < _pgTabLayers; ++i)
            {
                auto& alctr = (*m_pageTableLayerAllocators)[i];
                alctr = pg_tab_addr_alctr_t(m_pgTabAddrAlctr_reservedSpc+i*spacePerAllocator, 0u, 0u, pgtabSzSqr, pgtabSzSqr, 1u);
            }
        }

        if (_initSharedResources)
        {
            decltype(m_freePageTableLayerIDs)::container_type stackBackend;
            stackBackend.resize(_pgTabLayers);
            for (uint32_t i = 0u; i < _pgTabLayers; ++i)
                stackBackend[i] = _pgTabLayers-1u-i;
            m_freePageTableLayerIDs = decltype(m_freePageTableLayerIDs)(std::move(stackBackend));
        }
        {
            const uint32_t pgSzLog2 = getPageExtent_log2();
            bool ok = SMiptailPacker::computeMiptailOffsets(m_miptailOffsets, pgSzLog2, m_tilePadding);
            assert(ok);
        }
    }

    virtual STextureData pack(const image_t* _img, const IImage::SSubresourceRange& _subres, ISampler::E_TEXTURE_CLAMP _wrapu, ISampler::E_TEXTURE_CLAMP _wrapv, ISampler::E_TEXTURE_BORDER_COLOR _borderColor) = 0;

    virtual core::smart_refctd_ptr<image_view_t> createPageTableView() const = 0;

    image_t* const & getPageTable() const { return m_pageTable.get(); }
    uint32_t getPageTableExtent_log2() const { return m_pgSzxy_log2; }
    uint32_t getPageExtent() const { return m_pgSzxy; }
    uint32_t getPageExtent_log2() const { return core::findLSB(m_pgSzxy); }
    uint32_t getTilePadding() const { return m_tilePadding; }
    const core::stack<uint32_t>& getFreePageTableLayersStack() const { return m_freePageTableLayerIDs; }
    core::SRange<const uint32_t> getLayerToViewIndexMapping() const { return {m_layerToViewIndexMapping->begin(),m_layerToViewIndexMapping->end()}; }
    const auto& getResidentStorages() const { return m_storage; }
    typename SamplerArray::range_t getFloatViews() const  { return m_fsamplers.getViews(); }
    typename SamplerArray::range_t getIntViews() const { return m_isamplers.getViews(); }
    typename SamplerArray::range_t getUintViews() const { return m_usamplers.getViews(); }
};

class ICPUVirtualTexture final : public IVirtualTexture<ICPUImageView>
{
    using base_t = IVirtualTexture<ICPUImageView>;

public:
    class ICPUVTResidentStorage final : public base_t::IVTResidentStorage
    {
        using base_t = base_t::IVTResidentStorage;

    public:
        ICPUVTResidentStorage(E_FORMAT _format, uint32_t _extent, uint32_t _layers, uint32_t _tilesPerDim) :
            base_t(_layers, _tilesPerDim)
        {
            ICPUImage::SCreationParams params;
            params.extent = {_extent,_extent,1u};
            params.format = _format;
            params.arrayLayers = _layers;
            params.mipLevels = 1u;
            params.type = IImage::ET_2D;
            params.samples = IImage::ESCF_1_BIT;
            params.flags = static_cast<IImage::E_CREATE_FLAGS>(0);

            image = ICPUImage::create(std::move(params));
            {
                auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<IImage::SBufferCopy>>(1ull);
                auto& region = regions->front();
                region.imageSubresource.mipLevel = 0u;
                region.imageSubresource.baseArrayLayer = 0u;
                region.imageSubresource.layerCount = _layers;
                region.bufferOffset = 0u;
                region.bufferRowLength = _extent;
                region.bufferImageHeight = 0u; //tightly packed
                region.imageOffset = {0u,0u,0u};
                region.imageExtent = params.extent;
                auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(_format) * params.extent.width*params.extent.height*params.arrayLayers);
                image->setBufferAndRegions(std::move(buffer), regions);
            }
        }

    private:
        core::smart_refctd_ptr<ICPUImageView> createView_internal(ICPUImageView::SCreationParams&& _params) const override
        {
            return ICPUImageView::create(std::move(_params));
        }
    };

    ICPUVirtualTexture(
        const base_t::IVTResidentStorage::SCreationParams* _residentStorageParams,
        uint32_t _residentStorageCount,
        uint32_t _pgTabSzxy_log2 = 8u,
        uint32_t _pgTabLayers = 256u,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) : IVirtualTexture(
        _pgTabSzxy_log2, _pgTabLayers, _pgSzxy_log2, _tilePadding
    ) {
        m_pageTable = createPageTable(_pgTabSzxy_log2, _pgTabLayers, _pgSzxy_log2, _maxAllocatableTexSz_log2);
        initResidentStorage(_residentStorageParams, _residentStorageCount);
    }

    STextureData pack(const ICPUImage* _img, const IImage::SSubresourceRange& _subres, ISampler::E_TEXTURE_CLAMP _wrapu, ISampler::E_TEXTURE_CLAMP _wrapv, ISampler::E_TEXTURE_BORDER_COLOR _borderColor) override
    {
        const E_FORMAT format = _img->getCreationParameters().format;
        uint32_t smplrIndex = 0u;
        ICPUVTResidentStorage* storage = nullptr;
        {
            auto found = m_storage.find(getFormatClass(format));
            if (found==m_storage.end())
                return STextureData::invalid();
            storage = static_cast<ICPUVTResidentStorage*>(found->second.get());

            SamplerArray* views = nullptr;
            if (isFloatingPointFormat(format)||isNormalizedFormat(format)||isScaledFormat(format))
                views = &m_fsamplers;
            else if (isSignedFormat(format))
                views = &m_isamplers;
            else
                views = &m_usamplers;
            auto view_it = std::find_if(views->views->begin(), views->views->end(), [format](const core::smart_refctd_ptr<ICPUImageView>& _view) {return _view->getCreationParameters().format==format;});
            if (view_it==views->views->end()) //no physical page texture view/sampler for requested format
                return STextureData::invalid();
            smplrIndex = std::distance(views->views->begin(), view_it);
        }
        auto assignedLayers = storage->getPageTableLayers();

        page_tab_offset_t pgtOffset = page_tab_offset_invalid();
        for (auto it = assignedLayers.first; it != assignedLayers.second; ++it)
        {
            pgtOffset = alloc(_img, _subres, *it);
            if ((pgtOffset==page_tab_offset_invalid()).all())
                continue;
        }
        if ((pgtOffset==page_tab_offset_invalid()).all())
        {
            if (m_freePageTableLayerIDs.empty())
                return STextureData::invalid();
            const uint32_t pgtLayer = m_freePageTableLayerIDs.top();
            m_freePageTableLayerIDs.pop();
            pgtOffset = alloc(_img, _subres, pgtLayer);
            if ((pgtOffset==page_tab_offset_invalid()).all())//this would be super weird but let's check
                return STextureData::invalid();
            storage->addPageTableLayer(pgtLayer);

            (*m_layerToViewIndexMapping)[pgtLayer] = smplrIndex;
        }

        const auto extent = _img->getCreationParameters().extent;

        const uint32_t levelsTakingAtLeastOnePageCount = countLevelsTakingAtLeastOnePage(extent, _subres);
        const uint32_t levelsToPack = std::min(_subres.levelCount, m_pageTable->getCreationParameters().mipLevels+m_pgSzxy_log2);

        uint32_t miptailPgAddr = SPhysPgOffset::invalid_addr;

        using phys_pg_addr_alctr_t = ICPUVTResidentStorage::phys_pg_addr_alctr_t;
        //TODO up to this line, it's kinda common code for CPU and GPU, refactor later

        //fill page table and pack present mips into physical addr texture
        CFillImageFilter::state_type fill;
        fill.outImage = m_pageTable.get();
        fill.outRange.extent = { 1u,1u,1u };
        fill.subresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        fill.subresource.baseArrayLayer = pgtOffset.z;
        fill.subresource.layerCount = 1u;
        for (uint32_t i = 0u; i < levelsToPack; ++i)
        {
            const uint32_t w = neededPageCountForSide(extent.width, _subres.baseMipLevel+i);
            const uint32_t h = neededPageCountForSide(extent.height, _subres.baseMipLevel+i);

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t physPgAddr = phys_pg_addr_alctr_t::invalid_address;
                    if (i>=levelsTakingAtLeastOnePageCount)
                        physPgAddr = miptailPgAddr;
                    else
                    {
                        const uint32_t szAndAlignment = 1u;
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_alloc_addr(storage->tileAlctr, 1u, &physPgAddr, &szAndAlignment, &szAndAlignment, nullptr);
                    }
                    //assert(physPgAddr<SPhysPgOffset::invalid_addr);
                    if (physPgAddr==phys_pg_addr_alctr_t::invalid_address)
                    {
                        free(offsetToTextureData(pgtOffset, _img, _wrapu, _wrapv), _img, _subres);
                        return STextureData::invalid();
                    }

                    if (i==(levelsTakingAtLeastOnePageCount-1u) && levelsTakingAtLeastOnePageCount<_subres.levelCount)
                    {
                        assert(w==1u && h==1u);
                        uint32_t physMiptailPgAddr = phys_pg_addr_alctr_t::invalid_address;
                        const uint32_t szAndAlignment = 1u;
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_alloc_addr(storage->tileAlctr, 1u, &physMiptailPgAddr, &szAndAlignment, &szAndAlignment, nullptr);
                        assert(physMiptailPgAddr<SPhysPgOffset::invalid_addr);
                        miptailPgAddr = physMiptailPgAddr = (physMiptailPgAddr==phys_pg_addr_alctr_t::invalid_address) ? SPhysPgOffset::invalid_addr : physMiptailPgAddr;
                        physPgAddr |= (physMiptailPgAddr<<16);
                    }
                    else 
                        physPgAddr |= (SPhysPgOffset::invalid_addr<<16);
                    if (i < levelsTakingAtLeastOnePageCount)
                    {
                        fill.subresource.mipLevel = i;
                        fill.outRange.offset = {(pgtOffset.x>>i) + x, (pgtOffset.y>>i) + y, 0u};
                        fill.fillValue.asUint.x = physPgAddr;

                        CFillImageFilter::execute(&fill);
                    }

                    core::vector3du32_SIMD physPg = storage->pageCoords(physPgAddr, m_pgSzxy, m_tilePadding);
                    physPg -= core::vector2du32_SIMD(m_tilePadding, m_tilePadding);

                    const core::vector2du32_SIMD miptailOffset = (i>=levelsTakingAtLeastOnePageCount) ? core::vector2du32_SIMD(m_miptailOffsets[i-levelsTakingAtLeastOnePageCount].x,m_miptailOffsets[i-levelsTakingAtLeastOnePageCount].y) : core::vector2du32_SIMD(0u,0u);
                    physPg += miptailOffset;

                    CPaddedCopyImageFilter::state_type copy;
                    copy.outOffsetBaseLayer = (physPg).xyzz();/*physPg.z is layer*/ copy.outOffset.z = 0u;
                    copy.inOffsetBaseLayer = core::vector2du32_SIMD(x,y)*m_pgSzxy;
                    copy.extentLayerCount = core::vectorSIMDu32(m_pgSzxy, m_pgSzxy, 1u, 1u);
                    copy.relativeOffset = {0u,0u,0u};
                    if (x == w-1u)
                        copy.extentLayerCount.x = std::max(extent.width>>(_subres.baseMipLevel+i),1u)-copy.inOffsetBaseLayer.x;
                    if (y == h-1u)
                        copy.extentLayerCount.y = std::max(extent.height>>(_subres.baseMipLevel+i),1u)-copy.inOffsetBaseLayer.y;
                    memcpy(&copy.paddedExtent.width,(copy.extentLayerCount+core::vectorSIMDu32(2u*m_tilePadding)).pointer, 2u*sizeof(uint32_t));
                    copy.paddedExtent.depth = 1u;
                    if (w>1u)
                        copy.extentLayerCount.x += m_tilePadding;
                    if (x>0u && x<w-1u)
                        copy.extentLayerCount.x += m_tilePadding;
                    if (h>1u)
                        copy.extentLayerCount.y += m_tilePadding;
                    if (y>0u && y<h-1u)
                        copy.extentLayerCount.y += m_tilePadding;
                    if (x == 0u)
                        copy.relativeOffset.x = m_tilePadding;
                    else
                        copy.inOffsetBaseLayer.x -= m_tilePadding;
                    if (y == 0u)
                        copy.relativeOffset.y = m_tilePadding;
                    else
                        copy.inOffsetBaseLayer.y -= m_tilePadding;
                    copy.inMipLevel = _subres.baseMipLevel + i;
                    copy.outMipLevel = 0u;
                    copy.inImage = _img;
                    copy.outImage = storage->image.get();
                    copy.axisWraps[0] = _wrapu;
                    copy.axisWraps[1] = _wrapv;
                    copy.axisWraps[2] = ISampler::ETC_CLAMP_TO_EDGE;
                    copy.borderColor = _borderColor;
                    if (!CPaddedCopyImageFilter::execute(&copy))
                        _IRR_DEBUG_BREAK_IF(true);
                }
        }

        return offsetToTextureData(pgtOffset, _img, _wrapu, _wrapv);
    }

    bool free(const STextureData& _addr, const IImage* _img, const IImage::SSubresourceRange& _subres) override
    {
        const E_FORMAT format = _img->getCreationParameters().format;
        ICPUVTResidentStorage* storage = nullptr;
        {
            auto found = m_storage.find(getFormatClass(format));
            if (found==m_storage.end())
                return false;
            storage = static_cast<ICPUVTResidentStorage*>(found->second.get());
        }

        //free physical pages
        auto extent = _img->getCreationParameters().extent;
        const uint32_t levelCount = countLevelsTakingAtLeastOnePage(extent, _subres);

        CFillImageFilter::state_type fill;
        fill.outImage = m_pageTable.get();
        fill.subresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        fill.subresource.baseArrayLayer = _addr.pgTab_layer;
        fill.subresource.layerCount = 1u;
        fill.fillValue.asUint.x = SPhysPgOffset::invalid_addr;

        using phys_pg_addr_alctr_t = ICPUVTResidentStorage::phys_pg_addr_alctr_t;

        auto* const bufptr = reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer());
        for (uint32_t i = 0u; i < levelCount; ++i)
        {
            const uint32_t w = neededPageCountForSide(extent.width, _subres.baseMipLevel+i);
            const uint32_t h = neededPageCountForSide(extent.height, _subres.baseMipLevel+i);

            const auto& region = m_pageTable->getRegions().begin()[i];
            const auto strides = region.getByteStrides(TexelBlockInfo(m_pageTable->getCreationParameters().format));

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t* texelptr = reinterpret_cast<uint32_t*>(bufptr + region.getByteOffset(core::vector4du32_SIMD((_addr.pgTab_x>>i) + x, (_addr.pgTab_y>>i) + y, 0u, _addr.pgTab_layer), strides));
                    SPhysPgOffset physPgOffset = *texelptr;
                    if (storage->physPgOffset_valid(physPgOffset))
                    {
                        *texelptr = SPhysPgOffset::invalid_addr;

                        uint32_t addrs[2] { physPgOffset.addr&0xffffu, storage->physPgOffset_mipTailAddr(physPgOffset).addr&0xffffu };
                        const uint32_t szs[2]{ 1u,1u };
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_free_addr(storage->tileAlctr, storage->physPgOffset_hasMipTailAddr(physPgOffset) ? 2u : 1u, addrs, szs);
                    }
                }
            fill.subresource.mipLevel = i;
            fill.outRange.offset = {static_cast<uint32_t>(_addr.pgTab_x>>i),static_cast<uint32_t>(_addr.pgTab_y>>i),0u};
            fill.outRange.extent = {w,h,1u};
            CFillImageFilter::execute(&fill);
        }

        //free entries in page table
        if (!base_t::free(_addr, _img, _subres))
            return false;
        //in case when pgtab layer has no allocations, free it for use by another format
        if ((*m_pageTableLayerAllocators)[_addr.pgTab_layer].get_allocated_size()==0u)
        {
            (*m_pageTableLayerAllocators)[_addr.pgTab_layer].reset();//why actually do i need to reset it if its allocated size is 0 anyway?
            (*m_layerToViewIndexMapping)[_addr.pgTab_layer] = ~0u;
            storage->removeLayerAssignment(_addr.pgTab_layer);
            m_freePageTableLayerIDs.push(_addr.pgTab_layer);
        }

        return true;
    }

    core::smart_refctd_ptr<ICPUImageView> createPageTableView() const override
    {
        ICPUImageView::SCreationParams params;
        params.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0);
        params.format = m_pageTable->getCreationParameters().format;
        params.subresourceRange.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        params.subresourceRange.baseArrayLayer = 0u;
        params.subresourceRange.layerCount = m_pageTable->getCreationParameters().arrayLayers;
        params.subresourceRange.baseMipLevel = 0u;
        params.subresourceRange.levelCount = m_pageTable->getCreationParameters().mipLevels;
        params.viewType = IImageView<ICPUImage>::ET_2D_ARRAY;
        params.image = m_pageTable;

        return ICPUImageView::create(std::move(params));
    }

protected:
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(E_FORMAT _format, uint32_t _extent, uint32_t _layers, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<ICPUVTResidentStorage>(_format, _extent, _layers, _tilesPerDim);
    }
    core::smart_refctd_ptr<ICPUImage> createImage(ICPUImage::SCreationParams&& _params) const override
    {
        return ICPUImage::create(std::move(_params));
    }
};
template <typename image_view_t>
bool IVirtualTexture<image_view_t>::SMiptailPacker::computeMiptailOffsets(IVirtualTexture<image_view_t>::SMiptailPacker::rect* res, int log2SIZE, int padding)
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

class ICPUTexturePacker : public ITexturePacker
{
protected:
    virtual ~ICPUTexturePacker()
    {
        _IRR_ALIGNED_FREE(m_physPgAddrAlctr_reservedSpc);
    }

public:
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_PHYSICAL_PAGE_SIZE_LOG2 = 9u;
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_PHYSICAL_PAGE_SIZE = 1u<<MAX_PHYSICAL_PAGE_SIZE_LOG2;

    uint32_t maxAllocatableTextureSz() const { return m_pgSzxy<<m_pageTable->getCreationParameters().mipLevels; }

    uint32_t physAddrTexLayerSz() const
    {
        return m_tilesPerDim * (2u*m_tilePadding+m_pgSzxy);
    }
    //! @returns texel-wise offset of physical page
    core::vector3du32_SIMD pageCoords(SPhysPgOffset _txoffset, uint32_t _pgSz) const
    {
        core::vector3du32_SIMD coords(physPgOffset_x(_txoffset), physPgOffset_y(_txoffset), 0u);
        coords *= (_pgSz + 2u*m_tilePadding);
        coords += m_tilePadding;
        coords.z = physPgOffset_layer(_txoffset);
        return coords;
    }
    struct SMiptailPacker
    {
        struct rect
        {
            int x, y, mx, my;

            core::vector2du32_SIMD extent() const { return core::vector2du32_SIMD(mx, my)+core::vector2du32_SIMD(1u)-core::vector2du32_SIMD(x,y); }
        };
        static inline bool computeMiptailOffsets(rect* res, int log2SIZE, int padding);
    };

    static core::smart_refctd_ptr<ICPUImage> createPageTable(uint32_t _pgTabSzxy_log2 = 8u, uint32_t _pgTabLayers = 4u, uint32_t _pgSzxy_log2 = 7u, uint32_t _maxAllocatableTexSz_log2 = 14u)
    {
        assert(_pgTabSzxy_log2<=8u);//otherwise STextureData encoding falls apart
        assert(_pgTabLayers<=256u);

        const uint32_t sz = 1u<<_pgTabSzxy_log2;
        ICPUImage::SCreationParams params;
        params.arrayLayers = _pgTabLayers;
        params.extent = {sz,sz,1u};
        params.format = EF_R16G16_UINT;
        params.mipLevels = std::max(static_cast<int32_t>(_maxAllocatableTexSz_log2-_pgSzxy_log2+1u), 1);
        params.samples = IImage::ESCF_1_BIT;
        params.type = IImage::ET_2D;
        params.flags = static_cast<IImage::E_CREATE_FLAGS>(0);

        auto pgtab = ICPUImage::create(std::move(params));
        {
            const uint32_t pgTabSzxy = 1u<<_pgTabSzxy_log2;
            const uint32_t texelSz = getTexelOrBlockBytesize(pgtab->getCreationParameters().format);
        
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(pgtab->getCreationParameters().mipLevels);

            uint32_t bufOffset = 0u;
            for (uint32_t i = 0u; i < pgtab->getCreationParameters().mipLevels; ++i)
            {
                const uint32_t tilesPerLodDim = pgTabSzxy>>i;
                const uint32_t regionSz = _pgTabLayers*tilesPerLodDim*tilesPerLodDim*texelSz;
                auto& region = (*regions)[i];
                region.bufferOffset = bufOffset;
                region.bufferImageHeight = 0u;
                region.bufferRowLength = tilesPerLodDim;
                region.imageExtent = {tilesPerLodDim,tilesPerLodDim,1u};
                region.imageOffset = {0,0,0};
                region.imageSubresource.baseArrayLayer = 0u;
                region.imageSubresource.layerCount = _pgTabLayers;
                region.imageSubresource.mipLevel = i;
                region.imageSubresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);

                bufOffset += regionSz;
            }
            auto buf = core::make_smart_refctd_ptr<ICPUBuffer>(bufOffset);
            uint32_t* bufptr = reinterpret_cast<uint32_t*>(buf->getPointer());
            std::fill(bufptr, bufptr+bufOffset/sizeof(uint32_t), SPhysPgOffset::invalid_addr);

            pgtab->setBufferAndRegions(std::move(buf), regions);
        }
        return pgtab;
    }

    //! @param _pgtab Must be an image created by createPageTable()
    ICPUTexturePacker(E_FORMAT _format, core::smart_refctd_ptr<ICPUImage>&& _pgtab, const IImage::SSubresourceRange& _pgtabSubrange, uint32_t _pgSzxy_log2 = 7u, uint32_t _tilesPerDim_log2 = 5u, uint32_t _numLayers = 4u, uint32_t _tilePad = 9u/*max_aniso/2+1*/) :
        ITexturePacker(core::findLSB(_pgtab->getCreationParameters().extent.width), _pgtabSubrange.layerCount, _pgSzxy_log2, _tilesPerDim_log2),
        m_pageTable(std::move(_pgtab)),
        m_tilePadding(_tilePad),
        m_pgtabLayerOffset(_pgtabSubrange.baseArrayLayer),
        //m_physPgAddrAlctr_reservedSpc(core::make_refctd_dynamic_array<decltype(m_physPgAddrAlctr_reservedSpc)>(phys_pg_addr_alctr_t::reserved_size(1u, _numLayers*(1u<<_tilesPerDim_log2)*(1u<<_tilesPerDim_log2), 1u))),
        m_physPgAddrAlctr_reservedSpc(reinterpret_cast<uint8_t*>( _IRR_ALIGNED_MALLOC(phys_pg_addr_alctr_t::reserved_size(1u, _numLayers*(1u<<_tilesPerDim_log2)*(1u<<_tilesPerDim_log2), 1u), _IRR_SIMD_ALIGNMENT) )),
        m_physPgAddrAlctr(m_physPgAddrAlctr_reservedSpc, 0u, 0u, 1u, _numLayers*(1u<<_tilesPerDim_log2)*(1u<<_tilesPerDim_log2), 1u)
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
            region.imageExtent = params.extent;
            auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(_format) * params.extent.width*params.extent.height*params.arrayLayers);
            m_physAddrTex->setBufferAndRegions(std::move(buffer), regions);
        }

        const uint32_t pgSzLog2 = core::findMSB(m_pgSzxy);
        bool ok = SMiptailPacker::computeMiptailOffsets(m_miptailOffsets, pgSzLog2, m_tilePadding);
        assert(ok);
    }

    //! @param _addr Is expected to be offset returned from pack() -- i.e. absolute, not relative to layer offset for this packer object
    void free(page_tab_offset_t _addr, const IImage* _img, const IImage::SSubresourceRange& _subres) override
    {
        //free physical pages
        auto extent = _img->getCreationParameters().extent;
        const uint32_t levelCount = countLevelsTakingAtLeastOnePage(extent, _subres);

        CFillImageFilter::state_type fill;
        fill.outImage = m_pageTable.get();
        fill.subresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        fill.subresource.baseArrayLayer = _addr.z;
        fill.subresource.layerCount = 1u;
        fill.fillValue.asUint.x = SPhysPgOffset::invalid_addr;

        auto* const bufptr = reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer());
        for (uint32_t i = 0u; i < levelCount; ++i)
        {
            const uint32_t w = neededPageCountForSide(extent.width, _subres.baseMipLevel+i);
            const uint32_t h = neededPageCountForSide(extent.height, _subres.baseMipLevel+i);

            const auto& region = m_pageTable->getRegions().begin()[i];
            const auto strides = region.getByteStrides(TexelBlockInfo(m_pageTable->getCreationParameters().format));

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t* texelptr = reinterpret_cast<uint32_t*>(bufptr + region.getByteOffset(core::vector4du32_SIMD((_addr.x>>i) + x, (_addr.y>>i) + y, 0u, _addr.z), strides));
                    SPhysPgOffset physPgOffset = *texelptr;
                    if (physPgOffset_valid(physPgOffset))
                    {
                        *texelptr = SPhysPgOffset::invalid_addr;

                        uint32_t addrs[2] { physPgOffset.addr&0xffffu, physPgOffset_mipTailAddr(physPgOffset).addr&0xffffu };
                        const uint32_t szs[2]{ 1u,1u };
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_free_addr(m_physPgAddrAlctr, physPgOffset_hasMipTailAddr(physPgOffset) ? 2u : 1u, addrs, szs);
                    }
                }
            fill.subresource.mipLevel = i;
            fill.outRange.offset = {(_addr.x>>i),(_addr.y>>i),0u};
            fill.outRange.extent = {w,h,1u};
            CFillImageFilter::execute(&fill);
        }

        //free entries in page table
        ITexturePacker::free(_addr, _img, _subres);
    }

    STextureData offsetToTextureData(const page_tab_offset_t& _offset, const ICPUImage* _img, ISampler::E_TEXTURE_CLAMP _wrapu, ISampler::E_TEXTURE_CLAMP _wrapv)
    {
        STextureData texData;
        texData.origsize_x = _img->getCreationParameters().extent.width;
        texData.origsize_y = _img->getCreationParameters().extent.height;

		texData.pgTab_x = _offset.x;
		texData.pgTab_y = _offset.y;
        texData.pgTab_layer = _offset.z;

        //getCreationParameters().mipLevels doesnt necesarilly mean that there wasnt allocated space for higher non-existent mip levels
        texData.maxMip = _img->getCreationParameters().mipLevels-1u-m_pgSzxy_log2;

        auto ETC_to_int = [](ISampler::E_TEXTURE_CLAMP _etc) -> uint32_t {
            switch (_etc)
            {
            case ISampler::ETC_REPEAT:
                return STextureData::EWM_REPEAT;
            case ISampler::ETC_CLAMP_TO_EDGE:
            case ISampler::ETC_CLAMP_TO_BORDER:
                return STextureData::EWM_CLAMP;
            case ISampler::ETC_MIRROR:
            case ISampler::ETC_MIRROR_CLAMP_TO_EDGE:
            case ISampler::ETC_MIRROR_CLAMP_TO_BORDER:
                return STextureData::EWM_MIRROR;
            default:
                return STextureData::EWM_INVALID;
            }
        };

        texData.wrap_x = ETC_to_int(_wrapu);
        texData.wrap_y = ETC_to_int(_wrapv);

        return texData;
    }
    //! Returned offset is absolute (layer is not relative to layer offset of this packer object)
    page_tab_offset_t pack(const ICPUImage* _img, const ICPUImage::SSubresourceRange& _subres, ISampler::E_TEXTURE_CLAMP _wrapu, ISampler::E_TEXTURE_CLAMP _wrapv, ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
    {
        if (getFormatClass(_img->getCreationParameters().format)!=getFormatClass(m_physAddrTex->getCreationParameters().format))
            return page_tab_offset_invalid();

        const auto extent = _img->getCreationParameters().extent;

        if ((extent.width>>_subres.baseMipLevel) > maxAllocatableTextureSz() || (extent.height>>_subres.baseMipLevel) > maxAllocatableTextureSz())
            return page_tab_offset_invalid();

        page_tab_offset_t pgtOffset = alloc(_img, _subres);
        if ((pgtOffset==page_tab_offset_invalid()).all())
            return pgtOffset;
        pgtOffset.z += m_pgtabLayerOffset;

        const uint32_t levelsTakingAtLeastOnePageCount = countLevelsTakingAtLeastOnePage(extent, _subres);
        const uint32_t levelsToPack = std::min(_subres.levelCount, m_pageTable->getCreationParameters().mipLevels+m_pgSzxy_log2);

        uint32_t miptailPgAddr = SPhysPgOffset::invalid_addr;

        const uint32_t texelSz = getTexelOrBlockBytesize(m_physAddrTex->getCreationParameters().format);
        //fill page table and pack present mips into physical addr texture
        CFillImageFilter::state_type fill;
        fill.outImage = m_pageTable.get();
        fill.outRange.extent = {1u,1u,1u};
        fill.subresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        fill.subresource.baseArrayLayer = pgtOffset.z;
        fill.subresource.layerCount = 1u;
        for (uint32_t i = 0u; i < levelsToPack; ++i)
        {
            const uint32_t w = neededPageCountForSide(extent.width, _subres.baseMipLevel+i);
            const uint32_t h = neededPageCountForSide(extent.height, _subres.baseMipLevel+i);

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t physPgAddr = phys_pg_addr_alctr_t::invalid_address;
                    if (i>=levelsTakingAtLeastOnePageCount)
                        physPgAddr = miptailPgAddr;
                    else
                    {
                        const uint32_t szAndAlignment = 1u;
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_alloc_addr(m_physPgAddrAlctr, 1u, &physPgAddr, &szAndAlignment, &szAndAlignment, nullptr);
                    }
                    //assert(physPgAddr<SPhysPgOffset::invalid_addr);
                    if (physPgAddr==phys_pg_addr_alctr_t::invalid_address)
                    {
                        free(pgtOffset, _img, _subres);
                        return page_tab_offset_invalid();
                    }

                    if (i==(levelsTakingAtLeastOnePageCount-1u) && levelsTakingAtLeastOnePageCount<_subres.levelCount)
                    {
                        assert(w==1u && h==1u);
                        uint32_t physMiptailPgAddr = phys_pg_addr_alctr_t::invalid_address;
                        const uint32_t szAndAlignment = 1u;
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_alloc_addr(m_physPgAddrAlctr, 1u, &physMiptailPgAddr, &szAndAlignment, &szAndAlignment, nullptr);
                        assert(physMiptailPgAddr<SPhysPgOffset::invalid_addr);
                        miptailPgAddr = physMiptailPgAddr = (physMiptailPgAddr==phys_pg_addr_alctr_t::invalid_address) ? SPhysPgOffset::invalid_addr : physMiptailPgAddr;
                        physPgAddr |= (physMiptailPgAddr<<16);
                    }
                    else 
                        physPgAddr |= (SPhysPgOffset::invalid_addr<<16);
                    if (i < levelsTakingAtLeastOnePageCount)
                    {
                        fill.subresource.mipLevel = i;
                        fill.outRange.offset = {(pgtOffset.x>>i) + x, (pgtOffset.y>>i) + y, 0u};
                        fill.fillValue.asUint.x = physPgAddr;

                        CFillImageFilter::execute(&fill);
                    }

                    core::vector3du32_SIMD physPg = pageCoords(physPgAddr, m_pgSzxy);
                    physPg -= core::vector2du32_SIMD(m_tilePadding, m_tilePadding);

                    const core::vector2du32_SIMD miptailOffset = (i>=levelsTakingAtLeastOnePageCount) ? core::vector2du32_SIMD(m_miptailOffsets[i-levelsTakingAtLeastOnePageCount].x,m_miptailOffsets[i-levelsTakingAtLeastOnePageCount].y) : core::vector2du32_SIMD(0u,0u);
                    physPg += miptailOffset;

                    CPaddedCopyImageFilter::state_type copy;
                    copy.outOffsetBaseLayer = (physPg).xyzz();/*physPg.z is layer*/ copy.outOffset.z = 0u;
                    copy.inOffsetBaseLayer = core::vector2du32_SIMD(x,y)*m_pgSzxy;
                    copy.extentLayerCount = core::vectorSIMDu32(m_pgSzxy, m_pgSzxy, 1u, 1u);
                    copy.relativeOffset = {0u,0u,0u};
                    if (x == w-1u)
                        copy.extentLayerCount.x = std::max(extent.width>>(_subres.baseMipLevel+i),1u)-copy.inOffsetBaseLayer.x;
                    if (y == h-1u)
                        copy.extentLayerCount.y = std::max(extent.height>>(_subres.baseMipLevel+i),1u)-copy.inOffsetBaseLayer.y;
                    memcpy(&copy.paddedExtent.width,(copy.extentLayerCount+core::vectorSIMDu32(2u*m_tilePadding)).pointer, 2u*sizeof(uint32_t));
                    copy.paddedExtent.depth = 1u;
                    if (w>1u)
                        copy.extentLayerCount.x += m_tilePadding;
                    if (x>0u && x<w-1u)
                        copy.extentLayerCount.x += m_tilePadding;
                    if (h>1u)
                        copy.extentLayerCount.y += m_tilePadding;
                    if (y>0u && y<h-1u)
                        copy.extentLayerCount.y += m_tilePadding;
                    if (x == 0u)
                        copy.relativeOffset.x = m_tilePadding;
                    else
                        copy.inOffsetBaseLayer.x -= m_tilePadding;
                    if (y == 0u)
                        copy.relativeOffset.y = m_tilePadding;
                    else
                        copy.inOffsetBaseLayer.y -= m_tilePadding;
                    copy.inMipLevel = _subres.baseMipLevel + i;
                    copy.outMipLevel = 0u;
                    copy.inImage = _img;
                    copy.outImage = m_physAddrTex.get();
                    copy.axisWraps[0] = _wrapu;
                    copy.axisWraps[1] = _wrapv;
                    copy.axisWraps[2] = ISampler::ETC_CLAMP_TO_EDGE;
                    copy.borderColor = _borderColor;
                    if (!CPaddedCopyImageFilter::execute(&copy))
                        _IRR_DEBUG_BREAK_IF(true);
                }
        }

        return pgtOffset;
    }

    uint32_t physAddrTexelByteoffset(const core::vector3du32_SIMD& _offset, uint32_t _texelSz)
    {
        const uint32_t offset_lin = (m_physAddrTex->getCreationParameters().extent.width*m_physAddrTex->getCreationParameters().extent.height*_offset.z + _offset.y*m_physAddrTex->getCreationParameters().extent.width + _offset.x) * _texelSz;

        return offset_lin;
    }
    uint8_t* physAddrTexelPtr(const core::vector3du32_SIMD& _offset, uint32_t _texelSz)
    {
        return reinterpret_cast<uint8_t*>(m_physAddrTex->getBuffer()->getPointer()) + physAddrTexelByteoffset(_offset, _texelSz);
    }

    ICPUImage* getPhysicalAddressTexture() { return m_physAddrTex.get(); }
    ICPUImage* getPageTable() { return m_pageTable.get(); }

private:
    uint32_t countLevelsTakingAtLeastOnePage(const VkExtent3D& _extent, const ICPUImage::SSubresourceRange& _subres)
    {
        const uint32_t baseMaxDim = core::roundUpToPoT(core::max(_extent.width, _extent.height))>>_subres.baseMipLevel;
        const int32_t lastFullMip = core::findMSB(baseMaxDim) - static_cast<int32_t>(m_pgSzxy_log2);

        return core::clamp(lastFullMip+1, 0, static_cast<int32_t>(m_pageTable->getCreationParameters().mipLevels));
    }

    core::smart_refctd_ptr<ICPUImage> m_physAddrTex;
    core::smart_refctd_ptr<ICPUImage> m_pageTable;
    const uint32_t m_tilePadding;
    const uint32_t m_pgtabLayerOffset;

    using phys_pg_addr_alctr_t = core::PoolAddressAllocator<uint32_t>;
    //core::smart_refctd_dynamic_array<uint8_t> m_physPgAddrAlctr_reservedSpc;
    uint8_t* m_physPgAddrAlctr_reservedSpc;
    phys_pg_addr_alctr_t m_physPgAddrAlctr;

    SMiptailPacker::rect m_miptailOffsets[MAX_PHYSICAL_PAGE_SIZE_LOG2];
};

bool ICPUTexturePacker::SMiptailPacker::computeMiptailOffsets(ICPUTexturePacker::SMiptailPacker::rect* res, int log2SIZE, int padding)
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