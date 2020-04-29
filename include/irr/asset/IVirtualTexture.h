#ifndef __IRR_I_VIRTUAL_TEXTURE_H_INCLUDED__
#define __IRR_I_VIRTUAL_TEXTURE_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/PoolAddressAllocator.h"
#include "irr/core/math/morton.h"
#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/core/memory/memory.h"
#include "irr/asset/filters/CPaddedCopyImageFilter.h"
#include "irr/asset/filters/CFillImageFilter.h"

namespace irr {
namespace asset
{

template <typename image_view_t>
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
    _IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_PHYSICAL_PAGE_SIZE_LOG2 = 9u;
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

    void updateLayerToViewIndexLUT(uint32_t _layer, uint32_t _smplrIndex)
    {
        (*m_layerToViewIndexMapping)[_layer] = _smplrIndex;
        m_layer2viewWasUpdatedSinceLastQuery = true;
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
    mutable bool m_layer2viewWasUpdatedSinceLastQuery = true;
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
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_ADDR_BITLENGTH = 16u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_ADDR_X_BITS = 4u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_ADDR_X_MASK = (1u<<PAGE_ADDR_X_BITS)-1u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_ADDR_Y_BITS = 4u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_ADDR_Y_MASK = (1u<<PAGE_ADDR_Y_BITS)-1u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t PAGE_ADDR_LAYER_SHIFT = PAGE_ADDR_BITLENGTH - PAGE_ADDR_X_BITS - PAGE_ADDR_Y_BITS;

        _IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_TILES_PER_DIM = std::min(PAGE_ADDR_X_MASK,PAGE_ADDR_Y_MASK) + 1u;
        _IRR_STATIC_INLINE_CONSTEXPR uint32_t MAX_LAYERS = (1u<<(PAGE_ADDR_BITLENGTH-PAGE_ADDR_LAYER_SHIFT));

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
            m_decodeAddr_layerShift(core::findLSB(_tilesPerDim)<<1),
            m_decodeAddr_xMask((1u<<(m_decodeAddr_layerShift>>1))-1u)
        {
            assert(_tilesPerDim<=MAX_TILES_PER_DIM);
            assert(_layers<=MAX_LAYERS);
        }
        //TODO: this should also copy address allocator state, but from what i see our address allocators doesnt have copy ctors
        IVTResidentStorage(core::smart_refctd_ptr<image_t>&& _image, const core::vector<uint16_t>& _assignedLayers, uint32_t _layerShift, uint32_t _xmask) :
            image(std::move(_image)),
            m_assignedPageTableLayers(_assignedLayers),
            m_decodeAddr_layerShift(_layerShift),
            m_decodeAddr_xMask(_xmask)
        {

        }

        uint16_t encodePageAddress(uint16_t _addr) const
        {
            const uint16_t x = _addr & m_decodeAddr_xMask;
            const uint16_t y = (_addr>>(m_decodeAddr_layerShift>>1)) & m_decodeAddr_xMask;
            const uint16_t layer = _addr >> m_decodeAddr_layerShift;

            return x | (y<<PAGE_ADDR_X_BITS) | (layer<<PAGE_ADDR_LAYER_SHIFT);
        }

        core::smart_refctd_ptr<image_view_t> createView(E_FORMAT _format) const
        {
            auto found = m_viewsCache.find(_format);
            if (found!=m_viewsCache.end())
                return found->second;

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

            return m_viewsCache.insert({_format,createView_internal(std::move(params))}).first->second;
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

        inline uint32_t physPgOffset_x(SPhysPgOffset _offset) const { return _offset.addr & PAGE_ADDR_X_MASK; }
        inline uint32_t physPgOffset_y(SPhysPgOffset _offset) const { return (_offset.addr >> PAGE_ADDR_X_BITS) & PAGE_ADDR_Y_MASK; }
        inline uint32_t physPgOffset_layer(SPhysPgOffset _offset) const { return (_offset.addr & 0xffffu)>>PAGE_ADDR_LAYER_SHIFT; }
        inline bool physPgOffset_valid(SPhysPgOffset _offset) const { return (_offset.addr&0xffffu) != SPhysPgOffset::invalid_addr; }
        inline bool physPgOffset_hasMipTailAddr(SPhysPgOffset _offset) const { return physPgOffset_valid(physPgOffset_mipTailAddr(_offset)); }
        inline SPhysPgOffset physPgOffset_mipTailAddr(SPhysPgOffset _offset) const { return _offset.addr>>PAGE_ADDR_BITLENGTH; }
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
        const uint32_t m_decodeAddr_layerShift;
        const uint32_t m_decodeAddr_xMask;

    protected:
        virtual core::smart_refctd_ptr<image_view_t> createView_internal(typename image_view_t::SCreationParams&& _params) const = 0;

    private:
        mutable core::unordered_map<E_FORMAT, core::smart_refctd_ptr<image_view_t>> m_viewsCache;
    };
    //since c++14 std::hash specialization for all enum types are given by standard
    core::unordered_map<E_FORMAT_CLASS, core::smart_refctd_ptr<IVTResidentStorage>> m_storage;

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

    size_t getLayerToViewIndexLUTBytesize() const
    {
        return m_layerToViewIndexMapping->size()*sizeof(uint32_t);
    }
    void writeLayerToViewIndexLUTContents(void* _dst) const
    {
        memcpy(_dst, m_layerToViewIndexMapping->data(), getLayerToViewIndexLUTBytesize());
        m_layer2viewWasUpdatedSinceLastQuery = false;
    }
    bool layerToViewIndexLUTWasUpdated() const
    {
        return m_layer2viewWasUpdatedSinceLastQuery;
    }

    static std::string getGLSLExtensionsIncludePath()
    {
        return "irr/builtin/glsl/virtual_texturing/extensions.glsl";
    }
    std::string getGLSLDescriptorsIncludePath(uint32_t _set, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return "irr/builtin/glsl/virtual_texturing/descriptors.glsl/" +
            std::to_string(_set) + "/" +
            std::to_string(_pgtBinding) + "/" +
            std::to_string(_fsamplersBinding) + "/" +
            std::to_string(_isamplersBinding) + "/" +
            std::to_string(_usamplersBinding) + "/" +
            (m_fsamplers.views ? std::to_string(m_fsamplers.views->size()) : "0") + "/" +
            (m_isamplers.views ? std::to_string(m_isamplers.views->size()) : "0") + "/" +
            (m_usamplers.views ? std::to_string(m_usamplers.views->size()) : "0");
    }
    std::string getGLSLFunctionsIncludePath(const std::string& _get_pgtab_sz_log2_name, const std::string& _get_phys_pg_tex_sz_rcp_name, const std::string& _get_vtex_sz_rcp_name, const std::string& _get_layer2pid) const
    {
        //functions.glsl/pg_sz_log2/tile_padding/pgtab_tex_name/phys_pg_tex_name/get_pgtab_sz_log2_name/get_phys_pg_tex_sz_rcp_name/get_vtex_sz_rcp_name/get_layer2pid/(addr_x_bits/addr_y_bits)...
        std::string s = "irr/builtin/glsl/virtual_texturing/functions.glsl/";
        s += std::to_string(m_pgSzxy_log2) + "/";
        s += std::to_string(m_tilePadding) + "/";
        s += _get_pgtab_sz_log2_name + "/";
        s += _get_phys_pg_tex_sz_rcp_name + "/";
        s += _get_vtex_sz_rcp_name + "/";
        s += _get_layer2pid;

        return s;
    }

protected:
    template <typename DSlayout_t>
    core::smart_refctd_dynamic_array<typename DSlayout_t::SBinding> getDSlayoutBindings_internal(uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        using retval_t = core::smart_refctd_dynamic_array<typename DSlayout_t::SBinding>;
        auto retval = core::make_refctd_dynamic_array<retval_t>(1u+(getFloatViews().size()?1u:0u)+(getIntViews().size()?1u:0u)+(getUintViews().size()?1u:0u));
        auto* bindings = retval->data();

        auto fillBinding = [](auto& bnd, uint32_t _binding, uint32_t _count) {
            bnd.binding = _binding;
            bnd.count = _count;
            bnd.stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;
            bnd.type = asset::EDT_COMBINED_IMAGE_SAMPLER;
            bnd.samplers = nullptr; //samplers are left for user to specify at will
        };

        fillBinding(bindings[0], _pgtBinding, 1u);

        uint32_t i = 1u;
        if (getFloatViews().size())
        {
            fillBinding(bindings[i], _fsamplersBinding, getFloatViews().size());
            ++i;
        }
        if (getIntViews().size())
        {
            fillBinding(bindings[i], _isamplersBinding, getIntViews().size());
            ++i;
        }
        if (getUintViews().size())
        {
            fillBinding(bindings[i], _usamplersBinding, getUintViews().size());
        }

        return retval;
    }
    template <typename DS_t>
    std::pair<core::smart_refctd_dynamic_array<typename DS_t::SWriteDescriptorSet>, core::smart_refctd_dynamic_array<typename DS_t::SDescriptorInfo>>
        getDescriptorSetWrites_internal(DS_t* _dstSet, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        using writes_t = core::smart_refctd_dynamic_array<typename DS_t::SWriteDescriptorSet>;
        using info_t = core::smart_refctd_dynamic_array<typename DS_t::SDescriptorInfo>;
        using retval_t = std::pair<writes_t, info_t>;

        const uint32_t writeCount = 1u+(getFloatViews().size()?1u:0u)+(getIntViews().size()?1u:0u)+(getUintViews().size()?1u:0u);
        const uint32_t infoCount = 1u + getFloatViews().size() + getIntViews().size() + getUintViews().size();

        auto writes_array = core::make_refctd_dynamic_array<writes_t>(writeCount);
        auto* writes = writes_array->data();
        auto info_array = core::make_refctd_dynamic_array<info_t>(infoCount);
        auto* info = info_array->data();

        writes[0].binding = _pgtBinding;
        writes[0].arrayElement = 0u;
        writes[0].count = 1u;
        writes[0].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
        writes[0].dstSet = _dstSet;
        writes[0].info = info;
        info[0].desc = createPageTableView();
        info[0].image.imageLayout = EIL_UNDEFINED;
        info[0].image.sampler = nullptr; //samplers are left for user to specify at will

        uint32_t i = 1u, j = 1u;
        if (getFloatViews().size())
        {
            writes[i].binding = _fsamplersBinding;
            writes[i].arrayElement = 0u;
            writes[i].count = getFloatViews().size();
            writes[i].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
            writes[i].dstSet = _dstSet;
            writes[i].info = info+j;
            for (uint32_t j0 = j; (j-j0)<writes[i].count; ++j)
            {
                info[j].desc = getFloatViews().begin()[j-j0];
                info[j].image.imageLayout = EIL_UNDEFINED;
                info[j].image.sampler = nullptr;
            }
            ++i;
        }
        if (getIntViews().size())
        {
            writes[i].binding = _isamplersBinding;
            writes[i].arrayElement = 0u;
            writes[i].count = getIntViews().size();
            writes[i].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
            writes[i].dstSet = _dstSet;
            writes[i].info = info+j;
            for (uint32_t j0 = j; (j-j0)<writes[i].count; ++j)
            {
                info[j].desc = getIntViews().begin()[j-j0];
                info[j].image.imageLayout = EIL_UNDEFINED;
                info[j].image.sampler = nullptr;
            }
            ++i;
        }
        if (getUintViews().size())
        {
            writes[i].binding = _usamplersBinding;
            writes[i].arrayElement = 0u;
            writes[i].count = getUintViews().size();
            writes[i].descriptorType = EDT_COMBINED_IMAGE_SAMPLER;
            writes[i].dstSet = _dstSet;
            writes[i].info = info+j;
            for (uint32_t j0 = j; (j-j0)<writes[i].count; ++j)
            {
                info[j].desc = getUintViews().begin()[j-j0];
                info[j].image.imageLayout = EIL_UNDEFINED;
                info[j].image.sampler = nullptr;
            }
            ++i;
        }

        return retval_t{std::move(writes_array),std::move(info_array)};
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

}}

#endif