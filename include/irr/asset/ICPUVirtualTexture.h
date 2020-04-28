#ifndef __IRR_I_CPU_VIRTUAL_TEXTURE_H_INCLUDED__
#define __IRR_I_CPU_VIRTUAL_TEXTURE_H_INCLUDED__

#include <irr/asset/IVirtualTexture.h>
#include <irr/asset/ICPUImageView.h>
#include <irr/asset/ICPUDescriptorSet.h>

namespace irr {
namespace asset
{

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

                        if (!CFillImageFilter::execute(&fill))
                            _IRR_DEBUG_BREAK_IF(true);
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

    auto getDSlayoutBindings(uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return getDSlayoutBindings_internal<ICPUDescriptorSetLayout>(_pgtBinding, _fsamplersBinding, _isamplersBinding, _usamplersBinding);
    }

    auto getDescriptorSetWrites(ICPUDescriptorSet* _dstSet, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return getDescriptorSetWrites_internal<ICPUDescriptorSet>(_dstSet, _pgtBinding, _fsamplersBinding, _isamplersBinding, _usamplersBinding);
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

}}

#endif