// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_CPU_VIRTUAL_TEXTURE_H_INCLUDED__
#define __NBL_ASSET_I_CPU_VIRTUAL_TEXTURE_H_INCLUDED__

#include <nbl/asset/utils/IVirtualTexture.h>
#include <nbl/asset/ICPUImageView.h>
#include <nbl/asset/ICPUDescriptorSet.h>

#include "nbl/asset/filters/CMipMapGenerationImageFilter.h"

namespace nbl {
namespace asset
{

class ICPUVirtualTexture final : public IVirtualTexture<ICPUImageView, ICPUSampler>
{
    using base_t = IVirtualTexture<ICPUImageView, ICPUSampler>;

public:
    class ICPUVTResidentStorage final : public base_t::IVTResidentStorage
    {
        using base_t = base_t::IVTResidentStorage;

    public:
        ICPUVTResidentStorage(E_FORMAT _format, uint32_t _tilesPerDim) :
            base_t(_format, _tilesPerDim)
        {

        }
        ICPUVTResidentStorage(E_FORMAT _format, uint32_t _tileExtent, uint32_t _layers, uint32_t _tilesPerDim) :
            base_t(_format, _layers, _tilesPerDim)
        {
            deferredInitialization(_tileExtent, _layers);
        }

        void deferredInitialization(uint32_t tileExtent, uint32_t _layers) override
        {
            const uint32_t tilesPerDim = getTilesPerDim();
            const uint32_t extent = tileExtent * tilesPerDim;

            // deduce layer count from the need of physical space
            if (_layers == 0u)
            {
                const uint32_t tilesPerLayer = tilesPerDim * tilesPerDim;
                _layers = (m_tileCounter + tilesPerLayer - 1u) / tilesPerLayer;
            }

            base_t::deferredInitialization(tileExtent, _layers);

            if (image)
                return;

            ICPUImage::SCreationParams params;
            params.extent = { extent,extent,1u };
            params.format = imageFormat;
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
                region.bufferRowLength = extent;
                region.bufferImageHeight = 0u; //tightly packed
                region.imageOffset = { 0u,0u,0u };
                region.imageExtent = params.extent;
                auto buffer = core::make_smart_refctd_ptr<ICPUBuffer>(getTexelOrBlockBytesize(imageFormat) * params.extent.width * params.extent.height * params.arrayLayers);
                image->setBufferAndRegions(std::move(buffer), regions);
            }
        }

    private:
        core::smart_refctd_ptr<ICPUImageView> createView_internal(ICPUImageView::SCreationParams&& _params) const override
        {
            return ICPUImageView::create(std::move(_params));
        }
    };

    //! If there's a need, creates an subresource image rescaled to requested dimensions, otherwise returns `_img`
    //! Always call this before alloc() if you want to make sure it doesn't fail due to `isAllocatable` 
    core::smart_refctd_ptr<asset::ICPUImage> createResizedImage(const ICPUImage* _img, const VkExtent3D& newSubresourceExtent, const uint8_t baseMipLevel, const uint16_t baseArrayLayer=0u, const uint16_t depthOffset=0u) const
    {
        if (!_img || newSubresourceExtent.width==0u || newSubresourceExtent.height==0u || newSubresourceExtent.depth!=1u)
            return nullptr;

        const auto& params = _img->getCreationParameters();
        if (params.samples!=ICPUImage::ESCF_1_BIT)
            return nullptr;

        const auto _allocationExtent = _img->getMipSize(baseMipLevel);
        const VkExtent3D allocationExtent = {_allocationExtent.x,_allocationExtent.y,1u};
        if (allocationExtent==newSubresourceExtent)
        {
            asset::ICPUImage* img = const_cast<asset::ICPUImage*>(_img);
            return core::smart_refctd_ptr<asset::ICPUImage>(img);
        }

        core::smart_refctd_ptr<asset::ICPUImage> scaled_img;
        // create image
        {
            asset::ICPUImage::SCreationParams new_params = params;
            new_params.extent = newSubresourceExtent;
            new_params.mipLevels = 1u; // TODO: blit the mip-levels as well if present (but right now no point cause we never get input textures with any part of the pyramid anyway)
            new_params.arrayLayers = 1u;
            scaled_img = asset::ICPUImage::create(std::move(new_params));
        }
        // back its pixels with storage
        {
            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(1u);
            auto region = regions->begin();
            region->bufferOffset = 0u;
            region->bufferRowLength = newSubresourceExtent.width;
            region->bufferImageHeight = 0u;
            region->imageOffset = {0,0,0};
            region->imageExtent = newSubresourceExtent;
            region->imageSubresource.baseArrayLayer = 0u;
            region->imageSubresource.layerCount = 1u;
            region->imageSubresource.mipLevel = 0u;
            region->imageSubresource.aspectMask = _img->getRegions().begin()->imageSubresource.aspectMask;

            const size_t bufsz = scaled_img->getImageDataSizeInBytes(); // TODO: probably need a `IImage::SBufferCopy::getByteSize()` instead here
            auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufsz);
            scaled_img->setBufferAndRegions(std::move(buf),std::move(regions));
        }
        // run blit
        {
            using blit_filter_t = asset::CBlitImageFilter<asset::VoidSwizzle,asset::IdentityDither/*TODO: White Noise*/,void,false,asset::CMitchellImageFilterKernel<>>;
            blit_filter_t::state_type blit = {};
            blit.inOffset = {0u,0u,0u};
            blit.inBaseLayer = baseArrayLayer;
            blit.inExtent = params.extent;
            blit.inExtent.depth = 1u;
            blit.inLayerCount = 1u;
            blit.outOffsetBaseLayer = core::vectorSIMDu32(0u,0u,0u, 0u);
            blit.outExtent = newSubresourceExtent;
            blit.outLayerCount = 1u;
            blit.inMipLevel = baseMipLevel;
            blit.outMipLevel = 0u;
            blit.inImage = const_cast<asset::ICPUImage*>(_img);
            blit.outImage = scaled_img.get();
            blit.scratchMemoryByteSize = blit_filter_t::getRequiredScratchByteSize(&blit);

            auto scratch = std::make_unique<uint8_t[]>(blit.scratchMemoryByteSize);
            blit.scratchMemory = scratch.get();
            if (!blit_filter_t::execute(&blit))
                return nullptr;
        }

        return scaled_img;
    }

    //! Always call this before commit()
    //! Future TODO: @Erfan/@Achal remove this utility and just make the `commit` handle this padding and mip-chain generation "in-place" (create temporary images with regions "alised" to allocated pages :D )
    static std::pair<core::smart_refctd_ptr<asset::ICPUImage>, asset::VkExtent3D> createPoTPaddedSquareImageWithMipLevels(const ICPUImage* _img, ISampler::E_TEXTURE_CLAMP _wrapu, ISampler::E_TEXTURE_CLAMP _wrapv, ISampler::E_TEXTURE_BORDER_COLOR _borderColor)
    {
        if (!_img)
            return { nullptr, asset::VkExtent3D{0u,0u,0u} };

        const auto& params = _img->getCreationParameters();
        const auto originalExtent = params.extent;
        const uint32_t paddedExtent = core::roundUpToPoT(std::max<uint32_t>(params.extent.width,params.extent.height));

        //create PoT and square image with regions for all mips
        ICPUImage::SCreationParams paddedParams = params;
        paddedParams.extent = {paddedExtent,paddedExtent,1u};
        //in case of original extent being non-PoT, padding it to PoT gives us one extra mip level
        paddedParams.mipLevels = core::findLSB(paddedExtent) + 1u;
        auto paddedImg = asset::ICPUImage::create(std::move(paddedParams));
        {
            const uint32_t texelBytesize = asset::getTexelOrBlockBytesize(params.format);

            auto regions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::IImage::SBufferCopy>>(paddedImg->getCreationParameters().mipLevels);
            uint32_t bufoffset = 0u;
            for (uint32_t i = 0u; i < regions->size(); ++i)
            {
                auto& region = (*regions)[i];
                region.bufferImageHeight = 0u;
                region.bufferOffset = bufoffset;
                region.bufferRowLength = paddedExtent>>i;
                region.imageExtent = {paddedExtent>>i,paddedExtent>>i,1u};
                region.imageOffset = {0u,0u,0u};
                region.imageSubresource.baseArrayLayer = 0u;
                region.imageSubresource.layerCount = 1u;
                region.imageSubresource.mipLevel = i;

                bufoffset += texelBytesize*region.imageExtent.width*region.imageExtent.height;
            }
            auto buf = core::make_smart_refctd_ptr<asset::ICPUBuffer>(bufoffset);
            paddedImg->setBufferAndRegions(std::move(buf), regions);
        }

        //copy mip 0 to new image while filling padding according to wrapping modes
        asset::CPaddedCopyImageFilter::state_type copy;
        copy.axisWraps[0] = _wrapu;
        copy.axisWraps[1] = _wrapv;
        copy.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
        copy.borderColor = _borderColor;
        copy.extent = params.extent;
        copy.layerCount = 1u;
        copy.inMipLevel = 0u;
        copy.inOffset = {0u,0u,0u};
        copy.inBaseLayer = 0u;
        copy.outOffset = {0u,0u,0u};
        copy.outBaseLayer = 0u;
        copy.outMipLevel = 0u;
        copy.paddedExtent = {paddedExtent,paddedExtent,1u};
        copy.relativeOffset = {0u,0u,0u};
        copy.inImage = _img;
        copy.outImage = paddedImg.get();

        asset::CPaddedCopyImageFilter::execute(std::execution::par_unseq,&copy);

        using mip_gen_filter_t = asset::CMipMapGenerationImageFilter<
            VoidSwizzle,IdentityDither,void/*TODO: whitenoise*/,false,
            asset::CBoxImageFilterKernel, asset::CBoxImageFilterKernel,
            asset::CBoxImageFilterKernel, asset::CBoxImageFilterKernel,
            asset::CBoxImageFilterKernel, asset::CBoxImageFilterKernel
        >;
        //generate all mip levels
        {
            mip_gen_filter_t::state_type genmips;
            genmips.baseLayer = 0u;
            genmips.layerCount = 1u;
            genmips.startMipLevel = 1u;
            genmips.endMipLevel = paddedImg->getCreationParameters().mipLevels;
            genmips.inOutImage = paddedImg.get();
            genmips.scratchMemoryByteSize = mip_gen_filter_t::getRequiredScratchByteSize(&genmips);
            genmips.scratchMemory = reinterpret_cast<uint8_t*>(_NBL_ALIGNED_MALLOC(genmips.scratchMemoryByteSize,_NBL_SIMD_ALIGNMENT));
            genmips.axisWraps[0] = _wrapu;
            genmips.axisWraps[1] = _wrapv;
            genmips.axisWraps[2] = asset::ISampler::ETC_CLAMP_TO_EDGE;
            genmips.borderColor = _borderColor;
            mip_gen_filter_t::execute(std::execution::par_unseq,&genmips);
            _NBL_ALIGNED_FREE(genmips.scratchMemory);
        }

        return std::make_pair(std::move(paddedImg), originalExtent);
    }

    ICPUVirtualTexture(
        physical_tiles_per_dim_log2_callback_t&& _callback,
        const base_t::IVTResidentStorage::SCreationParams* _residentStorageParams,
        uint32_t _residentStorageCount,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _pgTabLayers = 32u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) : IVirtualTexture(
            std::move(_callback), _maxAllocatableTexSz_log2-_pgSzxy_log2, _pgTabLayers, _pgSzxy_log2, _tilePadding
        )
    {
        m_pageTable = createPageTable(m_pgtabSzxy_log2, _pgTabLayers, _pgSzxy_log2, _maxAllocatableTexSz_log2);
        initResidentStorage(_residentStorageParams, _residentStorageCount);
    }

    ICPUVirtualTexture(
        physical_tiles_per_dim_log2_callback_t&& _callback,
        uint32_t _pgSzxy_log2 = 7u,
        uint32_t _tilePadding = 9u,
        uint32_t _maxAllocatableTexSz_log2 = 14u
    ) : IVirtualTexture(
            std::move(_callback), _maxAllocatableTexSz_log2-_pgSzxy_log2, MAX_PAGE_TABLE_LAYERS, _pgSzxy_log2, _tilePadding
        )
    {
    }

    // TODO: thread safe commits?
    bool commit(const SMasterTextureData& _addr, const ICPUImage* _img, const IImage::SSubresourceRange& _subres, ISampler::E_TEXTURE_CLAMP _uwrap, ISampler::E_TEXTURE_CLAMP _vwrap, ISampler::E_TEXTURE_BORDER_COLOR _borderColor) override 
    {
        if (!validateCommit(_addr, _subres, _uwrap, _vwrap))
            return false;

        const page_tab_offset_t pgtOffset(_addr.pgTab_x, _addr.pgTab_y, _addr.pgTab_layer);

        ICPUVTResidentStorage* storage = nullptr;
        {
            uint32_t layer = pgtOffset.z;
            E_FORMAT format = getFormatInLayer(pgtOffset.z);
            E_FORMAT_CLASS fc = getFormatClass(format);
            auto found = m_storage.find(fc);
            if (found==m_storage.end())
                return false;
            storage = static_cast<ICPUVTResidentStorage*>(found->second.get());
        }

        const VkExtent3D extent = {static_cast<uint32_t>(_addr.origsize_x), static_cast<uint32_t>(_addr.origsize_y), 1u};

        const uint32_t levelsTakingAtLeastOnePageCount = countLevelsTakingAtLeastOnePage(extent);
        const uint32_t levelsToPack = std::min<uint32_t>(_subres.levelCount, m_pageTable->getCreationParameters().mipLevels+m_pgSzxy_log2);

        uint32_t miptailPgAddr = SPhysPgOffset::invalid_addr;

        using phys_pg_addr_alctr_t = ICPUVTResidentStorage::phys_pg_addr_alctr_t;

        if (levelsTakingAtLeastOnePageCount < _subres.levelCount)
        {
            uint32_t miptailPgAddr_tmp = phys_pg_addr_alctr_t::invalid_address;
            const uint32_t szAndAlignment = 1u;
            core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_alloc_addr(storage->tileAlctr, 1u, &miptailPgAddr_tmp, &szAndAlignment, &szAndAlignment, nullptr);
            miptailPgAddr_tmp = (miptailPgAddr_tmp == phys_pg_addr_alctr_t::invalid_address) ? SPhysPgOffset::invalid_addr : storage->encodePageAddress(miptailPgAddr_tmp);

            miptailPgAddr = miptailPgAddr_tmp;
        }

        const bool wholeTexGoesToMiptailPage = (levelsTakingAtLeastOnePageCount == 0u);

        //TODO up to this line, it's kinda common code for CPU and GPU, refactor later

        // TODO: parallelize over all 3 for loops
        for (uint32_t i = 0u; i < levelsToPack; ++i)
        {
            const uint32_t w = neededPageCountForSide(extent.width, i);
            const uint32_t h = neededPageCountForSide(extent.height, i);

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t physPgAddr = phys_pg_addr_alctr_t::invalid_address;
                    if (i>=levelsTakingAtLeastOnePageCount) // this `if` always executes in case of whole texture going into miptail page
                        physPgAddr = miptailPgAddr;
                    else
                    {
                        const uint32_t szAndAlignment = 1u;
                        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_alloc_addr(storage->tileAlctr, 1u, &physPgAddr, &szAndAlignment, &szAndAlignment, nullptr);
                        if (physPgAddr == phys_pg_addr_alctr_t::invalid_address)
                            physPgAddr = SPhysPgOffset::invalid_addr;
                        else
                            physPgAddr = storage->encodePageAddress(physPgAddr);
                    }

                    if (i==(levelsTakingAtLeastOnePageCount-1u) && levelsTakingAtLeastOnePageCount<_subres.levelCount)
                    {
                        assert(w==1u && h==1u);
     
                        physPgAddr |= (miptailPgAddr<<SPhysPgOffset::PAGE_ADDR_BITLENGTH);
                    }
                    else  // this `else` always executes in case of whole texture going into miptail page
                        physPgAddr |= (SPhysPgOffset::invalid_addr<<SPhysPgOffset::PAGE_ADDR_BITLENGTH);

                    if (i < levelsTakingAtLeastOnePageCount)
                    {
                        // physical double-address to write into page table
                        const uint32_t physAddrToWrite = physPgAddr;

                        const auto texelPos = core::vectorSIMDu32(pgtOffset.x>>i, pgtOffset.y>>i, 0u, pgtOffset.z) + core::vectorSIMDu32(x, y, 0u, 0u);
                        const auto* region = m_pageTable->getRegion(i, texelPos);
                        const uint64_t byteoffset = region->getByteOffset(texelPos, region->getByteStrides(m_pageTable->getTexelBlockInfo()));
                        uint8_t* bufptr = reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + byteoffset;
                        reinterpret_cast<uint32_t*>(bufptr)[0] = physAddrToWrite;
                    }

                    if (!SPhysPgOffset(physPgAddr).valid())
                        continue;

                    core::vector3du32_SIMD physPg = ICPUVTResidentStorage::pageCoords(physPgAddr, m_pgSzxy, m_tilePadding);
                    physPg -= core::vector2du32_SIMD(m_tilePadding, m_tilePadding);

                    const core::vector2du32_SIMD miptailOffset = (i>=levelsTakingAtLeastOnePageCount) ? core::vector2du32_SIMD(m_miptailOffsets[i-levelsTakingAtLeastOnePageCount].x,m_miptailOffsets[i-levelsTakingAtLeastOnePageCount].y) : core::vector2du32_SIMD(0u,0u);
                    physPg += miptailOffset;

                    CPaddedCopyImageFilter::state_type copy;
                    copy.outOffsetBaseLayer = (physPg).xyzz();/*physPg.z is layer*/ copy.outOffset.z = 0u;
                    copy.inOffsetBaseLayer = core::vector2du32_SIMD(x,y)*m_pgSzxy;
                    copy.extentLayerCount = core::vectorSIMDu32(m_pgSzxy, m_pgSzxy, 1u, 1u);
                    copy.relativeOffset = {0u,0u,0u};
                    if (x == w-1u)
                        copy.extentLayerCount.x = std::max<uint32_t>(extent.width>>i,1u)-copy.inOffsetBaseLayer.x;
                    if (y == h-1u)
                        copy.extentLayerCount.y = std::max<uint32_t>(extent.height>>i,1u)-copy.inOffsetBaseLayer.y;
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
                    copy.inOffsetBaseLayer.w = _subres.baseArrayLayer;
                    copy.inMipLevel = _subres.baseMipLevel + i;
                    copy.outMipLevel = 0u;
                    copy.inImage = _img;
                    copy.outImage = storage->image.get();
                    copy.axisWraps[0] = _uwrap;
                    copy.axisWraps[1] = _vwrap;
                    copy.axisWraps[2] = ISampler::ETC_CLAMP_TO_EDGE;
                    copy.borderColor = _borderColor;
                    if (!CPaddedCopyImageFilter::execute(std::execution::par_unseq,&copy))
                        assert(false);
                }
        }

        if (wholeTexGoesToMiptailPage)
        {
            // physical double-address to write into page table
            uint32_t physAddrToWrite = SPhysPgOffset::invalid_addr | (miptailPgAddr << SPhysPgOffset::PAGE_ADDR_BITLENGTH);

            const auto texelPos = core::vectorSIMDu32(pgtOffset.x, pgtOffset.y, 0u, pgtOffset.z);
            const auto* region = m_pageTable->getRegion(0u, texelPos);
            const uint64_t byteoffset = region->getByteOffset(texelPos, region->getByteStrides(m_pageTable->getTexelBlockInfo()));
            uint8_t* bufptr = reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer()) + byteoffset;
            reinterpret_cast<uint32_t*>(bufptr)[0] = physAddrToWrite;
        }

        return true;
    }

    SViewAliasTextureData createAlias(const SMasterTextureData& _addr, E_FORMAT _viewingFormat, const IImage::SSubresourceRange& _subresRelativeToMaster) override
    {
        if (!validateAliasCreation(_addr, _viewingFormat, _subresRelativeToMaster))
            return SViewAliasTextureData::invalid();

        const VkExtent3D extent = {
            static_cast<uint32_t>(_addr.origsize_x>>_subresRelativeToMaster.baseArrayLayer),
            static_cast<uint32_t>(_addr.origsize_y>>_subresRelativeToMaster.baseArrayLayer),
            1u};
        SMasterTextureData aliasAddr = alloc(_viewingFormat, VkExtent3D{static_cast<uint32_t>(_addr.origsize_x), static_cast<uint32_t>(_addr.origsize_y), 1u}, _subresRelativeToMaster, ISampler::ETC_CLAMP_TO_BORDER, ISampler::ETC_CLAMP_TO_BORDER);
        if (SMasterTextureData::is_invalid(aliasAddr))
            return SViewAliasTextureData::invalid();
        aliasAddr.wrap_x = _addr.wrap_x;
        aliasAddr.wrap_y = _addr.wrap_y;

        CCopyImageFilter::state_type copy;
        copy.inImage = m_pageTable.get();
        copy.outImage = m_pageTable.get();
        copy.outBaseLayer = aliasAddr.pgTab_layer;
        copy.inBaseLayer = _addr.pgTab_layer;
        copy.layerCount = 1u;
        for (uint32_t i = 0u; i < _subresRelativeToMaster.levelCount; ++i)
        {
            copy.inMipLevel = _subresRelativeToMaster.baseMipLevel+i;
            copy.outMipLevel = i;
            copy.extent = {std::max<uint32_t>(extent.width>>i,1u), std::max<uint32_t>(extent.height>>i,1u), 1u};
            copy.inOffset = {static_cast<uint32_t>(_addr.pgTab_x>>(copy.inMipLevel)),static_cast<uint32_t>(_addr.pgTab_y>>(copy.inMipLevel)),0u};
            copy.outOffset = {static_cast<uint32_t>(aliasAddr.pgTab_x>>i), static_cast<uint32_t>(aliasAddr.pgTab_y>>i), 0u};

            CCopyImageFilter::execute(std::execution::par_unseq,&copy);
        }

        //nasty trick
        return reinterpret_cast<SViewAliasTextureData*>(&aliasAddr)[0];
    }

    bool free(const SMasterTextureData& _addr) override
    {
        const E_FORMAT format = getFormatInLayer(_addr.pgTab_layer);
        ICPUVTResidentStorage* storage = static_cast<ICPUVTResidentStorage*>(getStorageForFormatClass(getFormatClass(format)));
        if (!storage)
            return false;

        //free physical pages
        VkExtent3D extent = {static_cast<uint32_t>(_addr.origsize_x), static_cast<uint32_t>(_addr.origsize_y), 1u};

#ifdef _NBL_DEBUG
        CFillImageFilter::state_type fill;
        fill.outImage = m_pageTable.get();
        fill.subresource.aspectMask = static_cast<IImage::E_ASPECT_FLAGS>(0);
        fill.subresource.baseArrayLayer = _addr.pgTab_layer;
        fill.subresource.layerCount = 1u;
        fill.fillValue.asUint.x = SPhysPgOffset::invalid_addr;
#endif

        using phys_pg_addr_alctr_t = ICPUVTResidentStorage::phys_pg_addr_alctr_t;

        uint32_t addrsOffset = 0u;
        std::fill(m_addrsArray->begin(), m_addrsArray->end(), IVTResidentStorage::phys_pg_addr_alctr_t::invalid_address);

        auto* const bufptr = reinterpret_cast<uint8_t*>(m_pageTable->getBuffer()->getPointer());
        auto levelCount = core::max(_addr.maxMip,1u);
        for (uint32_t i=0u; i<levelCount; ++i)
        {
            const uint32_t w = neededPageCountForSide(extent.width, i);
            const uint32_t h = neededPageCountForSide(extent.height, i);

            const auto& region = m_pageTable->getRegions().begin()[i];
            const auto strides = region.getByteStrides(TexelBlockInfo(m_pageTable->getCreationParameters().format));

            for (uint32_t y = 0u; y < h; ++y)
                for (uint32_t x = 0u; x < w; ++x)
                {
                    uint32_t* texelptr = reinterpret_cast<uint32_t*>(bufptr + region.getByteOffset(core::vector4du32_SIMD((_addr.pgTab_x>>i) + x, (_addr.pgTab_y>>i) + y, 0u, _addr.pgTab_layer), strides));
                    SPhysPgOffset physPgOffset = *texelptr;

                    (*m_addrsArray)[addrsOffset + y*w + x] = physPgOffset.addr&SPhysPgOffset::PAGE_ADDR_MASK;
                    if (physPgOffset.hasMipTailAddr())
                    {
                        assert(i==levelCount-1u && w==1u && h==1u);
                        (*m_addrsArray)[addrsOffset + y*w + x + 1u] = physPgOffset.mipTailAddr().addr;
                    }
                }

            addrsOffset += w*h;
#ifdef _NBL_DEBUG
            fill.subresource.mipLevel = i;
            fill.outRange.offset = {static_cast<uint32_t>(_addr.pgTab_x>>i),static_cast<uint32_t>(_addr.pgTab_y>>i),0u};
            fill.outRange.extent = {w,h,1u};
            CFillImageFilter::execute(&fill);
#endif
        }

        core::address_allocator_traits<phys_pg_addr_alctr_t>::multi_free_addr(storage->tileAlctr, m_addrsArray->size(), m_addrsArray->data(), m_addrsArray->data());

        //free entries in page table
        if (!base_t::free(_addr))
            return false;

        return true;
    }

    auto getDSlayoutBindings(ICPUDescriptorSetLayout::SBinding* _outBindings, core::smart_refctd_ptr<ICPUSampler>* _outSamplers, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return getDSlayoutBindings_internal<ICPUDescriptorSetLayout>(_outBindings, _outSamplers, _pgtBinding, _fsamplersBinding, _isamplersBinding, _usamplersBinding);
    }

    auto getDescriptorSetWrites(ICPUDescriptorSet::SWriteDescriptorSet* _outWrites, ICPUDescriptorSet::SDescriptorInfo* _outInfo, ICPUDescriptorSet* _dstSet, uint32_t _pgtBinding = 0u, uint32_t _fsamplersBinding = 1u, uint32_t _isamplersBinding = 2u, uint32_t _usamplersBinding = 3u) const
    {
        return getDescriptorSetWrites_internal<ICPUDescriptorSet>(_outWrites, _outInfo, _dstSet, _pgtBinding, _fsamplersBinding, _isamplersBinding, _usamplersBinding);
    }

protected:
    core::smart_refctd_ptr<ICPUImageView> createPageTableView() const override
    {
        return ICPUImageView::create(createPageTableViewCreationParams());
    }
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(E_FORMAT _format, uint32_t _tileExtent, uint32_t _layers, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<ICPUVTResidentStorage>(_format, _tileExtent, _layers, _tilesPerDim);
    }
    core::smart_refctd_ptr<IVTResidentStorage> createVTResidentStorage(E_FORMAT _format, uint32_t _tilesPerDim) override
    {
        return core::make_smart_refctd_ptr<ICPUVTResidentStorage>(_format, _tilesPerDim);
    }
    core::smart_refctd_ptr<ICPUImage> createPageTableImage(ICPUImage::SCreationParams&& _params) const override
    {
        return ICPUImage::create(std::move(_params));
    }
    core::smart_refctd_ptr<ICPUSampler> createSampler(const ISampler::SParams& _params) const override
    {
        return core::make_smart_refctd_ptr<ICPUSampler>(_params);
    }
};

}}

#endif
