// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED_
#define _NBL_VIDEO_I_GPU_OBJECT_FROM_ASSET_CONVERTER_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/core/alloc/LinearAddressAllocator.h"

#include "nbl/video/ISemaphore.h"
#include "nbl/video/ILogicalDevice.h"

#if 0
// TODO: rewrite after GPU polyphase implementation
auto IGPUObjectFromAssetConverter::create(const asset::ICPUImage** const _begin, const asset::ICPUImage** const _end, SParams& _params) -> created_gpu_object_array<asset::ICPUImage>
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImage> >(assetCount);

    // TODO: This should be the other way round because if a queue supports either compute or graphics but not the other way round
    const uint32_t transferFamIx = _params.perQueue[EQU_TRANSFER].queue->getFamilyIndex();
    const uint32_t computeFamIx = _params.perQueue[EQU_COMPUTE].queue ? _params.perQueue[EQU_COMPUTE].queue->getFamilyIndex() : transferFamIx;

    bool oneQueue = _params.perQueue[EQU_TRANSFER].queue == _params.perQueue[EQU_COMPUTE].queue;

    bool needToGenMips = false;
    
    core::unordered_map<const asset::ICPUImage*, core::smart_refctd_ptr<IGPUBuffer>> img2gpubuf;
    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUImage* cpuimg = _begin[i];
        if (cpuimg->getRegions().size() == 0ull)
            continue;

        // TODO: Why isn't this buffer cached and why are we not going through recursive asset creation and getting ICPUBuffer equivalents? 
        //(we can always discard/not cache the GPU Buffers created only for image data upload)
        IGPUBuffer::SCreationParams params = {};
        params.usage = core::bitflag(IGPUBuffer::EUF_TRANSFER_SRC_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
        const auto& cpuimgParams = cpuimg->getCreationParameters();
        params.size = cpuimg->getBuffer()->getSize();

        auto gpubuf = _params.device->createBuffer(std::move(params));
        auto mreqs = gpubuf->getMemoryReqs();
        mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto gpubufMem = _params.device->allocate(mreqs, gpubuf.get());

        img2gpubuf.insert({ cpuimg, std::move(gpubuf) });

        const auto format = cpuimg->getCreationParameters().format;
        if (!asset::isIntegerFormat(format) && !asset::isBlockCompressionFormat(format))
            needToGenMips = true;
    }

    bool oneSubmitPerBatch = !needToGenMips || oneQueue;

    auto& transfer_fence = _params.fences[EQU_TRANSFER]; 
    auto cmdbuf_transfer = _params.perQueue[EQU_TRANSFER].cmdbuf;
    auto cmdbuf_compute = _params.perQueue[EQU_COMPUTE].cmdbuf;

    if (img2gpubuf.size())
    {
        transfer_fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));

        // User will call begin on cmdbuf now
        // cmdbuf_transfer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        assert(cmdbuf_transfer && cmdbuf_transfer->getState() == IGPUCommandBuffer::STATE::RECORDING);
        if (oneQueue)
        {
            cmdbuf_compute = cmdbuf_transfer;
        }
        else if (needToGenMips)
        {
            assert(cmdbuf_compute && cmdbuf_compute->getState() == IGPUCommandBuffer::STATE::RECORDING);
            // User will call begin on cmdbuf now
            // cmdbuf_compute->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        }
    }

    auto needToCompMipsForThisImg = [](const asset::ICPUImage* img) -> bool
    {
        if (img->getRegions().empty())
            return false;
        auto format = img->getCreationParameters().format;
        if (asset::isIntegerFormat(format) || asset::isBlockCompressionFormat(format))
            return false;
        // its enough to define a single mipmap region above the base level to prevent automatic computation
        for (auto& region : img->getRegions())
        if (region.imageSubresource.mipLevel)
            return false;
        return true;
    };
    
    IQueue::SSubmitInfo submit_transfer;
    {
        submit_transfer.commandBufferCount = 1u;
        submit_transfer.commandBuffers = &cmdbuf_transfer.get();
        // buffer and image written and copied to are fresh (or in the case of streaming buffer, at least fenced before freeing), no need to wait for anything external
        submit_transfer.waitSemaphoreCount = 0u;
        submit_transfer.pWaitSemaphores = nullptr;
        submit_transfer.pWaitDstStageMask = nullptr;
    }
    auto cmdComputeMip = [&](const asset::ICPUImage* cpuimg, IGPUImage* gpuimg, IGPUImage::LAYOUT newLayout) -> void
    {
        const auto& realParams = gpuimg->getCreationParameters();
        // TODO when we have compute shader mips generation:
        /*computeCmdbuf->bindPipeline();
        computeCmdbuf->bindDescriptorSets();
        computeCmdbuf->pushConstants();
        computeCmdbuf->dispatch();*/

        IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
        decltype(info)::image_barrier_t barrier = {};
        info.imgBarriers = &barrier;
        info.imgBarrierCount = 1u;

        barrier.barrier.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE;
        barrier.barrier.otherQueueFamilyIndex = cmdbuf_compute->getQueueFamilyIndex();
        barrier.image = gpuimg;
        // TODO this is probably wrong (especially in case of depth/stencil formats), but i think i can leave it like this since we'll never have any depth/stencil images loaded (right?)
        barrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT; // not hardcode
        barrier.subresourceRange.levelCount = 1u;
        barrier.subresourceRange.layerCount = realParams.arrayLayers;
        barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;

        IGPUCommandBuffer::SImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.srcSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.srcSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.srcOffsets[0] = { 0, 0, 0 };

        blitRegion.dstSubresource.aspectMask = barrier.subresourceRange.aspectMask;
        blitRegion.dstSubresource.baseArrayLayer = barrier.subresourceRange.baseArrayLayer;
        blitRegion.dstSubresource.layerCount = barrier.subresourceRange.layerCount;
        blitRegion.dstOffsets[0] = { 0, 0, 0 };

        // Compute mips
        auto mipsize = cpuimg->getMipSize(cpuimg->getCreationParameters().mipLevels);
        uint32_t mipWidth = mipsize.x;
        uint32_t mipHeight = mipsize.y;
        uint32_t mipDepth = mipsize.z;
        for (uint32_t i = cpuimg->getCreationParameters().mipLevels; i < gpuimg->getCreationParameters().mipLevels; ++i)
        {
            const uint32_t srcLoD = i - 1u;

            // TODO: with compute blit these will have to be COMPUTE 2 COMPUTE barriers from DST to newLayout (with an intermediate transition to GENERAL for storage)
            barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
            barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
            barrier.newLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
            barrier.subresourceRange.baseMipLevel = srcLoD;
            barrier.subresourceRange.levelCount = 1u;

            cmdbuf_transfer->pipelineBarrier(asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT, asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,
                static_cast<asset::E_DEPENDENCY_FLAGS>(0u), 0u, nullptr, 0u, nullptr, 1u, &barrier);

            const auto srcMipSz = cpuimg->getMipSize(srcLoD);

            blitRegion.srcSubresource.mipLevel = srcLoD;
            blitRegion.srcOffsets[1] = { srcMipSz.x, srcMipSz.y, srcMipSz.z };

            blitRegion.dstSubresource.mipLevel = i;
            blitRegion.dstOffsets[1] = { mipWidth, mipHeight, mipDepth };

            // TODO: Remove the requirement that the transfer queue has graphics caps,
            cmdbuf_transfer->blitImage(gpuimg, IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL, gpuimg,
                IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL, 1u, &blitRegion, asset::ISampler::ETF_LINEAR);

            if (mipWidth > 1u) mipWidth /= 2u;
            if (mipHeight > 1u) mipHeight /= 2u;
            if (mipDepth > 1u) mipDepth /= 2u;
        }
        
        barrier.subresourceRange.baseMipLevel = cpuimg->getCreationParameters().mipLevels - 1u;
        barrier.subresourceRange.levelCount = gpuimg->getCreationParameters().mipLevels - 1u;
        barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
        barrier.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
        barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = newLayout;

        cmdbuf_transfer->pipelineBarrier(
            asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,
            finalStageMask,
            static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
            0u, nullptr,
            0u, nullptr,
            1u, &barrier);

        // Transition the last mip level to correct layout
        barrier.subresourceRange.baseMipLevel = gpuimg->getCreationParameters().mipLevels - 1u;
        barrier.subresourceRange.levelCount = 1u;
        barrier.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
        cmdbuf_transfer->pipelineBarrier(
            asset::PIPELINE_STAGE_FLAGS::TRANSFER_BIT,
            finalStageMask,
            static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
            0u, nullptr,
            0u, nullptr,
            1u, &barrier);
    };

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUImage* cpuimg = _begin[i];
        IGPUImage::SCreationParams params = {};
        params = cpuimg->getCreationParameters();
        
        IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
        promotionRequest.originalFormat = params.format;
        promotionRequest.usages = {};

        // override the mip-count if its not an integer format and there was no mip-pyramid specified 
        if (params.mipLevels==1u && !asset::isIntegerFormat(params.format))
            params.mipLevels = 1u + static_cast<uint32_t>(std::log2(static_cast<float>(core::max<uint32_t>(core::max<uint32_t>(params.extent.width, params.extent.height), params.extent.depth))));

        if (cpuimg->getRegions().size())
            params.usage |= asset::IImage::EUF_TRANSFER_DST_BIT;
        
        const bool computeMips = needToCompMipsForThisImg(cpuimg);
        if (computeMips)
        {
            params.usage |= asset::IImage::EUF_TRANSFER_SRC_BIT; // this is for blit
            // I'm already adding usage flags for mip-mapping compute shader
            params.usage |= asset::IImage::EUF_SAMPLED_BIT; // to read source mips
            // but we don't add the STORAGE USAGE, yet
            // TODO: will change when we do the blit on compute shader.
            promotionRequest.usages.blitDst = true;
            promotionRequest.usages.blitSrc = true;
        }
        
        auto physDev = _params.device->getPhysicalDevice();
        promotionRequest.usages = promotionRequest.usages | params.usage;
        auto newFormat = physDev->promoteImageFormat(promotionRequest, IGPUImage::TILING::OPTIMAL);
        auto newFormatIsStorable = physDev->getImageFormatUsagesOptimalTiling()[newFormat].storageImage;
        
        // If Format Promotion failed try the same usages but with linear tiling.
        if (newFormat == asset::EF_UNKNOWN)
        {
            newFormat = physDev->promoteImageFormat(promotionRequest, IGPUImage::TILING::LINEAR);
            newFormatIsStorable = physDev->getImageFormatUsagesLinearTiling()[newFormat].storageImage;
            params.tiling = IGPUImage::TILING::LINEAR;
        }

        assert(newFormat != asset::EF_UNKNOWN); // No feasible supported format found for creating this image
        params.format = newFormat;

        // now add the STORAGE USAGE
        if (computeMips)
        {
            // formats like SRGB etc. can't be stored to
            params.usage |= asset::IImage::EUF_STORAGE_BIT;
            // but image views with formats that are store-able can be created
            if (!newFormatIsStorable)
            {
                params.flags |= asset::IImage::ECF_MUTABLE_FORMAT_BIT;
                params.flags |= asset::IImage::ECF_EXTENDED_USAGE_BIT;
                if (asset::isBlockCompressionFormat(newFormat))
                    params.flags |= asset::IImage::ECF_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
            }
        }

        auto gpuimg = _params.device->createImage(std::move(params));
        auto gpuimgMemReqs = gpuimg->getMemoryReqs();
        gpuimgMemReqs.memoryTypeBits &= physDev->getDeviceLocalMemoryTypeBits();
        auto gpuimgMem = _params.device->allocate(gpuimgMemReqs, gpuimg.get());

		res->operator[](i) = std::move(gpuimg);
    }

    if (img2gpubuf.size() == 0ull)
        return res;

    auto it = _begin;
    auto doBatch = [&]() -> void
    {
        constexpr uint32_t pipeliningDepth = 8u;

        IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {};
        decltype(info)::image_barrier_t imgbarriers[pipeliningDepth];
        info.imgBarriers = imgbarriers;

        const uint32_t n = it - _begin;

        auto oldIt = it;
        for (uint32_t i = 0u; i < pipeliningDepth && it != _end; ++i)
        {
            auto* cpuimg = *(it++);
            auto* gpuimg = (*res)[n+i].get();
            // Creation of a GPU image (on Vulkan) can fail for several reasons, one of them, for
            // example is unsupported formats
            if (!gpuimg)
                continue;

            // There should be no pipeline barrier before this, because the first usage implicitly acquires the resource for the queue!
            submit_transfer = _params.utilities->updateImageViaStagingBuffer(
                cpuimg->getBuffer(), cpuimg->getCreationParameters().format, gpuimg, IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL, cpuimg->getRegions(),
                _params.perQueue[EQU_TRANSFER].queue, transfer_fence.get(), submit_transfer
            );
            
            IGPUImage::LAYOUT newLayout;
            const auto& realParams = gpuimg->getCreationParameters();
            if (realParams.usage.hasFlags(asset::IImage::EUF_SAMPLED_BIT))
                newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
            else
                newLayout = IGPUImage::LAYOUT::GENERAL;

            if (needToCompMipsForThisImg(cpuimg))
            {
                assert(_params.device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[realParams.format].sampledImage);
                assert(asset::isFloatingPointFormat(realParams.format) || asset::isNormalizedFormat(realParams.format));
                cmdComputeMip(cpuimg, gpuimg, newLayout);
            }
            else
            {
                auto& b = imgbarriers[info.imgBarrierCount++] = {};
                b.barrier.dep.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT;
                b.barrier.dep.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT;
                if (ownershipTfer)
                {
                    b.barrier.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE;
                    b.barrier.otherQueueFamilyIndex = computeFamIx;
                }
                else
                {
                    b.barrier.dep.dstStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
                    b.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS|asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
                }
                b.image = gpuimg;
                if (asset::isDepthOrStencilFormat(realParams.format))
                {
                    if (!asset::isStencilOnlyFormat(realParams.format))
                        b.subresourceRange.aspectMask |= IGPUImage::EAF_DEPTH_BIT;
                    if (!asset::isDepthOnlyFormat(realParams.format))
                        b.subresourceRange.aspectMask |= IGPUImage::EAF_STENCIL_BIT;
                }
                else
                    b.subresourceRange.aspectMask = IGPUImage::EAF_COLOR_BIT;
                b.subresourceRange.levelCount = realParams.mipLevels;
                b.subresourceRange.layerCount = realParams.arrayLayers;
                b.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
                b.newLayout = newLayout;
            }
        }

        // ownership transition release or just a barrier
        cmdbuf_transfer->pipelineBarrier(asset::EDF_NONE,info);

        if (transferFamIx!=computeFamIx && cmdbuf_compute) // need to do ownership transition
        {
            // ownership transition acquire
            for (auto j=0u; j<info.imgBarrierCount; j++)
            {
                auto& b = imgbarriers[j];
                b.barrier.dep = {};
                b.barrier.dep.dstStageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
                b.barrier.dep.dstAccessMask = asset::ACCESS_FLAGS::MEMORY_READ_BITS|asset::ACCESS_FLAGS::MEMORY_WRITE_BITS;
                b.barrier.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE;
                b.barrier.otherQueueFamilyIndex = transferFamIx;
            }
            cmdbuf_transfer->pipelineBarrier(asset::EDF_NONE,info);
        }

        cmdbuf_transfer->end();
        if (needToGenMips && !oneQueue)
            cmdbuf_compute->end();

        auto transfer_sem = _params.device->createSemaphore();
        auto* transfer_sem_ptr = transfer_sem.get();

        auto batch_final_fence = transfer_fence;

        submit_transfer.signalSemaphoreCount = 1u;
        submit_transfer.pSignalSemaphores = &transfer_sem_ptr;
        _params.perQueue[EQU_TRANSFER].queue->submit(1u, &submit_transfer, batch_final_fence.get());

        if (_params.perQueue[EQU_TRANSFER].semaphore)
            _params.perQueue[EQU_TRANSFER].semaphore[0] = transfer_sem;
        if (_params.perQueue[EQU_COMPUTE].semaphore && oneQueue && needToGenMips)
            _params.perQueue[EQU_COMPUTE].semaphore[0] = transfer_sem;

         // must be outside `if` scope to not get deleted after `batch_final_fence = compute_fence_ptr;` assignment
        auto & compute_fence = _params.fences[EQU_COMPUTE];
        if (!oneSubmitPerBatch)
        {
            core::smart_refctd_ptr<IGPUSemaphore> compute_sem;
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                compute_sem = _params.device->createSemaphore();
            auto* compute_sem_ptr = compute_sem.get();
            compute_fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            auto* compute_fence_ptr = compute_fence.get();

            const auto dstWait = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            auto* cb_comp = cmdbuf_compute.get();
            IQueue::SSubmitInfo submit_compute;
            submit_compute.commandBufferCount = 1u;
            submit_compute.commandBuffers = &cb_comp;
            submit_compute.waitSemaphoreCount = 1u;
            submit_compute.pWaitDstStageMask = &dstWait;
            submit_compute.pWaitSemaphores = &transfer_sem_ptr;
            submit_compute.signalSemaphoreCount = compute_sem?1u:0u;
            submit_compute.pSignalSemaphores = compute_sem?&compute_sem_ptr:nullptr;
            _params.perQueue[EQU_COMPUTE].queue->submit(1u, &submit_compute, compute_fence_ptr);

            if (_params.perQueue[EQU_COMPUTE].semaphore)
                _params.perQueue[EQU_COMPUTE].semaphore[0] = compute_sem;

#if 0 //TODO: (!) enable when mips are in fact computed on `cmdbuf_compute` (currently they are done with blits on cmdbuf_transfer)
            batch_final_fence = compute_fence;
#endif
        }

        // wait to finish all batch work in order to safely reset command buffers
        _params.device->waitForFences(1u, &batch_final_fence.get(), false, 9999999999ull);

        if (!oneSubmitPerBatch)
            _params.device->waitForFences(1u, &compute_fence.get(), false, 9999999999ull);

        // separate cmdbufs per batch instead?
        if (it != _end)
        {
            cmdbuf_transfer->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
            cmdbuf_transfer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            _params.device->resetFences(1u, &transfer_fence.get());
            
            if (!oneSubmitPerBatch)
            {
                cmdbuf_compute->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
                cmdbuf_compute->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
                _params.device->resetFences(1u, &compute_fence.get());
            }
        }
    };

    while (it != _end)
    {
        doBatch();
    }

    return res;
}
inline created_gpu_object_array<asset::ICPUImageView> IGPUObjectFromAssetConverter::create(const asset::ICPUImageView** const _begin, const asset::ICPUImageView** const _end, SParams& _params)
{
    const auto assetCount = std::distance(_begin, _end);
    auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUImageView> >(assetCount);

    core::vector<asset::ICPUImage*> cpuDeps;
    cpuDeps.reserve(res->size());

    const asset::ICPUImageView** it = _begin;
    while (it != _end)
    {
        cpuDeps.push_back((*it)->getCreationParameters().image.get());
        ++it;
    }

    core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuDeps);

    auto gpuDeps = getGPUObjectsFromAssets<asset::ICPUImage>(cpuDeps.data(), cpuDeps.data() + cpuDeps.size(), _params);
    const auto physDev = _params.device->getPhysicalDevice();
    const auto& optimalUsages = physDev->getImageFormatUsagesOptimalTiling();
    const auto& linearUsages = physDev->getImageFormatUsagesLinearTiling();
    for (ptrdiff_t i = 0; i < assetCount; ++i)
    {
        if (gpuDeps->begin()[redirs[i]])
        {
            const auto& cpuParams = _begin[i]->getCreationParameters();

            IGPUImageView::SCreationParams params = {};
            params.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(cpuParams.flags);
            params.viewType = static_cast<IGPUImageView::E_TYPE>(cpuParams.viewType);
            params.image = (*gpuDeps)[redirs[i]];
            const auto& gpuImgParams = params.image->getCreationParameters();
            // override the view's format if the source image got promoted, a bit crude, but don't want to scratch my head about how to promote the views and guess semantics
            const bool formatGotPromoted = cpuParams.format!=gpuImgParams.format;
            params.format = formatGotPromoted ? gpuImgParams.format:cpuParams.format;
            params.subUsages = cpuParams.subUsages;
            // TODO: In Asset Converter 2.0 we'd pass through all descriptor sets etc and propagate the adding usages backwards to views, but here we need to trim the image's usages instead
            {
                IPhysicalDevice::SFormatImageUsages::SUsage validUsages(gpuImgParams.usage);
                if (params.image->getTiling()!=IGPUImage::TILING::LINEAR)
                    validUsages = validUsages & optimalUsages[params.format];
                else
                    validUsages = validUsages & linearUsages[params.format];
                // add them after trimming
                if (validUsages.sampledImage)
                    params.subUsages |= IGPUImage::EUF_SAMPLED_BIT;
                if (validUsages.storageImage)
                    params.subUsages |= IGPUImage::EUF_STORAGE_BIT;
                if (validUsages.attachment)
                    params.subUsages |= IGPUImage::EUF_RENDER_ATTACHMENT_BIT;
                if (validUsages.transferSrc)
                    params.subUsages |= IGPUImage::EUF_TRANSFER_SRC_BIT;
                if (validUsages.transferDst)
                    params.subUsages |= IGPUImage::EUF_TRANSFER_DST_BIT;
                // stuff thats not dependent on device caps
                const auto uncappedUsages = IGPUImage::EUF_TRANSIENT_ATTACHMENT_BIT|IGPUImage::EUF_INPUT_ATTACHMENT_BIT|IGPUImage::EUF_SHADING_RATE_ATTACHMENT_BIT|IGPUImage::EUF_FRAGMENT_DENSITY_MAP_BIT;
                params.subUsages |= gpuImgParams.usage&uncappedUsages;
            }
            memcpy(&params.components, &cpuParams.components, sizeof(params.components));
            params.subresourceRange = cpuParams.subresourceRange;
            // TODO: Undo this, make all loaders set the level and layer counts on image views to `ICPUImageView::remaining_...`
            params.subresourceRange.levelCount = gpuImgParams.mipLevels-params.subresourceRange.baseMipLevel;
            (*res)[i] = _params.device->createImageView(std::move(params));
        }
    }

    return res;
}

inline created_gpu_object_array<asset::ICPUDescriptorSet> IGPUObjectFromAssetConverter::create(const asset::ICPUDescriptorSet** const _begin, const asset::ICPUDescriptorSet** const _end, SParams& _params)
{
                else if (isSampledImgViewDesc(type))
                {
                    auto cpuImgView = static_cast<asset::ICPUImageView*>(descriptor);
                    auto cpuImg = cpuImgView->getCreationParameters().image;
                    if (cpuImg)
                        cpuImg->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);
                    cpuImgViews.push_back(cpuImgView);
                    if (info->info.image.sampler)
                        cpuSamplers.push_back(info->info.image.sampler.get());
                }
                else if (isStorageImgDesc(type))
                {
                    auto cpuImgView = static_cast<asset::ICPUImageView*>(descriptor);
                    auto cpuImg = cpuImgView->getCreationParameters().image;
                    if (cpuImg)
                        cpuImg->addImageUsageFlags(asset::IImage::EUF_STORAGE_BIT);
                    cpuImgViews.push_back(cpuImgView);
                }
}

auto IGPUObjectFromAssetConverter::create(const asset::ICPUAccelerationStructure** _begin, const asset::ICPUAccelerationStructure** _end, SParams& _params) -> created_gpu_object_array<asset::ICPUAccelerationStructure>
{
	const size_t assetCount = std::distance(_begin, _end);
	auto res = core::make_refctd_dynamic_array<created_gpu_object_array<asset::ICPUAccelerationStructure> >(assetCount);
	auto toCreateAndBuild = std::vector<const asset::ICPUAccelerationStructure*>();
    auto buildRangeInfos = std::vector<IGPUAccelerationStructure::BuildRangeInfo*>();
    toCreateAndBuild.reserve(assetCount);
    buildRangeInfos.reserve(assetCount);
    // Lambda function: creates the acceleration structure and It's buffer
    auto allocateBufferAndCreateAccelerationStructure = [&](size_t asSize, const asset::ICPUAccelerationStructure* cpuas)
    {
        // Create buffer with cpuas->getAccelerationStructureSize
        IGPUBuffer::SCreationParams gpuBufParams = {};
        gpuBufParams.size = asSize;
        gpuBufParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
        auto gpubuf = _params.device->createBuffer(std::move(gpuBufParams));
        auto mreqs = gpubuf->getMemoryReqs();
        mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
        auto gpubufMem = _params.device->allocate(mreqs, gpubuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
        assert(gpubufMem.isValid());

        // Create GPUAccelerationStructure with that buffer
        IGPUAccelerationStructure::SCreationParams creatationParams = {};
        creatationParams.bufferRange.buffer = gpubuf;
        creatationParams.bufferRange.offset = 0;
        creatationParams.bufferRange.size = asSize;
        creatationParams.flags = cpuas->getCreationParameters().flags;
        creatationParams.type = cpuas->getCreationParameters().type;
        return _params.device->createAccelerationStructure(std::move(creatationParams));
    };

    for (ptrdiff_t i = 0u; i < assetCount; ++i)
    {
        const asset::ICPUAccelerationStructure* cpuas = _begin[i];

        if(cpuas->hasBuildInfo())
        {
            // Add to toBuild vector of ICPUAccelerationStructure
            toCreateAndBuild.push_back(cpuas);
            buildRangeInfos.push_back(const_cast<IGPUAccelerationStructure::BuildRangeInfo*>(cpuas->getBuildRanges().begin()));
        }
        else if(cpuas->getAccelerationStructureSize() > 0)
        {
            res->operator[](i) = allocateBufferAndCreateAccelerationStructure(cpuas->getAccelerationStructureSize(), cpuas);
        }
    }

    if(toCreateAndBuild.empty() == false)
    {
        bool hostBuildCommands = false; // get from SFeatures
        if(hostBuildCommands)
        {
            _NBL_TODO();
        }
        else
        {
            core::vector<const asset::ICPUBuffer*> cpuBufferDeps;
            constexpr uint32_t MaxGeometryPerBuildInfo = 16;
            constexpr uint32_t MaxBuffersPerGeometry = 3; // TrianglesData ->  vertex+index+transformation
            cpuBufferDeps.reserve(assetCount * MaxGeometryPerBuildInfo * MaxBuffersPerGeometry);

            // Get CPUBuffer Dependencies
            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = toCreateAndBuild[i];
            
                auto buildInfo = cpuas->getBuildInfo();
                assert(buildInfo != nullptr);

                auto geoms = buildInfo->getGeometries().begin();
                auto geomsCount = buildInfo->getGeometries().size();
                if(geomsCount == 0)
                {
                    assert(false);
                    continue;
                }

                for(uint32_t g = 0; g < geomsCount; ++g) 
                {
                    const auto& geom = geoms[g];
                    if(geom.type == asset::IAccelerationStructure::EGT_TRIANGLES)
                    {
                        if(geom.data.triangles.indexData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.indexData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                        if(geom.data.triangles.vertexData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.vertexData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                        if(geom.data.triangles.transformData.isValid())
                        {
                            auto cpuBuf = geom.data.triangles.transformData.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                    else if(geom.type == asset::IAccelerationStructure::EGT_AABBS)
                    {
                        if(geom.data.aabbs.data.isValid())
                        {
                            auto cpuBuf = geom.data.aabbs.data.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                    else if(geom.type == asset::IAccelerationStructure::EGT_INSTANCES)
                    {
                        if(geom.data.instances.data.isValid())
                        {
                            auto cpuBuf = geom.data.instances.data.buffer.get();
                            cpuBuf->addUsageFlags(core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT);
                            cpuBufferDeps.push_back(cpuBuf);
                        }
                    }
                }
            }

            // Convert CPUBuffer Deps to GPUBuffers
            core::vector<size_t> redirs = eliminateDuplicatesAndGenRedirs(cpuBufferDeps);
            auto gpuBufs = getGPUObjectsFromAssets<asset::ICPUBuffer>(cpuBufferDeps.data(), cpuBufferDeps.data()+cpuBufferDeps.size(), _params);
            _params.waitForCreationToComplete();
            _params.beginCommandBuffers();
            size_t bufIter = 0ull;

            // Fill buildGeomInfos partially (to later ge Get AS Size before build command)
            std::vector<IGPUAccelerationStructure::DeviceBuildGeometryInfo> buildGeomInfos(toCreateAndBuild.size());
     
            using GPUGeometry = IGPUAccelerationStructure::Geometry<IGPUAccelerationStructure::DeviceAddressType>;
            std::vector<GPUGeometry> gpuGeoms;
            gpuGeoms.reserve(assetCount * MaxGeometryPerBuildInfo);

            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = toCreateAndBuild[i];
            
                auto cpuBuildInfo = cpuas->getBuildInfo();
                auto & gpuBuildInfo = buildGeomInfos[i];

                gpuBuildInfo.type = cpuBuildInfo->type;
                gpuBuildInfo.buildFlags = cpuBuildInfo->buildFlags;
                gpuBuildInfo.buildMode = cpuBuildInfo->buildMode;
                assert(cpuBuildInfo->buildMode == asset::IAccelerationStructure::EBM_BUILD);

                // Fill Later:
                gpuBuildInfo.srcAS = nullptr;
                gpuBuildInfo.dstAS = nullptr;
                gpuBuildInfo.scratchAddr = {};
                
                auto cpu_geoms = cpuBuildInfo->getGeometries().begin();
                auto geomsCount = cpuBuildInfo->getGeometries().size();
                if(geomsCount == 0)
                {
                    assert(false);
                    continue;
                }

                size_t startGeom = gpuGeoms.size();
                size_t endGeom = gpuGeoms.size() + geomsCount;

                for(uint32_t g = 0; g < geomsCount; ++g) 
                {
                    const auto& cpu_geom = cpu_geoms[g];

                    GPUGeometry gpu_geom = {};
                    gpu_geom.type = cpu_geom.type;
                    gpu_geom.flags = cpu_geom.flags;

                    if(cpu_geom.type == asset::IAccelerationStructure::EGT_TRIANGLES)
                    {
                        gpu_geom.data.triangles.vertexFormat = cpu_geom.data.triangles.vertexFormat;
                        gpu_geom.data.triangles.vertexStride = cpu_geom.data.triangles.vertexStride;
                        gpu_geom.data.triangles.maxVertex = cpu_geom.data.triangles.maxVertex;
                        gpu_geom.data.triangles.indexType = cpu_geom.data.triangles.indexType;

                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.indexData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.indexData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.indexData.offset;
                        }
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.vertexData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.vertexData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.vertexData.offset;
                        }
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.triangles.transformData.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.triangles.transformData.offset = gpubuf->getOffset() + cpu_geom.data.triangles.transformData.offset;
                        }
                    }
                    else if(cpu_geom.type == asset::IAccelerationStructure::EGT_AABBS)
                    {
                        gpu_geom.data.aabbs.stride = cpu_geom.data.aabbs.stride;
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.aabbs.data.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.aabbs.data.offset = gpubuf->getOffset() + cpu_geom.data.aabbs.data.offset;
                        }
                    }
                    else if(cpu_geom.type == asset::IAccelerationStructure::EGT_INSTANCES)
                    {
                        {
                            IGPUOffsetBufferPair* gpubuf = (*gpuBufs)[redirs[bufIter++]].get();
                            gpu_geom.data.instances.data.buffer = core::smart_refctd_ptr<IGPUBuffer>(gpubuf->getBuffer());
                            gpu_geom.data.instances.data.offset = gpubuf->getOffset() + cpu_geom.data.instances.data.offset;
                        }
                    }

                    gpuGeoms.push_back(gpu_geom);
                }

                gpuBuildInfo.geometries = core::SRange<GPUGeometry>(gpuGeoms.data() + startGeom, gpuGeoms.data() + endGeom);
            }
            
            // Get SizeInfo for each CPUAS -> Create the AS -> Get Total Scratch Buffer Size 
            std::vector<IGPUAccelerationStructure::BuildSizes> buildSizes(toCreateAndBuild.size());
            uint64_t totalScratchBufferSize = 0ull;
            uint64_t maxScratchBufferSize = 0ull;
            for (ptrdiff_t i = 0u, toBuildIndex = 0u; i < assetCount; ++i)
            {
                const asset::ICPUAccelerationStructure* cpuas = _begin[i];
                if(cpuas->hasBuildInfo() == false)
                {
                    // Only those with buildInfo (index in toCreateAndBuild vector) will get passed
                    continue;
                }

                assert(cpuas == toCreateAndBuild[toBuildIndex]);
                assert(toBuildIndex < toCreateAndBuild.size());

                auto buildRanges = cpuas->getBuildRanges().begin();
                auto buildRangesCount = cpuas->getBuildRanges().size();

                auto & gpuBuildInfo = buildGeomInfos[toBuildIndex];
                
                std::vector<uint32_t> maxPrimCount(buildRangesCount);
                for(auto b = 0; b < buildRangesCount; b++)
                  maxPrimCount[b] = buildRanges[b].primitiveCount;

                auto buildSize = _params.device->getAccelerationStructureBuildSizes(gpuBuildInfo, maxPrimCount.data());
                buildSizes[i] = buildSize;

                auto gpuAS = allocateBufferAndCreateAccelerationStructure(buildSize.accelerationStructureSize, cpuas);
                res->operator[](i) = gpuAS;

                // complete the buildGeomInfos (now only thing left is to allocate and set scratchAddr.buffer)
                buildGeomInfos[toBuildIndex].dstAS = gpuAS.get();
                buildGeomInfos[toBuildIndex].scratchAddr.offset = totalScratchBufferSize;

                totalScratchBufferSize += buildSize.buildScratchSize;
                core::max(maxScratchBufferSize, buildSize.buildScratchSize); // maxScratchBufferSize has no use now (unless we changed this function to build 1 by 1 instead of batch builds or have some kind of memory limit?)
                ++toBuildIndex;
            }

            // Allocate Scratch Buffer
            IGPUBuffer::SCreationParams gpuScratchBufParams = {};
            gpuScratchBufParams.size = totalScratchBufferSize;
            gpuScratchBufParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | asset::IBuffer::EUF_STORAGE_BUFFER_BIT; 
            auto gpuScratchBuf = _params.device->createBuffer(std::move(gpuScratchBufParams));
            auto mreqs = gpuScratchBuf->getMemoryReqs();
            mreqs.memoryTypeBits &= _params.device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto gpuScratchBufMem = _params.device->allocate(mreqs, gpuScratchBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);


            for (ptrdiff_t i = 0u; i < toCreateAndBuild.size(); ++i)
            {
                auto & gpuBuildInfo = buildGeomInfos[i];
                gpuBuildInfo.scratchAddr.buffer = gpuScratchBuf;
            }

            // Record CommandBuffer for Building (We have Completed buildInfos + buildRanges for each CPUAS)
            auto & fence = _params.fences[EQU_COMPUTE];
            fence = _params.device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf = _params.perQueue[EQU_COMPUTE].cmdbuf;

            IQueue::SSubmitInfo submit;
            {
                submit.commandBufferCount = 1u;
                submit.commandBuffers = &cmdbuf.get();
                submit.waitSemaphoreCount = 0u;
                submit.pWaitDstStageMask = nullptr;
                submit.pWaitSemaphores = nullptr;
                uint32_t waitSemaphoreCount = 0u;
            }
            
            assert(cmdbuf->getState() == IGPUCommandBuffer::STATE::RECORDING);
            cmdbuf->buildAccelerationStructures({buildGeomInfos.data(),buildGeomInfos.data()+buildGeomInfos.size()},buildRangeInfos.data());
            cmdbuf->end();

            // TODO for future to make this function more sophisticated: Compaction, MemoryLimit for Build

            core::smart_refctd_ptr<IGPUSemaphore> sem;
            
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                sem = _params.device->createSemaphore();

            auto* sem_ptr = sem.get();
            auto* fence_ptr = fence.get();

            submit.signalSemaphoreCount = sem_ptr?1u:0u;
            submit.pSignalSemaphores = sem_ptr?&sem_ptr:nullptr;

            _params.perQueue[EQU_COMPUTE].queue->submit(1u, &submit, fence_ptr);
            if (_params.perQueue[EQU_COMPUTE].semaphore)
                _params.perQueue[EQU_COMPUTE].semaphore[0] = std::move(sem);
        }
    }

    return res;
}
#endif

#endif
