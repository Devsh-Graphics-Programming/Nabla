#ifndef __NBL_VIDEO_I_UTILITIES_H_INCLUDED__
#define __NBL_VIDEO_I_UTILITIES_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/IGPUImage.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/utilities/CPropertyPoolHandler.h"
#include "nbl/video/utilities/CScanner.h"

namespace nbl::video
{

class IUtilities : public core::IReferenceCounted
{
    public:
        IUtilities(core::smart_refctd_ptr<ILogicalDevice>&& _device, size_t downstreamSize=0x4000000ull, size_t upstreamSize=0x4000000ull) : m_device(std::move(_device))
        {
            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            {
                auto reqs = m_device->getDownStreamingMemoryReqs();
                reqs.vulkanReqs.size = downstreamSize;
                reqs.vulkanReqs.alignment = 64u * 1024u; // if you need larger alignments then you're not right in the head
                m_defaultDownloadBuffer = core::make_smart_refctd_ptr<StreamingTransientDataBufferMT<> >(m_device.get(), reqs);
            }
            {
                auto reqs = m_device->getUpStreamingMemoryReqs();
                reqs.vulkanReqs.size = upstreamSize;
                reqs.vulkanReqs.alignment = 64u * 1024u; // if you need larger alignments then you're not right in the head
                m_defaultUploadBuffer = core::make_smart_refctd_ptr<StreamingTransientDataBufferMT<> >(m_device.get(), reqs);
            }
            m_propertyPoolHandler = core::make_smart_refctd_ptr<CPropertyPoolHandler>(core::smart_refctd_ptr(m_device));
            // smaller workgroups fill occupancy gaps better, especially on new Nvidia GPUs, but we don't want too small workgroups on mobile
            // TODO: investigate whether we need to clamp against 256u instead of 128u on mobile
            const auto scan_workgroup_size = core::max(core::roundDownToPoT(limits.maxWorkgroupSize[0])>>1u,128u);
            m_scanner = core::make_smart_refctd_ptr<CScanner>(core::smart_refctd_ptr(m_device),scan_workgroup_size);
        }

        //!
        inline ILogicalDevice* getLogicalDevice() const { return m_device.get(); }

        //!
        inline StreamingTransientDataBufferMT<>* getDefaultUpStreamingBuffer()
        {
            return m_defaultUploadBuffer.get();
        }
        inline StreamingTransientDataBufferMT<>* getDefaultDownStreamingBuffer()
        {
            return m_defaultDownloadBuffer.get();
        }

        //!
        virtual CPropertyPoolHandler* getDefaultPropertyPoolHandler() const
        {
            return m_propertyPoolHandler.get();
        }

        //!
        virtual CScanner* getDefaultScanner() const
        {
            return m_scanner.get();
        }

        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUBuffer> createFilledDeviceLocalGPUBufferOnDedMem(IGPUQueue* queue, size_t size, const void* data)
        {
            IGPUBuffer::SCreationParams params = {};
            auto retval = m_device->createDeviceLocalGPUBufferOnDedMem(params, size);
            updateBufferRangeViaStagingBuffer(queue,asset::SBufferRange<IGPUBuffer>{0u,size,retval},data);
            return retval;
        }

        // TODO: Some utility in ILogical Device that can upload the image via the streaming buffer just from the regions without creating a whole intermediate huge GPU Buffer
        //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUCommandBuffer* cmdbuf, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions)
        {
            // This API check is temporary (or not?) since getFormatProperties is not
            // yet implemented on OpenGL
            if (srcBuffer->getAPIType() == EAT_VULKAN)
            {
                const auto reqFormatFeature = asset::EFF_TRANSFER_DST_BIT;
                const auto& formatProps = srcBuffer->getOriginDevice()->getPhysicalDevice()->getFormatProperties(params.format);
                if ((params.tiling == asset::IImage::ET_OPTIMAL) && (formatProps.optimalTilingFeatures & reqFormatFeature).value == 0)
                    return nullptr;
                if ((params.tiling == asset::IImage::ET_LINEAR) && (formatProps.linearTilingFeatures & reqFormatFeature).value == 0)
                    return nullptr;
            }

            const auto finalLayout = params.initialLayout;

            if (!((params.usage & asset::IImage::EUF_TRANSFER_DST_BIT).value))
                params.usage |= asset::IImage::EUF_TRANSFER_DST_BIT;

            auto retval = m_device->createDeviceLocalGPUImageOnDedMem(std::move(params));

            assert(cmdbuf->getState()==IGPUCommandBuffer::ES_RECORDING);

            IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
            barrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.oldLayout = asset::EIL_UNDEFINED;
            barrier.newLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = ~0u;
            barrier.dstQueueFamilyIndex = ~0u;
            barrier.image = retval;
            barrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT; // need this from input, infact this family of functions would be more usable if we take in a SSubresourceRange to operate on
            barrier.subresourceRange.baseArrayLayer = 0u;
            barrier.subresourceRange.layerCount = retval->getCreationParameters().arrayLayers;
            barrier.subresourceRange.baseMipLevel = 0u;
            barrier.subresourceRange.levelCount = retval->getCreationParameters().mipLevels;
            cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);

            cmdbuf->copyBufferToImage(srcBuffer,retval.get(),asset::EIL_TRANSFER_DST_OPTIMAL,regionCount,pRegions);

            if (finalLayout != asset::EIL_TRANSFER_DST_OPTIMAL)
            {
                barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
                barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
                barrier.oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = finalLayout;
                cmdbuf->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
            }

            return retval;
        }
        //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
            IGPUFence* fence, IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions,
            const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            auto cmdpool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_TRANSIENT_BIT);
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
            m_device->createCommandBuffers(cmdpool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
            assert(cmdbuf);
            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
            auto retval = createFilledDeviceLocalGPUImageOnDedMem(cmdbuf.get(),std::move(params),srcBuffer,regionCount,pRegions);
            cmdbuf->end();
            IGPUQueue::SSubmitInfo submit;
            submit.commandBufferCount = 1u;
            submit.commandBuffers = &cmdbuf.get();
            assert(!signalSemaphoreCount || semaphoresToSignal);
            submit.signalSemaphoreCount = signalSemaphoreCount;
            submit.pSignalSemaphores = semaphoresToSignal;
            assert(!waitSemaphoreCount || semaphoresToWaitBeforeExecution&&stagesToWaitForPerSemaphore);
            submit.waitSemaphoreCount = waitSemaphoreCount;
            submit.pWaitSemaphores = semaphoresToWaitBeforeExecution;
            submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
            queue->submit(1u,&submit,fence);
            m_device->waitForFences(1u, &fence, false, 9999999999ull);
            return retval;
        }
        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
            IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions,
            const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            auto* fenceptr = fence.get();
            auto retval = createFilledDeviceLocalGPUImageOnDedMem(
                fenceptr,queue,std::move(params),srcBuffer,regionCount,pRegions,
                waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore,
                signalSemaphoreCount,semaphoresToSignal
            );
            m_device->waitForFences(1u,&fenceptr,false,9999999999ull);
            return retval;
        }

        //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUCommandBuffer* cmdbuf, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions)
        {
            // This API check is temporary (or not?) since getFormatProperties is not
            // yet implemented on OpenGL
            if (srcImage->getAPIType() == EAT_VULKAN)
            {
                const auto* physicalDevice = srcImage->getOriginDevice()->getPhysicalDevice();
                const auto validateFormatFeature = [&params, physicalDevice](const auto format, const auto reqFormatFeature) -> bool
                {
                    const auto& formatProps = physicalDevice->getFormatProperties(params.format);

                    if ((params.tiling == asset::IImage::ET_OPTIMAL) && (formatProps.optimalTilingFeatures & reqFormatFeature).value == 0)
                        return false;
                    if ((params.tiling == asset::IImage::ET_LINEAR) && (formatProps.linearTilingFeatures & reqFormatFeature).value == 0)
                        return false;

                    return true;
                };

                if (!validateFormatFeature(srcImage->getCreationParameters().format, asset::EFF_TRANSFER_SRC_BIT))
                    return nullptr;

                if (!validateFormatFeature(params.format, asset::EFF_TRANSFER_DST_BIT))
                    return nullptr;
            }

            const auto finalLayout = params.initialLayout;

            if (!((params.usage & asset::IImage::EUF_TRANSFER_DST_BIT).value))
                params.usage |= asset::IImage::EUF_TRANSFER_DST_BIT;

            auto retval = m_device->createDeviceLocalGPUImageOnDedMem(std::move(params));

            assert(cmdbuf->getState()==IGPUCommandBuffer::ES_RECORDING);

            IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
            barrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.oldLayout = asset::EIL_UNDEFINED;
            barrier.newLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = ~0u;
            barrier.dstQueueFamilyIndex = ~0u;
            barrier.image = retval;
            barrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT; // need this from input, infact this family of functions would be more usable if we take in a SSubresourceRange to operate on
            barrier.subresourceRange.baseArrayLayer = 0u;
            barrier.subresourceRange.layerCount = retval->getCreationParameters().arrayLayers;
            barrier.subresourceRange.baseMipLevel = 0u;
            barrier.subresourceRange.levelCount = retval->getCreationParameters().mipLevels;
            cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);

            cmdbuf->copyImage(srcImage,asset::EIL_TRANSFER_SRC_OPTIMAL,retval.get(),asset::EIL_TRANSFER_DST_OPTIMAL,regionCount,pRegions);

            if (finalLayout != asset::EIL_TRANSFER_DST_OPTIMAL)
            {
                barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
                barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
                barrier.oldLayout = asset::EIL_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = finalLayout;
                cmdbuf->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
            }
            return retval;
        }
        //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
            IGPUFence* fence, IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions,
            const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            auto cmdpool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_TRANSIENT_BIT);
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
            m_device->createCommandBuffers(cmdpool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
            assert(cmdbuf);
            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
            auto retval = createFilledDeviceLocalGPUImageOnDedMem(cmdbuf.get(),std::move(params),srcImage,regionCount,pRegions);
            cmdbuf->end();
            IGPUQueue::SSubmitInfo submit;
            submit.commandBufferCount = 1u;
            submit.commandBuffers = &cmdbuf.get();
            assert(!signalSemaphoreCount || semaphoresToSignal);
            submit.signalSemaphoreCount = signalSemaphoreCount;
            submit.pSignalSemaphores = semaphoresToSignal;
            assert(!waitSemaphoreCount || semaphoresToWaitBeforeExecution&&stagesToWaitForPerSemaphore);
            submit.waitSemaphoreCount = waitSemaphoreCount;
            submit.pWaitSemaphores = semaphoresToWaitBeforeExecution;
            submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
            queue->submit(1u,&submit,fence);
            m_device->waitForFences(1u, &fence, false, 9999999999ull);
            return retval;
        }
        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
            IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions,
            const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            auto* fenceptr = fence.get();
            auto retval = createFilledDeviceLocalGPUImageOnDedMem(
                fenceptr,queue,std::move(params),srcImage,regionCount,pRegions,
                waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore,
                signalSemaphoreCount,semaphoresToSignal
            );
            m_device->waitForFences(1u,&fenceptr,false,9999999999ull);
            return retval;
        }
        
        // --------------
        // updateBufferRangeViaStagingBuffer
        // --------------
        
        //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
        // `cmdbuf` needs to be already begun and from a pool that allows for resetting commandbuffers individually
        // `fence` needs to be in unsignalled state
        // `queue` must have the transfer capability
        // `semaphoresToWaitBeforeOverwrite` and `stagesToWaitForPerSemaphore` are references which will be set to null (consumed) if we needed to perform a submit
        inline void updateBufferRangeViaStagingBuffer(
            IGPUCommandBuffer* cmdbuf, IGPUFence* fence, IGPUQueue* queue, const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data,
            uint32_t& waitSemaphoreCount, IGPUSemaphore*const * &semaphoresToWaitBeforeOverwrite, const asset::E_PIPELINE_STAGE_FLAGS* &stagesToWaitForPerSemaphore
        )
        {
            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            auto* cmdpool = cmdbuf->getPool();
            assert(cmdpool->getCreationFlags()&IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
            assert(cmdpool->getQueueFamilyIndex()==queue->getFamilyIndex());

            for (size_t uploadedSize=0ull; uploadedSize<bufferRange.size;)
            {
                const void* dataPtr = reinterpret_cast<const uint8_t*>(data)+uploadedSize;
                uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_address;
                const uint32_t alignment = static_cast<uint32_t>(limits.nonCoherentAtomSize);
                const uint32_t subSize = static_cast<uint32_t>(core::min<uint64_t>(core::alignDown(m_defaultUploadBuffer.get()->max_size(),alignment), bufferRange.size-uploadedSize));
                const uint32_t paddedSize = core::alignUp(subSize,alignment);
                // cannot use `multi_place` because of the extra padding size we could have added
                m_defaultUploadBuffer.get()->multi_alloc(std::chrono::high_resolution_clock::now()+std::chrono::microseconds(500u),1u,&localOffset,&paddedSize,&alignment);
                // copy only the unpadded part
                if (localOffset!=video::StreamingTransientDataBufferMT<>::invalid_address)
                    memcpy(reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+localOffset,dataPtr,subSize);

                // keep trying again
                if (localOffset == video::StreamingTransientDataBufferMT<>::invalid_address)
                {
                    // but first sumbit the already buffered up copies
                    cmdbuf->end();
                    IGPUQueue::SSubmitInfo submit;
                    submit.commandBufferCount = 1u;
                    submit.commandBuffers = &cmdbuf;
                    submit.signalSemaphoreCount = 0u;
                    submit.pSignalSemaphores = nullptr;
                    assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
                    submit.waitSemaphoreCount = waitSemaphoreCount;
                    submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
                    submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
                    queue->submit(1u,&submit,fence);
                    m_device->blockForFences(1u,&fence);
                    waitSemaphoreCount = 0u;
                    semaphoresToWaitBeforeOverwrite = nullptr;
                    stagesToWaitForPerSemaphore = nullptr;
                    // before resetting we need poll all events in the allocator's deferred free list
                    m_defaultUploadBuffer->cull_frees();
                    // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
                    m_device->resetFences(1u,&fence);
                    cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
                    cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                    continue;
                }
                // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
                if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate())
                {
                    IDriverMemoryAllocation::MappedMemoryRange flushRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(),localOffset,paddedSize);
                    m_device->flushMappedMemoryRanges(1u,&flushRange);
                }
                // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
                asset::SBufferCopy copy;
                copy.srcOffset = localOffset;
                copy.dstOffset = bufferRange.offset+uploadedSize;
                copy.size = subSize;
                cmdbuf->copyBuffer(m_defaultUploadBuffer.get()->getBuffer(),bufferRange.buffer.get(),1u,&copy);
                // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
                m_defaultUploadBuffer.get()->multi_free(1u,&localOffset,&paddedSize,core::smart_refctd_ptr<IGPUFence>(fence),&cmdbuf); // can queue with a reset but not yet pending fence, just fine
                uploadedSize += subSize;
            }
        }
        //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        // `fence` needs to be in unsignalled state
        inline void updateBufferRangeViaStagingBuffer(
            IGPUFence* fence, IGPUQueue* queue, const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data,
            uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            core::smart_refctd_ptr<IGPUCommandPool> pool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
            m_device->createCommandBuffers(pool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
            assert(cmdbuf);
            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
            updateBufferRangeViaStagingBuffer(cmdbuf.get(),fence,queue,bufferRange,data,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore);
            cmdbuf->end();
            IGPUQueue::SSubmitInfo submit;
            submit.commandBufferCount = 1u;
            submit.commandBuffers = &cmdbuf.get();
            submit.signalSemaphoreCount = signalSemaphoreCount;
            submit.pSignalSemaphores = semaphoresToSignal;
            assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
            submit.waitSemaphoreCount = waitSemaphoreCount;
            submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
            submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
            queue->submit(1u,&submit,fence);
        }
        //! WARNING: This function blocks and stalls the GPU!
        inline void updateBufferRangeViaStagingBuffer(
            IGPUQueue* queue, const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data,
            uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            updateBufferRangeViaStagingBuffer(fence.get(),queue,bufferRange,data,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,signalSemaphoreCount,semaphoresToSignal);
            auto* fenceptr = fence.get();
            m_device->blockForFences(1u,&fenceptr);
        }
        

        // --------------
        // updateImageViaStagingBuffer
        // --------------

        inline void updateImageViaStagingBuffer(
            IGPUCommandBuffer* cmdbuf, IGPUFence* fence, IGPUQueue* queue,
            asset::ICPUBuffer const* srcBuffer, const core::SRange<const asset::IImage::SBufferCopy>& regions, video::IGPUImage* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout,
            uint32_t& waitSemaphoreCount, IGPUSemaphore*const * &semaphoresToWaitBeforeOverwrite, const asset::E_PIPELINE_STAGE_FLAGS* &stagesToWaitForPerSemaphore)
        {
            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            auto* cmdpool = cmdbuf->getPool();
            assert(cmdpool->getCreationFlags()&IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
            assert(cmdpool->getQueueFamilyIndex()==queue->getFamilyIndex());
            
            auto texelBlockInfo = asset::TexelBlockInfo(dstImage->getCreationParameters().format);
            auto texelBlockDim = texelBlockInfo.getDimension();
            auto queueFamProps = m_device->getPhysicalDevice()->getQueueFamilyProperties()[0];
            auto minImageTransferGranularity = queueFamProps.minImageTransferGranularity;
            
            // Queues supporting graphics and/or compute operations must report (1,1,1) in minImageTransferGranularity, meaning that there are no additional restrictions on the granularity of image transfer operations for these queues.
            // Other queues supporting image transfer operations are only required to support whole mip level transfers, thus minImageTransferGranularity for queues belonging to such queue families may be (0,0,0)
            bool canTransferMipLevelsPartially = !(minImageTransferGranularity.width == 0 && minImageTransferGranularity.height == 0 && minImageTransferGranularity.depth == 0);

            // Block Offsets 
            // (1 blockInRow = texelBlockDimensions.x texels)
            // (1 rowInSlice = texelBlockDimensions.y texel rows)
            // (1 sliceInLayer = texelBlockDimensions.z texel depths)
            uint32_t currentBlockInRow = 0u;
            uint32_t currentRowInSlice = 0u;
            uint32_t currentSliceInLayer = 0u;
            uint32_t currentLayerInRegion = 0u;
            uint32_t currentRegion = 0u;
            
            // bufferOffsetAlignment: TODO
                // [ ] If Depth/Stencil -> must be multiple of 4
                // [ ] If multi-planar -> bufferOffset must be a multiple of the element size of the compatible format for the aspectMask of imagesubresource
                // [ ] If Queue doesn't support GRAPHICS_BIT and COMPUTE_BIT ->  must be multiple of 4
                // [x] bufferOffset must be a multiple of texel block size in bytes
            const uint32_t bufferOffsetAlignment = texelBlockInfo.getBlockByteSize();

            while (currentRegion < regions.size())
            {
                size_t memoryNeededForRemainingRegions = 0ull;
                for (uint32_t i = currentRegion; i < regions.size(); ++i)
                {
                    memoryNeededForRemainingRegions = core::alignUp(memoryNeededForRemainingRegions, bufferOffsetAlignment);
                    
                    const asset::IImage::SBufferCopy & region = regions[i];
                    
                    auto subresourceSize = dstImage->getMipSize(region.imageSubresource.mipLevel);

                    assert(static_cast<uint32_t>(region.imageSubresource.aspectMask) != 0u);
                    assert(core::isPoT(static_cast<uint32_t>(region.imageSubresource.aspectMask)) && "region.aspectMask should only have a single bit set.");
                    // Validate Region
                    // canTransferMipLevelsPartially
                    if(!canTransferMipLevelsPartially)
                    {
                        assert(region.imageOffset.x == 0 && region.imageOffset.y == 0 && region.imageOffset.z == 0);
                        assert(region.imageExtent.width == subresourceSize.x && region.imageExtent.height == subresourceSize.y && region.imageExtent.depth == subresourceSize.z);
                    }

                    // region.imageOffset.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size
                    bool isImageOffsetValid =
                        region.imageOffset.x == core::alignUp(region.imageOffset.x, minImageTransferGranularity.width  * texelBlockDim.x) &&
                        region.imageOffset.y == core::alignUp(region.imageOffset.y, minImageTransferGranularity.height  * texelBlockDim.y) &&
                        region.imageOffset.z == core::alignUp(region.imageOffset.z, minImageTransferGranularity.depth  * texelBlockDim.z);
                    assert(isImageOffsetValid);

                    // region.imageExtent.{xyz} should be multiple of minImageTransferGranularity.{xyz} scaled up by block size,
                    // OR ELSE (region.imageOffset.{x/y/z} + region.imageExtent.{width/height/depth}) MUST be equal to subresource{Width,Height,Depth}
                    bool isImageExtentValid = 
                        (region.imageExtent.width  == core::alignUp(region.imageExtent.width , minImageTransferGranularity.width  * texelBlockDim.x) || (region.imageOffset.x + region.imageExtent.width   == subresourceSize.x)) && 
                        (region.imageExtent.height == core::alignUp(region.imageExtent.height, minImageTransferGranularity.height * texelBlockDim.y) || (region.imageOffset.y + region.imageExtent.height  == subresourceSize.y)) &&
                        (region.imageExtent.depth  == core::alignUp(region.imageExtent.depth , minImageTransferGranularity.depth  * texelBlockDim.z) || (region.imageOffset.z + region.imageExtent.depth   == subresourceSize.z));
                    assert(isImageExtentValid);

                    auto alignedImageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth));
                    auto imageExtentBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(alignedImageExtentInBlocks);
                    if(i == currentRegion)
                    {
                        auto remainingBlocksInRow = alignedImageExtentInBlocks.x - currentBlockInRow;
                        auto remainingRowsInSlice = alignedImageExtentInBlocks.y - currentRowInSlice;
                        auto remainingSlicesInLayer = alignedImageExtentInBlocks.z - currentSliceInLayer;
                        auto remainingLayersInRegion = region.imageSubresource.layerCount - currentLayerInRegion;

                        if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && remainingLayersInRegion > 0)
                            memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * remainingLayersInRegion;
                        else if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer > 0)
                        {
                            memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[2] * remainingSlicesInLayer;
                            if(remainingLayersInRegion > 1u)
                                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u);
                        }
                        else if (currentBlockInRow == 0 && currentRowInSlice > 0)
                        {
                            memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[1] * remainingRowsInSlice;
                            if(remainingSlicesInLayer > 1u)
                                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[2] * (remainingSlicesInLayer - 1u);
                            if(remainingLayersInRegion > 1u)
                                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u);
                        }
                        else if (currentBlockInRow > 0)
                        {
                            memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[0] * remainingBlocksInRow;
                            if(remainingRowsInSlice > 1u)
                                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[1] * (remainingRowsInSlice - 1u);
                            if(remainingSlicesInLayer > 1u)
                                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[2] * (remainingSlicesInLayer - 1u);
                            if(remainingLayersInRegion > 1u)
                                memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * (remainingLayersInRegion - 1u);
                        }
                    }
                    else
                    {
                        memoryNeededForRemainingRegions += imageExtentBlockStridesInBytes[3] * region.imageSubresource.layerCount; // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z * region.imageSubresource.layerCount
                    }
                }

                uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_address;
                const uint32_t alignment = static_cast<uint32_t>(limits.nonCoherentAtomSize);
                const uint32_t subSize = static_cast<uint32_t>(core::min<uint64_t>(core::alignDown(m_defaultUploadBuffer.get()->max_size(), alignment), memoryNeededForRemainingRegions));
                const uint32_t uploadBufferSize = core::alignUp(subSize, alignment);
                // cannot use `multi_place` because of the extra padding size we could have added
                m_defaultUploadBuffer.get()->multi_alloc(std::chrono::high_resolution_clock::now()+std::chrono::microseconds(500u), 1u, &localOffset, &uploadBufferSize, &alignment);

                // keep trying again
                if (localOffset == video::StreamingTransientDataBufferMT<>::invalid_address)
                {
                    // but first sumbit the already buffered up copies
                    cmdbuf->end();
                    IGPUQueue::SSubmitInfo submit;
                    submit.commandBufferCount = 1u;
                    submit.commandBuffers = &cmdbuf;
                    submit.signalSemaphoreCount = 0u;
                    submit.pSignalSemaphores = nullptr;
                    assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
                    submit.waitSemaphoreCount = waitSemaphoreCount;
                    submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
                    submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
                    queue->submit(1u, &submit, fence);
                    m_device->blockForFences(1u, &fence);
                    waitSemaphoreCount = 0u;
                    semaphoresToWaitBeforeOverwrite = nullptr;
                    stagesToWaitForPerSemaphore = nullptr;
                    // before resetting we need poll all events in the allocator's deferred free list
                    m_defaultUploadBuffer->cull_frees();
                    // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
                    m_device->resetFences(1u, &fence);
                    cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
                    cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                    continue;
                }
                else
                {
                    // Start CmdCopying Regions and Copying Data to m_defaultUploadBuffer
                    uint32_t currentUploadBufferOffset = 0u;
                    uint32_t availableUploadBufferMemory = 0u;
                    auto addToCurrentUploadBufferOffset = [&](uint32_t size) -> bool 
                    {
                        currentUploadBufferOffset += size;
                        currentUploadBufferOffset = core::alignUp(currentUploadBufferOffset, bufferOffsetAlignment);
                        if(currentUploadBufferOffset - localOffset <= uploadBufferSize)
                        {
                            availableUploadBufferMemory = uploadBufferSize - (currentUploadBufferOffset - localOffset);
                            return true;
                        }
                        else
                            return false;
                    };

                    // currentUploadBufferOffset = localOffset
                    // currentUploadBufferOffset = alignUp(currentUploadBufferOffset, bufferOffsetAlignment)
                    // availableUploadBufferMemory = uploadBufferSize
                    addToCurrentUploadBufferOffset(localOffset);

                    bool anyTransferRecorded = false;
                    core::vector<asset::IImage::SBufferCopy> regionsToCopy;

                    // Worst case iterations: remaining blocks --> remaining rows --> remaining slices --> full layers
                    constexpr uint32_t maxIterations = regions.size() * 4u; 
                    for (uint32_t d = 0u; d < maxIterations && currentRegion < regions.size(); ++d)
                    {
                        const asset::IImage::SBufferCopy & region = regions[currentRegion];

                        auto subresourceSize = dstImage->getMipSize(region.imageSubresource.mipLevel);
                        auto subresourceSizeInBlocks = texelBlockInfo.convertTexelsToBlocks(subresourceSize);

                        // regionBlockStrides = <BufferRowLengthInBlocks, BufferImageHeightInBlocks, ImageDepthInBlocks>
                        auto regionBlockStrides = region.getBlockStrides(texelBlockInfo);
                        // regionBlockStridesInBytes = <BlockByteSize,
                        //                              BlockBytesSize * BufferRowLengthInBlocks,
                        //                              BlockBytesSize * BufferRowLengthInBlocks * BufferImageHeightInBlocks,
                        //                              BlockBytesSize * BufferRowLengthInBlocks * BufferImageHeightInBlocks * ImageDepthInBlocks>
                        auto regionBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(regionBlockStrides);

                        auto imageExtent = core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth);
                        auto imageOffset = core::vector3du32_SIMD(region.imageOffset.x, region.imageOffset.y, region.imageOffset.z);
                        auto imageOffsetInBlocks = texelBlockInfo.convertTexelsToBlocks(imageOffset);
                        auto imageExtentInBlocks = texelBlockInfo.convertTexelsToBlocks(imageExtent);
                        auto alignedImageExtentBlockStridesInBytes = texelBlockInfo.convert3DBlockStridesTo1DByteStrides(imageExtentInBlocks);

                        // region <-> region.imageSubresource.layerCount <-> imageExtentInBlocks.z <-> imageExtentInBlocks.y <-> imageExtentInBlocks.x
                        auto updateCurrentOffsets = [&]() -> void
                        {
                            if(currentBlockInRow >= imageExtentInBlocks.x) 
                            {
                                currentBlockInRow = 0u;
                                currentRowInSlice++;
                            }
                            if(currentRowInSlice >= imageExtentInBlocks.y)
                            {
                                assert(currentBlockInRow == 0);
                                currentRowInSlice = 0u;
                                currentSliceInLayer++;
                            }
                            if(currentSliceInLayer >= imageExtentInBlocks.z)
                            {
                                assert(currentBlockInRow == 0 && currentRowInSlice == 0);
                                currentSliceInLayer = 0u;
                                currentLayerInRegion++;
                            }
                            if(currentLayerInRegion >= region.imageSubresource.layerCount) 
                            {
                                assert(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0);
                                currentLayerInRegion = 0u; 
                                currentRegion++;
                            }
                        };
                        
                        uint32_t eachBlockNeededMemory = alignedImageExtentBlockStridesInBytes[0];  // = blockByteSize
                        uint32_t eachRowNeededMemory = alignedImageExtentBlockStridesInBytes[1];    // = blockByteSize * imageExtentInBlocks.x
                        uint32_t eachSliceNeededMemory = alignedImageExtentBlockStridesInBytes[2];  // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y
                        uint32_t eachLayerNeededMemory = alignedImageExtentBlockStridesInBytes[3];  // = blockByteSize * imageExtentInBlocks.x * imageExtentInBlocks.y * imageExtentInBlocks.z

                        auto tryFillRow = [&]() -> bool
                        {
                            bool ret = false;
                            // C: There is remaining slices left in layer -> Copy Blocks
                            uint32_t uploadableBlocks = availableUploadBufferMemory / eachBlockNeededMemory;
                            uint32_t remainingBlocks = imageExtentInBlocks.x - currentBlockInRow;
                            uploadableBlocks = core::min(uploadableBlocks, remainingBlocks);
                            if(uploadableBlocks + currentBlockInRow < subresourceSizeInBlocks.x)
                                uploadableBlocks = core::alignDown(uploadableBlocks, minImageTransferGranularity.width);

                            if(uploadableBlocks > 0)
                            {
                                uint32_t blocksToUploadMemorySize = eachBlockNeededMemory * uploadableBlocks;
                                auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(imageOffsetInBlocks.x + currentBlockInRow, imageOffsetInBlocks.y + currentRowInSlice, imageOffsetInBlocks.z + currentSliceInLayer, currentLayerInRegion);
                                uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                                memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                        reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                        blocksToUploadMemorySize);

                                asset::IImage::SBufferCopy bufferCopy;
                                bufferCopy.bufferOffset = currentUploadBufferOffset;
                                bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                                bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                                bufferCopy.imageSubresource.mipLevel = region.imageSubresource.baseArrayLayer;
                                bufferCopy.imageSubresource.baseArrayLayer += currentLayerInRegion;
                                bufferCopy.imageOffset.x = currentBlockInRow * texelBlockDim.x;
                                bufferCopy.imageOffset.y = currentRowInSlice * texelBlockDim.y;
                                bufferCopy.imageOffset.z = currentSliceInLayer * texelBlockDim.z;
                                bufferCopy.imageExtent.width = uploadableBlocks * texelBlockDim.x;
                                bufferCopy.imageExtent.height = 1u * texelBlockDim.y;
                                bufferCopy.imageExtent.depth = 1u * texelBlockDim.z;
                                bufferCopy.imageSubresource.layerCount = 1u;
                                regionsToCopy.push_back(bufferCopy);

                                addToCurrentUploadBufferOffset(blocksToUploadMemorySize);

                                currentBlockInRow += uploadableBlocks;
                                ret = true;
                            }

                            updateCurrentOffsets();

                            return ret;
                        };
                        
                        auto tryFillSlice = [&]() -> bool
                        {
                            bool ret = false;
                            // B: There is remaining slices left in layer -> Copy Rows
                            uint32_t uploadableRows = availableUploadBufferMemory / eachRowNeededMemory;
                            uint32_t remainingRows = imageExtentInBlocks.y - currentRowInSlice;
                            uploadableRows = core::min(uploadableRows, remainingRows);
                            if(uploadableRows + currentRowInSlice < subresourceSizeInBlocks.y)
                                uploadableRows = core::alignDown(uploadableRows, minImageTransferGranularity.height);

                            if(uploadableRows > 0)
                            {
                                uint32_t rowsToUploadMemorySize = eachRowNeededMemory * uploadableRows;
                                
                                if(regionBlockStrides.x != imageExtentInBlocks.x)
                                {
                                    // Can't copy all rows at once, there is padding, copy row by row
                                    for(uint32_t y = 0; y < uploadableRows; ++y)
                                    {
                                        auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(imageOffsetInBlocks.x, imageOffsetInBlocks.y + currentRowInSlice + y, imageOffsetInBlocks.z + currentSliceInLayer, currentLayerInRegion);
                                        uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                        uint64_t offsetInUploadBuffer = currentUploadBufferOffset + y*eachRowNeededMemory;
                                        memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                                reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                                eachRowNeededMemory);
                                    }
                                }
                                else
                                {
                                    // We can copy all rows at once, because imageExtent is fit to rowLength
                                    assert(imageOffsetInBlocks.x == 0);
                                    auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(0u, imageOffsetInBlocks.y + currentRowInSlice, imageOffsetInBlocks.z + currentSliceInLayer, currentLayerInRegion);
                                    uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                    uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                                    memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                            reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                            rowsToUploadMemorySize);
                                }


                                asset::IImage::SBufferCopy bufferCopy;
                                bufferCopy.bufferOffset = currentUploadBufferOffset;
                                bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                                bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                                bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                                bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                                bufferCopy.imageOffset.x = 0u; assert(currentBlockInRow == 0);
                                bufferCopy.imageOffset.y = currentRowInSlice * texelBlockDim.y;
                                bufferCopy.imageOffset.z = currentSliceInLayer * texelBlockDim.z;
                                bufferCopy.imageExtent.width = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.imageExtent.height = uploadableRows * texelBlockDim.y;
                                bufferCopy.imageExtent.depth = 1u * texelBlockDim.z;
                                bufferCopy.imageSubresource.layerCount = 1u;
                                regionsToCopy.push_back(bufferCopy);

                                addToCurrentUploadBufferOffset(rowsToUploadMemorySize);

                                currentRowInSlice += uploadableRows;
                                ret = true;
                            }
                            
                            if(currentRowInSlice < imageExtentInBlocks.z)
                            {
                                bool filledAnyBlocksInRow = tryFillRow();
                                if(filledAnyBlocksInRow)
                                    ret = true;
                            }
                            
                            updateCurrentOffsets();

                            return ret;
                        };
                        
                        auto tryFillLayer = [&]() -> bool
                        {
                            bool ret = false;
                            // A: There is remaining layers left in region -> Copy Slices (Depths)
                            uint32_t uploadableSlices = availableUploadBufferMemory / eachSliceNeededMemory;
                            uint32_t remainingSlices = imageExtentInBlocks.z - currentSliceInLayer;
                            uploadableSlices = core::min(uploadableSlices, remainingSlices);
                            if(uploadableSlices + currentSliceInLayer < subresourceSizeInBlocks.z)
                                uploadableSlices = core::alignDown(uploadableSlices, minImageTransferGranularity.depth);

                            if(uploadableSlices > 0)
                            {
                                uint32_t slicesToUploadMemorySize = eachSliceNeededMemory * uploadableSlices;

                                if(regionBlockStrides.x != imageExtentInBlocks.x)
                                {
                                    // Can't copy all rows at once, there is more padding at the end of rows, copy row by row:
                                    for(uint32_t z = 0; z < uploadableSlices; ++z)
                                    {
                                        for(uint32_t y = 0; y < imageExtentInBlocks.y; ++y)
                                        {
                                            auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(imageOffsetInBlocks.x, imageOffsetInBlocks.y + y, imageOffsetInBlocks.z + currentSliceInLayer + z, currentLayerInRegion);
                                            uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                            uint64_t offsetInUploadBuffer = currentUploadBufferOffset + z * eachSliceNeededMemory + y * eachRowNeededMemory;
                                            memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                                    reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                                    eachRowNeededMemory);
                                        }
                                    }
                                }
                                else if (regionBlockStrides.y != imageExtentInBlocks.y)
                                {
                                    assert(imageOffsetInBlocks.x == 0u);
                                    // Can't copy all slices at once, there is more padding at the end of slices, copy slice by slice
                                    for(uint32_t z = 0; z < uploadableSlices; ++z)
                                    {
                                        auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(0u, imageOffsetInBlocks.y, imageOffsetInBlocks.z + currentSliceInLayer + z, currentLayerInRegion);
                                        uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                        uint64_t offsetInUploadBuffer = currentUploadBufferOffset + z * eachSliceNeededMemory;
                                        memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                                reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                                eachSliceNeededMemory);
                                    }
                                }
                                else
                                {
                                    // We can copy all arrays and slices at once, because imageExtent is fit to bufferRowLength and bufferImageHeight
                                    assert(imageOffsetInBlocks.x == 0u);
                                    assert(imageOffsetInBlocks.y == 0u);
                                    auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(0u, 0u, imageOffsetInBlocks.z + currentSliceInLayer, currentLayerInRegion);
                                    uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                    uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                                    memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                            reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                            slicesToUploadMemorySize);
                                }
                                
                                asset::IImage::SBufferCopy bufferCopy;
                                bufferCopy.bufferOffset = currentUploadBufferOffset;
                                bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                                bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                                bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                                bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                                bufferCopy.imageOffset.x = 0u; assert(currentBlockInRow == 0);
                                bufferCopy.imageOffset.y = 0u; assert(currentRowInSlice == 0);
                                bufferCopy.imageOffset.z = currentSliceInLayer * texelBlockDim.z;
                                bufferCopy.imageExtent.width = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.imageExtent.height = imageExtentInBlocks.y * texelBlockDim.y;
                                bufferCopy.imageExtent.depth = uploadableSlices * texelBlockDim.z;
                                bufferCopy.imageSubresource.layerCount = 1u;
                                regionsToCopy.push_back(bufferCopy);

                                addToCurrentUploadBufferOffset(slicesToUploadMemorySize);

                                currentSliceInLayer += uploadableSlices;
                                ret = true;
                            }
                            
                            if(currentSliceInLayer < imageExtentInBlocks.z)
                            {
                                bool filledAnyRowsOrBlocksInSlice = tryFillSlice();
                                if(filledAnyRowsOrBlocksInSlice)
                                    ret = true;
                            }
                            
                            updateCurrentOffsets();

                            return ret;
                        };

                        auto tryFillRegion = [&]() -> bool
                        {
                            bool ret = false;
                            uint32_t uploadableArrayLayers = availableUploadBufferMemory / eachLayerNeededMemory;
                            uint32_t remainingLayers = region.imageSubresource.layerCount - currentLayerInRegion;
                            uploadableArrayLayers = core::min(uploadableArrayLayers, remainingLayers);

                            if(uploadableArrayLayers > 0)
                            {
                                uint32_t layersToUploadMemorySize = eachLayerNeededMemory * uploadableArrayLayers;

                                if(regionBlockStrides.x != imageExtentInBlocks.x)
                                {
                                    // Can't copy all rows at once, there is more padding at the end of rows, copy row by row:
                                    for(uint32_t layer = 0; layer < uploadableArrayLayers; ++layer)
                                    {
                                        for(uint32_t z = 0; z < imageExtentInBlocks.z; ++z)
                                        {
                                            for(uint32_t y = 0; y < imageExtentInBlocks.y; ++y)
                                            {
                                                auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(imageOffsetInBlocks.x, imageOffsetInBlocks.y + y, imageOffsetInBlocks.z + z, currentLayerInRegion + layer);
                                                uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                                uint64_t offsetInUploadBuffer = currentUploadBufferOffset + layer * eachLayerNeededMemory + z * eachSliceNeededMemory + y * eachRowNeededMemory;
                                                memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                                        reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                                        eachRowNeededMemory);
                                            }
                                        }
                                    }
                                }
                                else if (regionBlockStrides.y != imageExtentInBlocks.y)
                                {
                                    assert(imageOffsetInBlocks.x == 0u);
                                    // Can't copy all slices at once, there is more padding at the end of slices, copy slice by slice
                                    
                                    for(uint32_t layer = 0; layer < uploadableArrayLayers; ++layer)
                                    {
                                        for(uint32_t z = 0; z < imageExtentInBlocks.z; ++z)
                                        {
                                            auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(0u, imageOffsetInBlocks.y, imageOffsetInBlocks.z + z, currentLayerInRegion + layer);
                                            uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                            uint64_t offsetInUploadBuffer = currentUploadBufferOffset + layer * eachLayerNeededMemory + z * eachSliceNeededMemory;
                                            memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                                    reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                                    eachSliceNeededMemory);
                                        }
                                    }
                                }
                                else
                                {
                                    // We can copy all arrays and slices at once, because imageExtent is fit to bufferRowLength and bufferImageHeight
                                    assert(imageOffsetInBlocks.x == 0u);
                                    assert(imageOffsetInBlocks.y == 0u);
                                    auto imageOffsetInCPUBuffer = core::vector3du32_SIMD(0u, 0u, imageOffsetInBlocks.z, currentLayerInRegion);
                                    uint64_t offsetInCPUBuffer = region.bufferOffset + core::dot(imageOffsetInCPUBuffer, regionBlockStridesInBytes)[0];
                                    uint64_t offsetInUploadBuffer = currentUploadBufferOffset;
                                    memcpy( reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer())+offsetInUploadBuffer,
                                            reinterpret_cast<uint8_t const*>(srcBuffer->getPointer())+offsetInCPUBuffer,
                                            layersToUploadMemorySize);
                                }
                                
                                asset::IImage::SBufferCopy bufferCopy;
                                bufferCopy.bufferOffset = currentUploadBufferOffset;
                                bufferCopy.bufferRowLength = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.bufferImageHeight = imageExtentInBlocks.y * texelBlockDim.y;
                                bufferCopy.imageSubresource.aspectMask = region.imageSubresource.aspectMask;
                                bufferCopy.imageSubresource.mipLevel = region.imageSubresource.mipLevel;
                                bufferCopy.imageSubresource.baseArrayLayer = region.imageSubresource.baseArrayLayer + currentLayerInRegion;
                                bufferCopy.imageOffset.x = 0u; assert(currentBlockInRow == 0);
                                bufferCopy.imageOffset.y = 0u; assert(currentRowInSlice == 0);
                                bufferCopy.imageOffset.z = 0u; assert(currentSliceInLayer == 0);
                                bufferCopy.imageExtent.width = imageExtentInBlocks.x * texelBlockDim.x;
                                bufferCopy.imageExtent.height = imageExtentInBlocks.y * texelBlockDim.y;
                                bufferCopy.imageExtent.depth = imageExtentInBlocks.z * texelBlockDim.z;
                                bufferCopy.imageSubresource.layerCount = uploadableArrayLayers;
                                regionsToCopy.push_back(bufferCopy);

                                addToCurrentUploadBufferOffset(layersToUploadMemorySize);

                                currentLayerInRegion += uploadableArrayLayers;
                                ret = true;
                            }

                            // currentLayerInRegion is respective to region.imageSubresource.baseArrayLayer so It's not in the calculations until the cmdCopy.
                            if(currentLayerInRegion < region.imageSubresource.layerCount && canTransferMipLevelsPartially)
                            {
                                bool filledAnySlicesOrRowsOrBlocksInLayer = tryFillLayer();
                                if(filledAnySlicesOrRowsOrBlocksInLayer)
                                    ret = true;
                            }
                            
                            updateCurrentOffsets();

                            return ret;
                        };

                         // There is remaining layers in region that needs copying
                        auto remainingLayersInRegion = region.imageSubresource.layerCount - currentLayerInRegion;
                        if(currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer == 0 && remainingLayersInRegion > 0)
                        {
                            bool success = tryFillRegion();
                            if(success)
                                anyTransferRecorded = true;
                            else
                                break; // not enough mem -> break
                        }
                        // There is remaining slices in layer that needs copying
                        else if (currentBlockInRow == 0 && currentRowInSlice == 0 && currentSliceInLayer > 0)
                        {
                            assert(canTransferMipLevelsPartially);
                            bool success = tryFillLayer();
                            if(success)
                                anyTransferRecorded = true;
                            else
                                break; // not enough mem -> break
                        }
                        // There is remaining rows in slice that needs copying
                        else if (currentBlockInRow == 0 && currentRowInSlice > 0)
                        {
                            assert(canTransferMipLevelsPartially);
                            bool success = tryFillSlice();
                            if(success)
                                anyTransferRecorded = true;
                            else
                                break; // not enough mem -> break
                        }
                        // There is remaining blocks in row that needs copying
                        else if (currentBlockInRow > 0)
                        {
                            assert(canTransferMipLevelsPartially);
                            bool success = tryFillRow();
                            if(success)
                                anyTransferRecorded = true;
                            else
                                break; // not enough mem -> break
                        }
                    }

                    if(!regionsToCopy.empty())
                    {
                        cmdbuf->copyBufferToImage(m_defaultUploadBuffer.get()->getBuffer(), dstImage, dstImageLayout, regionsToCopy.size(), regionsToCopy.data());
                    }

                    assert(anyTransferRecorded && "uploadBufferSize is not enough to support the smallest possible transferable units to image, may be caused if your queueFam's minImageTransferGranularity is large or equal to <0,0,0>.");
                }

                // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
                if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate()) {
                    IDriverMemoryAllocation::MappedMemoryRange flushRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(), localOffset, uploadBufferSize);
                    m_device->flushMappedMemoryRanges(1u, &flushRange);
                }

                // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
                m_defaultUploadBuffer.get()->multi_free(1u, &localOffset, &uploadBufferSize, core::smart_refctd_ptr<IGPUFence>(fence), &cmdbuf); // can queue with a reset but not yet pending fence, just fine
            }
        }

        inline void updateImageViaStagingBuffer(
            IGPUFence* fence, IGPUQueue* queue,
            asset::ICPUBuffer const* srcBuffer, const core::SRange<const asset::IImage::SBufferCopy>& regions, video::IGPUImage* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout,
            uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            core::smart_refctd_ptr<IGPUCommandPool> pool = m_device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
            m_device->createCommandBuffers(pool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
            assert(cmdbuf);
            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
            updateImageViaStagingBuffer(cmdbuf.get(),fence,queue,srcBuffer,regions,dstImage,dstImageLayout,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore);
            cmdbuf->end();
            IGPUQueue::SSubmitInfo submit;
            submit.commandBufferCount = 1u;
            submit.commandBuffers = &cmdbuf.get();
            submit.signalSemaphoreCount = signalSemaphoreCount;
            submit.pSignalSemaphores = semaphoresToSignal;
            assert(!waitSemaphoreCount || semaphoresToWaitBeforeOverwrite && stagesToWaitForPerSemaphore);
            submit.waitSemaphoreCount = waitSemaphoreCount;
            submit.pWaitSemaphores = semaphoresToWaitBeforeOverwrite;
            submit.pWaitDstStageMask = stagesToWaitForPerSemaphore;
            queue->submit(1u,&submit,fence);
        }
        //! WARNING: This function blocks and stalls the GPU!
        inline void updateImageViaStagingBuffer(
            IGPUQueue* queue,
            asset::ICPUBuffer const* srcBuffer, const core::SRange<const asset::IImage::SBufferCopy>& regions, video::IGPUImage* dstImage, asset::E_IMAGE_LAYOUT dstImageLayout,
            uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeOverwrite=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
            const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
        )
        {
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            updateImageViaStagingBuffer(fence.get(),queue,srcBuffer,regions,dstImage,dstImageLayout,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,signalSemaphoreCount,semaphoresToSignal);
            auto* fenceptr = fence.get();
            m_device->blockForFences(1u,&fenceptr);
        }

    protected:
        core::smart_refctd_ptr<ILogicalDevice> m_device;

        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultDownloadBuffer;
        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultUploadBuffer;

        core::smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
        core::smart_refctd_ptr<CScanner> m_scanner;
};

}


#endif