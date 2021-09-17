#ifndef __NBL_VIDEO_I_UTILITIES_H_INCLUDED__
#define __NBL_VIDEO_I_UTILITIES_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/utilities/CPropertyPoolHandler.h"

namespace nbl::video
{

class IUtilities : public core::IReferenceCounted
{
    public:
        IUtilities(core::smart_refctd_ptr<ILogicalDevice>&& _device, size_t downstreamSize=0x4000000ull, size_t upstreamSize=0x4000000ull) : m_device(std::move(_device))
        {
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
            // Todo(achal): 
            // dstImage's format should support VK_FORMAT_FEATURE_TRANSFER_DST_BIT

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
            // Todo(achal): srcImage's format should support VK_FORMAT_FEATURE_TRANSFER_SRC_BIT,
            // dstImage's format should support VK_FORMAT_FEATURE_TRANSFER_DST_BIT

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
            auto* cmdpool = cmdbuf->getPool();
            assert(cmdpool->getCreationFlags()&IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
            assert(cmdpool->getQueueFamilyIndex()==queue->getFamilyIndex());

            for (size_t uploadedSize=0ull; uploadedSize<bufferRange.size;)
            {
                const void* dataPtr = reinterpret_cast<const uint8_t*>(data)+uploadedSize;
                uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_address;
                uint32_t alignment = 64u; // smallest mapping alignment capability
                uint32_t subSize = static_cast<uint32_t>(core::min<uint32_t>(core::alignDown(m_defaultUploadBuffer.get()->max_size(), alignment), bufferRange.size-uploadedSize));
                m_defaultUploadBuffer.get()->multi_place(std::chrono::high_resolution_clock::now()+std::chrono::microseconds(500u), 1u, (const void* const*)&dataPtr, &localOffset, &subSize, &alignment);

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
                    m_device->waitForFences(1u,&fence,false,9999999999ull);
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
                    IDriverMemoryAllocation::MappedMemoryRange flushRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(),localOffset,subSize);
                    m_device->flushMappedMemoryRanges(1u,&flushRange);
                }
                // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
                asset::SBufferCopy copy;
                copy.srcOffset = localOffset;
                copy.dstOffset = bufferRange.offset+uploadedSize;
                copy.size = subSize;
                cmdbuf->copyBuffer(m_defaultUploadBuffer.get()->getBuffer(),bufferRange.buffer.get(),1u,&copy);
                // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
                m_defaultUploadBuffer.get()->multi_free(1u,&localOffset,&subSize,core::smart_refctd_ptr<IGPUFence>(fence),&cmdbuf); // can queue with a reset but not yet pending fence, just fine
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
            m_device->waitForFences(1u,&fenceptr,false,9999999999ull);
        }

    protected:
        core::smart_refctd_ptr<ILogicalDevice> m_device;

        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultDownloadBuffer;
        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultUploadBuffer;

        core::smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
};

}


#endif