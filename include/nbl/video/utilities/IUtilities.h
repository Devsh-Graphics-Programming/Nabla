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
#include "nbl/video/utilities/CComputeBlit.h"

namespace nbl::video
{

class NBL_API2 IUtilities : public core::IReferenceCounted
{
    protected:
        constexpr static inline uint32_t maxStreamingBufferAllocationAlignment = 64u*1024u; // if you need larger alignments then you're not right in the head
        constexpr static inline uint32_t minStreamingBufferAllocationSize = 1024u;

        uint32_t m_allocationAlignment = 0u;
        uint32_t m_allocationAlignmentForBufferImageCopy = 0u;

        nbl::system::logger_opt_smart_ptr m_logger;

    public:
        IUtilities(core::smart_refctd_ptr<ILogicalDevice>&& device, nbl::system::logger_opt_smart_ptr&& logger = nullptr, const uint32_t downstreamSize = 0x4000000u, const uint32_t upstreamSize = 0x4000000u)
            : m_device(std::move(device))
            , m_logger(std::move(logger))
        {
            auto physicalDevice = m_device->getPhysicalDevice();
            const auto& limits = physicalDevice->getLimits();

            auto queueFamProps = physicalDevice->getQueueFamilyProperties();
            uint32_t minImageTransferGranularityVolume = 1u; // minImageTransferGranularity.width * height * depth

            for (uint32_t i = 0; i < queueFamProps.size(); i++)
            {
                uint32_t volume = queueFamProps[i].minImageTransferGranularity.width * queueFamProps[i].minImageTransferGranularity.height * queueFamProps[i].minImageTransferGranularity.depth;
                if(minImageTransferGranularityVolume < volume)
                    minImageTransferGranularityVolume = volume;
            }

            // host-mapped device memory needs to have this alignment in flush/invalidate calls, therefore this is the streaming buffer's "allocationAlignment".
            m_allocationAlignment = static_cast<uint32_t>(limits.nonCoherentAtomSize);
            m_allocationAlignmentForBufferImageCopy = core::max(static_cast<uint32_t>(limits.optimalBufferCopyOffsetAlignment), m_allocationAlignment);

            const uint32_t bufferOptimalTransferAtom = limits.maxResidentInvocations*sizeof(uint32_t);
            const uint32_t maxImageOptimalTransferAtom = limits.maxResidentInvocations * asset::TexelBlockInfo(asset::EF_R64G64B64A64_SFLOAT).getBlockByteSize() * minImageTransferGranularityVolume;
            const uint32_t minImageOptimalTransferAtom = limits.maxResidentInvocations * asset::TexelBlockInfo(asset::EF_R8_UINT).getBlockByteSize();;
            const uint32_t maxOptimalTransferAtom = core::max(bufferOptimalTransferAtom, maxImageOptimalTransferAtom);
            const uint32_t minOptimalTransferAtom = core::min(bufferOptimalTransferAtom, minImageOptimalTransferAtom);

            // allocationAlignment <= minBlockSize <= minOptimalTransferAtom <= maxOptimalTransferAtom <= stagingBufferSize/4
            assert(m_allocationAlignment <= minStreamingBufferAllocationSize);
            assert(m_allocationAlignmentForBufferImageCopy <= minStreamingBufferAllocationSize);

            assert(minStreamingBufferAllocationSize <= minOptimalTransferAtom);

            assert(maxOptimalTransferAtom * 4u <= upstreamSize);
            assert(maxOptimalTransferAtom * 4u <= downstreamSize);

            assert(minStreamingBufferAllocationSize % m_allocationAlignment == 0u);
            assert(minStreamingBufferAllocationSize % m_allocationAlignmentForBufferImageCopy == 0u);

            IGPUBuffer::SCreationParams streamingBufferCreationParams = {};
            bool shaderDeviceAddressSupport = false; //TODO(Erfan)
            auto commonUsages = core::bitflag(IGPUBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT)|IGPUBuffer::EUF_STORAGE_BUFFER_BIT|IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
            if(shaderDeviceAddressSupport)
                commonUsages |= IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
            
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags(IDeviceMemoryAllocation::EMAF_NONE);
            if(shaderDeviceAddressSupport)
                allocateFlags |= IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT;

            {
                IGPUBuffer::SCreationParams streamingBufferCreationParams = {};
                streamingBufferCreationParams.size = downstreamSize;
                streamingBufferCreationParams.usage = commonUsages|IGPUBuffer::EUF_TRANSFER_DST_BIT|IGPUBuffer::EUF_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT|IGPUBuffer::EUF_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT|IGPUBuffer::EUF_CONDITIONAL_RENDERING_BIT_EXT; // GPU write to RAM usages
                auto buffer = m_device->createBuffer(std::move(streamingBufferCreationParams));
                auto reqs = buffer->getMemoryReqs();
                reqs.memoryTypeBits &= physicalDevice->getDownStreamingMemoryTypeBits();

                auto memOffset = m_device->allocate(reqs, buffer.get(), allocateFlags);
                auto mem = memOffset.memory;

                core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access(IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
                const auto memProps = mem->getMemoryPropertyFlags();
                if (memProps.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
                    access |= IDeviceMemoryAllocation::EMCAF_READ;
                if (memProps.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
                    access |= IDeviceMemoryAllocation::EMCAF_WRITE;
                assert(access.value);
                IDeviceMemoryAllocation::MappedMemoryRange memoryRange = {mem.get(),0ull,mem->getAllocationSize()};
                m_device->mapMemory(memoryRange, access);

                m_defaultDownloadBuffer = core::make_smart_refctd_ptr<StreamingTransientDataBufferMT<>>(asset::SBufferRange<video::IGPUBuffer>{0ull,downstreamSize,std::move(buffer)},maxStreamingBufferAllocationAlignment,minStreamingBufferAllocationSize);
            }
            {
                IGPUBuffer::SCreationParams streamingBufferCreationParams = {};
                streamingBufferCreationParams.size = upstreamSize;
                streamingBufferCreationParams.usage = commonUsages|IGPUBuffer::EUF_TRANSFER_SRC_BIT|IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT|IGPUBuffer::EUF_UNIFORM_BUFFER_BIT|IGPUBuffer::EUF_INDEX_BUFFER_BIT|IGPUBuffer::EUF_VERTEX_BUFFER_BIT|IGPUBuffer::EUF_INDIRECT_BUFFER_BIT|IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT|IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT;
                auto buffer = m_device->createBuffer(std::move(streamingBufferCreationParams));

                auto reqs = buffer->getMemoryReqs();
                reqs.memoryTypeBits &= physicalDevice->getUpStreamingMemoryTypeBits();
                auto memOffset = m_device->allocate(reqs, buffer.get(), allocateFlags);

                auto mem = memOffset.memory;
                core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access(IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
                const auto memProps = mem->getMemoryPropertyFlags();
                if (memProps.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
                    access |= IDeviceMemoryAllocation::EMCAF_READ;
                if (memProps.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
                    access |= IDeviceMemoryAllocation::EMCAF_WRITE;
                assert(access.value);
                IDeviceMemoryAllocation::MappedMemoryRange memoryRange = {mem.get(),0ull,mem->getAllocationSize()};
                m_device->mapMemory(memoryRange, access);

                m_defaultUploadBuffer = core::make_smart_refctd_ptr<StreamingTransientDataBufferMT<>>(asset::SBufferRange<video::IGPUBuffer>{0ull,upstreamSize,std::move(buffer)},maxStreamingBufferAllocationAlignment,minStreamingBufferAllocationSize);
            }
            m_propertyPoolHandler = core::make_smart_refctd_ptr<CPropertyPoolHandler>(core::smart_refctd_ptr(m_device));
            // smaller workgroups fill occupancy gaps better, especially on new Nvidia GPUs, but we don't want too small workgroups on mobile
            // TODO: investigate whether we need to clamp against 256u instead of 128u on mobile
            const auto scan_workgroup_size = core::max(core::roundDownToPoT(limits.maxWorkgroupSize[0]) >> 1u, 128u);
            m_scanner = core::make_smart_refctd_ptr<CScanner>(core::smart_refctd_ptr(m_device), scan_workgroup_size);
        }

        ~IUtilities()
        {
            m_device->unmapMemory(m_defaultDownloadBuffer->getBuffer()->getBoundMemory());
            m_device->unmapMemory(m_defaultUploadBuffer->getBuffer()->getBoundMemory());
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
        
        //! This function provides some guards against streamingBuffer fragmentation or allocation failure
        static uint32_t getAllocationSizeForStreamingBuffer(const size_t size, const uint64_t alignment, uint32_t maxFreeBlock, const uint32_t optimalTransferAtom)
        {
            // due to coherent flushing atom sizes, we need to pad
            const size_t paddedSize = core::alignUp(size,alignment);
            // if we aim to make a "slightly" smaller allocation we need to assume worst case about fragmentation
            if (!core::is_aligned_to(maxFreeBlock,alignment) || maxFreeBlock>paddedSize)
            {
                // two freeblocks might be spawned, one for the front (due to alignment) and one for the end
                const auto maxWastedSpace = (minStreamingBufferAllocationSize<<1)+alignment-1u;
                if (maxFreeBlock>maxWastedSpace)
                    maxFreeBlock = core::alignDown(maxFreeBlock-maxWastedSpace,alignment);
                else
                    maxFreeBlock = 0;
            }
            // don't want to be stuck doing tiny copies, better defragment the allocator by forcing an allocation failure
            const bool largeEnoughTransfer = maxFreeBlock>=paddedSize || maxFreeBlock>=optimalTransferAtom;
            // how big of an allocation we'll make
            const uint32_t allocationSize = static_cast<uint32_t>(core::min<size_t>(
                largeEnoughTransfer ? maxFreeBlock:optimalTransferAtom,paddedSize
            ));
            return allocationSize;
        }

        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUBuffer> createFilledDeviceLocalBufferOnDedMem(IGPUQueue* queue, IGPUBuffer::SCreationParams&& params, const void* data)
        {
            if(!params.usage.hasFlags(IGPUBuffer::EUF_TRANSFER_DST_BIT))
            {
                assert(false);
                return nullptr;
            }
            auto buffer = m_device->createBuffer(std::move(params));
            auto mreqs = buffer->getMemoryReqs();
            mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto mem = m_device->allocate(mreqs, buffer.get());
            updateBufferRangeViaStagingBufferAutoSubmit(asset::SBufferRange<IGPUBuffer>{0u, params.size, core::smart_refctd_ptr(buffer)}, data, queue);
            return buffer;
        }

        // ! Create Filled Image from IGPUBuffer
        // TODO: Look into removing this set of functions (next 3) now that we have uploadImageViaStagingBuffer? Because every usage of this function creates a big IGPUBuffer only to feed into this.

        //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalImageOnDedMem(IGPUCommandBuffer* cmdbuf, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions)
        {
            // Todo: Remove this API check once OpenGL(ES) does its format usage reporting correctly
            if (srcBuffer->getAPIType() == EAT_VULKAN)
            {
                const auto& formatUsages = m_device->getPhysicalDevice()->getImageFormatUsagesOptimalTiling()[params.format];
                if (!formatUsages.transferDst)
                    return nullptr;
            }

            const auto finalLayout = params.initialLayout;
        
            if(!params.usage.hasFlags(asset::IImage::EUF_TRANSFER_DST_BIT))
            {
                assert(false);
                return nullptr;
            }

            auto retImg = m_device->createImage(std::move(params));
            auto retImgMemReqs = retImg->getMemoryReqs();
            retImgMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto retImgMem = m_device->allocate(retImgMemReqs, retImg.get());

            assert(cmdbuf->getState() == IGPUCommandBuffer::ES_RECORDING);

            IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
            barrier.barrier.srcAccessMask = asset::EAF_NONE;
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.oldLayout = asset::IImage::EL_UNDEFINED;
            barrier.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = ~0u;
            barrier.dstQueueFamilyIndex = ~0u;
            barrier.image = retImg;
            barrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT; // need this from input, infact this family of functions would be more usable if we take in a SSubresourceRange to operate on
            barrier.subresourceRange.baseArrayLayer = 0u;
            barrier.subresourceRange.layerCount = retImg->getCreationParameters().arrayLayers;
            barrier.subresourceRange.baseMipLevel = 0u;
            barrier.subresourceRange.levelCount = retImg->getCreationParameters().mipLevels;
            cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);

            cmdbuf->copyBufferToImage(srcBuffer, retImg.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, regionCount, pRegions);

            if (finalLayout != asset::IImage::EL_TRANSFER_DST_OPTIMAL)
            {
                // Cannot transition to UNDEFINED and PREINITIALIZED
                // TODO: Take an extra parameter that let's the user choose the newLayout or an output parameter that tells the user the final layout
                if(finalLayout != asset::IImage::EL_UNDEFINED && finalLayout != asset::IImage::EL_PREINITIALIZED)
                {
                    barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
                    barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
                    barrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout = finalLayout;
                    cmdbuf->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
                }
            }

            return retImg;
        }
        
        //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalImageOnDedMem(
            IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions,
            IGPUQueue* submissionQueue, IGPUQueue::SSubmitInfo submitInfo = {}
        )
        {
            if (!submitInfo.isValid())
            {
                // TODO: log error
                assert(false);
                return nullptr;
            }
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            auto* fenceptr = fence.get();
            CSubmitInfoPatcher submitInfoPatcher;
            submitInfoPatcher.patchAndBegin(submitInfo, m_device, submissionQueue->getFamilyIndex());
            auto retval = createFilledDeviceLocalImageOnDedMem(submitInfoPatcher.getRecordingCommandBuffer(), std::move(params), srcBuffer, regionCount, pRegions);
            submitInfoPatcher.end();
            submissionQueue->submit(1u, &submitInfo, fenceptr);
            m_device->blockForFences(1u, &fenceptr);
            return retval;
        }

        // ! Create Filled Image from another IGPUImage

        //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalImageOnDedMem(IGPUCommandBuffer* cmdbuf, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions)
        {
            // Todo(achal): Remove this API check once OpenGL(ES) does its format usage reporting correctly
            if (srcImage->getAPIType() == EAT_VULKAN)
            {
                auto* physicalDevice = m_device->getPhysicalDevice();
                const auto validateFormatFeature = [&params, physicalDevice](const auto format, const auto reqFormatUsages) -> bool
                {
                    if (params.tiling == IGPUImage::ET_OPTIMAL)
                        return (physicalDevice->getImageFormatUsagesOptimalTiling()[params.format] & reqFormatUsages) == reqFormatUsages;
                    else
                        return (physicalDevice->getImageFormatUsagesLinearTiling()[params.format] & reqFormatUsages) == reqFormatUsages;
                };

                IPhysicalDevice::SFormatImageUsages::SUsage requiredFormatUsage = {};
                requiredFormatUsage.transferSrc = 1;
                if (!validateFormatFeature(srcImage->getCreationParameters().format, requiredFormatUsage))
                    return nullptr;

                requiredFormatUsage.transferSrc = 0;
                requiredFormatUsage.transferDst = 1;
                if (!validateFormatFeature(params.format, requiredFormatUsage))
                    return nullptr;
            }

            const auto finalLayout = params.initialLayout;
            
            if(!params.usage.hasFlags(asset::IImage::EUF_TRANSFER_DST_BIT))
            {
                assert(false);
                return nullptr;
            }

            auto retImg = m_device->createImage(std::move(params));
            auto retImgMemReqs = retImg->getMemoryReqs();
            retImgMemReqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto retImgMem = m_device->allocate(retImgMemReqs, retImg.get());

            assert(cmdbuf->getState() == IGPUCommandBuffer::ES_RECORDING);

            IGPUCommandBuffer::SImageMemoryBarrier barrier = {};
            barrier.barrier.srcAccessMask = asset::EAF_NONE;
            barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
            barrier.oldLayout = asset::IImage::EL_UNDEFINED;
            barrier.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = ~0u;
            barrier.dstQueueFamilyIndex = ~0u;
            barrier.image = retImg;
            barrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT; // need this from input, infact this family of functions would be more usable if we take in a SSubresourceRange to operate on
            barrier.subresourceRange.baseArrayLayer = 0u;
            barrier.subresourceRange.layerCount = retImg->getCreationParameters().arrayLayers;
            barrier.subresourceRange.baseMipLevel = 0u;
            barrier.subresourceRange.levelCount = retImg->getCreationParameters().mipLevels;
            cmdbuf->pipelineBarrier(asset::EPSF_TOP_OF_PIPE_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);

            cmdbuf->copyImage(srcImage, asset::IImage::EL_TRANSFER_SRC_OPTIMAL, retImg.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, regionCount, pRegions);

            if (finalLayout != asset::IImage::EL_TRANSFER_DST_OPTIMAL)
            {
                barrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
                barrier.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
                barrier.oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
                barrier.newLayout = finalLayout;
                cmdbuf->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &barrier);
            }
            return retImg;
        }
        
        //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalImageOnDedMem(
            IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions,
            IGPUQueue* submissionQueue, IGPUQueue::SSubmitInfo submitInfo = {}
        )
        {
            if (!submitInfo.isValid())
            {
                // TODO: log error
                assert(false);
                return nullptr;
            }

            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            auto* fenceptr = fence.get();

            CSubmitInfoPatcher submitInfoPatcher;
            submitInfoPatcher.patchAndBegin(submitInfo, m_device, submissionQueue->getFamilyIndex());
            auto retval = createFilledDeviceLocalImageOnDedMem(submitInfoPatcher.getRecordingCommandBuffer(), std::move(params), srcImage, regionCount, pRegions);
            submitInfoPatcher.end();
            submissionQueue->submit(1u, &submitInfo, fenceptr);
            m_device->blockForFences(1u, &fenceptr);
            return retval;
        }

        // --------------
        // updateBufferRangeViaStagingBuffer
        // --------------

        //! Copies `data` to stagingBuffer and Records the commands needed to copy the data from stagingBuffer to `bufferRange.buffer`
        //! If the allocation from staging memory fails due to large buffer size or fragmentation then This function may need to submit the command buffer via the `submissionQueue`. 
        //! Returns:
        //!     IGPUQueue::SSubmitInfo to use for command buffer submission instead of `intendedNextSubmit`. 
        //!         for example: in the case the `SSubmitInfo::waitSemaphores` were already signalled, the new SSubmitInfo will have it's waitSemaphores emptied from `intendedNextSubmit`.
        //!     Make sure to submit with the new SSubmitInfo returned by this function
        //! Parameters:
        //!     - bufferRange: contains offset + size into bufferRange::buffer that will be copied from `data` (offset doesn't affect how `data` is accessed)
        //!     - data: raw pointer to data that will be copied to bufferRange::buffer
        //!     - intendedNextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers.
        //!         ** The last command buffer will be used to record the copy commands
        //!     - submissionQueue: IGPUQueue used to submit, when needed. 
        //!         Note: This parameter is required but may not be used if there is no need to submit
        //!     - submissionFence: 
        //!         - This is the fence you will use to submit the copies to, this allows freeing up space in stagingBuffer when the fence is signalled, indicating that the copy has finished.
        //!         - This fence will be in `UNSIGNALED` state after exiting the function. (It will reset after each implicit submit)
        //!         - This fence may be used for CommandBuffer submissions using `submissionQueue` inside the function.
        //!         ** NOTE: This fence will be signalled everytime there is a submission inside this function, which may be more than one until the job is finished.
        //! Valid Usage:
        //!     * data must not be nullptr
        //!     * bufferRange should be valid (see SBufferRange::isValid())
        //!     * intendedNextSubmit::commandBufferCount must be > 0
        //!     * The commandBuffers should have been allocated from a CommandPool with the same queueFamilyIndex as `submissionQueue`
        //!     * The last command buffer should be in `RECORDING` state.
        //!     * The last command buffer should be must've called "begin()" with `IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT` flag
        //!         The reason is the commands recorded into the command buffer would not be valid for a second submission and the stagingBuffer memory wouldv'e been freed/changed.
        //!     * The last command buffer should be "resettable". See `ICommandBuffer::E_STATE` comments
        //!     * To ensure correct execution order, (if any) all the command buffers except the last one should be in `EXECUTABLE` state.
        //!     * submissionQueue must point to a valid IGPUQueue
        //!     * submissionFence must point to a valid IGPUFence
        //!     * submissionFence must be in `UNSIGNALED` state
        //!     ** IUtility::getDefaultUpStreamingBuffer()->cull_frees() should be called before reseting the submissionFence and after fence is signaled. 
        [[nodiscard("Use The New IGPUQueue::SubmitInfo")]] inline IGPUQueue::SSubmitInfo updateBufferRangeViaStagingBuffer(
            const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data,
            IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo intendedNextSubmit
        )
        {
            if(!intendedNextSubmit.isValid() || intendedNextSubmit.commandBufferCount <= 0u)
            {
                // TODO: log error -> intendedNextSubmit is invalid
                assert(false);
                return intendedNextSubmit;
            }

            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            const uint32_t optimalTransferAtom = limits.maxResidentInvocations*sizeof(uint32_t);
            
            // Use the last command buffer in intendedNextSubmit, it should be in recording state
            auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount-1];
            auto* cmdpool = cmdbuf->getPool();
            assert(cmdbuf->isResettable());
            assert(cmdpool->getQueueFamilyIndex() == submissionQueue->getFamilyIndex());
            assert(cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT));
            assert(bufferRange.buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_TRANSFER_DST_BIT));

            // no pipeline barriers necessary because write and optional flush happens before submit, and memory allocation is reclaimed after fence signal
            for (size_t uploadedSize = 0ull; uploadedSize < bufferRange.size;)
            {
                // how much hasn't been uploaded yet
                const size_t size = bufferRange.size-uploadedSize;
                // how large we can make the allocation
                uint32_t maxFreeBlock = m_defaultUploadBuffer.get()->max_size();
                // get allocation size
                const uint32_t allocationSize = getAllocationSizeForStreamingBuffer(size, m_allocationAlignment, maxFreeBlock, optimalTransferAtom);
                // make sure we dont overrun the destination buffer due to padding
                const uint32_t subSize = core::min(allocationSize,size);
                // cannot use `multi_place` because of the extra padding size we could have added
                uint32_t localOffset = StreamingTransientDataBufferMT<>::invalid_value;
                m_defaultUploadBuffer.get()->multi_allocate(std::chrono::steady_clock::now()+std::chrono::microseconds(500u),1u,&localOffset,&allocationSize,&m_allocationAlignment);
                // copy only the unpadded part
                if (localOffset != StreamingTransientDataBufferMT<>::invalid_value)
                {
                    const void* dataPtr = reinterpret_cast<const uint8_t*>(data) + uploadedSize;
                    memcpy(reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer()) + localOffset, dataPtr, subSize);
                }
                // keep trying again
                if (localOffset == StreamingTransientDataBufferMT<>::invalid_value)
                {
                    // but first sumbit the already buffered up copies
                    cmdbuf->end();
                    IGPUQueue::SSubmitInfo submit = intendedNextSubmit;
                    submit.signalSemaphoreCount = 0u;
                    submit.pSignalSemaphores = nullptr;
                    assert(submit.isValid());
                    submissionQueue->submit(1u, &submit, submissionFence);
                    m_device->blockForFences(1u, &submissionFence);
                    intendedNextSubmit.commandBufferCount = 1u;
                    intendedNextSubmit.commandBuffers = &cmdbuf;
                    intendedNextSubmit.waitSemaphoreCount = 0u;
                    intendedNextSubmit.pWaitSemaphores = nullptr;
                    intendedNextSubmit.pWaitDstStageMask = nullptr;
                    // before resetting we need poll all events in the allocator's deferred free list
                    m_defaultUploadBuffer->cull_frees();
                    // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
                    m_device->resetFences(1u, &submissionFence);
                    cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
                    cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                    continue;
                }
                // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
                if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate())
                {
                    auto flushRange = AlignedMappedMemoryRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(),localOffset,subSize,limits.nonCoherentAtomSize);
                    m_device->flushMappedMemoryRanges(1u,&flushRange);
                }
                // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
                asset::SBufferCopy copy;
                copy.srcOffset = localOffset;
                copy.dstOffset = bufferRange.offset + uploadedSize;
                copy.size = subSize;
                cmdbuf->copyBuffer(m_defaultUploadBuffer.get()->getBuffer(), bufferRange.buffer.get(), 1u, &copy);
                // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
                m_defaultUploadBuffer.get()->multi_deallocate(1u,&localOffset,&allocationSize,core::smart_refctd_ptr<IGPUFence>(submissionFence),&cmdbuf); // can queue with a reset but not yet pending fence, just fine
                uploadedSize += subSize;
            }
            return intendedNextSubmit;
        }

        //! This function is an specialization of the `updateBufferRangeViaStagingBuffer` function above.
        //! Submission of the commandBuffer to submissionQueue happens automatically, no need for the user to handle submit
        //! WARNING: Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        //! Parameters:
        //! - `submitInfo`: IGPUQueue::SSubmitInfo used to submit the copy operations.
        //!     * Use this parameter to wait for previous operations to finish using submitInfo::waitSemaphores or signal new semaphores using submitInfo::signalSemaphores
        //!     * Fill submitInfo::commandBuffers with the commandbuffers you want to be submitted before the copy in this struct as well, for example pipeline barrier commands.
        //!     * Empty by default: waits for no semaphore and signals no semaphores.
        //! Patches the submitInfo::commandBuffers
        //! If submitInfo::commandBufferCount == 0, it will create an implicit command buffer to use for recording copy commands
        //! If submitInfo::commandBufferCount > 0 the last command buffer is in `EXECUTABLE` state, It will add another temporary command buffer to end of the array and use it for recording and submission
        //! If submitInfo::commandBufferCount > 0 the last command buffer is in `RECORDING` or `INITIAL` state, It won't add another command buffer and uses the last command buffer for the copy commands.
        //! WARNING: If commandBufferCount > 0, The last commandBuffer won't be in the same state as it was before entering the function, because it needs to be `end()`ed and submitted
        //! Valid Usage:
        //!     * If submitInfo::commandBufferCount > 0 and the last command buffer must be in one of these stages: `EXECUTABLE`, `INITIAL`, `RECORDING`
        //! For more info on command buffer states See `ICommandBuffer::E_STATE` comments.
        inline void updateBufferRangeViaStagingBufferAutoSubmit(
            const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data,
            IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo submitInfo = {}
        )
        {
            if(!submitInfo.isValid())
            {
                // TODO: log error
                assert(false);
                return;
            }

            CSubmitInfoPatcher submitInfoPatcher;
            submitInfoPatcher.patchAndBegin(submitInfo, m_device, submissionQueue->getFamilyIndex());
            submitInfo = updateBufferRangeViaStagingBuffer(bufferRange,data,submissionQueue,submissionFence,submitInfo);
            submitInfoPatcher.end();

            assert(submitInfo.isValid());
            submissionQueue->submit(1u,&submitInfo,submissionFence);
        }
        
        //! This function is an specialization of the `updateBufferRangeViaStagingBufferAutoSubmit` function above.
        //! Additionally waits for the fence
        //! WARNING: This function blocks CPU and stalls the GPU!
        inline void updateBufferRangeViaStagingBufferAutoSubmit(
            const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data,
            IGPUQueue* submissionQueue, const IGPUQueue::SSubmitInfo& submitInfo = {}
        )
        {
            if(!submitInfo.isValid())
            {
                // TODO: log error
                assert(false);
                return;
            }

            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            updateBufferRangeViaStagingBufferAutoSubmit(bufferRange, data, submissionQueue, fence.get(), submitInfo);
            m_device->blockForFences(1u, &fence.get());
        }
        
        // --------------
        // downloadBufferRangeViaStagingBuffer
        // --------------
        
        /* callback signature used for downstreaming requests */
        using data_consumption_callback_t = void(const size_t /*dstOffset*/, const void* /*srcPtr*/, const size_t /*size*/);

        struct default_data_consumption_callback_t
        {
            default_data_consumption_callback_t(void* dstPtr) :
                m_dstPtr(dstPtr)
            {}

            inline void operator()(const size_t dstOffset, const void* srcPtr, const size_t size)
            {
                uint8_t* dst = reinterpret_cast<uint8_t*>(m_dstPtr) + dstOffset;
                memcpy(dst, srcPtr, size);
            }

            void* m_dstPtr;
        };

        //! Calls the callback to copy the data to a destination Offset
        //! * IMPORTANT: To make the copies ready, IUtility::getDefaultDownStreamingBuffer()->cull_frees() should be called after the `submissionFence` is signaled.
        //! If the allocation from staging memory fails due to large image size or fragmentation then This function may need to submit the command buffer via the `submissionQueue` and then signal the fence. 
        //! Returns:
        //!     IGPUQueue::SSubmitInfo to use for command buffer submission instead of `intendedNextSubmit`. 
        //!         for example: in the case the `SSubmitInfo::waitSemaphores` were already signalled, the new SSubmitInfo will have it's waitSemaphores emptied from `intendedNextSubmit`.
        //!     Make sure to submit with the new SSubmitInfo returned by this function
        //! Parameters:
        //!     - consumeCallback: it's a std::function called when the data is ready to be copied (see `data_consumption_callback_t`)
        //!     - srcBufferRange: the buffer range (buffer + size) to be copied from.
        //!     - intendedNextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers.
        //!         ** The last command buffer will be used to record the copy commands
        //!     - submissionQueue: IGPUQueue used to submit, when needed. 
        //!         Note: This parameter is required but may not be used if there is no need to submit
        //!     - submissionFence: 
        //!         - This is the fence you will use to submit the copies to, this allows freeing up space in stagingBuffer when the fence is signalled, indicating that the copy has finished.
        //!         - This fence will be in `UNSIGNALED` state after exiting the function. (It will reset after each implicit submit)
        //!         - This fence may be used for CommandBuffer submissions using `submissionQueue` inside the function.
        //!         ** NOTE: This fence will be signalled everytime there is a submission inside this function, which may be more than one until the job is finished.
        //! Valid Usage:
        //!     * srcBuffer must point to a valid ICPUBuffer
        //!     * srcBuffer->getPointer() must not be nullptr.
        //!     * dstImage must point to a valid IGPUImage
        //!     * regions.size() must be > 0
        //!     * intendedNextSubmit::commandBufferCount must be > 0
        //!     * The commandBuffers should have been allocated from a CommandPool with the same queueFamilyIndex as `submissionQueue`
        //!     * The last command buffer should be in `RECORDING` state.
        //!     * The last command buffer should be must've called "begin()" with `IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT` flag
        //!         The reason is the commands recorded into the command buffer would not be valid for a second submission and the stagingBuffer memory wouldv'e been freed/changed.
        //!     * The last command buffer should be "resettable". See `ICommandBuffer::E_STATE` comments
        //!     * To ensure correct execution order, (if any) all the command buffers except the last one should be in `EXECUTABLE` state.
        //!     * submissionQueue must point to a valid IGPUQueue
        //!     * submissionFence must point to a valid IGPUFence
        //!     * submissionFence must be in `UNSIGNALED` state
        [[nodiscard("Use The New IGPUQueue::SubmitInfo")]] inline IGPUQueue::SSubmitInfo downloadBufferRangeViaStagingBuffer(
            const std::function<data_consumption_callback_t>& consumeCallback, const asset::SBufferRange<IGPUBuffer>& srcBufferRange,
            IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo intendedNextSubmit = {}
        )
        {
            if (!intendedNextSubmit.isValid() || intendedNextSubmit.commandBufferCount <= 0u)
            {
                // TODO: log error -> intendedNextSubmit is invalid
                assert(false);
                return intendedNextSubmit;
            }

            // Use the last command buffer in intendedNextSubmit, it should be in recording state
            auto& cmdbuf = intendedNextSubmit.commandBuffers[intendedNextSubmit.commandBufferCount - 1];

            assert(cmdbuf->getState() == IGPUCommandBuffer::ES_RECORDING && cmdbuf->isResettable());
            assert(cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT));

            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            const uint32_t optimalTransferAtom = limits.maxResidentInvocations*sizeof(uint32_t);

            auto* cmdpool = cmdbuf->getPool();
            assert(cmdpool->getQueueFamilyIndex() == submissionQueue->getFamilyIndex());
 
            // Basically downloadedSize is downloadRecordedIntoCommandBufferSize :D
            for (size_t downloadedSize = 0ull; downloadedSize < srcBufferRange.size;)
            {
                const size_t notDownloadedSize = srcBufferRange.size - downloadedSize;
                // how large we can make the allocation
                uint32_t maxFreeBlock = m_defaultDownloadBuffer.get()->max_size();
                // get allocation size
                const uint32_t allocationSize = getAllocationSizeForStreamingBuffer(notDownloadedSize, m_allocationAlignment, maxFreeBlock, optimalTransferAtom);
                const uint32_t copySize = core::min(allocationSize, notDownloadedSize);

                uint32_t localOffset = StreamingTransientDataBufferMT<>::invalid_value;
                m_defaultDownloadBuffer.get()->multi_allocate(std::chrono::steady_clock::now()+std::chrono::microseconds(500u),1u,&localOffset,&allocationSize,&m_allocationAlignment);
                
                if (localOffset != StreamingTransientDataBufferMT<>::invalid_value)
                {
                    asset::SBufferCopy copy;
                    copy.srcOffset = srcBufferRange.offset + downloadedSize;
                    copy.dstOffset = localOffset;
                    copy.size = copySize;
                    cmdbuf->copyBuffer(srcBufferRange.buffer.get(), m_defaultDownloadBuffer.get()->getBuffer(), 1u, &copy);

                    auto dataConsumer = core::make_smart_refctd_ptr<CDownstreamingDataConsumer>(downloadedSize, IDeviceMemoryAllocation::MemoryRange(localOffset, copySize), consumeCallback, cmdbuf, m_defaultDownloadBuffer.get(), core::smart_refctd_ptr(m_device));

                    m_defaultDownloadBuffer.get()->multi_deallocate(1u, &localOffset, &allocationSize, core::smart_refctd_ptr<IGPUFence>(submissionFence), &dataConsumer.get());

                    downloadedSize += copySize;
                }
                else
                {
                    // but first sumbit the already buffered up copies
                    cmdbuf->end();
                    IGPUQueue::SSubmitInfo submit = intendedNextSubmit;
                    submit.signalSemaphoreCount = 0u;
                    submit.pSignalSemaphores = nullptr;
                    assert(submit.isValid());
                    submissionQueue->submit(1u, &submit, submissionFence);
                    m_device->blockForFences(1u, &submissionFence);

                    intendedNextSubmit.commandBufferCount = 1u;
                    intendedNextSubmit.commandBuffers = &cmdbuf;
                    intendedNextSubmit.waitSemaphoreCount = 0u;
                    intendedNextSubmit.pWaitSemaphores = nullptr;
                    intendedNextSubmit.pWaitDstStageMask = nullptr;

                    // before resetting we need poll all events in the allocator's deferred free list
                    m_defaultDownloadBuffer->cull_frees();
                    // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
                    m_device->resetFences(1u, &submissionFence);
                    cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
                    cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                }
            }
            return intendedNextSubmit;
        }

        //! This function is an specialization of the `downloadBufferRangeViaStagingBufferAutoSubmit` function above.
        //! Submission of the commandBuffer to submissionQueue happens automatically, no need for the user to handle submit
        //! Parameters:
        //! - `submitInfo`: IGPUQueue::SSubmitInfo used to submit the copy operations.
        //!     * Use this parameter to wait for previous operations to finish using submitInfo::waitSemaphores or signal new semaphores using submitInfo::signalSemaphores
        //!     * Fill submitInfo::commandBuffers with the commandbuffers you want to be submitted before the copy in this struct as well, for example pipeline barrier commands.
        //!     * Empty by default: waits for no semaphore and signals no semaphores.
        //! Patches the submitInfo::commandBuffers
        //! If submitInfo::commandBufferCount == 0, it will create an implicit command buffer to use for recording copy commands
        //! If submitInfo::commandBufferCount > 0 the last command buffer is in `EXECUTABLE` state, It will add another temporary command buffer to end of the array and use it for recording and submission
        //! If submitInfo::commandBufferCount > 0 the last command buffer is in `RECORDING` or `INITIAL` state, It won't add another command buffer and uses the last command buffer for the copy commands.
        //! WARNING: If commandBufferCount > 0, The last commandBuffer won't be in the same state as it was before entering the function, because it needs to be `end()`ed and submitted
        //! Valid Usage:
        //!     * If submitInfo::commandBufferCount > 0 and the last command buffer must be in one of these stages: `EXECUTABLE`, `INITIAL`, `RECORDING`
        //! For more info on command buffer states See `ICommandBuffer::E_STATE` comments.
        inline void downloadBufferRangeViaStagingBufferAutoSubmit(
            const std::function<data_consumption_callback_t>& consumeCallback, const asset::SBufferRange<IGPUBuffer>& srcBufferRange,
            IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo submitInfo = {}
        )
        {
            if (!submitInfo.isValid())
            {
                // TODO: log error
                assert(false);
                return;
            }

            CSubmitInfoPatcher submitInfoPatcher;
            submitInfoPatcher.patchAndBegin(submitInfo, m_device, submissionQueue->getFamilyIndex());
            submitInfo = downloadBufferRangeViaStagingBuffer(consumeCallback, srcBufferRange, submissionQueue, submissionFence, submitInfo);
            submitInfoPatcher.end();

            assert(submitInfo.isValid());
            submissionQueue->submit(1u, &submitInfo, submissionFence);
        }

        //! This function is an specialization of the `downloadBufferRangeViaStagingBufferAutoSubmit` function above.
        //! Additionally waits for the fence
        //! WARNING: This function blocks CPU and stalls the GPU!
        inline void downloadBufferRangeViaStagingBufferAutoSubmit(
            const asset::SBufferRange<IGPUBuffer>& srcBufferRange, void* data,
            IGPUQueue* submissionQueue, const IGPUQueue::SSubmitInfo& submitInfo = {}
        )
        {
            if (!submitInfo.isValid())
            {
                // TODO: log error
                assert(false);
                return;
            }
            

            auto fence = m_device->createFence(IGPUFence::ECF_UNSIGNALED);
            downloadBufferRangeViaStagingBufferAutoSubmit(std::function<data_consumption_callback_t>(default_data_consumption_callback_t(data)), srcBufferRange, submissionQueue, fence.get(), submitInfo);
            auto* fenceptr = fence.get();
            m_device->blockForFences(1u, &fenceptr);

            m_defaultDownloadBuffer->cull_frees();
        }
        
        // --------------
        // buildAccelerationStructures
        // --------------

        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline void buildAccelerationStructures(IGPUQueue* queue, const core::SRange<IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
        {
            core::smart_refctd_ptr<IGPUCommandPool> pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
            m_device->createCommandBuffers(pool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
            IGPUQueue::SSubmitInfo submit;
            {
                submit.commandBufferCount = 1u;
                submit.commandBuffers = &cmdbuf.get();
                submit.waitSemaphoreCount = 0u;
                submit.pWaitDstStageMask = nullptr;
                submit.pWaitSemaphores = nullptr;
            }

            cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
            {
                cmdbuf->buildAccelerationStructures(pInfos, ppBuildRangeInfos);
            }
            cmdbuf->end();

            queue->submit(1u, &submit, fence.get());
        
            m_device->blockForFences(1u,&fence.get());
        }

        // --------------
        // updateImageViaStagingBuffer
        // --------------

        //! Copies `srcBuffer` to stagingBuffer and Records the commands needed to copy the image from stagingBuffer to `dstImage`
        //! If the allocation from staging memory fails due to large image size or fragmentation then This function may need to submit the command buffer via the `submissionQueue` and then signal the fence. 
        //! Returns:
        //!     IGPUQueue::SSubmitInfo to use for command buffer submission instead of `intendedNextSubmit`. 
        //!         for example: in the case the `SSubmitInfo::waitSemaphores` were already signalled, the new SSubmitInfo will have it's waitSemaphores emptied from `intendedNextSubmit`.
        //!     Make sure to submit with the new SSubmitInfo returned by this function
        //! Parameters:
        //!     - srcBuffer: source buffer to copy image from
        //!     - srcFormat: The image format the `srcBuffer` is laid out in memory.
        //          In the case that dstImage has a different format this function will make the necessary conversions.
        //          If `srcFormat` is EF_UNKOWN, it will be assumed to have the same format `dstImage` was created with.
        //!     - dstImage: destination image to copy image to
        //!     - currentDstImageLayout: the image layout of `dstImage` at the time of submission.
        //!     - regions: regions to copy `srcBuffer`
        //!     - intendedNextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers.
        //!         ** The last command buffer will be used to record the copy commands
        //!     - submissionQueue: IGPUQueue used to submit, when needed. 
        //!         Note: This parameter is required but may not be used if there is no need to submit
        //!     - submissionFence: 
        //!         - This is the fence you will use to submit the copies to, this allows freeing up space in stagingBuffer when the fence is signalled, indicating that the copy has finished.
        //!         - This fence will be in `UNSIGNALED` state after exiting the function. (It will reset after each implicit submit)
        //!         - This fence may be used for CommandBuffer submissions using `submissionQueue` inside the function.
        //!         ** NOTE: This fence will be signalled everytime there is a submission inside this function, which may be more than one until the job is finished.
        //! Valid Usage:
        //!     * srcBuffer must point to a valid ICPUBuffer
        //!     * srcBuffer->getPointer() must not be nullptr.
        //!     * dstImage must point to a valid IGPUImage
        //!     * regions.size() must be > 0
        //!     * intendedNextSubmit::commandBufferCount must be > 0
        //!     * The commandBuffers should have been allocated from a CommandPool with the same queueFamilyIndex as `submissionQueue`
        //!     * The last command buffer should be in `RECORDING` state.
        //!     * The last command buffer should be must've called "begin()" with `IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT` flag
        //!         The reason is the commands recorded into the command buffer would not be valid for a second submission and the stagingBuffer memory wouldv'e been freed/changed.
        //!     * The last command buffer should be "resettable". See `ICommandBuffer::E_STATE` comments
        //!     * To ensure correct execution order, (if any) all the command buffers except the last one should be in `EXECUTABLE` state.
        //!     * submissionQueue must point to a valid IGPUQueue
        //!     * submissionFence must point to a valid IGPUFence
        //!     * submissionFence must be in `UNSIGNALED` state
        //!     ** IUtility::getDefaultUpStreamingBuffer()->cull_frees() should be called before reseting the submissionFence and after `submissionFence` is signaled. 
        [[nodiscard("Use The New IGPUQueue::SubmitInfo")]] IGPUQueue::SSubmitInfo updateImageViaStagingBuffer(
            asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, asset::IImage::E_LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
            IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo intendedNextSubmit);
        
        //! This function is an specialization of the `updateImageViaStagingBuffer` function above.
        //! Submission of the commandBuffer to submissionQueue happens automatically, no need for the user to handle submit
        //! Parameters:
        //! - `submitInfo`: IGPUQueue::SSubmitInfo used to submit the copy operations.
        //!     * Use this parameter to wait for previous operations to finish using submitInfo::waitSemaphores or signal new semaphores using submitInfo::signalSemaphores
        //!     * Fill submitInfo::commandBuffers with the commandbuffers you want to be submitted before the copy in this struct as well, for example pipeline barrier commands.
        //!     * Empty by default: waits for no semaphore and signals no semaphores.
        //! Patches the submitInfo::commandBuffers
        //! If submitInfo::commandBufferCount == 0, it will create an implicit command buffer to use for recording copy commands
        //! If submitInfo::commandBufferCount > 0 the last command buffer is in `EXECUTABLE` state, It will add another temporary command buffer to end of the array and use it for recording and submission
        //! If submitInfo::commandBufferCount > 0 the last command buffer is in `RECORDING` or `INITIAL` state, It won't add another command buffer and uses the last command buffer for the copy commands.
        //! WARNING: If commandBufferCount > 0, The last commandBuffer won't be in the same state as it was before entering the function, because it needs to be `end()`ed and submitted
        //! Valid Usage:
        //!     * If submitInfo::commandBufferCount > 0 and the last command buffer must be in one of these stages: `EXECUTABLE`, `INITIAL`, `RECORDING`
        //! For more info on command buffer states See `ICommandBuffer::E_STATE` comments.
        void updateImageViaStagingBufferAutoSubmit(
            asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, asset::IImage::E_LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
            IGPUQueue* submissionQueue, IGPUFence* submissionFence, IGPUQueue::SSubmitInfo submitInfo = {});

        //! This function is an specialization of the `updateImageViaStagingBufferAutoSubmit` function above.
        //! Additionally waits for the fence
        //! WARNING: This function blocks CPU and stalls the GPU!
        void updateImageViaStagingBufferAutoSubmit(
            asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, asset::IImage::E_LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
            IGPUQueue* submissionQueue, const IGPUQueue::SSubmitInfo& submitInfo = {}
        );

    protected:
        
        // The application must round down the start of the range to the nearest multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize,
        // and round the end of the range up to the nearest multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize.
        static IDeviceMemoryAllocation::MappedMemoryRange AlignedMappedMemoryRange(IDeviceMemoryAllocation* mem, const size_t& off, const size_t& len, size_t nonCoherentAtomSize)
        {
            IDeviceMemoryAllocation::MappedMemoryRange range = {};
            range.memory = mem;
            range.offset = core::alignDown(off, nonCoherentAtomSize);
            range.length = core::min(core::alignUp(len, nonCoherentAtomSize), mem->getAllocationSize());
            return range;
        }

        //! Internal tool used to patch command buffers in submit info.
        class CSubmitInfoPatcher
        {
        public:
            //! Patches the submitInfo::commandBuffers and then makes sure the last command buffer is in recording state
            //! If submitInfo::commandBufferCount == 0, it will create an implicit command buffer to use for recording copy commands
            //! If submitInfo::commandBufferCount > 0 the last command buffer is in `EXECUTABLE` state, It will add another temporary command buffer to end of the array and use it for recording and submission
            //! If submitInfo::commandBufferCount > 0 the last command buffer is in `RECORDING` or `INITIAL` state, It won't add another command buffer and uses the last command buffer for the copy commands.
            //! Params:
            //!     - submitInfo: IGPUQueue::SSubmitInfo to patch
            //!     - device: logical device to create new command pool and command buffer if necessary.
            //!     - newCommandPoolFamIdx: family index to create commandPool with if necessary.
            inline void patchAndBegin(IGPUQueue::SSubmitInfo& submitInfo, core::smart_refctd_ptr<ILogicalDevice> device, uint32_t newCommandPoolFamIdx)
            {
                bool needToCreateNewCommandBuffer = false;

                if (submitInfo.commandBufferCount <= 0u)
                    needToCreateNewCommandBuffer = true;
                else
                {
                    auto lastCmdBuf = submitInfo.commandBuffers[submitInfo.commandBufferCount - 1u];
                    if (lastCmdBuf->getState() == IGPUCommandBuffer::ES_EXECUTABLE)
                        needToCreateNewCommandBuffer = true;
                }

                // commandBuffer used to record the commands
                if (needToCreateNewCommandBuffer)
                {
                    core::smart_refctd_ptr<IGPUCommandPool> pool = device->createCommandPool(newCommandPoolFamIdx, IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
                    device->createCommandBuffers(pool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &m_newCommandBuffer);

                    const uint32_t newCommandBufferCount = (needToCreateNewCommandBuffer) ? submitInfo.commandBufferCount + 1 : submitInfo.commandBufferCount;
                    m_allCommandBuffers.resize(newCommandBufferCount);

                    for (uint32_t i = 0u; i < submitInfo.commandBufferCount; ++i)
                        m_allCommandBuffers[i] = submitInfo.commandBuffers[i];

                    m_recordCommandBuffer = m_newCommandBuffer.get();
                    m_allCommandBuffers[newCommandBufferCount - 1u] = m_recordCommandBuffer;

                    submitInfo.commandBufferCount = newCommandBufferCount;
                    submitInfo.commandBuffers = m_allCommandBuffers.data();

                    m_recordCommandBuffer->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                }
                else
                {
                    m_recordCommandBuffer = submitInfo.commandBuffers[submitInfo.commandBufferCount - 1u];
                    // If the last command buffer is in INITIAL state, bring it to RECORDING state
                    if (m_recordCommandBuffer->getState() == IGPUCommandBuffer::ES_INITIAL)
                        m_recordCommandBuffer->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                }
            }
            inline void end()
            {
                m_recordCommandBuffer->end();
            }
            inline IGPUCommandBuffer* getRecordingCommandBuffer() { return m_recordCommandBuffer; }

        private:
            IGPUCommandBuffer* m_recordCommandBuffer;
            core::vector<IGPUCommandBuffer*> m_allCommandBuffers;
            core::smart_refctd_ptr<IGPUCommandBuffer> m_newCommandBuffer; // if necessary, then need to hold reference to.
        };

        //! Used in downloadBufferRangeViaStagingBuffer multi_deallocate objectsToHold, 
        //! Calls the std::function callback in destructor because allocator will hold on to this object and drop it when it's safe (fence is singnalled and submit has finished)
        class CDownstreamingDataConsumer final : public core::IReferenceCounted
        {
        public:
            CDownstreamingDataConsumer(
                size_t dstOffset,
                const IDeviceMemoryAllocation::MemoryRange& copyRange,
                const std::function<data_consumption_callback_t>& consumeCallback,
                IGPUCommandBuffer* cmdBuffer,
                StreamingTransientDataBufferMT<>* downstreamingBuffer,
                core::smart_refctd_ptr<ILogicalDevice>&& device
            )
                : m_dstOffset(dstOffset)
                , m_copyRange(copyRange)
                , m_consumeCallback(consumeCallback)
                , m_cmdBuffer(core::smart_refctd_ptr<IGPUCommandBuffer>(cmdBuffer))
                , m_downstreamingBuffer(downstreamingBuffer)
                , m_device(std::move(device))
            {}

            ~CDownstreamingDataConsumer()
            {
                if (m_downstreamingBuffer != nullptr)
                {
                    if (m_downstreamingBuffer->needsManualFlushOrInvalidate())
                    {
                        const auto nonCoherentAtomSize = m_device->getPhysicalDevice()->getLimits().nonCoherentAtomSize;
                        auto flushRange = AlignedMappedMemoryRange(m_downstreamingBuffer->getBuffer()->getBoundMemory(), m_copyRange.offset, m_copyRange.length, nonCoherentAtomSize);
                        m_device->invalidateMappedMemoryRanges(1u, &flushRange);
                    }
                    // Call the function
                    const uint8_t* copySrc = reinterpret_cast<uint8_t*>(m_downstreamingBuffer->getBufferPointer()) + m_copyRange.offset;
                    m_consumeCallback(m_dstOffset, copySrc, m_copyRange.length);
                }
                else
                {
                    assert(false);
                }
            }
        private:
            const size_t m_dstOffset;
            const IDeviceMemoryAllocation::MemoryRange m_copyRange;
            std::function<data_consumption_callback_t> m_consumeCallback;
            core::smart_refctd_ptr<ILogicalDevice> m_device;
            const core::smart_refctd_ptr<const IGPUCommandBuffer> m_cmdBuffer; // because command buffer submiting the copy shouldn't go out of scope when copy isn't finished
            StreamingTransientDataBufferMT<>* m_downstreamingBuffer = nullptr;
        };

        core::smart_refctd_ptr<ILogicalDevice> m_device;

        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultDownloadBuffer;
        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultUploadBuffer;

        core::smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
        core::smart_refctd_ptr<CScanner> m_scanner;
    };

class NBL_API ImageRegionIterator
{
public:
    ImageRegionIterator(
        const core::SRange<const asset::IImage::SBufferCopy>& copyRegions,
        IPhysicalDevice::SQueueFamilyProperties queueFamilyProps,
        asset::ICPUBuffer const* srcBuffer,
        asset::E_FORMAT srcImageFormat,
        video::IGPUImage* const dstImage,
        size_t optimalRowPitchAlignment
    );
    
    // ! Memory you need to allocate to transfer the remaining regions in one submit.
    // ! WARN: It's okay to use less memory than the return value of this function for your staging memory, in that usual case more than 1 copy regions will be needed to transfer the remaining regions.
    size_t getMemoryNeededForRemainingRegions() const;

    // ! Gives `regionToCopyNext` based on `availableMemory`
    // ! memcopies the data from `srcBuffer` to `stagingBuffer`, preparing it for launch and submit to copy to GPU buffer
    // ! updates `availableMemory` (availableMemory -= consumedMemory)
    // ! updates `stagingBufferOffset` based on consumed memory and alignment requirements
    // ! this function may do format conversions when copying from `srcBuffer` to `stagingBuffer` if srcBufferFormat != dstImage->Format passed as constructor parameters
    bool advanceAndCopyToStagingBuffer(asset::IImage::SBufferCopy& regionToCopyNext, uint32_t& availableMemory, uint32_t& stagingBufferOffset, void* stagingBufferPointer);

    // ! returns true when there is no more regions left over to copy
    bool isFinished() const { return currentRegion == regions.size(); }
    uint32_t getCurrentBlockInRow() const { return currentBlockInRow; }
    uint32_t getCurrentRowInSlice() const { return currentRowInSlice; }
    uint32_t getCurrentSliceInLayer() const { return currentSliceInLayer; }
    uint32_t getCurrentLayerInRegion() const { return currentLayerInRegion; }
    uint32_t getCurrentRegion() const { return currentRegion; }

    inline core::vector3du32_SIMD getOptimalCopyTexelStrides(const asset::VkExtent3D& copyExtents) const
    {
        return core::vector3du32_SIMD(
            core::alignUp(copyExtents.width, optimalRowPitchAlignment),
            copyExtents.height,
            copyExtents.depth);
    }

private:

    core::SRange<const asset::IImage::SBufferCopy> regions;

    // Mock CPU Images used to copy cpu buffer to staging buffer
    std::vector<core::smart_refctd_ptr<asset::ICPUImage>> imageFilterInCPUImages;
    core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy> outCPUImageRegions; // needs to be updated before each upload
    std::vector<core::smart_refctd_ptr<asset::ICPUImage>> imageFilterOutCPUImages;

    size_t optimalRowPitchAlignment = 1u;
    bool canTransferMipLevelsPartially = false;
    asset::VkExtent3D minImageTransferGranularity = {};
    uint32_t bufferOffsetAlignment = 1u;

    asset::E_FORMAT srcImageFormat;
    asset::E_FORMAT dstImageFormat;
    asset::ICPUBuffer const* srcBuffer;
    video::IGPUImage* const dstImage;
    
    // Block Offsets 
    uint16_t currentBlockInRow = 0u;
    uint16_t currentRowInSlice = 0u;
    uint16_t currentSliceInLayer = 0u;
    uint16_t currentLayerInRegion = 0u;
    uint16_t currentRegion = 0u;
};

}

#endif