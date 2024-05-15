// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_UTILITIES_H_INCLUDED_
#define _NBL_VIDEO_I_UTILITIES_H_INCLUDED_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/utilities/SIntendedSubmitInfo.h"
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
        constexpr static inline uint32_t OptimalCoalescedInvocationXferSize = sizeof(uint32_t);

        uint32_t m_allocationAlignment = 0u;
        uint32_t m_allocationAlignmentForBufferImageCopy = 0u;

        nbl::system::logger_opt_smart_ptr m_logger;

    public:
        IUtilities(core::smart_refctd_ptr<ILogicalDevice>&& device, nbl::system::logger_opt_smart_ptr&& logger=nullptr, const uint32_t downstreamSize=0x4000000u, const uint32_t upstreamSize=0x4000000u)
            : m_device(core::smart_refctd_ptr(device)), m_logger(nbl::system::logger_opt_smart_ptr(logger))
        {
            auto physicalDevice = m_device->getPhysicalDevice();
            const auto& limits = physicalDevice->getLimits();

            auto queueFamProps = physicalDevice->getQueueFamilyProperties();
            uint32_t minImageTransferGranularityVolume = 1u; // minImageTransferGranularity.width * height * depth

            for (auto& qf : queueFamProps)
            {
                uint32_t volume = qf.minImageTransferGranularity.width*qf.minImageTransferGranularity.height*qf.minImageTransferGranularity.depth;
                if(minImageTransferGranularityVolume<volume)
                    minImageTransferGranularityVolume = volume;
            }

            // host-mapped device memory needs to have this alignment in flush/invalidate calls, therefore this is the streaming buffer's "allocationAlignment".
            m_allocationAlignment = limits.nonCoherentAtomSize;
            m_allocationAlignmentForBufferImageCopy = core::max<uint32_t>(limits.optimalBufferCopyOffsetAlignment,m_allocationAlignment);

            const uint32_t bufferOptimalTransferAtom = limits.maxResidentInvocations * OptimalCoalescedInvocationXferSize;
            const uint32_t maxImageOptimalTransferAtom = limits.maxResidentInvocations * asset::TexelBlockInfo(asset::EF_R64G64B64A64_SFLOAT).getBlockByteSize() * minImageTransferGranularityVolume;
            const uint32_t minImageOptimalTransferAtom = limits.maxResidentInvocations * asset::TexelBlockInfo(asset::EF_R8_UINT).getBlockByteSize();
            const uint32_t maxOptimalTransferAtom = core::max(bufferOptimalTransferAtom,maxImageOptimalTransferAtom);
            const uint32_t minOptimalTransferAtom = core::min(bufferOptimalTransferAtom,minImageOptimalTransferAtom);

            // allocationAlignment <= minBlockSize <= minOptimalTransferAtom <= maxOptimalTransferAtom
            assert(m_allocationAlignment <= minStreamingBufferAllocationSize);
            assert(m_allocationAlignmentForBufferImageCopy <= minStreamingBufferAllocationSize);

            assert(minStreamingBufferAllocationSize <= minOptimalTransferAtom);
            assert(minOptimalTransferAtom <= maxOptimalTransferAtom);

            assert(minStreamingBufferAllocationSize % m_allocationAlignment == 0u);
            assert(minStreamingBufferAllocationSize % m_allocationAlignmentForBufferImageCopy == 0u);

            const auto& enabledFeatures = m_device->getEnabledFeatures();

            IGPUBuffer::SCreationParams streamingBufferCreationParams = {};
            auto commonUsages = core::bitflag(IGPUBuffer::EUF_STORAGE_TEXEL_BUFFER_BIT)|IGPUBuffer::EUF_STORAGE_BUFFER_BIT|IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
            if (enabledFeatures.accelerationStructure)
                commonUsages |= IGPUBuffer::EUF_ACCELERATION_STRUCTURE_STORAGE_BIT;
            
            core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags(IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

            {
                IGPUBuffer::SCreationParams streamingBufferCreationParams = {};
                streamingBufferCreationParams.size = downstreamSize;
                // GPU write to RAM usages
                streamingBufferCreationParams.usage = commonUsages|IGPUBuffer::EUF_TRANSFER_DST_BIT;
                if (enabledFeatures.conditionalRendering)
                    streamingBufferCreationParams.usage |= IGPUBuffer::EUF_CONDITIONAL_RENDERING_BIT_EXT;
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
                mem->map({0ull,reqs.size},access);

                m_defaultDownloadBuffer = core::make_smart_refctd_ptr<StreamingTransientDataBufferMT<>>(asset::SBufferRange<video::IGPUBuffer>{0ull,downstreamSize,std::move(buffer)},maxStreamingBufferAllocationAlignment,minStreamingBufferAllocationSize);
                m_defaultDownloadBuffer->getBuffer()->setObjectDebugName(("Default Download Buffer of Utilities "+std::to_string(ptrdiff_t(this))).c_str());
            }
            {
                IGPUBuffer::SCreationParams streamingBufferCreationParams = {};
                streamingBufferCreationParams.size = upstreamSize;
                streamingBufferCreationParams.usage = commonUsages|IGPUBuffer::EUF_TRANSFER_SRC_BIT|IGPUBuffer::EUF_UNIFORM_TEXEL_BUFFER_BIT|IGPUBuffer::EUF_UNIFORM_BUFFER_BIT|IGPUBuffer::EUF_INDEX_BUFFER_BIT|IGPUBuffer::EUF_VERTEX_BUFFER_BIT|IGPUBuffer::EUF_INDIRECT_BUFFER_BIT;
                if (enabledFeatures.accelerationStructure)
                    streamingBufferCreationParams.usage |= IGPUBuffer::EUF_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT;
                if (enabledFeatures.rayTracingPipeline)
                    streamingBufferCreationParams.usage |= IGPUBuffer::EUF_SHADER_BINDING_TABLE_BIT;
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
                mem->map({0ull,reqs.size},access);

                m_defaultUploadBuffer = core::make_smart_refctd_ptr<StreamingTransientDataBufferMT<>>(asset::SBufferRange<video::IGPUBuffer>{0ull,upstreamSize,std::move(buffer)},maxStreamingBufferAllocationAlignment,minStreamingBufferAllocationSize);
                m_defaultUploadBuffer->getBuffer()->setObjectDebugName(("Default Upload Buffer of Utilities "+std::to_string(ptrdiff_t(this))).c_str());
            }
#if 0 // TODO: port
            m_propertyPoolHandler = core::make_smart_refctd_ptr<CPropertyPoolHandler>(core::smart_refctd_ptr(m_device));
            // smaller workgroups fill occupancy gaps better, especially on new Nvidia GPUs, but we don't want too small workgroups on mobile
            // TODO: investigate whether we need to clamp against 256u instead of 128u on mobile
            const auto scan_workgroup_size = core::max(core::roundDownToPoT(limits.maxWorkgroupSize[0]) >> 1u, 128u);
            m_scanner = core::make_smart_refctd_ptr<CScanner>(core::smart_refctd_ptr(m_device), scan_workgroup_size);
#endif
        }

        inline ~IUtilities()
        {
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

#if 0 // TODO: port
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
#endif
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


        //! This method lets you wrap any other function following the "submit on overflow" pattern with the final submission
        //! to `intendedSubmit.queue` happening automatically, no need for the user to handle the submit at the end.
        //! WARNING: Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        //!  like the `updateBufferRangeViaStagingBufferAutoSubmit` function below.
        //! Parameters:
        //! - `intendedSubmit`: more lax than regular `SIntendedSubmitInfo::valid()`, only needs a valid queue.
        //!     If you don't specify a scratch semaphore, we'll patch and create one internally.
        //!     If you don't have a commandbuffer usable as scratch as the last one, we'll patch internally.
        //!         WARNING: If this particular case happens the `commandBuffers` span will be emptied out!
        template<typename IntendedSubmitInfo> requires std::is_same_v<std::decay_t<IntendedSubmitInfo>,SIntendedSubmitInfo>
        inline ISemaphore::future_t<IQueue::RESULT> autoSubmit(
            IntendedSubmitInfo&& intendedSubmit,
            const std::function<bool(SIntendedSubmitInfo&)>& what,
            const std::span<IQueue::SSubmitInfo::SSemaphoreInfo> extraSignalSemaphores={}
        )
        {
            auto queue = intendedSubmit.queue;
            if (!queue)
            {
                // TODO: log error
                return IQueue::RESULT::OTHER_ERROR;
            }
            // backup in-case we need to restore to unmodified state
            SIntendedSubmitInfo patchedSubmit;
            memcpy(&patchedSubmit,&intendedSubmit,sizeof(SIntendedSubmitInfo));

            core::smart_refctd_ptr<ISemaphore> patchedSemaphore;
            if (!patchedSubmit.scratchSemaphore.semaphore)
            {
                auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
                patchedSemaphore = device->createSemaphore(0);
                patchedSubmit.scratchSemaphore = {patchedSemaphore.get(),0,asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS};
            }
            
            // patch the commandbuffers if needed
            core::vector<IQueue::SSubmitInfo::SCommandBufferInfo> patchedCmdBufs;
            auto patchCmdBuf = [&]()->void{patchedCmdBufs.resize(patchedSubmit.commandBuffers.size()+1);};
            if (auto* candidateScratch=patchedSubmit.getScratchCommandBuffer(); candidateScratch)
            switch(candidateScratch->getState())
            {
                case IGPUCommandBuffer::STATE::INITIAL:
                case IGPUCommandBuffer::STATE::INVALID:
                    if (candidateScratch->isResettable() && candidateScratch->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                        break;
                    patchCmdBuf();
                    break;
                case IGPUCommandBuffer::STATE::RECORDING:
                    if (candidateScratch->isResettable() && candidateScratch->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                        break;
                    candidateScratch->end();
                    patchCmdBuf();
                    break;
                default:
                    patchCmdBuf();
                    break;
            }
            else
                patchCmdBuf();

            core::smart_refctd_ptr<IGPUCommandBuffer> newScratch;
            if (!patchedCmdBufs.empty())
            {
                // allocate a span one larger than the original
                const auto origCmdBufs = patchedSubmit.commandBuffers;
                patchedSubmit.commandBuffers = patchedCmdBufs;
                // copy the original commandbuffers
                std::copy(origCmdBufs.begin(),origCmdBufs.end(),patchedCmdBufs.begin());
                // create the scratch commandbuffer (the patching)
                {
                    auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
                    auto pool = device->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
                    if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&newScratch,1}))
                    {
                        // TODO: log error
                        return IQueue::RESULT::OTHER_ERROR;
                    }
                    if (!newScratch->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                    {
                        // TODO: log error
                        return IQueue::RESULT::OTHER_ERROR;
                    }
                }
                patchedCmdBufs[origCmdBufs.size()] = {newScratch.get()};
                patchedSubmit.commandBuffers = patchedCmdBufs;
            }

            if (!patchedSubmit.valid())
            {
                // TODO: log error
                return IQueue::RESULT::OTHER_ERROR;
            }

            if (!what(patchedSubmit))
            {
                // TODO: log error
                return IQueue::RESULT::OTHER_ERROR;
            }
            // no way back now, have to modify the intended submit
            memcpy(&intendedSubmit,&patchedSubmit,sizeof(intendedSubmit));
            intendedSubmit.getScratchCommandBuffer()->end();

            const auto submit = intendedSubmit.popSubmit(extraSignalSemaphores);
            if (newScratch)
                intendedSubmit.commandBuffers = {};
            if (const auto error=queue->submit(submit); error!=IQueue::RESULT::SUCCESS)
            {
                if (patchedSemaphore)
                {
                    intendedSubmit.waitSemaphores = {};
                    intendedSubmit.scratchSemaphore = {};
                }
                return error;
            }

            ISemaphore::future_t<IQueue::RESULT> retval(IQueue::RESULT::SUCCESS);
            retval.set({intendedSubmit.scratchSemaphore.semaphore,intendedSubmit.scratchSemaphore.value});
            if (patchedSemaphore)
            {
                intendedSubmit.waitSemaphores = {};
                intendedSubmit.scratchSemaphore = {};
            }
            return retval;
        }

        // --------------
        // updateBufferRangeViaStagingBuffer
        // --------------

        //! Copies `data` to stagingBuffer and Records the commands needed to copy the data from stagingBuffer to `bufferRange.buffer`
        //! If the allocation from staging memory fails due to large buffer size or fragmentation then This function may need to submit the command buffer via the `submissionQueue`. 
        //! Returns:
        //!     True on successful recording of copy commands and handling of overflows if any, and false on failure for any reason.
        //!     Make sure to submit with `nextSubmit.popSubmit()` after this function returns.
        //! Parameters:
        //!     - nextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers with, it will be modified if overflow occurred @see SIntendedSubmitInfo
        //!     - bufferRange: contains offset + size into bufferRange::buffer that will be copied from `data` (offset doesn't affect how `data` is accessed)
        //!     - data: raw pointer to data that will be copied to bufferRange::buffer
        //! Valid Usage:
        //!     * nextSubmit must be valid (see `SIntendedSubmitInfo::valid()`)
        //!     * bufferRange must be valid (see `SBufferRange::isValid()`)
        //!     * data must not be nullptr
        template<typename IntendedSubmitInfo> requires std::is_same_v<std::decay_t<IntendedSubmitInfo>,SIntendedSubmitInfo>
        inline bool updateBufferRangeViaStagingBuffer(IntendedSubmitInfo&& nextSubmit, const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data)
        {
            if (!bufferRange.isValid() || !bufferRange.buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_TRANSFER_DST_BIT))
            {
                m_logger.log("Invalid `bufferRange` or buffer has no `EUF_TRANSFER_DST_BIT` usage flag, cannot `updateBufferRangeViaStagingBuffer`!", system::ILogger::ELL_ERROR);
                return false;
            }

            if (!commonTransferValidation(nextSubmit))
                return false;

            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            // TODO: Why did we settle on `/4` ? It definitely wasn't about the uint32_t size!
            const uint32_t optimalTransferAtom = core::min<uint32_t>(limits.maxResidentInvocations*OptimalCoalescedInvocationXferSize,m_defaultUploadBuffer->get_total_size()/4);

            auto cmdbuf = nextSubmit.getScratchCommandBuffer();
            // no pipeline barriers necessary because write and optional flush happens before submit, and memory allocation is reclaimed after fence signal
            for (size_t uploadedSize=0ull; uploadedSize<bufferRange.size;)
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
                if (localOffset!=StreamingTransientDataBufferMT<>::invalid_value)
                {
                    const void* dataPtr = reinterpret_cast<const uint8_t*>(data) + uploadedSize;
                    memcpy(reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer()) + localOffset, dataPtr, subSize);
                }
                else
                {
                    nextSubmit.overflowSubmit();
                    continue; // keep trying again
                }
                // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
                if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate())
                {
                    auto flushRange = AlignedMappedMemoryRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory().memory,localOffset,subSize,limits.nonCoherentAtomSize);
                    m_device->flushMappedMemoryRanges(1u,&flushRange);
                }
                // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
                IGPUCommandBuffer::SBufferCopy copy;
                copy.srcOffset = localOffset;
                copy.dstOffset = bufferRange.offset+uploadedSize;
                copy.size = subSize;
                cmdbuf->copyBuffer(m_defaultUploadBuffer.get()->getBuffer(), bufferRange.buffer.get(), 1u, &copy);
                // this doesn't actually free the memory, the memory is queued up to be freed only after the `scratchSemaphore` reaches a value a future submit will signal
                m_defaultUploadBuffer.get()->multi_deallocate(1u,&localOffset,&allocationSize,nextSubmit.getFutureScratchSemaphore(),&cmdbuf);
                uploadedSize += subSize;
            }
            return true;
        }

        //! This only needs a valid queue in `submit`
        template<typename IntendedSubmitInfo> requires std::is_same_v<std::decay_t<IntendedSubmitInfo>,SIntendedSubmitInfo>
        inline ISemaphore::future_t<core::smart_refctd_ptr<IGPUBuffer>> createFilledDeviceLocalBufferOnDedMem(
            IntendedSubmitInfo&& submit,
            IGPUBuffer::SCreationParams&& params,
            const void* data,
            const std::span<IQueue::SSubmitInfo::SSemaphoreInfo> extraSignalSemaphores={}
        )
        {
            auto buffer = m_device->createBuffer(std::move(params));
            auto mreqs = buffer->getMemoryReqs();
            mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto mem = m_device->allocate(mreqs,buffer.get());

            auto submitSuccess = autoSubmit(
                submit,
                [&](auto& info)->bool
                {
                    return updateBufferRangeViaStagingBuffer(info,asset::SBufferRange<IGPUBuffer>{0u,params.size,core::smart_refctd_ptr(buffer)},data);
                },
                extraSignalSemaphores
            );
            // probably error
            if (!submitSuccess.blocking())
                return {};

            ISemaphore::future_t<core::smart_refctd_ptr<IGPUBuffer>> retval(std::move(buffer));
            static_cast<ISemaphore::future_base_t&>(retval) = std::move(submitSuccess);
            return retval;
        }

        // pipelineBarrierAutoSubmit?

        // --------------
        // downloadBufferRangeViaStagingBuffer
        // --------------
        
        /* callback signature used for downstreaming requests */
        using data_consumption_callback_t = void(const size_t /*dstOffset*/, const void* /*srcPtr*/, const size_t /*size*/);

        struct default_data_consumption_callback_t
        {
            default_data_consumption_callback_t(void* dstPtr) : m_dstPtr(dstPtr) {}

            inline void operator()(const size_t dstOffset, const void* srcPtr, const size_t size)
            {
                uint8_t* dst = reinterpret_cast<uint8_t*>(m_dstPtr) + dstOffset;
                memcpy(dst, srcPtr, size);
            }

            void* m_dstPtr;
        };

        //! Used in downloadBufferRangeViaStagingBuffer multi_deallocate objectsToHold, 
        //! Calls the std::function callback in destructor because allocator will hold on to this object and drop it when it's safe (fence is singnalled and submit has finished)
        class CDownstreamingDataConsumer final : public core::IReferenceCounted
        {
            public:
                CDownstreamingDataConsumer(
                    const IDeviceMemoryAllocation::MemoryRange& copyRange,
                    const std::function<data_consumption_callback_t>& consumeCallback,
                    core::smart_refctd_ptr<IGPUCommandBuffer>&& cmdBuffer,
                    StreamingTransientDataBufferMT<>* downstreamingBuffer,
                    size_t dstOffset=0
                ) : m_copyRange(copyRange)
                    , m_consumeCallback(consumeCallback)
                    , m_cmdBuffer(core::smart_refctd_ptr<IGPUCommandBuffer>(cmdBuffer))
                    , m_downstreamingBuffer(downstreamingBuffer)
                    , m_dstOffset(dstOffset)
                {}

                ~CDownstreamingDataConsumer()
                {
                    assert(m_downstreamingBuffer);
                    auto device = const_cast<ILogicalDevice*>(m_downstreamingBuffer->getBuffer()->getOriginDevice());
                    if (m_downstreamingBuffer->needsManualFlushOrInvalidate())
                    {
                        const auto nonCoherentAtomSize = device->getPhysicalDevice()->getLimits().nonCoherentAtomSize;
                        auto flushRange = AlignedMappedMemoryRange(m_downstreamingBuffer->getBuffer()->getBoundMemory().memory,m_copyRange.offset,m_copyRange.length,nonCoherentAtomSize);
                        device->invalidateMappedMemoryRanges(1u,&flushRange);
                    }
                    // Call the function
                    const uint8_t* copySrc = reinterpret_cast<uint8_t*>(m_downstreamingBuffer->getBufferPointer()) + m_copyRange.offset;
                    m_consumeCallback(m_dstOffset, copySrc, m_copyRange.length);
                }

            private:
                const IDeviceMemoryAllocation::MemoryRange m_copyRange;
                std::function<data_consumption_callback_t> m_consumeCallback;
                const core::smart_refctd_ptr<const IGPUCommandBuffer> m_cmdBuffer; // because command buffer submiting the copy shouldn't go out of scope when copy isn't finished
                StreamingTransientDataBufferMT<>* m_downstreamingBuffer;
                const size_t m_dstOffset;
        };

        //! Calls the callback to copy the data to a destination Offset
        //! * IMPORTANT: To make all the callbacks execute, `IUtility::getDefaultDownStreamingBuffer()->cull_frees()` should be called after the `nextSubmit.scratchSemaphore` is signaled.
        //! If the allocation from staging memory fails due to large image size or fragmentation then This function may need to submit the command buffer via the `submissionQueue` and then signal the fence. 
        //! Returns:
        //!     Boolean whether successfully enqueued whole buffer range for download.
        //! Parameters:
        //!     - consumeCallback: it's a std::function called when the data is ready to be copied (see `data_consumption_callback_t`)
        //!     - srcBufferRange: the buffer range (buffer + size) to be copied from.
        //!     - nextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers.
        //!         ** The last command buffer will be used to record the copy commands
        //! Valid Usage:
        //!     * `srcBufferRange` must be valid
        //!     * `nextSubmit` must be valid
        //!     * The commandBuffers should have been allocated from a CommandPool with the same queueFamilyIndex as `nextSubmit.queue`
        //!     * The last command buffer should be in `RECORDING` state.
        //!     * The last command buffer should be must've called "begin()" with `IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT` flag
        //!         The reason is the commands recorded into the command buffer would not be valid for a second submission and the stagingBuffer memory wouldv'e been freed/changed.
        //!     * The last command buffer should be "resettable". See `ICommandBuffer::E_STATE` comments
        //!     * To ensure correct execution order, (if any) all the command buffers except the last one should be in `EXECUTABLE` state.
        //!     * `nextSubmit.queue` must point to a valid `IQueue`
        //!     * `nextSubmit.scratchSemaphore` must be a valid `IQueue::SSubmitInfo::SSemaphoreInfo` (the `value` member must be equal or higher than current and any pending signal operation)
        template<typename IntendedSubmitInfo> requires std::is_same_v<std::decay_t<IntendedSubmitInfo>, SIntendedSubmitInfo>
        inline bool downloadBufferRangeViaStagingBuffer(const std::function<data_consumption_callback_t>& consumeCallback, IntendedSubmitInfo&& nextSubmit, const asset::SBufferRange<IGPUBuffer>& srcBufferRange)
        {
            if (!srcBufferRange.isValid() || !srcBufferRange.buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_TRANSFER_SRC_BIT))
            {
                m_logger.log("Invalid `srcBufferRange` or buffer has no `EUF_TRANSFER_SRC_BIT` usage flag, cannot `downloadBufferRangeViaStagingBuffer`!",system::ILogger::ELL_ERROR);
                return false;
            }

            if (!commonTransferValidation(nextSubmit))
                return false;

            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            // TODO: Why did we settle on `/4` ? It definitely wasn't about the uint32_t size!
            const uint32_t optimalTransferAtom = core::min<uint32_t>(limits.maxResidentInvocations*OptimalCoalescedInvocationXferSize,m_defaultDownloadBuffer->get_total_size()/4);

            auto cmdbuf = nextSubmit.getScratchCommandBuffer();
            // Basically downloadedSize is downloadRecordedIntoCommandBufferSize :D
            for (size_t downloadedSize=0ull; downloadedSize<srcBufferRange.size;)
            {
                const size_t notDownloadedSize = srcBufferRange.size - downloadedSize;
                // how large we can make the allocation
                const uint32_t maxFreeBlock = m_defaultDownloadBuffer->max_size();
                // get allocation size
                const uint32_t allocationSize = getAllocationSizeForStreamingBuffer(notDownloadedSize,m_allocationAlignment,maxFreeBlock,optimalTransferAtom);
                const uint32_t copySize = core::min(allocationSize,notDownloadedSize);

                uint32_t localOffset = StreamingTransientDataBufferMT<>::invalid_value;
                m_defaultDownloadBuffer.get()->multi_allocate(std::chrono::steady_clock::now()+std::chrono::microseconds(500u),1u,&localOffset,&allocationSize,&m_allocationAlignment);
                
                if (localOffset!=StreamingTransientDataBufferMT<>::invalid_value)
                {
                    IGPUCommandBuffer::SBufferCopy copy;
                    copy.srcOffset = srcBufferRange.offset + downloadedSize;
                    copy.dstOffset = localOffset;
                    copy.size = copySize;
                    cmdbuf->copyBuffer(srcBufferRange.buffer.get(),m_defaultDownloadBuffer->getBuffer(),1u,&copy);

                    auto dataConsumer = core::make_smart_refctd_ptr<CDownstreamingDataConsumer>(
                        IDeviceMemoryAllocation::MemoryRange(localOffset,copySize),
                        consumeCallback,
                        core::smart_refctd_ptr<IGPUCommandBuffer>(cmdbuf),
                        m_defaultDownloadBuffer.get(),
                        downloadedSize
                    );
                    m_defaultDownloadBuffer.get()->multi_deallocate(1u,&localOffset,&allocationSize,nextSubmit.getFutureScratchSemaphore(),&dataConsumer.get());

                    downloadedSize += copySize;
                }
                else // but first sumbit the already buffered up copies
                    nextSubmit.overflowSubmit();
            }
            return true;
        }

        //! This function is an specialization of the `downloadBufferRangeViaStagingBufferAutoSubmit` function above.
        //! Additionally waits for the fence
        //! WARNING: This function blocks CPU and stalls the GPU!
        template<typename IntendedSubmitInfo> requires std::is_same_v<std::decay_t<IntendedSubmitInfo>,SIntendedSubmitInfo>
        inline bool downloadBufferRangeViaStagingBufferAutoSubmit(IntendedSubmitInfo&& submit, const asset::SBufferRange<IGPUBuffer>& srcBufferRange, void* data)
        {
            auto lambda = [&](SIntendedSubmitInfo& nextSubmit)->bool
            {
                return downloadBufferRangeViaStagingBuffer(default_data_consumption_callback_t(data),nextSubmit,srcBufferRange);
            };
            if (autoSubmit(submit,lambda).copy<IQueue::RESULT>()!=IQueue::RESULT::SUCCESS)
                return false;

            //! NOTE this method cannot be turned into a pure autoSubmitAndBlock + lambda because there's stuff to do AFTER the semaphore wait~! 
            m_defaultDownloadBuffer->cull_frees();
            return true;
        }

        // --------------
        // buildAccelerationStructures
        // --------------
#if 0 // TODO: port later when we have an example
        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline void buildAccelerationStructures(IQueue* queue, const core::SRange<const IGPUAccelerationStructure::DeviceBuildGeometryInfo>& pInfos, IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
        {
            core::smart_refctd_ptr<IGPUCommandPool> pool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
            auto fence = m_device->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
            core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
            m_device->createCommandBuffers(pool.get(), IGPUCommandBuffer::LEVEL::PRIMARY, 1u, &cmdbuf);
            IQueue::SSubmitInfo submit;
            {
                submit.commandBufferCount = 1u;
                submit.commandBuffers = &cmdbuf.get();
                submit.waitSemaphoreCount = 0u;
                submit.pWaitDstStageMask = nullptr;
                submit.pWaitSemaphores = nullptr;
            }

            cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            cmdbuf->buildAccelerationStructures(pInfos,ppBuildRangeInfos);
            cmdbuf->end();

            queue->submit(1u, &submit, fence.get());
        
            m_device->blockForFences(1u,&fence.get());
        }
#endif
        // --------------
        // updateImageViaStagingBuffer
        // --------------
        //! Copies `srcBuffer` to stagingBuffer and Records the commands needed to copy the image from stagingBuffer to `dstImage`
        //! If the allocation from staging memory fails due to large image size or fragmentation then this function may need to submit the command buffer
        //! on the `nextSubmit.queue` and then signal the scratch semaphore. 
        //! Returns:
        //!     True on successful recording of copy commands and handling of overflows if any, and false on failure for any reason.
        //!     Make sure to submit with `nextSubmit.popSubmit()` after this function returns.
        //! Parameters:
        //!     - srcData: source data to copy image from
        //!     - srcFormat: The image format the `srcBuffer` is laid out in memory.
        //          In the case that dstImage has a different format this function will make the necessary conversions.
        //          If `srcFormat` is EF_UNKOWN, it will be assumed to have the same format `dstImage` was created with.
        //!     - dstImage: destination image to copy image to
        //!     - currentDstImageLayout: the image layout of `dstImage` will be at the time of submission of the copy commands
        //!     - regions: regions to copy `srcBuffer`
        //!     - nextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers with, it will be modified if overflow occurred @see SIntendedSubmitInfo
        //! Valid Usage:
        //!     * nextSubmit must be valid (see `SIntendedSubmitInfo::valid()`)
        //!     * srcBuffer must be valid (see `SBufferRange::isValid()`)
        //!     * dstImage must point to a valid IGPUImage
        //!     * regions.size() must be > 0
        bool updateImageViaStagingBuffer(
            SIntendedSubmitInfo& nextSubmit, const void* srcData, asset::E_FORMAT srcFormat, 
            IGPUImage* dstImage, IGPUImage::LAYOUT currentDstImageLayout,
            const std::span<const asset::IImage::SBufferCopy> regions
        );

        // wrapper for old API
        [[deprecated]] inline bool updateImageViaStagingBuffer(
            SIntendedSubmitInfo& submit, asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat,
            IGPUImage* dstImage, IGPUImage::LAYOUT currentDstImageLayout,
            const core::SRange<const asset::IImage::SBufferCopy>& regions
        )
        {
            return updateImageViaStagingBuffer(submit,srcBuffer->getPointer(),srcFormat,dstImage,currentDstImageLayout,{regions.begin(),regions.size()});
        }

        // wrapper for R-value reference submit infos
        template<typename... Args>
        inline bool updateImageViaStagingBuffer(SIntendedSubmitInfo&& nextSubmit, Args&&... args)
        {
            return updateImageViaStagingBuffer(nextSubmit,std::forward<Args>(args)...);
        }
        
        //! Auto-Submit utility
        template<typename IntendedSubmitInfo, typename... Args> requires std::is_same_v<std::decay_t<IntendedSubmitInfo>,SIntendedSubmitInfo>
        inline ISemaphore::future_t<IQueue::RESULT> updateImageViaStagingBufferAutoSubmit(IntendedSubmitInfo&& submit, Args&&... args)
        {
            return autoSubmit(submit,[&](SIntendedSubmitInfo& nextSubmit)->bool{return updateImageViaStagingBuffer(nextSubmit,std::forward<Args>(args)...);});
        }

        // --------------
        // downloadImageViaStagingBuffer
        // --------------
        bool downloadImageViaStagingBuffer(
            SIntendedSubmitInfo& nextSubmit, const IGPUImage* srcImage, const IGPUImage::LAYOUT currentSrcImageLayout,
            void* dest, const std::span<const asset::IImage::SBufferCopy> regions
        );

        // wrapper for R-value reference submit infos
        template<typename... Args>
        inline bool downloadImageViaStagingBuffer(SIntendedSubmitInfo&& nextSubmit, Args&&... args)
        {
            return downloadImageViaStagingBuffer(nextSubmit,std::forward<Args>(args)...);
        }

    protected:
        //
        inline bool commonTransferValidation(const SIntendedSubmitInfo& intendedNextSubmit)
        {
            if (!intendedNextSubmit.valid())
            {
                m_logger.log("Invalid `intendedNextSubmit`.", nbl::system::ILogger::ELL_ERROR);
                return false;
            }

            assert(intendedNextSubmit.queue);
            auto queueFamProps = m_device->getPhysicalDevice()->getQueueFamilyProperties()[intendedNextSubmit.queue->getFamilyIndex()];
            if (!queueFamProps.queueFlags.hasFlags(IQueue::FAMILY_FLAGS::TRANSFER_BIT))
            {
                m_logger.log("Invalid `intendedNextSubmit.queue` is not capable of transfer operations!", nbl::system::ILogger::ELL_ERROR);
                return false;
            }

            return true;
        }

        // The application must round down the start of the range to the nearest multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize,
        // and round the end of the range up to the nearest multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize.
        static ILogicalDevice::MappedMemoryRange AlignedMappedMemoryRange(IDeviceMemoryAllocation* mem, const size_t& off, const size_t& len, size_t nonCoherentAtomSize)
        {
            ILogicalDevice::MappedMemoryRange range = {};
            range.memory = mem;
            range.offset = core::alignDown(off, nonCoherentAtomSize);
            range.length = core::min(core::alignUp(len, nonCoherentAtomSize), mem->getAllocationSize());
            return range;
        }


        core::smart_refctd_ptr<ILogicalDevice> m_device;

        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultDownloadBuffer;
        core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultUploadBuffer;

#if 0 // TODO: port
        core::smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
        core::smart_refctd_ptr<CScanner> m_scanner;
#endif
};

}

#endif