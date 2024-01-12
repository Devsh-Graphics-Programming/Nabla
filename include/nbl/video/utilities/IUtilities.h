// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_I_UTILITIES_H_INCLUDED_
#define _NBL_VIDEO_I_UTILITIES_H_INCLUDED_

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IGPUBuffer.h"
#include "nbl/video/IGPUImage.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"
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
        IUtilities(core::smart_refctd_ptr<ILogicalDevice>&& device, nbl::system::logger_opt_smart_ptr&& logger=nullptr, const uint32_t downstreamSize=0x4000000u, const uint32_t upstreamSize=0x4000000u)
            : m_device(std::move(device)), m_logger(std::move(logger))
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

            constexpr uint32_t OptimalCoalescedInvocationXferSize = sizeof(uint32_t);
            const uint32_t bufferOptimalTransferAtom = limits.maxResidentInvocations * OptimalCoalescedInvocationXferSize;
            const uint32_t maxImageOptimalTransferAtom = limits.maxResidentInvocations * asset::TexelBlockInfo(asset::EF_R64G64B64A64_SFLOAT).getBlockByteSize() * minImageTransferGranularityVolume;
            const uint32_t minImageOptimalTransferAtom = limits.maxResidentInvocations * asset::TexelBlockInfo(asset::EF_R8_UINT).getBlockByteSize();
            const uint32_t maxOptimalTransferAtom = core::max(bufferOptimalTransferAtom,maxImageOptimalTransferAtom);
            const uint32_t minOptimalTransferAtom = core::min(bufferOptimalTransferAtom,minImageOptimalTransferAtom);

            // allocationAlignment <= minBlockSize <= minOptimalTransferAtom <= maxOptimalTransferAtom <= stagingBufferSize/4
            assert(m_allocationAlignment <= minStreamingBufferAllocationSize);
            assert(m_allocationAlignmentForBufferImageCopy <= minStreamingBufferAllocationSize);

            assert(minStreamingBufferAllocationSize <= minOptimalTransferAtom);

            assert(maxOptimalTransferAtom*OptimalCoalescedInvocationXferSize <= upstreamSize);
            assert(maxOptimalTransferAtom*OptimalCoalescedInvocationXferSize <= downstreamSize);

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
        
        //! Struct meant to be used with any Utility (not just `IUtilities`) which exhibits "submit on overflow" behaviour.
        //! Such functions are non-blocking (unless overflow) and take `SIntendedSubmitInfo` by reference and patch it accordingly. 
        //! MAKE SURE to do a submit to `queue` by yourself with a submit info obtained by casting `this` to `IQueue::SSubmitInfo` !
        //!     for example: in the case the `frontHalf.waitSemaphores` were already waited upon, the struct will be modified to have it's `waitSemaphores` emptied.
        struct SIntendedSubmitInfo final
        {
            public:
                inline bool valid() const
                {
                    if (!frontHalf.valid() || frontHalf.commandBuffers.empty() || signalSemaphores.empty())
                        return false;
                    const auto* scratch = frontHalf.getScratchCommandBuffer();
                    // Must be resettable so we can end, submit, wait, reset and continue recording commands into it as-if nothing happened 
                    if (!scratch->isResettable())
                        return false;
                    // It makes no sense to reuse the same commands for a second submission.
                    // Moreover its dangerous because the utilities record their own internal commands which might use subresources for which
                    // frees have already been latched on the scratch semaphore you must signal anyway.
                    if (!scratch->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                        return false;
                    if (scratch->getState()!=IGPUCommandBuffer::STATE::INITIAL)
                        return false;
                    return true;
                }

                inline ISemaphore::SWaitInfo getScratchSemaphoreNextWait() const {return {signalSemaphores.front().semaphore,signalSemaphores.front().value};}

                inline operator IQueue::SSubmitInfo() const
                {
                    return {
                        .waitSemaphores = frontHalf.waitSemaphores,
                        .commandBuffers = frontHalf.commandBuffers,
                        .signalSemaphores = signalSemaphores
                    };
                }

                // One thing you might notice is that this results in a few implicit Memory and Execution Dependencies
                // So there's a little bit of non-deterministic behaviour we won't fight (will not insert a barrier every time you "could-have" overflown)
                inline void overflowSubmit()
                {
                    auto cmdbuf = frontHalf.getScratchCommandBuffer();
                    auto& scratchSemaphore = signalSemaphores.front();
                    // but first sumbit the already buffered up copies
                    cmdbuf->end();
                    IQueue::SSubmitInfo submit = *this;
                    // we only signal the last semaphore which is used as scratch
                    submit.signalSemaphores = {&scratchSemaphore,1};
                    assert(submit.isValid());
                    frontHalf.queue->submit({&submit,1});
                    // We wait (stall) on the immediately preceeding submission timeline semaphore signal value and increase it for the next signaller
                    {
                        const ISemaphore::SWaitInfo info = {scratchSemaphore.semaphore,scratchSemaphore.value++};
                        const_cast<ILogicalDevice*>(cmdbuf->getOriginDevice())->blockForSemaphores({&info,1});
                    }
                    // we've already waited on the Host for the semaphores, no use waiting twice
                    frontHalf.waitSemaphores = {};
                    // since all the commandbuffers have submitted already we only reuse the last one
                    frontHalf.commandBuffers = {&frontHalf.commandBuffers.back(),1};
                    // we will still signal the same set in the future
                    cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
                    cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
                }


                //! The last CommandBuffer will be used to record the copy commands    
                struct SFrontHalf final
                {
                    //! Need a valid queue and all the command buffers except the last one should be in `EXECUTABLE` state.
                    inline bool valid() const
                    {
                        if (!queue)
                            return false;
                        if (!commandBuffers.empty())
                        for (size_t i=0; i<commandBuffers.size()-1; i++)
                        if (commandBuffers[i].cmdbuf->getState()==IGPUCommandBuffer::STATE::EXECUTABLE)
                            return false;
                        return true;
                    }

                    //! Little class to hold the storage for the modified commandbuffer span until submission time.
                    class CRAIISpanPatch final : core::Uncopyable
                    {
                        public:
                            inline ~CRAIISpanPatch()
                            {
                                toNullify->commandBuffers = {};
                            }
                            inline CRAIISpanPatch(CRAIISpanPatch&& other) : CRAIISpanPatch() {operator=(std::move(other));}
                            inline CRAIISpanPatch& operator=(CRAIISpanPatch&& rhs)
                            {
                                commandBuffersStorage = std::move(rhs.commandBuffersStorage);
                                return *this;
                            }

                            inline operator bool() const {return m_recordingCommandBuffer.get();}

                        private:
                            friend SFrontHalf;
                            inline CRAIISpanPatch() = default;
                            inline CRAIISpanPatch(SFrontHalf* _toNull) : commandBuffersStorage(_toNull->commandBuffers.size()+1), toNullify(_toNull) {}

                            core::vector<IQueue::SSubmitInfo::SCommandBufferInfo> commandBuffersStorage;
                            // If we made a new commandbuffer we need to nullify the span so it doesn't point at stale mem
                            SFrontHalf* toNullify = nullptr;
                            // If new one made, then need to hold reference to it, else its just an extra ref, but whatever
                            core::smart_refctd_ptr<IGPUCommandBuffer> m_recordingCommandBuffer;
                    };
                    //! Patches the `commandBuffers` and then makes sure the last command buffer is resettable, in recording state begun with ONE_TIME_SUBMIT
                    //! If we can't make the last cmdbuffer that way, we make a new one and add it onto the end (hence the name "patching")
                    //! If `commandBuffers.empty()`, it will create an implicit command buffer to use for recording commands,
                    //! else if the last command buffer is not feasible to use as scratch for whatever reason,
                    //! it will add another temporary command buffer to end of `commandBuffers` and use it for recording.
                    //! WARNING: If patching occurs:
                    //!     - a submission must occur before the return value goes out of scope!
                    //!     - if `!commandBuffers.empty()`, the last CommandBuffer won't be in the same state as it was before entering the function,
                    //!         because it needs to be `end()`ed before the submission
                    //!     - the destructor of the return value will clear `commandBuffers` span
                    //! For more info on command buffer states See `ICommandBuffer::E_STATE` comments.
                    [[nodiscard("The RAII object returned by `patch()` provides lifetimes to your spans!")]]
                    inline CRAIISpanPatch patch()
                    {
                        if (auto* candidateScratch = getScratchCommandBuffer(); candidateScratch && candidateScratch->isResettable())
                        switch(candidateScratch->getState())
                        {
                            case IGPUCommandBuffer::STATE::INITIAL:
                                if (!candidateScratch->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                                    break;
                                [[fallthrough]];
                            case IGPUCommandBuffer::STATE::RECORDING:
                                if (!candidateScratch->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                                    break;
                                {
                                    CRAIISpanPatch retval;
                                    retval.m_recordingCommandBuffer = core::smart_refctd_ptr<IGPUCommandBuffer>(candidateScratch);
                                    return retval;
                                }
                                break;
                            default:
                                break;
                        }

                        CRAIISpanPatch retval(this);
                        std::copy(commandBuffers.begin(),commandBuffers.end(),retval.commandBuffersStorage.begin());
                        {
                            auto pool = const_cast<ILogicalDevice*>(queue->getOriginDevice())->createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
                            if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&retval.m_recordingCommandBuffer,1}))
                                return {};
                            if (!retval.m_recordingCommandBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
                                return {};
                            retval.commandBuffersStorage.back().cmdbuf = retval.m_recordingCommandBuffer.get();
                        }
                        commandBuffers = retval.commandBuffersStorage;
                        return retval;
                    }

                    // Use the last command buffer in intendedNextSubmit, it should be in recording state
                    inline IGPUCommandBuffer* getScratchCommandBuffer() {return commandBuffers.empty() ? nullptr:commandBuffers.back().cmdbuf;}
                    inline const IGPUCommandBuffer* getScratchCommandBuffer() const {return commandBuffers.empty() ? nullptr:commandBuffers.back().cmdbuf;}

                    // This parameter is required but may be unused if there is no need to submit
                    IQueue* queue = {};
                    // Use this parameter to wait for previous operations to finish before whatever commands the Utility you're using records
                    std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores = {};
                    // Fill the commandbuffers you want to run before the first command the Utility records to run in the same submit,
                    // for example baked command buffers with pipeline barrier commands.
                    // Also remember that even though the last CommandBuffer is scratch, it you can record commands into it as well.
                    std::span<IQueue::SSubmitInfo::SCommandBufferInfo> commandBuffers = {};
                } frontHalf = {};
                //! The first Semaphore will be used as a scratch, so don't choose the values for waits and signals yourself as we can advance the counter an arbitrary amount!
                //! You can actually examine the change in `signalSemaphore.front().value` to figure out how many overflows occurred.
                //! This semaphore is needed to "stitch together" additional submits if they occur so they occur before and after the original intended waits and signals.
                //! We use the first semaphore to keep the intended order of original semaphore signal and waits unchanged no matter how many overflows occur.
                //! You do however, NEED TO KEEP IT in the signal set of the last submit you're supposed to do manually, this allows freeing any resources used
                //! after the submit is done, indicating that your streaming routine is done.  
                //! * Also use this parameter to signal new semaphores so that other submits know your Utility method is done.
                std::span<IQueue::SSubmitInfo::SSemaphoreInfo> signalSemaphores = {};

            private:
                friend class IUtilities;
                static const char* ErrorText;
        };


        //! This method lets you wrap any other function following the "submit on overflow" pattern with the final submission
        //! to `intendedSubmit.queue` happening automatically, no need for the user to handle the submit at the end.
        //! WARNING: Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
        //!  of the `updateBufferRangeViaStagingBufferAutoSubmit` function above.
        //! Parameters:
        //! - `intendedSubmit`: more lax than regular `SIntendedSubmitInfo::valid()`, only needs a valid queue and at least one semaphore to use as scratch and signal.
        //!     if you don't have a commandbuffer usable as scratch as the last one, we'll patch internally.
        inline IQueue::RESULT autoSubmit(SIntendedSubmitInfo& intendedSubmit, const std::function<bool(SIntendedSubmitInfo&)>& what)
        {
            if (!intendedSubmit.frontHalf.valid() || intendedSubmit.signalSemaphores.empty())
            {
                // TODO: log error
                return IQueue::RESULT::OTHER_ERROR;
            }

            const auto raii = intendedSubmit.frontHalf.patch();
            if (!raii)
            {
                // TODO: log error
                return IQueue::RESULT::OTHER_ERROR;
            }

            if (!what(intendedSubmit))
                return IQueue::RESULT::OTHER_ERROR;
            intendedSubmit.frontHalf.getScratchCommandBuffer()->end();

            const IQueue::SSubmitInfo submit = intendedSubmit;
            if (const auto error=intendedSubmit.frontHalf.queue->submit({&submit,1}); error!=IQueue::RESULT::SUCCESS)
                return error;
            // If there's any subsequent submit in a chain, make sure it waits for this one to finish
            // (to achieve a command ordering in the cmdbuffer transparent to overflow submits)
            intendedSubmit.frontHalf.waitSemaphores = {&intendedSubmit.signalSemaphores.front(),1};
            intendedSubmit.signalSemaphores = {};
            return IQueue::RESULT::SUCCESS;
        }

        //! This function is an specialization of the `autoSubmit` function above, it will additionally wait on the Host (CPU) for the final submit to finish.
        //! WARNING: This function blocks CPU and stalls the GPU!
        inline bool autoSubmitAndBlock(const SIntendedSubmitInfo::SFrontHalf& submit, const std::function<bool(SIntendedSubmitInfo&)>& what)
        {            
            auto semaphore = m_device->createSemaphore(0);
            // so we begin latching everything on the value of 1, but if we overflow it increases
            IQueue::SSubmitInfo::SSemaphoreInfo info = {semaphore.get(),1};

            SIntendedSubmitInfo intendedSubmit = {.frontHalf=submit,.signalSemaphores={&info,1}};
            if (autoSubmit(intendedSubmit,what)!=IQueue::RESULT::SUCCESS)
                return false;
            
            // Watch carefully and note that we might not be waiting on the value of `1` for why @see `SIntendedSubmitInfo::signalSemaphores`
            const ISemaphore::SWaitInfo waitInfo = {info.semaphore,info.value};
            m_device->blockForSemaphores({&waitInfo,1});
            return true;
        }  

        // --------------
        // updateBufferRangeViaStagingBuffer
        // --------------

        //! Copies `data` to stagingBuffer and Records the commands needed to copy the data from stagingBuffer to `bufferRange.buffer`
        //! If the allocation from staging memory fails due to large buffer size or fragmentation then This function may need to submit the command buffer via the `submissionQueue`. 
        //! Returns:
        //!     True on successful recording of copy commands and handling of overflows, false on failure for any reason.
        //! Parameters:
        //!     - nextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers with, it will be patched if overflow occurred @see SIntendedSubmitInfo
        //!     - bufferRange: contains offset + size into bufferRange::buffer that will be copied from `data` (offset doesn't affect how `data` is accessed)
        //!     - data: raw pointer to data that will be copied to bufferRange::buffer
        //! Valid Usage:
        //!     * nextSubmit must be valid (see `SIntendedSubmitInfo::valid()`)
        //!     * bufferRange must be valid (see `SBufferRange::isValid()`)
        //!     * data must not be nullptr
        inline bool updateBufferRangeViaStagingBuffer(SIntendedSubmitInfo& nextSubmit, const asset::SBufferRange<IGPUBuffer>& bufferRange, const void* data)
        {
            if (!bufferRange.isValid() || !bufferRange.buffer->getCreationParams().usage.hasFlags(asset::IBuffer::EUF_TRANSFER_DST_BIT))
            {
                m_logger.log("Invalid `bufferRange` or buffer has no `EUF_TRANSFER_DST_BIT` usage flag, cannot `updateBufferRangeViaStagingBuffer`!", system::ILogger::ELL_ERROR);
                return false;
            }

            if (!nextSubmit.valid())
            {
                m_logger.log(nextSubmit.ErrorText,system::ILogger::ELL_ERROR);
                return false;
            }

            const auto& limits = m_device->getPhysicalDevice()->getLimits();
            const uint32_t optimalTransferAtom = limits.maxResidentInvocations * sizeof(uint32_t);

            auto cmdbuf = nextSubmit.frontHalf.getScratchCommandBuffer();
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
                if (localOffset != StreamingTransientDataBufferMT<>::invalid_value)
                {
                    const void* dataPtr = reinterpret_cast<const uint8_t*>(data) + uploadedSize;
                    memcpy(reinterpret_cast<uint8_t*>(m_defaultUploadBuffer->getBufferPointer()) + localOffset, dataPtr, subSize);
                }
                // keep trying again
                if (localOffset == StreamingTransientDataBufferMT<>::invalid_value)
                {
                    nextSubmit.overflowSubmit();
                    continue;
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
                m_defaultUploadBuffer.get()->multi_deallocate(1u,&localOffset,&allocationSize,nextSubmit.getScratchSemaphoreNextWait(),&cmdbuf);
                uploadedSize += subSize;
            }
            return true;
        }

        //! WARNING: This function blocks the CPU and stalls the GPU!
        inline core::smart_refctd_ptr<IGPUBuffer> createFilledDeviceLocalBufferOnDedMem(const SIntendedSubmitInfo::SFrontHalf& submit, IGPUBuffer::SCreationParams&& params, const void* data)
        {
            auto buffer = m_device->createBuffer(std::move(params));
            auto mreqs = buffer->getMemoryReqs();
            mreqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            auto mem = m_device->allocate(mreqs,buffer.get());
            if (!autoSubmitAndBlock(submit,[&](auto& info){return updateBufferRangeViaStagingBuffer(info,asset::SBufferRange<IGPUBuffer>{0u,params.size,core::smart_refctd_ptr(buffer)},data);}))
                return nullptr;
            return buffer;
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
#if 0 // TODO: port
        //! Calls the callback to copy the data to a destination Offset
        //! * IMPORTANT: To make the copies ready, IUtility::getDefaultDownStreamingBuffer()->cull_frees() should be called after the `submissionFence` is signaled.
        //! If the allocation from staging memory fails due to large image size or fragmentation then This function may need to submit the command buffer via the `submissionQueue` and then signal the fence. 
        //! Returns:
        //!     IQueue::SSubmitInfo to use for command buffer submission instead of `intendedNextSubmit`. 
        //!         for example: in the case the `SSubmitInfo::waitSemaphores` were already signalled, the new SSubmitInfo will have it's waitSemaphores emptied from `intendedNextSubmit`.
        //!     Make sure to submit with the new SSubmitInfo returned by this function
        //! Parameters:
        //!     - consumeCallback: it's a std::function called when the data is ready to be copied (see `data_consumption_callback_t`)
        //!     - srcBufferRange: the buffer range (buffer + size) to be copied from.
        //!     - intendedNextSubmit:
        //!         Is the SubmitInfo you intended to submit your command buffers.
        //!         ** The last command buffer will be used to record the copy commands
        //!     - submissionQueue: IQueue used to submit, when needed. 
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
        //!     * The last command buffer should be must've called "begin()" with `IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT` flag
        //!         The reason is the commands recorded into the command buffer would not be valid for a second submission and the stagingBuffer memory wouldv'e been freed/changed.
        //!     * The last command buffer should be "resettable". See `ICommandBuffer::E_STATE` comments
        //!     * To ensure correct execution order, (if any) all the command buffers except the last one should be in `EXECUTABLE` state.
        //!     * submissionQueue must point to a valid IQueue
        //!     * submissionFence must point to a valid IGPUFence
        //!     * submissionFence must be in `UNSIGNALED` state
        [[nodiscard("Use The New IQueue::SubmitInfo")]] inline IQueue::SSubmitInfo downloadBufferRangeViaStagingBuffer(
            const std::function<data_consumption_callback_t>& consumeCallback, const asset::SBufferRange<IGPUBuffer>& srcBufferRange,
            IQueue* submissionQueue, IGPUFence* submissionFence, IQueue::SSubmitInfo intendedNextSubmit = {}
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

            assert(cmdbuf->getState() == IGPUCommandBuffer::STATE::RECORDING && cmdbuf->isResettable());
            assert(cmdbuf->getRecordingFlags().hasFlags(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT));

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
                    IGPUCommandBuffer::SBufferCopy copy;
                    copy.srcOffset = srcBufferRange.offset + downloadedSize;
                    copy.dstOffset = localOffset;
                    copy.size = copySize;
                    cmdbuf->copyBuffer(srcBufferRange.buffer.get(), m_defaultDownloadBuffer.get()->getBuffer(), 1u, &copy);

                    auto dataConsumer = core::make_smart_refctd_ptr<CDownstreamingDataConsumer>(
                        IDeviceMemoryAllocation::MemoryRange(localOffset,copySize),
                        consumeCallback,
                        core::smart_refctd_ptr<IGPUCommandBuffer>(cmdbuf),
                        m_defaultDownloadBuffer.get(),
                        downloadedSize
                    );
                    m_defaultDownloadBuffer.get()->multi_deallocate(1u, &localOffset, &allocationSize, core::smart_refctd_ptr<IGPUFence>(submissionFence), &dataConsumer.get());

                    downloadedSize += copySize;
                }
                else
                {
                    // but first sumbit the already buffered up copies
                    cmdbuf->end();
                    IQueue::SSubmitInfo submit = intendedNextSubmit;
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
                    cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
                    cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
                }
            }
            return intendedNextSubmit;
        }

        //! This function is an specialization of the `downloadBufferRangeViaStagingBufferAutoSubmit` function above.
        //! Additionally waits for the fence
        //! WARNING: This function blocks CPU and stalls the GPU!
        inline void downloadBufferRangeViaStagingBufferAutoSubmit(
            const asset::SBufferRange<IGPUBuffer>& srcBufferRange, void* data,
            IQueue* submissionQueue, const IQueue::SSubmitInfo& submitInfo = {}
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

            //! TODO: NOTE this method cannot be turned into a pure autoSubmitAndBlock + lambda because there's stuff to do AFTER the semaphore wait~! 
            m_defaultDownloadBuffer->cull_frees(); // its while(poll()) {} now IIRC
        }
#endif        
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
#if 0 // TODO: port
        //! Copies `srcBuffer` to stagingBuffer and Records the commands needed to copy the image from stagingBuffer to `dstImage`
        //! If the allocation from staging memory fails due to large image size or fragmentation then This function may need to submit the command buffer via the `submissionQueue` and then signal the fence. 
        //! Returns:
        //!     IQueue::SSubmitInfo to use for command buffer submission instead of `intendedNextSubmit`. 
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
        //!     - submissionQueue: IQueue used to submit, when needed. 
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
        //!     * The last command buffer should be must've called "begin()" with `IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT` flag
        //!         The reason is the commands recorded into the command buffer would not be valid for a second submission and the stagingBuffer memory wouldv'e been freed/changed.
        //!     * The last command buffer should be "resettable". See `ICommandBuffer::E_STATE` comments
        //!     * To ensure correct execution order, (if any) all the command buffers except the last one should be in `EXECUTABLE` state.
        //!     * submissionQueue must point to a valid IQueue
        //!     * submissionFence must point to a valid IGPUFence
        //!     * submissionFence must be in `UNSIGNALED` state
        //!     ** IUtility::getDefaultUpStreamingBuffer()->cull_frees() should be called before reseting the submissionFence and after `submissionFence` is signaled. 
        [[nodiscard("Use The New IQueue::SubmitInfo")]] IQueue::SSubmitInfo updateImageViaStagingBuffer(
            asset::ICPUBuffer const* srcBuffer, asset::E_FORMAT srcFormat, video::IGPUImage* dstImage, IGPUImage::LAYOUT currentDstImageLayout, const core::SRange<const asset::IImage::SBufferCopy>& regions,
            IQueue* submissionQueue, IGPUFence* submissionFence, IQueue::SSubmitInfo intendedNextSubmit);
#endif

    protected:        
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

class ImageRegionIterator
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