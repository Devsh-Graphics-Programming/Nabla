#ifndef __NBL_I_GPU_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_I_GPU_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/CThreadSafeGPUQueueAdapter.h"
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUCommandPool.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/ISwapchain.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/video/IGPUPipelineCache.h"
#include "nbl/video/EApiType.h"
#include "nbl/video/alloc/StreamingTransientDataBuffer.h"
#include "nbl/video/CPropertyPoolHandler.h"
#include "nbl/video/IDeferredOperation.h"

namespace nbl::video
{

class ILogicalDevice : public core::IReferenceCounted
{
public:
    struct SQueueCreationParams
    {
        IGPUQueue::E_CREATE_FLAGS flags;
        uint32_t familyIndex;
        uint32_t count;
        const float* priorities;
    };
    struct SCreationParams
    {
        uint32_t queueParamsCount;
        const SQueueCreationParams* queueCreateInfos;
        // ???:
        //uint32_t enabledExtensionCount;
        //const char* const* ppEnabledExtensionNames;
        //const VkPhysicalDeviceFeatures* pEnabledFeatures;
    };

    struct SDescriptorSetCreationParams
    {
        IDescriptorPool* descriptorPool;
        uint32_t descriptorSetCount;
        IGPUDescriptorSetLayout** pSetLayouts;
    };

    struct SBindBufferMemoryInfo
    {
        IGPUBuffer* buffer;
        IDriverMemoryAllocation* memory;
        size_t offset;
    };
    struct SBindImageMemoryInfo
    {
        IGPUImage* image;
        IDriverMemoryAllocation* memory;
        size_t offset;
    };

    virtual ~ILogicalDevice()
    {
        if (m_queues && m_queues->empty())
        {
            for (uint32_t i = 0u; i < m_queues->size(); ++i)
                delete (*m_queues)[i];
        }
    }

    inline IPhysicalDevice* getPhysicalDevice() const { return m_physicalDevice; }

    E_API_TYPE getAPIType() const;

    inline IGPUQueue* getQueue(uint32_t _familyIx, uint32_t _ix)
    {
        const uint32_t offset = (*m_offsets)[_familyIx];
        return (*m_queues)[offset+_ix]->getUnderlyingQueue();
    }

    // Using the same queue as both a threadsafe queue and a normal queue invalidates the safety.
    inline CThreadSafeGPUQueueAdapter* getThreadSafeQueue(uint32_t _familyIx, uint32_t _ix)
    {
        const uint32_t offset = (*m_offsets)[_familyIx];
        return (*m_queues)[offset + _ix];
    }

    inline StreamingTransientDataBufferMT<>* getDefaultUpStreamingBuffer()
    {
        return m_defaultUploadBuffer.get();
    }

    virtual core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() = 0;

    virtual core::smart_refctd_ptr<IGPUEvent> createEvent(IGPUEvent::E_CREATE_FLAGS flags) = 0;
    virtual IGPUEvent::E_STATUS getEventStatus(const IGPUEvent* _event) = 0;
    virtual IGPUEvent::E_STATUS resetEvent(IGPUEvent* _event) = 0;
    virtual IGPUEvent::E_STATUS setEvent(IGPUEvent* _event) = 0;

    virtual core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS _flags) = 0;
    virtual IGPUFence::E_STATUS getFenceStatus(IGPUFence* _fence) = 0;
    virtual void resetFences(uint32_t _count, IGPUFence*const * _fences) = 0;
    virtual IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence* const* _fences, bool _waitAll, uint64_t _timeout) = 0;

    virtual const core::smart_refctd_dynamic_array<std::string> getSupportedGLSLExtensions() const = 0;

    bool createCommandBuffers(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _outCmdBufs)
    {
        if (!_cmdPool->wasCreatedBy(this))
            return false;
        return createCommandBuffers_impl(_cmdPool, _level, _count, _outCmdBufs);
    }
    bool freeCommandBuffers(IGPUCommandBuffer** _cmdbufs, uint32_t _count)
    {
        for (uint32_t i = 0u; i < _count; ++i)
            if (!_cmdbufs[i]->wasCreatedBy(this))
                return false;
        return freeCommandBuffers_impl(_cmdbufs, _count);
    }

    virtual core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, std::underlying_type_t<IGPUCommandPool::E_CREATE_FLAGS> flags) = 0;
    virtual core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) = 0;

    core::smart_refctd_ptr<IGPUFramebuffer> createGPUFramebuffer(IGPUFramebuffer::SCreationParams&& params)
    {
        if (!params.renderpass->wasCreatedBy(this))
            return nullptr;
        if (!IGPUFramebuffer::validate(params))
            return nullptr;
        return createGPUFramebuffer_impl(std::move(params));
    }

    virtual core::smart_refctd_ptr<IGPURenderpass> createGPURenderpass(const IGPURenderpass::SCreationParams& params) = 0;

    static inline IDriverMemoryBacked::SDriverMemoryRequirements getDeviceLocalGPUMemoryReqs()
    {
        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 0;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        return reqs;
    }
    static inline IDriverMemoryBacked::SDriverMemoryRequirements getSpilloverGPUMemoryReqs()
    {
        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 0;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        return reqs;
    }
    static inline IDriverMemoryBacked::SDriverMemoryRequirements getUpStreamingMemoryReqs()
    {
        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 0;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        return reqs;
    }
    static inline IDriverMemoryBacked::SDriverMemoryRequirements getDownStreamingMemoryReqs()
    {
        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 0;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ | IDriverMemoryAllocation::EMCF_CACHED;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        return reqs;
    }
    static inline IDriverMemoryBacked::SDriverMemoryRequirements getCPUSideGPUVisibleGPUMemoryReqs()
    {
        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
        reqs.vulkanReqs.alignment = 0;
        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_NOT_DEVICE_LOCAL;
        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_READ | IDriverMemoryAllocation::EMCF_CAN_MAP_FOR_WRITE | IDriverMemoryAllocation::EMCF_COHERENT | IDriverMemoryAllocation::EMCF_CACHED;
        reqs.prefersDedicatedAllocation = true;
        reqs.requiresDedicatedAllocation = true;
        return reqs;
    }

    //! Best for Mesh data, UBOs, SSBOs, etc.
    virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateDeviceLocalMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) { return nullptr; }

    //! If cannot or don't want to use device local memory, then this memory can be used
    /** If the above fails (only possible on vulkan) or we have perfomance hitches due to video memory oversubscription.*/
    virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateSpilloverMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) { return nullptr; }

    //! Best for staging uploads to the GPU, such as resource streaming, and data to update the above memory with
    virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateUpStreamingMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) { return nullptr; }

    //! Best for staging downloads from the GPU, such as query results, Z-Buffer, video frames for recording, etc.
    virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateDownStreamingMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) { return nullptr; }

    //! Should be just as fast to play around with on the CPU as regular malloc'ed memory, but slowest to access with GPU
    virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateCPUSideGPUVisibleMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& additionalReqs) { return nullptr; }


    //! For memory allocations without the video::IDriverMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the CPU writes to become GPU visible
    void flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges)
    {
        core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges{ pMemoryRanges, pMemoryRanges + memoryRangeCount };
        return flushMappedMemoryRanges(ranges);
    }

    //! Utility wrapper for the pointer based func
    virtual void flushMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) = 0;

    //! For memory allocations without the video::IDriverMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the GPU writes to become CPU visible (slow on OpenGL)
    void invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges)
    {
        core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges{ pMemoryRanges, pMemoryRanges + memoryRangeCount };
        return invalidateMappedMemoryRanges(ranges);
    }

    //! Utility wrapper for the pointer based func
    virtual void invalidateMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) = 0;

    virtual core::smart_refctd_ptr<IGPUBuffer> createGPUBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false) { return nullptr; }

    //! Binds memory allocation to provide the backing for the resource.
    /** Available only on Vulkan, in OpenGL all resources create their own memory implicitly,
    so pooling or aliasing memory for different resources is not possible.
    There is no unbind, so once memory is bound it remains bound until you destroy the resource object.
    Actually all resource classes in OpenGL implement both IDriverMemoryBacked and IDriverMemoryAllocation,
    so effectively the memory is pre-bound at the time of creation.
    \return true on success, always false under OpenGL.*/
    virtual bool bindBufferMemory(uint32_t bindInfoCount, const SBindBufferMemoryInfo* pBindInfos) { return false; }

    //! Creates the buffer, allocates memory dedicated memory and binds it at once.
    inline core::smart_refctd_ptr<IGPUBuffer> createDeviceLocalGPUBufferOnDedMem(size_t size)
    {
        auto reqs = getDeviceLocalGPUMemoryReqs();
        reqs.vulkanReqs.size = size;
        return this->createGPUBufferOnDedMem(reqs, false);
    }

    //! Creates the buffer, allocates memory dedicated memory and binds it at once.
    inline core::smart_refctd_ptr<IGPUBuffer> createSpilloverGPUBufferOnDedMem(size_t size)
    {
        auto reqs = getSpilloverGPUMemoryReqs();
        reqs.vulkanReqs.size = size;
        return this->createGPUBufferOnDedMem(reqs, false);
    }

    //! Creates the buffer, allocates memory dedicated memory and binds it at once.
    inline core::smart_refctd_ptr<IGPUBuffer> createUpStreamingGPUBufferOnDedMem(size_t size)
    {
        auto reqs = getUpStreamingMemoryReqs();
        reqs.vulkanReqs.size = size;
        return this->createGPUBufferOnDedMem(reqs, false);
    }

    //! Creates the buffer, allocates memory dedicated memory and binds it at once.
    inline core::smart_refctd_ptr<IGPUBuffer> createDownStreamingGPUBufferOnDedMem(size_t size)
    {
        auto reqs = getDownStreamingMemoryReqs();
        reqs.vulkanReqs.size = size;
        return this->createGPUBufferOnDedMem(reqs, false);
    }

    //! Creates the buffer, allocates memory dedicated memory and binds it at once.
    inline core::smart_refctd_ptr<IGPUBuffer> createCPUSideGPUVisibleGPUBufferOnDedMem(size_t size)
    {
        auto reqs = getCPUSideGPUVisibleGPUMemoryReqs();
        reqs.vulkanReqs.size = size;
        return this->createGPUBufferOnDedMem(reqs, false);
    }

    //! Low level function used to implement the above, use with caution
    virtual core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false) { return nullptr; }

    //! WARNING: This function blocks the CPU and stalls the GPU!
    inline core::smart_refctd_ptr<IGPUBuffer> createFilledDeviceLocalGPUBufferOnDedMem(IGPUQueue* queue, size_t size, const void* data)
	{
		auto retval = createDeviceLocalGPUBufferOnDedMem(size);
        updateBufferRangeViaStagingBuffer(queue,asset::SBufferRange<IGPUBuffer>{0u,size,retval},data);
		return retval;
	}

    virtual core::smart_refctd_ptr<IGPUShader> createGPUShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) = 0;

    core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt = nullptr)
    {
        if (!_unspecialized->wasCreatedBy(this))
            return nullptr;
        return createGPUSpecializedShader_impl(_unspecialized, _specInfo, _spvopt);
    }

    //! Create a BufferView, to a shader; a fake 1D texture with no interpolation (@see ICPUBufferView)
    core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer)
    {
        if (!_underlying->wasCreatedBy(this))
            return nullptr;
        return createGPUBufferView_impl(_underlying, _fmt, _offset, _size);
    }


    //! Creates an Image (@see ICPUImage)
    virtual core::smart_refctd_ptr<IGPUImage> createGPUImage(asset::IImage::SCreationParams&& params) { return nullptr; }

    //! The counterpart of @see bindBufferMemory for images
    virtual bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) { return false; }

    //! Creates the Image, allocates dedicated memory and binds it at once.
    inline core::smart_refctd_ptr<IGPUImage> createDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params)
    {
        auto reqs = getDeviceLocalGPUMemoryReqs();
        return this->createGPUImageOnDedMem(std::move(params), reqs);
    }

    //!
    virtual core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) = 0;

    // TODO: Some utility in ILogical Device that can upload the image via the streaming buffer just from the regions without creating a whole intermediate huge GPU Buffer
    //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUCommandBuffer* cmdbuf, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions)
    {
        const auto finalLayout = params.initialLayout;
        params.initialLayout = asset::EIL_TRANSFER_DST_OPTIMAL; // TODO: @achal verify my layouts
        auto retval = createDeviceLocalGPUImageOnDedMem(std::move(params));
        assert(cmdbuf->getState()==IGPUCommandBuffer::ES_RECORDING);
        cmdbuf->copyBufferToImage(srcBuffer,retval.get(),finalLayout,regionCount,pRegions);
        return retval;
    }
    //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
        IGPUFence* fence, IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions,
        const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
        const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
    )
    {
        auto cmdpool = createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_TRANSIENT_BIT);
        core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
        createCommandBuffers(cmdpool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
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
        return retval;
    }
    //! WARNING: This function blocks the CPU and stalls the GPU!
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
        IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions,
        const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
        const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
    )
    {
        auto fence = this->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
        auto* fenceptr = fence.get();
        auto retval = createFilledDeviceLocalGPUImageOnDedMem(
            fenceptr,queue,std::move(params),srcBuffer,regionCount,pRegions,
            waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore,
            signalSemaphoreCount,semaphoresToSignal
        );
        waitForFences(1u,&fenceptr,false,9999999999ull);
        return retval;
    }

    //! Remember to ensure a memory dependency between the command recorded here and any users (so fence wait, semaphore when submitting, pipeline barrier or event)
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUCommandBuffer* cmdbuf, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions)
    {
        const auto finalLayout = params.initialLayout;
        params.initialLayout = asset::EIL_TRANSFER_DST_OPTIMAL; // TODO: @achal verify my layouts
        auto retval = createDeviceLocalGPUImageOnDedMem(std::move(params));
        assert(cmdbuf->getState()==IGPUCommandBuffer::ES_RECORDING);
        cmdbuf->copyImage(srcImage,asset::EIL_TRANSFER_SRC_OPTIMAL,retval.get(),finalLayout,regionCount,pRegions);
        return retval;
    }
    //! Don't use this function in hot loops or to do batch updates, its merely a convenience for one-off uploads
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
        IGPUFence* fence, IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions,
        const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
        const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
    )
    {
        auto cmdpool = createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_TRANSIENT_BIT);
        core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
        createCommandBuffers(cmdpool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
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
        return retval;
    }
    //! WARNING: This function blocks the CPU and stalls the GPU!
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(
        IGPUQueue* queue, IGPUImage::SCreationParams&& params, const IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions,
        const uint32_t waitSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToWaitBeforeExecution=nullptr, const asset::E_PIPELINE_STAGE_FLAGS* stagesToWaitForPerSemaphore=nullptr,
        const uint32_t signalSemaphoreCount=0u, IGPUSemaphore* const* semaphoresToSignal=nullptr
    )
    {
        auto fence = this->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
        auto* fenceptr = fence.get();
        auto retval = createFilledDeviceLocalGPUImageOnDedMem(
            fenceptr,queue,std::move(params),srcImage,regionCount,pRegions,
            waitSemaphoreCount,semaphoresToWaitBeforeExecution,stagesToWaitForPerSemaphore,
            signalSemaphoreCount,semaphoresToSignal
        );
        waitForFences(1u,&fenceptr,false,9999999999ull);
        return retval;
    }


    //! Create an ImageView that can actually be used by shaders (@see ICPUImageView)
    core::smart_refctd_ptr<IGPUImageView> createGPUImageView(IGPUImageView::SCreationParams&& params)
    {
        if (!params.image->wasCreatedBy(this))
            return nullptr;
        return createGPUImageView_impl(std::move(params));
    }

    core::smart_refctd_ptr<IGPUDescriptorSet> createGPUDescriptorSet(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout)
    {
        if (!pool->wasCreatedBy(this))
            return nullptr;
        if (!layout->wasCreatedBy(this))
            return nullptr;
        return createGPUDescriptorSet_impl(pool, std::move(layout));
    }

    core::smart_refctd_ptr<IDescriptorPool> createDescriptorPoolForDSLayouts(core::smart_refctd_ptr<IGPUDescriptorSetLayout>* begin, core::smart_refctd_ptr<IGPUDescriptorSetLayout>* end)
    {
        size_t setCount = end - begin;
        std::vector<IDescriptorPool::SDescriptorPoolSize> poolSizes;
        for (auto* curLayout = begin; curLayout != end; curLayout++)
        {
            auto bindings = curLayout->get()->getBindings();
            for (const auto& binding : bindings)
            {
                auto ps = std::find_if(poolSizes.begin(), poolSizes.end(), [&](const IDescriptorPool::SDescriptorPoolSize& poolSize) { return poolSize.type == binding.type; });
                if (ps != poolSizes.end())
                {
                    ++ps->count;
                }
                else
                {
                    poolSizes.push_back(IDescriptorPool::SDescriptorPoolSize { binding.type, 1 });
                }
            }

        }
        
        core::smart_refctd_ptr<IDescriptorPool> dsPool = createDescriptorPool(IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT, setCount, poolSizes.size(), poolSizes.data());
        return dsPool;
    }

    void createGPUDescriptorSets(IDescriptorPool* pool, uint32_t count, const IGPUDescriptorSetLayout** _layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
    {
        core::SRange<const IGPUDescriptorSetLayout*> layouts{ _layouts, _layouts + count };
        createGPUDescriptorSets(pool, layouts, output);
    }
    void createGPUDescriptorSets(IDescriptorPool* pool, core::SRange<const IGPUDescriptorSetLayout*> layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
    {
        uint32_t i = 0u;
        for (const IGPUDescriptorSetLayout* layout_ : layouts)
        {
            auto layout = core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout_);
            output[i++] = createGPUDescriptorSet(pool, std::move(layout));
        }
    }

    //! Fill out the descriptor sets with descriptors
    virtual void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) = 0;

    //! Create a sampler object to use with images
    virtual core::smart_refctd_ptr<IGPUSampler> createGPUSampler(const IGPUSampler::SParams& _params) = 0;

    //! Create a pipeline cache object
    virtual core::smart_refctd_ptr<IGPUPipelineCache> createGPUPipelineCache() { return nullptr; }

    //! Create a descriptor set layout (@see ICPUDescriptorSetLayout)
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end)
    {
        for (auto b = _begin; b != _end; ++b)
        {
            if (b->type == asset::EDT_COMBINED_IMAGE_SAMPLER && b->samplers)
            {
                auto* samplers = b->samplers;
                for (uint32_t i = 0u; i < b->count; ++i)
                    if (!samplers[i]->wasCreatedBy(this))
                        return nullptr;
            }
        }
        return createGPUDescriptorSetLayout_impl(_begin, _end);
    }

    //! Create a pipeline layout (@see ICPUPipelineLayout)
    core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout(
        const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
    )
    {
        if (_layout0 && !_layout0->wasCreatedBy(this))
            return nullptr;
        if (_layout1 && !_layout1->wasCreatedBy(this))
            return nullptr;
        if (_layout2 && !_layout2->wasCreatedBy(this))
            return nullptr;
        if (_layout3 && !_layout3->wasCreatedBy(this))
            return nullptr;
        return createGPUPipelineLayout_impl(_pcRangesBegin, _pcRangesEnd, std::move(_layout0), std::move(_layout1), std::move(_layout2), std::move(_layout3));
    }

    core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
    ) {
        if (_pipelineCache && !_pipelineCache->wasCreatedBy(this))
            return nullptr;
        if (!_layout->wasCreatedBy(this))
            return nullptr;
        if (!_shader->wasCreatedBy(this))
            return nullptr;
        return createGPUComputePipeline_impl(_pipelineCache, std::move(_layout), std::move(_shader));
    }

    bool createGPUComputePipelines(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output
    ) {
        if (pipelineCache && !pipelineCache->wasCreatedBy(this))
            return false;
        for (const auto& ci : createInfos)
            if (!ci.layout->wasCreatedBy(this) || !ci.shader->wasCreatedBy(this))
                return false;
        return createGPUComputePipelines_impl(pipelineCache, createInfos, output);
    }

    bool createGPUComputePipelines(
        IGPUPipelineCache* pipelineCache,
        uint32_t count,
        const IGPUComputePipeline::SCreationParams* createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output
    ) 
    {
        auto ci = core::SRange<const IGPUComputePipeline::SCreationParams>{createInfos, createInfos+count};
        return createGPUComputePipelines(pipelineCache, ci, output);
    }

    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createGPURenderpassIndependentPipeline(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        IGPUSpecializedShader** _shaders, IGPUSpecializedShader** _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams,
        const asset::SBlendParams& _blendParams,
        const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams
    ) {
        if (_pipelineCache && !_pipelineCache->wasCreatedBy(this))
            return nullptr;
        if (!_layout->wasCreatedBy(this))
            return nullptr;
        for (auto s = _shaders; s != _shadersEnd; ++s)
        {
            if (!(*s)->wasCreatedBy(this))
                return nullptr;
        }
        return createGPURenderpassIndependentPipeline_impl(_pipelineCache, std::move(_layout), _shaders, _shadersEnd, _vertexInputParams, _blendParams, _primAsmParams, _rasterParams);
    }

    bool createGPURenderpassIndependentPipelines(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
    ) {
        if (pipelineCache && !pipelineCache->wasCreatedBy(this))
            return false;
        for (const IGPURenderpassIndependentPipeline::SCreationParams& ci : createInfos)
        {
            if (!ci.layout->wasCreatedBy(this))
                return false;
            for (auto& s : ci.shaders)
                if (s && !s->wasCreatedBy(this))
                    return false;
        }
        return createGPURenderpassIndependentPipelines_impl(pipelineCache, createInfos, output);
    }

    bool createGPURenderpassIndependentPipelines(
        IGPUPipelineCache* pipelineCache,
        uint32_t count,
        const IGPURenderpassIndependentPipeline::SCreationParams* createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
    )
    {
        auto ci = core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams>{createInfos, createInfos+count};
        return createGPURenderpassIndependentPipelines(pipelineCache, ci, output);
    }

    core::smart_refctd_ptr<IGPUGraphicsPipeline> createGPUGraphicsPipeline(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params)
    {
        if (pipelineCache && !pipelineCache->wasCreatedBy(this))
            return nullptr;
        if (!params.renderpass->wasCreatedBy(this))
            return nullptr;
        if (!params.renderpassIndependent->wasCreatedBy(this))
            return nullptr;
        if (!IGPUGraphicsPipeline::validate(params))
            return nullptr;
        return createGPUGraphicsPipeline_impl(pipelineCache, std::move(params));
    }

    bool createGPUGraphicsPipelines(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
    {
        if (pipelineCache && !pipelineCache->wasCreatedBy(this))
            return false;
        for (const auto& ci : params)
        {
            if (!ci.renderpass->wasCreatedBy(this))
                return false;
            if (!ci.renderpassIndependent->wasCreatedBy(this))
                return false;
            if (!IGPUGraphicsPipeline::validate(ci))
                return false;
        }
        return createGPUGraphicsPipelines_impl(pipelineCache, params, output);
    }

    bool createGPUGraphicsPipelines(IGPUPipelineCache* pipelineCache, uint32_t count, const IGPUGraphicsPipeline::SCreationParams* params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
    {
        auto ci = core::SRange<const IGPUGraphicsPipeline::SCreationParams>{ params, params + count };
        return createGPUGraphicsPipelines(pipelineCache, ci, output);
    }

    virtual core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) = 0;

    virtual void waitIdle() = 0;

    virtual void* mapMemory(const IDriverMemoryAllocation::MappedMemoryRange& memory, IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG accessHint = IDriverMemoryAllocation::EMCAF_READ_AND_WRITE) = 0;

    virtual void unmapMemory(IDriverMemoryAllocation* memory) = 0;

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
            m_defaultUploadBuffer.get()->multi_place(std::chrono::high_resolution_clock::now() + std::chrono::microseconds(500u), 1u, (const void* const*)&dataPtr, &localOffset, &subSize, &alignment);

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
                waitForFences(1u,&fence,false,9999999999ull);
                waitSemaphoreCount = 0u;
                semaphoresToWaitBeforeOverwrite = nullptr;
                stagesToWaitForPerSemaphore = nullptr;
                // we can reset the fence and commandbuffer because we fully wait for the GPU to finish here
                resetFences(1u,&fence);
                cmdbuf->reset(IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
                cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
                continue;
            }
            // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
            if (m_defaultUploadBuffer.get()->needsManualFlushOrInvalidate())
            {
                IDriverMemoryAllocation::MappedMemoryRange flushRange(m_defaultUploadBuffer.get()->getBuffer()->getBoundMemory(),localOffset,subSize);
                flushMappedMemoryRanges(1u,&flushRange);
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
        core::smart_refctd_ptr<IGPUCommandPool> pool = createCommandPool(queue->getFamilyIndex(),IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
        core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
        createCommandBuffers(pool.get(),IGPUCommandBuffer::EL_PRIMARY,1u,&cmdbuf);
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
        auto fence = this->createFence(static_cast<IGPUFence::E_CREATE_FLAGS>(0));
        updateBufferRangeViaStagingBuffer(fence.get(),queue,bufferRange,data,waitSemaphoreCount,semaphoresToWaitBeforeOverwrite,stagesToWaitForPerSemaphore,signalSemaphoreCount,semaphoresToSignal);
        auto* fenceptr = fence.get();
        waitForFences(1u,&fenceptr,false,9999999999ull);
    }

    virtual core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() = 0;
    // Not implemented stuff:
    //vkCreateGraphicsPipelines // no graphics pipelines yet (just renderpass independent)
    //vkGetBufferMemoryRequirements // wonder how it works with dedicated memory XD
    //vkGetDescriptorSetLayoutSupport
    //vkTrimCommandPool // for this you need to Optimize OpenGL commandrecording to use linked list
    //vkGetPipelineCacheData //as pipeline cache method?? (why not)
    //vkMergePipelineCaches //as pipeline cache method (why not)
    //vkCreateQueryPool //????
    //vkCreateShaderModule //????
#if 0
    //!
    virtual CPropertyPoolHandler* getDefaultPropertyPoolHandler() const
    {
        return m_propertyPoolHandler;
    }
#endif
protected:
    ILogicalDevice(IPhysicalDevice* physicalDevice, const SCreationParams& params) : m_physicalDevice(physicalDevice)
    {
        uint32_t qcnt = 0u;
        uint32_t greatestFamNum = 0u;
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            greatestFamNum = (std::max)(greatestFamNum, params.queueCreateInfos[i].familyIndex);
            qcnt += params.queueCreateInfos[i].count;
        }

        m_queues = core::make_refctd_dynamic_array<queues_array_t>(qcnt);
        m_offsets = core::make_refctd_dynamic_array<q_offsets_array_t>(greatestFamNum + 1u, 0u);

        for (const auto& qci : core::SRange<const SQueueCreationParams>(params.queueCreateInfos, params.queueCreateInfos + params.queueParamsCount))
        {
            if (qci.familyIndex == greatestFamNum)
                continue;

            (*m_offsets)[qci.familyIndex + 1u] = qci.count;
        }
        std::inclusive_scan(m_offsets->begin(),m_offsets->end(),m_offsets->begin());
    }

    // must be called by implementations of mapMemory()
    static void post_mapMemory(IDriverMemoryAllocation* memory, void* ptr, IDriverMemoryAllocation::MemoryRange rng, IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG access) 
    {
        memory->postMapSetMembers(ptr, rng, access);
    }
    // must be called by implementations of unmapMemory()
    static void post_unmapMemory(IDriverMemoryAllocation* memory)
    {
        post_mapMemory(memory, nullptr, { 0,0 }, IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
    }

    void initDefaultDownloadBuffer(size_t size = 0x4000000ull)
    {
        auto reqs = getDownStreamingMemoryReqs();
        reqs.vulkanReqs.size = size;
        reqs.vulkanReqs.alignment = 64u * 1024u; // if you need larger alignments then you're not right in the head
        m_defaultDownloadBuffer = core::make_smart_refctd_ptr<video::StreamingTransientDataBufferMT<> >(this, reqs);
    }
    void initDefaultUploadBuffer(size_t size = 0x4000000ull)
    {
        auto reqs = getUpStreamingMemoryReqs();
        reqs.vulkanReqs.size = size;
        reqs.vulkanReqs.alignment = 64u * 1024u; // if you need larger alignments then you're not right in the head
        m_defaultUploadBuffer = core::make_smart_refctd_ptr<video::StreamingTransientDataBufferMT<> >(this, reqs);
    }

    virtual bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _outCmdBufs) = 0;
    virtual bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) = 0;
    virtual core::smart_refctd_ptr<IGPUFramebuffer> createGPUFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) = 0;
    virtual core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt) = 0;
    virtual core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) = 0;
    virtual core::smart_refctd_ptr<IGPUImageView> createGPUImageView_impl(IGPUImageView::SCreationParams&& params) = 0;
    virtual core::smart_refctd_ptr<IGPUDescriptorSet> createGPUDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout) = 0;
    virtual core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) = 0;
    virtual core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout_impl(
        const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
    ) = 0;
    virtual core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline_impl(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
    ) = 0;
    virtual bool createGPUComputePipelines_impl(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output
    ) = 0;
    virtual core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createGPURenderpassIndependentPipeline_impl(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        IGPUSpecializedShader** _shaders, IGPUSpecializedShader** _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams,
        const asset::SBlendParams& _blendParams,
        const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams
    ) = 0;
    virtual bool createGPURenderpassIndependentPipelines_impl(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
    ) = 0;
    virtual core::smart_refctd_ptr<IGPUGraphicsPipeline> createGPUGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params) = 0;
    virtual bool createGPUGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) = 0;

    core::smart_refctd_ptr<IAPIConnection> m_api;
    IPhysicalDevice* m_physicalDevice;

    using queues_array_t = core::smart_refctd_dynamic_array<CThreadSafeGPUQueueAdapter*>;
    queues_array_t m_queues;
    using q_offsets_array_t = core::smart_refctd_dynamic_array<uint32_t>;
    q_offsets_array_t m_offsets;

    core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultDownloadBuffer;
    core::smart_refctd_ptr<StreamingTransientDataBufferMT<> > m_defaultUploadBuffer;

    //core::smart_refctd_ptr<CPropertyPoolHandler> m_propertyPoolHandler;
};

}


#endif