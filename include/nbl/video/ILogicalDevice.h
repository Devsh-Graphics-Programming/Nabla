#ifndef __NBL_I_GPU_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_I_GPU_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IGPUQueue.h"
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IDescriptorPool.h"
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUCommandPool.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/ISwapchain.h"

namespace nbl {
namespace video
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

    ILogicalDevice(const SCreationParams& params)
    {
        uint32_t qcnt = 0u;
        uint32_t greatestFamNum = 0u;
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            greatestFamNum = std::max(greatestFamNum, params.queueCreateInfos[i].familyIndex);
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
        // compute prefix sum
        for (uint32_t i = 1u; i < m_offsets->size(); ++i)
        {
            (*m_offsets)[i] += (*m_offsets)[i - 1u];
        }
    }

    IGPUQueue* getQueue(uint32_t _familyIx, uint32_t _ix)
    {
        const uint32_t offset = (*m_offsets)[_familyIx];

        return (*m_queues)[offset+_ix].get();
    }

    virtual core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() = 0;

    virtual core::smart_refctd_ptr<IGPUEvent> createEvent() const = 0;
    virtual IGPUEvent::E_STATUS getEventStatus(const IGPUEvent* _event) const = 0;
    virtual IGPUEvent::E_STATUS resetEvent(IGPUEvent* _event) const = 0;
    virtual IGPUEvent::E_STATUS setEvent(IGPUEvent* _event) const = 0;

    virtual core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS _flags) const = 0;
    virtual void resetFences(uint32_t _count, IGPUFence** _fences) = 0;
    virtual IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence** _fences, bool _waitAll, uint64_t _timeout) = 0;

    virtual void createCommandBuffers(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer*>* _outCmdBufs) = 0;
    virtual void freeCommandBuffers(IGPUCommandBuffer** _cmdbufs, uint32_t _count) = 0;
    virtual void createDescriptorSets(IDescriptorPool* _descPool, uint32_t _count, IGPUDescriptorSetLayout** _layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* _outDescSets) = 0;

    virtual core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, IGPUCommandPool::E_CREATE_FLAGS flags) = 0;
    virtual core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) = 0;

    virtual core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params) = 0;

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
    virtual void flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) {}

    //! Utility wrapper for the pointer based func
    inline void flushMappedMemoryRanges(const core::vector<video::IDriverMemoryAllocation::MappedMemoryRange>& ranges)
    {
        this->flushMappedMemoryRanges(static_cast<uint32_t>(ranges.size()), ranges.data());
    }

    //! For memory allocations without the video::IDriverMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the GPU writes to become CPU visible (slow on OpenGL)
    virtual void invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDriverMemoryAllocation::MappedMemoryRange* pMemoryRanges) {}

    //! Utility wrapper for the pointer based func
    inline void invalidateMappedMemoryRanges(const core::vector<video::IDriverMemoryAllocation::MappedMemoryRange>& ranges)
    {
        this->invalidateMappedMemoryRanges(static_cast<uint32_t>(ranges.size()), ranges.data());
    }

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

/*
    inline core::smart_refctd_ptr<IGPUBuffer> createFilledDeviceLocalGPUBufferOnDedMem(size_t size, const void* data)
    {
        auto retval = createDeviceLocalGPUBufferOnDedMem(size);

        updateBufferRangeViaStagingBuffer(retval.get(), 0u, size, data);

        return retval;
    }
*/

    //! Create a BufferView, to a shader; a fake 1D texture with no interpolation (@see ICPUBufferView)
    virtual core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) { return nullptr; }


    //! Creates an Image (@see ICPUImage)
    virtual core::smart_refctd_ptr<IGPUImage> createGPUImage(asset::IImage::SCreationParams&& params) = 0;

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

    //!
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params, IGPUBuffer* srcBuffer, uint32_t regionCount, const IGPUImage::SBufferCopy* pRegions)
    {
        auto retval = createDeviceLocalGPUImageOnDedMem(std::move(params));
        // TODO, copyBufferToImage() is a command, so this whole function sholdnt be in ILogicalDevice probably
        //this->copyBufferToImage(srcBuffer, retval.get(), regionCount, pRegions);
        return retval;
    }
    inline core::smart_refctd_ptr<IGPUImage> createFilledDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params, IGPUImage* srcImage, uint32_t regionCount, const IGPUImage::SImageCopy* pRegions)
    {
        auto retval = createDeviceLocalGPUImageOnDedMem(std::move(params));
        // TODO, copyImage() is a command, so this whole function sholdnt be in ILogicalDevice probably
        //this->copyImage(srcImage, retval.get(), regionCount, pRegions);
        return retval;
    }


    //! Create an ImageView that can actually be used by shaders (@see ICPUImageView)
    virtual core::smart_refctd_ptr<IGPUImageView> createGPUImageView(IGPUImageView::SCreationParams&& params) = 0;

//! Fill out the descriptor sets with descriptors
    virtual void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) = 0;

    //! Create a sampler object to use with images
    virtual core::smart_refctd_ptr<IGPUSampler> createGPUSampler(const IGPUSampler::SParams& _params) = 0;

    //! Create a pipeline cache object
    virtual core::smart_refctd_ptr<IGPUPipelineCache> createGPUPipelineCache() { return nullptr; }

    //! Create a descriptor set layout (@see ICPUDescriptorSetLayout)
    virtual core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) = 0;

    //! Create a pipeline layout (@see ICPUPipelineLayout)
    virtual core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout(
        const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
    )
    {
        return nullptr;
    }

    virtual core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
    ) = 0;

    virtual bool createGPUComputePipelines(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output
    ) = 0;

    virtual bool createGPUComputePipelines(
        IGPUPipelineCache* pipelineCache,
        uint32_t count,
        const IGPUComputePipeline::SCreationParams* createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output
    ) 
    {
        auto ci = core::SRange<const IGPUComputePipeline::SCreationParams>{createInfos, createInfos+count};
        return createGPUComputePipelines(pipelineCache, ci, output);
    }

    virtual core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createGPURenderpassIndependentPipeline(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        IGPUSpecializedShader** _shaders, IGPUSpecializedShader** _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams,
        const asset::SBlendParams& _blendParams,
        const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams
    ) = 0;

    virtual bool createGPURenderpassIndependentPipelines(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
    ) = 0;

    virtual bool createGPURenderpassIndependentPipelines(
        IGPUPipelineCache* pipelineCache,
        uint32_t count,
        const IGPURenderpassIndependentPipeline::SCreationParams* createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
    )
    {
        auto ci = core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams>{createInfos, createInfos+count};
        return createGPURenderpassIndependentPipelines(pipelineCache, ci, output);
    }

    virtual core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) = 0;

    // Not implemented stuff:
    //vkCreateGraphicsPipelines //no graphics pipelines yet (just renderpass independent)
    //vkGetBufferMemoryRequirements
    //vkGetDescriptorSetLayoutSupport
    //vkMapMemory
    //vkUnmapMemory
    //vkTrimCommandPool
    //vkGetPipelineCacheData //as pipeline cache method??
    //vkMergePipelineCaches //as pipeline cache method
    //vkCreateQueryPool //????
    //vkCreateShaderModule //????

protected:
    using queues_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<IGPUQueue>>;
    queues_array_t m_queues;
    using q_offsets_array_t = core::smart_refctd_dynamic_array<uint32_t>;
    q_offsets_array_t m_offsets;
};

}
}


#endif