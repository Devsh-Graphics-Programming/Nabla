#ifndef __NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IGPUFence.h"
/*
#include "nbl/video/IGPUSemaphore.h"
#include "nbl/video/IGPUEvent.h"
*/
#include "nbl/video/IGPUShader.h"
#include "nbl/video/IDescriptorPool.h"
/*
#include "nbl/video/IGPUDescriptorSet.h"
#include "nbl/video/IGPUGraphicsPipeline.h"
#include "nbl/video/IGPUCommandPool.h"
#include "nbl/video/IGPUFramebuffer.h"
*/
#include "nbl/video/IGPUPipelineCache.h"
#include "nbl/video/IGPUQueue.h"
#include "nbl/video/ISwapchain.h"
#include "nbl/video/IDeferredOperation.h"
#include "nbl/video/IGPUAccelerationStructure.h"
#include "nbl/video/IQueryPool.h"

// TODO: undo the circular ref
#include "nbl/video/CThreadSafeGPUQueueAdapter.h"
#include "nbl/video/IDriverMemoryAllocator.h"

namespace nbl::video
{

class IDescriptorPool;
class IPhysicalDevice;

class ILogicalDevice : public core::IReferenceCounted, public IDriverMemoryAllocator
{
    public:
        enum E_FEATURE
        {
            EF_SWAPCHAIN = 0,
            EF_DEFERRED_HOST_OPERATIONS,
            EF_BUFFER_DEVICE_ADDRESS,
            EF_DESCRIPTOR_INDEXING,
            EF_ACCELERATION_STRUCTURE,
            EF_SHADER_FLOAT_CONTROLS,
            EF_SPIRV_1_4,
            EF_RAY_TRACING_PIPELINE,
            EF_RAY_QUERY,
            EF_FRAGMENT_SHADER_INTERLOCK,
            EF_COUNT
        };

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
            const SQueueCreationParams* queueParams;
            uint32_t requiredFeatureCount;
            E_FEATURE* requiredFeatures;
            uint32_t optionalFeatureCount;
            E_FEATURE* optionalFeatures;
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
            if (m_queues && !m_queues->empty())
            {
                for (uint32_t i = 0u; i < m_queues->size(); ++i)
                    delete (*m_queues)[i];
            }
        }

        inline IPhysicalDevice* getPhysicalDevice() const { return m_physicalDevice; }

        E_API_TYPE getAPIType() const;

        //
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

        virtual core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() = 0;

        virtual core::smart_refctd_ptr<IGPUEvent> createEvent(IGPUEvent::E_CREATE_FLAGS flags) = 0;
        virtual IGPUEvent::E_STATUS getEventStatus(const IGPUEvent* _event) = 0;
        virtual IGPUEvent::E_STATUS resetEvent(IGPUEvent* _event) = 0;
        virtual IGPUEvent::E_STATUS setEvent(IGPUEvent* _event) = 0;

        virtual core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS _flags) = 0;
        virtual IGPUFence::E_STATUS getFenceStatus(IGPUFence* _fence) = 0;
        virtual bool resetFences(uint32_t _count, IGPUFence*const * _fences) = 0;
        virtual IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence* const* _fences, bool _waitAll, uint64_t _timeout) = 0;
        // Forever waiting variant if you're confident that the fence will eventually be signalled
        inline bool blockForFences(uint32_t _count, IGPUFence* const* _fences, bool _waitAll = true)
        {
            if (_count)
            for (IGPUFence::E_STATUS waitStatus=IGPUFence::ES_NOT_READY; waitStatus!=IGPUFence::ES_SUCCESS;)
            {
                waitStatus = waitForFences(_count,_fences,_waitAll,999999999ull);
                if (waitStatus==video::IGPUFence::ES_ERROR)
                {
                    assert(false);
                    return false;
                }
            }
            return true;
        }

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
        
        virtual core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() = 0;
        virtual core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags) = 0;
        virtual core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) = 0;

        core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params)
        {
            if (!params.renderpass->wasCreatedBy(this))
                return nullptr;
            if (!IGPUFramebuffer::validate(params))
                return nullptr;
            return createFramebuffer_impl(std::move(params));
        }

        virtual core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params) = 0;

        static inline IDriverMemoryBacked::SDriverMemoryRequirements getDeviceLocalGPUMemoryReqs()
        {
            IDriverMemoryBacked::SDriverMemoryRequirements reqs;
            reqs.vulkanReqs.size = 0;
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

        virtual core::smart_refctd_ptr<IDriverMemoryAllocation> allocateGPUMemory(const IDriverMemoryBacked::SDriverMemoryRequirements& reqs, core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDriverMemoryAllocation::EMAF_NONE) { return nullptr; }

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

        virtual core::smart_refctd_ptr<IGPUBuffer> createGPUBuffer(const IGPUBuffer::SCreationParams& creationParams, const size_t size) { return nullptr; }
        virtual core::smart_refctd_ptr<IGPUBuffer> createBuffer(const IGPUBuffer::SCreationParams& creationParams) { return nullptr; }

        //! Binds memory allocation to provide the backing for the resource.
        /** Available only on Vulkan, in OpenGL all resources create their own memory implicitly,
        so pooling or aliasing memory for different resources is not possible.
        There is no unbind, so once memory is bound it remains bound until you destroy the resource object.
        Actually all resource classes in OpenGL implement both IDriverMemoryBacked and IDriverMemoryAllocation,
        so effectively the memory is pre-bound at the time of creation.
        \return true on success, always false under OpenGL.*/
        virtual bool bindBufferMemory(uint32_t bindInfoCount, const SBindBufferMemoryInfo* pBindInfos) { return false; }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createDeviceLocalGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& params, const size_t size)
        {
            auto reqs = getDeviceLocalGPUMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(params, reqs);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createSpilloverGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& params, const size_t size)
        {
            auto reqs = getSpilloverGPUMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(params, reqs);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createUpStreamingGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& params, const size_t size)
        {
            auto reqs = getUpStreamingMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(params, reqs);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createDownStreamingGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& params, const size_t size)
        {
            auto reqs = getDownStreamingMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(params, reqs);
        }

        //! Creates the buffer, allocates memory dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUBuffer> createCPUSideGPUVisibleGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& params, const size_t size)
        {
            auto reqs = getCPUSideGPUVisibleGPUMemoryReqs();
            reqs.vulkanReqs.size = size;
            return this->createGPUBufferOnDedMem(params, reqs);
        }

        //! Low level function used to implement the above, use with caution
        virtual core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& creationParams, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) { return nullptr; }

        virtual core::smart_refctd_ptr<IGPUShader> createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) = 0;

        core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt = nullptr)
        {
            if (!_unspecialized->wasCreatedBy(this))
                return nullptr;
            auto retval =  createSpecializedShader_impl(_unspecialized, _specInfo, _spvopt);
            const auto path = _unspecialized->getFilepathHint();
            if (retval && !path.empty())
                retval->setObjectDebugName(path.c_str());
            return retval;
        }

        //! Create a BufferView, to a shader; a fake 1D texture with no interpolation (@see ICPUBufferView)
        core::smart_refctd_ptr<IGPUBufferView> createBufferView(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer)
        {
            if (!_underlying->wasCreatedBy(this))
                return nullptr;
            return createBufferView_impl(_underlying, _fmt, _offset, _size);
        }


        //! Creates an Image (@see ICPUImage)
        virtual core::smart_refctd_ptr<IGPUImage> createGPUImage(asset::IImage::SCreationParams&& params) { return nullptr; }
        virtual core::smart_refctd_ptr<IGPUImage> createImage(asset::IImage::SCreationParams&& params) { return nullptr; }

        //! The counterpart of @see bindBufferMemory for images
        virtual bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) { return false; }

        //! Creates the Image, allocates dedicated memory and binds it at once.
        inline core::smart_refctd_ptr<IGPUImage> createDeviceLocalGPUImageOnDedMem(IGPUImage::SCreationParams&& params)
        {
            auto reqs = getDeviceLocalGPUMemoryReqs();
            params.initialLayout = asset::EIL_UNDEFINED; // initialLayout in Vulkan could either be UNDEFINED or PREINITIALIZED, can't find a use case for PREINITIALIZED yet
            return this->createGPUImageOnDedMem(std::move(params), reqs);
        }

        //!
        virtual core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) = 0;


        //! Create an ImageView that can actually be used by shaders (@see ICPUImageView)
        core::smart_refctd_ptr<IGPUImageView> createImageView(IGPUImageView::SCreationParams&& params)
        {
            if (!params.image->wasCreatedBy(this))
                return nullptr;
            return createImageView_impl(std::move(params));
        }

        core::smart_refctd_ptr<IGPUAccelerationStructure> createAccelerationStructure(IGPUAccelerationStructure::SCreationParams&& params)
        {
            if (!params.bufferRange.buffer->wasCreatedBy(this))
                return nullptr;
            return createAccelerationStructure_impl(std::move(params));
        }

        core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout)
        {
            if (!pool->wasCreatedBy(this))
                return nullptr;
            if (!layout->wasCreatedBy(this))
                return nullptr;
            return createDescriptorSet_impl(pool, std::move(layout));
        }

        core::smart_refctd_ptr<IDescriptorPool> createDescriptorPoolForDSLayouts(const IDescriptorPool::E_CREATE_FLAGS flags, const IGPUDescriptorSetLayout* const* const begin, const IGPUDescriptorSetLayout* const* const end, const uint32_t* setCounts=nullptr)
        {
            uint32_t totalSetCount = 0;
            std::vector<IDescriptorPool::SDescriptorPoolSize> poolSizes; // TODO: use a map
            auto setCountsIt = setCounts;
            for (auto* curLayout = begin; curLayout!=end; curLayout++,setCountsIt++)
            {
                const auto setCount = setCounts ? (*setCountsIt):1u;
                totalSetCount += setCount;

                auto bindings = (*curLayout)->getBindings();
                for (const auto& binding : bindings)
                {
                    auto ps = std::find_if(poolSizes.begin(), poolSizes.end(), [&](const IDescriptorPool::SDescriptorPoolSize& poolSize) { return poolSize.type == binding.type; });
                    if (ps != poolSizes.end())
                    {
                        ps->count += setCount*binding.count;
                    }
                    else
                    {
                        poolSizes.push_back(IDescriptorPool::SDescriptorPoolSize { binding.type, setCount*binding.count });
                    }
                }

            }
        
            core::smart_refctd_ptr<IDescriptorPool> dsPool = createDescriptorPool(flags, totalSetCount, poolSizes.size(), poolSizes.data());
            return dsPool;
        }

        void createDescriptorSets(IDescriptorPool* pool, uint32_t count, const IGPUDescriptorSetLayout* const* _layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
        {
            core::SRange<const IGPUDescriptorSetLayout* const> layouts{ _layouts, _layouts + count };
            createDescriptorSets(pool, layouts, output);
        }
        void createDescriptorSets(IDescriptorPool* pool, core::SRange<const IGPUDescriptorSetLayout* const> layouts, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
        {
            uint32_t i = 0u;
            for (const IGPUDescriptorSetLayout* layout_ : layouts)
            {
                auto layout = core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout_);
                output[i++] = createDescriptorSet(pool, std::move(layout));
            }
        }

        //! Fill out the descriptor sets with descriptors
        virtual void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) = 0;

        //! Create a sampler object to use with images
        virtual core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) = 0;

        //! Create a pipeline cache object
        virtual core::smart_refctd_ptr<IGPUPipelineCache> createPipelineCache() { return nullptr; }

        //! Create a descriptor set layout (@see ICPUDescriptorSetLayout)
        core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end);

        //! Create a pipeline layout (@see ICPUPipelineLayout)
        core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout(
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
            return createPipelineLayout_impl(_pcRangesBegin, _pcRangesEnd, std::move(_layout0), std::move(_layout1), std::move(_layout2), std::move(_layout3));
        }

        core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline(
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
            const char* debugName = _shader->getObjectDebugName();
            auto retval = createComputePipeline_impl(_pipelineCache, std::move(_layout), std::move(_shader));
            if (retval && debugName[0])
                retval->setObjectDebugName(debugName);
            return retval;
        }

        bool createComputePipelines(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* output
        ) {
            if (pipelineCache && !pipelineCache->wasCreatedBy(this))
                return false;
            for (const auto& ci : createInfos)
                if (!ci.layout->wasCreatedBy(this) || !ci.shader->wasCreatedBy(this))
                    return false;
            return createComputePipelines_impl(pipelineCache, createInfos, output);
        }

        bool createComputePipelines(
            IGPUPipelineCache* pipelineCache,
            uint32_t count,
            const IGPUComputePipeline::SCreationParams* createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* output
        ) 
        {
            auto ci = core::SRange<const IGPUComputePipeline::SCreationParams>{createInfos, createInfos+count};
            return createComputePipelines(pipelineCache, ci, output);
        }

        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            IGPUSpecializedShader* const* _shaders, IGPUSpecializedShader* const* _shadersEnd,
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
            return createRenderpassIndependentPipeline_impl(_pipelineCache, std::move(_layout), _shaders, _shadersEnd, _vertexInputParams, _blendParams, _primAsmParams, _rasterParams);
        }

        bool createRenderpassIndependentPipelines(
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
            return createRenderpassIndependentPipelines_impl(pipelineCache, createInfos, output);
        }

        bool createRenderpassIndependentPipelines(
            IGPUPipelineCache* pipelineCache,
            uint32_t count,
            const IGPURenderpassIndependentPipeline::SCreationParams* createInfos,
            core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
        )
        {
            auto ci = core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams>{createInfos, createInfos+count};
            return createRenderpassIndependentPipelines(pipelineCache, ci, output);
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
            return createGraphicsPipeline_impl(pipelineCache, std::move(params));
        }

        bool createGraphicsPipelines(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
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
            return createGraphicsPipelines_impl(pipelineCache, params, output);
        }

        bool createGraphicsPipelines(IGPUPipelineCache* pipelineCache, uint32_t count, const IGPUGraphicsPipeline::SCreationParams* params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output)
        {
            auto ci = core::SRange<const IGPUGraphicsPipeline::SCreationParams>{ params, params + count };
            return createGraphicsPipelines(pipelineCache, ci, output);
        }

        virtual core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) = 0;

        virtual void waitIdle() = 0;

        //
        virtual void* mapMemory(const IDriverMemoryAllocation::MappedMemoryRange& memory, core::bitflag<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> accessHint = IDriverMemoryAllocation::EMCAF_READ_AND_WRITE) = 0;

        //
        virtual void unmapMemory(IDriverMemoryAllocation* memory) = 0;

        // Not implemented stuff:
        //vkCreateGraphicsPipelines // no graphics pipelines yet (just renderpass independent)
        //vkGetDescriptorSetLayoutSupport
        //vkTrimCommandPool // for this you need to Optimize OpenGL commandrecording to use linked list
        //vkGetPipelineCacheData //as pipeline cache method?? (why not)
        //vkMergePipelineCaches //as pipeline cache method (why not)
        
        virtual core::smart_refctd_ptr<IQueryPool> createQueryPool(IQueryPool::SCreationParams&& params) { return nullptr; }

        virtual bool getQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void * pData, uint64_t stride, core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags) { return false;}

        virtual bool buildAccelerationStructures(
            core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation,
            const core::SRange<IGPUAccelerationStructure::HostBuildGeometryInfo>& pInfos,
            IGPUAccelerationStructure::BuildRangeInfo* const* ppBuildRangeInfos)
        {
            return false;
        }

        virtual bool copyAccelerationStructure(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::CopyInfo& copyInfo)
        {
            return false;
        }
    
        virtual bool copyAccelerationStructureToMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyToMemoryInfo& copyInfo)
        {
            return false;
        }

        virtual bool copyAccelerationStructureFromMemory(core::smart_refctd_ptr<IDeferredOperation>&& deferredOperation, const IGPUAccelerationStructure::HostCopyFromMemoryInfo& copyInfo)
        {
            return false;
        }

        virtual IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::HostBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts)
        {
            return IGPUAccelerationStructure::BuildSizes{};
        }

        virtual IGPUAccelerationStructure::BuildSizes getAccelerationStructureBuildSizes(const IGPUAccelerationStructure::DeviceBuildGeometryInfo& pBuildInfo, const uint32_t* pMaxPrimitiveCounts)
        {
            return IGPUAccelerationStructure::BuildSizes{};
        }

        // OpenGL: const egl::CEGL::Context*
        // Vulkan: const VkDevice*
        virtual const void* getNativeHandle() const = 0;

    protected:
        ILogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, IPhysicalDevice* physicalDevice, const SCreationParams& params)
            : m_api(api), m_physicalDevice(physicalDevice)
        {
            uint32_t qcnt = 0u;
            uint32_t greatestFamNum = 0u;
            for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
            {
                greatestFamNum = (std::max)(greatestFamNum, params.queueParams[i].familyIndex);
                qcnt += params.queueParams[i].count;
            }

            m_queues = core::make_refctd_dynamic_array<queues_array_t>(qcnt);
            m_offsets = core::make_refctd_dynamic_array<q_offsets_array_t>(greatestFamNum + 1u, 0u);

            for (const auto& qci : core::SRange<const SQueueCreationParams>(params.queueParams, params.queueParams + params.queueParamsCount))
            {
                if (qci.familyIndex == greatestFamNum)
                    continue;

                (*m_offsets)[qci.familyIndex + 1u] = qci.count;
            }
            std::inclusive_scan(m_offsets->begin(),m_offsets->end(),m_offsets->begin());
        }

        // must be called by implementations of mapMemory()
        static void post_mapMemory(IDriverMemoryAllocation* memory, void* ptr, IDriverMemoryAllocation::MemoryRange rng, core::bitflag<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access) 
        {
            memory->postMapSetMembers(ptr, rng, access);
        }
        // must be called by implementations of unmapMemory()
        static void post_unmapMemory(IDriverMemoryAllocation* memory)
        {
            post_mapMemory(memory, nullptr, { 0,0 }, IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
        }

        virtual bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _outCmdBufs) = 0;
        virtual bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) = 0;
        virtual core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt) = 0;
        virtual core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) = 0;
        virtual core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout) = 0;
        virtual core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) = 0;
        virtual core::smart_refctd_ptr<IGPUAccelerationStructure> createAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
            const asset::SPushConstantRange* const _pcRangesBegin = nullptr, const asset::SPushConstantRange* const _pcRangesEnd = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1 = nullptr,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2 = nullptr, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3 = nullptr
        ) = 0;
        virtual core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline_impl(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
        ) = 0;
        virtual bool createComputePipelines_impl(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPUComputePipeline>* output
        ) = 0;
        virtual core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline_impl(
            IGPUPipelineCache* _pipelineCache,
            core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
            IGPUSpecializedShader* const* _shaders, IGPUSpecializedShader* const* _shadersEnd,
            const asset::SVertexInputParams& _vertexInputParams,
            const asset::SBlendParams& _blendParams,
            const asset::SPrimitiveAssemblyParams& _primAsmParams,
            const asset::SRasterizationParams& _rasterParams
        ) = 0;
        virtual bool createRenderpassIndependentPipelines_impl(
            IGPUPipelineCache* pipelineCache,
            core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
            core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
        ) = 0;
        virtual core::smart_refctd_ptr<IGPUGraphicsPipeline> createGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params) = 0;
        virtual bool createGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) = 0;

        core::smart_refctd_ptr<IAPIConnection> m_api;
        IPhysicalDevice* m_physicalDevice;

        using queues_array_t = core::smart_refctd_dynamic_array<CThreadSafeGPUQueueAdapter*>;
        queues_array_t m_queues;
        using q_offsets_array_t = core::smart_refctd_dynamic_array<uint32_t>;
        q_offsets_array_t m_offsets;
};

}


#endif