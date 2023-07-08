#ifndef __NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_VIDEO_I_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/asset/asset.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "nbl/asset/utils/CCompilerSet.h"

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
#include "nbl/video/IDeviceMemoryAllocator.h"

#include "nbl/video/SPhysicalDeviceFeatures.h"

namespace nbl::video
{

class IDescriptorPool;
class IPhysicalDevice;

class NBL_API2 ILogicalDevice : public core::IReferenceCounted, public IDeviceMemoryAllocator
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
            const SQueueCreationParams* queueParams;
            SPhysicalDeviceFeatures featuresToEnable;
            core::smart_refctd_ptr<asset::CCompilerSet> compilerSet = nullptr;
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
            IDeviceMemoryAllocation* memory;
            size_t offset;
        };
        struct SBindImageMemoryInfo
        {
            IGPUImage* image;
            IDeviceMemoryAllocation* memory;
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

        inline const SPhysicalDeviceFeatures& getEnabledFeatures() const { return m_enabledFeatures; }

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
        virtual core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::SCreateInfo&& createInfo) = 0;

        core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params)
        {
            if (!params.renderpass->wasCreatedBy(this))
                return nullptr;
            if (!IGPUFramebuffer::validate(params))
                return nullptr;
            return createFramebuffer_impl(std::move(params));
        }

        virtual core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params) = 0;

        //! For memory allocations without the video::IDeviceMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the CPU writes to become GPU visible
        void flushMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDeviceMemoryAllocation::MappedMemoryRange* pMemoryRanges)
        {
            core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges{ pMemoryRanges, pMemoryRanges + memoryRangeCount };
            return flushMappedMemoryRanges(ranges);
        }

        //! Utility wrapper for the pointer based func
        virtual void flushMappedMemoryRanges(core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges) = 0;

        //! For memory allocations without the video::IDeviceMemoryAllocation::EMCF_COHERENT mapping capability flag you need to call this for the GPU writes to become CPU visible (slow on OpenGL)
        void invalidateMappedMemoryRanges(uint32_t memoryRangeCount, const video::IDeviceMemoryAllocation::MappedMemoryRange* pMemoryRanges)
        {
            core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges{ pMemoryRanges, pMemoryRanges + memoryRangeCount };
            return invalidateMappedMemoryRanges(ranges);
        }

        //! Utility wrapper for the pointer based func
        virtual void invalidateMappedMemoryRanges(core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges) = 0;

        virtual core::smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams&& creationParams) { return nullptr; }

        virtual uint64_t getBufferDeviceAddress(IGPUBuffer* buffer) { return ~0ull; }

        //! Binds memory allocation to provide the backing for the resource.
        /** Available only on Vulkan, in OpenGL all resources create their own memory implicitly,
        so pooling or aliasing memory for different resources is not possible.
        There is no unbind, so once memory is bound it remains bound until you destroy the resource object.
        Actually all resource classes in OpenGL implement both IDeviceMemoryBacked and IDeviceMemoryAllocation,
        so effectively the memory is pre-bound at the time of creation.
        \return true on success, always false under OpenGL.*/
        virtual bool bindBufferMemory(uint32_t bindInfoCount, const SBindBufferMemoryInfo* pBindInfos) { return false; }

        virtual core::smart_refctd_ptr<IGPUShader> createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) = 0;

        core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo)
        {
            if (!_unspecialized->wasCreatedBy(this))
                return nullptr;
            auto retval =  createSpecializedShader_impl(_unspecialized, _specInfo);
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
        virtual core::smart_refctd_ptr<IGPUImage> createImage(IGPUImage::SCreationParams&& params) { return nullptr; }

        //! The counterpart of @see bindBufferMemory for images
        virtual bool bindImageMemory(uint32_t bindInfoCount, const SBindImageMemoryInfo* pBindInfos) { return false; }

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

        core::smart_refctd_ptr<IDescriptorPool> createDescriptorPoolForDSLayouts(const IDescriptorPool::E_CREATE_FLAGS flags, const IGPUDescriptorSetLayout* const* const begin, const IGPUDescriptorSetLayout* const* const end, const uint32_t* setCounts=nullptr)
        {
            IDescriptorPool::SCreateInfo createInfo;

            auto setCountsIt = setCounts;
            for (auto* curLayout = begin; curLayout!=end; curLayout++,setCountsIt++)
            {
                const auto setCount = setCounts ? (*setCountsIt):1u;
                createInfo.maxSets += setCount;

                for (uint32_t t = 0u; t < static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COUNT); ++t)
                {
                    const auto type = static_cast<asset::IDescriptor::E_TYPE>(t);
                    const auto& redirect = (*curLayout)->getDescriptorRedirect(type);
                    createInfo.maxDescriptorCount[t] += setCount * redirect.getTotalCount();
                }
            }
        
            auto dsPool = createDescriptorPool(std::move(createInfo));
            return dsPool;
        }

        //! Fill out the descriptor sets with descriptors
        bool updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies);

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

        core::smart_refctd_ptr<IGPUGraphicsPipeline> createGraphicsPipeline(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params)
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

        virtual void waitIdle() = 0;

        //
        virtual void* mapMemory(const IDeviceMemoryAllocation::MappedMemoryRange& memory, core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> accessHint = IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE) = 0;

        //
        virtual void unmapMemory(IDeviceMemoryAllocation* memory) = 0;

        virtual void* getExternalMemoryHandle(IDeviceMemoryBacked* obj) const = 0;

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
        
        // these are the defines which shall be added to any IGPUShader which has its source as GLSL
        inline core::SRange<const char* const> getExtraShaderDefines() const
        {
            const char* const* begin = m_extraShaderDefines.data();
            return {begin,begin+m_extraShaderDefines.size()};
        }

    protected:
        ILogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, IPhysicalDevice* physicalDevice, const SCreationParams& params)
            : m_api(api), m_physicalDevice(physicalDevice), m_enabledFeatures(params.featuresToEnable), m_compilerSet(params.compilerSet)
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
        static void post_mapMemory(IDeviceMemoryAllocation* memory, void* ptr, IDeviceMemoryAllocation::MemoryRange rng, core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access) 
        {
            // rewind pointer so 0 offset is a real start to the memory
            memory->postMapSetMembers(reinterpret_cast<uint8_t*>(ptr)-rng.offset, rng, access);
        }
        // must be called by implementations of unmapMemory()
        static void post_unmapMemory(IDeviceMemoryAllocation* memory)
        {
            post_mapMemory(memory, nullptr, { 0,0 }, IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
        }

        virtual bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _outCmdBufs) = 0;
        virtual bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) = 0;
        virtual core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) = 0;
        virtual core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo) = 0;
        virtual core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) = 0;
        virtual core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) = 0;
        virtual void updateDescriptorSets_impl(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) = 0;
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
        
        void addCommonShaderDefines(std::ostringstream& pool, const bool runningInRenderDoc);

        template<typename... Args>
        inline void addShaderDefineToPool(std::ostringstream& pool, const char* define, Args&&... args)
        {
            const ptrdiff_t pos = pool.tellp();
            m_extraShaderDefines.push_back(reinterpret_cast<const char*>(pos));
            pool << define << " ";
            ((pool << std::forward<Args>(args)), ...);
        }
        inline void finalizeShaderDefinePool(std::ostringstream&& pool)
        {
            m_ShaderDefineStringPool.resize(static_cast<size_t>(pool.tellp())+m_extraShaderDefines.size());
            const auto data = ptrdiff_t(m_ShaderDefineStringPool.data());

            const auto str = pool.str();
            size_t nullCharsWritten = 0u;
            for (auto i=0u; i<m_extraShaderDefines.size(); i++)
            {
                auto& dst = m_extraShaderDefines[i];
                const auto len = (i!=(m_extraShaderDefines.size()-1u) ? ptrdiff_t(m_extraShaderDefines[i+1]):str.length())-ptrdiff_t(dst);
                const char* src = str.data()+ptrdiff_t(dst);
                dst += data+(nullCharsWritten++);
                memcpy(const_cast<char*>(dst),src,len);
                const_cast<char*>(dst)[len] = 0;
            }
        }

        core::vector<char> m_ShaderDefineStringPool;
        core::vector<const char*> m_extraShaderDefines;

        core::smart_refctd_ptr<asset::CCompilerSet> m_compilerSet;
        core::smart_refctd_ptr<IAPIConnection> m_api;
        SPhysicalDeviceFeatures m_enabledFeatures;
        IPhysicalDevice* m_physicalDevice;

        using queues_array_t = core::smart_refctd_dynamic_array<CThreadSafeGPUQueueAdapter*>;
        queues_array_t m_queues;
        using q_offsets_array_t = core::smart_refctd_dynamic_array<uint32_t>;
        q_offsets_array_t m_offsets;
};

}


#endif