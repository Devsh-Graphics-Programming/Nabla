#ifndef __NBL_C_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/system/ILogger.h"

#include <chrono>

#include "nbl/video/IOpenGL_LogicalDevice.h"

#include "nbl/video/utilities/renderdoc.h"
#include "nbl/video/COpenGLFramebuffer.h"
#include "nbl/video/COpenGLRenderpass.h"
#include "nbl/video/COpenGLDescriptorSet.h"
#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/COpenGLEvent.h"
#include "nbl/video/COpenGLSemaphore.h"

namespace nbl::video
{

template <typename QueueType_>
class COpenGL_LogicalDevice : public IOpenGL_LogicalDevice
{
    template <E_REQUEST_TYPE DestroyReqType>
    SRequest& destroyGlObjects(uint32_t count, const GLuint names[MaxGlNamesForSingleObject])
    {
        assert(count <= MaxGlNamesForSingleObject);
        using req_params_t = SRequest_Destroy<DestroyReqType>;

        req_params_t params;
        params.count = count;
        std::copy_n(names,count,params.glnames);

        auto& req = m_threadHandler.request(std::move(params));
        // dont need to wait on this
        return req;
    }

    static inline constexpr bool IsGLES = (QueueType_::FunctionTableType::EGL_API_TYPE == EGL_OPENGL_ES_API);
    static_assert(QueueType_::FunctionTableType::EGL_API_TYPE == EGL_OPENGL_API || QueueType_::FunctionTableType::EGL_API_TYPE == EGL_OPENGL_ES_API);

    static uint32_t getTotalQueueCount(const SCreationParams& params)
    {
        uint32_t count = 0u;
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
            count += params.queueParams[i].count;
        return count;
    }

public:
    using QueueType = QueueType_;
    using FunctionTableType = typename QueueType::FunctionTableType;
    using FeaturesType = COpenGLFeatureMap;

    COpenGL_LogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, IPhysicalDevice* physicalDevice, renderdoc_api_t* rdoc, const SCreationParams& params, const egl::CEGL* _egl, const FeaturesType* _features, EGLConfig config, EGLint major, EGLint minor) :
        IOpenGL_LogicalDevice(std::move(api),physicalDevice,params,_egl),
        m_rdoc_api(rdoc),
        m_threadHandler(
            this,&m_masterContextSync,&m_masterContextCallsReturned,_egl,_features,
            getTotalQueueCount(params),
            createWindowlessGLContext(FunctionTableType::EGL_API_TYPE,_egl,major,minor,config),
            static_cast<COpenGLDebugCallback*>(physicalDevice->getDebugCallback())
        ),
        m_glfeatures(_features),
        m_config(config),
        m_gl_ver(major, minor)
    {
        uint32_t totalQCount = getTotalQueueCount(params);
        assert(totalQCount <= MaxQueueCount);

        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueParams[i];
            const uint32_t famIx = qci.familyIndex;
            const uint32_t offset = (*m_offsets)[famIx];
            const auto flags = qci.flags;

            for (uint32_t j = 0u; j < qci.count; ++j)
            {
                const float priority = qci.priorities[j];

                auto glctx = createWindowlessGLContext(FunctionTableType::EGL_API_TYPE, _egl, major, minor, config, m_threadHandler.glctx.ctx);

                const uint32_t ix = offset + j;
                const uint32_t ctxid = 1u + ix; // +1 because one ctx is here, in logical device (consider if it means we have to have another spec shader GL name for it, probably not) -- [TODO]
                
                (*m_queues)[ix] = new CThreadSafeGPUQueueAdapter
                (
                    this,
                    (IGPUQueue*)new QueueType(this, rdoc, _egl, m_glfeatures, ctxid,
                        glctx, famIx, flags, priority,
                        static_cast<COpenGLDebugCallback*>(physicalDevice->getDebugCallback()))
                );
            }
        }
        // wait for all queues to start before we set out master context
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueParams[i];
            const uint32_t famIx = qci.familyIndex;
            const uint32_t offset = (*m_offsets)[famIx];
            for (uint32_t j = 0u; j < params.queueParams[i].count; ++j)
            {
                const uint32_t ix = offset + j;
                // wait until queue's internal thread finish context creation
                static_cast<QueueType*>((*m_queues)[ix]->getUnderlyingQueue())->waitForInitComplete();

                // TODO(achal): Just some debug code. Remove.
                std::cout << "Queue's internal thread ID: " << static_cast<QueueType*>((*m_queues)[ix]->getUnderlyingQueue())->getUnderlyingThreadID() << "\n";
            }
        }

        m_threadHandler.start();
        m_threadHandler.waitForInitComplete();
        // TODO(achal): Just some debug code. Remove.
        std::cout << "Master context thread ID: " << m_threadHandler.getThreadID() << "\n";
    }

    core::smart_refctd_ptr<IGPUImage> createImage(IGPUImage::SCreationParams&& params) override final
    {
        if (!asset::IImage::validateCreationParameters(params))
            return nullptr;
        if constexpr (IsGLES)
        {
            if (params.type == IGPUImage::ET_1D)
                return nullptr;
        }

        core::smart_refctd_ptr<IGPUImage> retval;

        SRequestImageCreate reqParams;
        reqParams.deviceLocalMemoryTypeBits = m_physicalDevice->getDeviceLocalMemoryTypeBits();
        reqParams.creationParams = std::move(params);
        auto& req = m_threadHandler.request(std::move(reqParams), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestImageCreate>(req);

        return retval;
    }

    core::smart_refctd_ptr<IGPUSampler> createSampler(const IGPUSampler::SParams& _params) override final
    {
        core::smart_refctd_ptr<IGPUSampler> retval;

        SRequestSamplerCreate req_params;
        req_params.params = _params;
        req_params.is_gles = IsGLES;
        auto& req = m_threadHandler.template request<SRequestSamplerCreate>(std::move(req_params), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestSamplerCreate>(req);

        return retval;
    }

    core::smart_refctd_ptr<IGPUShader> createShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) override final
    {
        auto source = cpushader->getSPVorGLSL();
        auto clone = core::smart_refctd_ptr_static_cast<asset::ICPUBuffer>(source->clone(1u));
        if (cpushader->containsGLSL())
            return core::make_smart_refctd_ptr<COpenGLShader>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(clone), IGPUShader::buffer_contains_glsl, cpushader->getStage(), std::string(cpushader->getFilepathHint()));
        else
            return core::make_smart_refctd_ptr<COpenGLShader>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(clone), cpushader->getStage(), std::string(cpushader->getFilepathHint()));
    }

    core::smart_refctd_ptr<IGPURenderpass> createRenderpass(const IGPURenderpass::SCreationParams& params) override final
    {
        return core::make_smart_refctd_ptr<COpenGLRenderpass>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), params);
    }

    core::smart_refctd_ptr<IDeferredOperation> createDeferredOperation() override
    {
        assert(false && "not implemented");
        return nullptr;
    }

    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags) override
    {
        return core::make_smart_refctd_ptr<COpenGLCommandPool>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), flags, _familyIx);
    }

    core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) override
    {
        return core::make_smart_refctd_ptr<IDescriptorPool>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this),flags, maxSets, poolSizeCount, poolSizes);
    }
    
    SMemoryOffset allocate(const SAllocateInfo& info) override
    {
        SMemoryOffset ret =  {nullptr, IDeviceMemoryAllocator::InvalidMemoryOffset};
        if(info.dedication)
        {
            IOpenGLMemoryAllocation* glAllocation = nullptr;
            if(info.dedication->getObjectType() == IDeviceMemoryBacked::EOT_BUFFER)
            {
                COpenGLBuffer* buffer = static_cast<COpenGLBuffer*>(info.dedication);
                glAllocation = static_cast<IOpenGLMemoryAllocation*>(buffer);
            }
            else if(info.dedication->getObjectType() == IDeviceMemoryBacked::EOT_IMAGE)
            {
                COpenGLImage* image = static_cast<COpenGLImage*>(info.dedication);
                glAllocation = static_cast<IOpenGLMemoryAllocation*>(image);
            }

            if(info.memoryTypeIndex < m_physicalDevice->getMemoryProperties().memoryTypeCount)
            {
                SRequestAllocate reqParams;
                reqParams.dedicationAsAllocation = glAllocation;
                reqParams.memoryAllocateFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(info.flags);
                reqParams.memoryPropertyFlags = m_physicalDevice->getMemoryProperties().memoryTypes[info.memoryTypeIndex].propertyFlags;

                auto& req = m_threadHandler.request(std::move(reqParams),&ret);
                m_masterContextCallsInvoked++;
                m_threadHandler.template waitForRequestCompletion<SRequestAllocate>(req);
                assert(ret.memory && ret.offset != IDeviceMemoryAllocator::InvalidMemoryOffset);
            }
            else
            {
                assert(false);
            }
        }
        else
        {
            assert(false);
        }
        return ret;
    }

    core::smart_refctd_ptr<IGPUBuffer> createBuffer(IGPUBuffer::SCreationParams&& creationParams) override
    {
        SRequestBufferCreate reqParams;
        reqParams.creationParams = std::move(creationParams);
        core::smart_refctd_ptr<IGPUBuffer> output;
        auto& req = m_threadHandler.request(std::move(reqParams),&output);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestBufferCreate>(req);

        return output;
    }

    core::smart_refctd_ptr<IGPUSemaphore> createSemaphore() override final
    {
        return core::make_smart_refctd_ptr<COpenGLSemaphore>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this));
    }

    core::smart_refctd_ptr<IGPUEvent> createEvent(IGPUEvent::E_CREATE_FLAGS flags) override
    {
        return core::make_smart_refctd_ptr<COpenGLEvent>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), flags);
    }
    IGPUEvent::E_STATUS getEventStatus(const IGPUEvent* _event) override
    {
        assert((_event->getFlags()&IGPUEvent::ECF_DEVICE_ONLY_BIT) == 0);
        // only support DEVICE_ONLY events for now
        return IGPUEvent::ES_FAILURE;
    }
    IGPUEvent::E_STATUS resetEvent(IGPUEvent* _event) override
    {
        assert((_event->getFlags() & IGPUEvent::ECF_DEVICE_ONLY_BIT) == 0);
        // only support DEVICE_ONLY events for now
        return IGPUEvent::ES_FAILURE;
    }
    IGPUEvent::E_STATUS setEvent(IGPUEvent* _event) override
    {
        assert((_event->getFlags() & IGPUEvent::ECF_DEVICE_ONLY_BIT) == 0);
        // only support DEVICE_ONLY events for now
        return IGPUEvent::ES_FAILURE;
    }

    core::smart_refctd_ptr<IGPUFence> createFence(IGPUFence::E_CREATE_FLAGS _flags) override final
    {
        if (_flags & IGPUFence::ECF_SIGNALED_BIT)
        {
            SRequestFenceCreate params;
            params.flags = _flags;
            core::smart_refctd_ptr<IGPUFence> retval;
            auto& req = m_threadHandler.request(std::move(params), &retval);
            m_threadHandler.template waitForRequestCompletion<SRequestFenceCreate>(req);

            return retval;
        }
        return core::make_smart_refctd_ptr<COpenGLFence>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this));
    }

    IGPUFence::E_STATUS getFenceStatus(IGPUFence* _fence) override final
    {
        SRequestGetFenceStatus req_params;
        req_params.fence = _fence;

        IGPUFence::E_STATUS retval;

        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_threadHandler.template waitForRequestCompletion<SRequestGetFenceStatus>(req);

        return retval;
    }

    bool resetFences(uint32_t _count, IGPUFence*const * _fences) override final
    {
        for (uint32_t i = 0u; i < _count; ++i)
            IBackendObject::device_compatibility_cast<COpenGLFence*>(_fences[i], this)->reset();
        return true;
    }

    IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence* const* _fences, bool _waitAll, uint64_t _timeout) override final
    {
#ifdef _NBL_DEBUG
        for (uint32_t i = 0u; i < _count; ++i)
        {
            assert(_fences[i]);
        }
#endif
        auto tmp = SRequestWaitForFences::clock_t::now();
        const auto end = tmp+std::chrono::nanoseconds(_timeout);
        
        // dont hog the queue, let other requests jump in every 50us (20000 device non-queue requests/second if something is polling)
        constexpr uint64_t pollingQuanta = 50000u;
        IGPUFence::E_STATUS retval;
        do
        {
            tmp += std::chrono::nanoseconds(pollingQuanta);
            SRequestWaitForFences params{ {_fences,_fences+_count},core::min(tmp,end),_waitAll };
            auto& req = m_threadHandler.request(std::move(params),&retval);
            m_threadHandler.template waitForRequestCompletion<SRequestWaitForFences>(req);
        } while (retval==IGPUFence::ES_TIMEOUT && SRequestWaitForFences::clock_t::now()<end);

        return retval;
    }

    void updateDescriptorSets_impl(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) override final
    {
        for (uint32_t i = 0u; i < descriptorWriteCount; i++)
            static_cast<COpenGLDescriptorSet*>(pDescriptorWrites[i].dstSet)->writeDescriptorSet(pDescriptorWrites[i]);
        for (uint32_t i = 0u; i < descriptorCopyCount; i++)
            static_cast<COpenGLDescriptorSet*>(pDescriptorCopies[i].dstSet)->copyDescriptorSet(pDescriptorCopies[i]);
    }

    bool freeDescriptorSets_impl(IDescriptorPool* pool, const uint32_t descriptorSetCount, IGPUDescriptorSet *const *const descriptorSets) override final
    {
        _NBL_TODO();
        return false;
    }

    void flushMappedMemoryRanges(core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges) override final
    {
        SRequestFlushMappedMemoryRanges req_params{ ranges };
        auto& req = m_threadHandler.request(std::move(req_params));
        m_masterContextCallsInvoked++;
        // TODO: if we actually copied the range parameter we wouldn't have to wait
        m_threadHandler.template waitForRequestCompletion<SRequestFlushMappedMemoryRanges>(req);
    }

    void invalidateMappedMemoryRanges(core::SRange<const video::IDeviceMemoryAllocation::MappedMemoryRange> ranges) override final
    {
        SRequestInvalidateMappedMemoryRanges req_params{ ranges };
        auto& req = m_threadHandler.request(std::move(req_params));
        m_threadHandler.template waitForRequestCompletion<SRequestInvalidateMappedMemoryRanges>(req);
    }

    void* mapMemory(const IDeviceMemoryAllocation::MappedMemoryRange& memory, core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access = IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE) override final
    {
        if (memory.memory == nullptr || memory.memory->getAPIType() != (IsGLES ? EAT_OPENGL_ES:EAT_OPENGL))
            return nullptr;

        assert(!memory.memory->isCurrentlyMapped());
        assert(IDeviceMemoryAllocation::isMappingAccessConsistentWithMemoryType(access, memory.memory->getMemoryPropertyFlags()));

        auto* buf = static_cast<COpenGLBuffer*>(memory.memory);
        const GLbitfield storageFlags = buf->getOpenGLStorageFlags();

        GLbitfield flags = GL_MAP_PERSISTENT_BIT | (access.hasFlags(IDeviceMemoryAllocation::EMCAF_READ) ? GL_MAP_READ_BIT : 0);
        if (storageFlags & GL_MAP_COHERENT_BIT)
            flags |= GL_MAP_COHERENT_BIT | (access.hasFlags(IDeviceMemoryAllocation::EMCAF_WRITE) ? GL_MAP_WRITE_BIT : 0);
        else if (access.hasFlags(IDeviceMemoryAllocation::EMCAF_WRITE))
            flags |= GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_WRITE_BIT;

        SRequestMapBufferRange req_params;
        req_params.buf = core::smart_refctd_ptr<IDeviceMemoryAllocation>(memory.memory);
        req_params.offset = memory.offset;
        req_params.size = memory.length;
        req_params.flags = flags;

        void* retval = nullptr;
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestMapBufferRange>(req);

        core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> actualAccess(0u);
        if (flags & GL_MAP_READ_BIT)
            actualAccess |= IDeviceMemoryAllocation::EMCAF_READ;
        if (flags & GL_MAP_WRITE_BIT)
            actualAccess |= IDeviceMemoryAllocation::EMCAF_WRITE;
        if (retval)
            post_mapMemory(memory.memory, retval, memory.range, actualAccess);

        return memory.memory->getMappedPointer(); // so pointer is rewound
    }

    void unmapMemory(IDeviceMemoryAllocation* memory) override final
    {
        assert(memory->isCurrentlyMapped());

        SRequestUnmapBuffer req_params;
        req_params.buf = core::smart_refctd_ptr<IDeviceMemoryAllocation>(memory);

        auto& req = m_threadHandler.request(std::move(req_params));
        m_masterContextCallsInvoked++;

        post_unmapMemory(memory);
    }

    core::smart_refctd_ptr<IQueryPool> createQueryPool(IQueryPool::SCreationParams&& params) override
    {
        core::smart_refctd_dynamic_array<GLuint> queries[IOpenGLPhysicalDeviceBase::MaxQueues];
        
        uint32_t glQueriesPerQuery = 0u;
        GLenum glQueryType = 0u;
        if(params.queryType == IQueryPool::EQT_OCCLUSION)
        {
            glQueriesPerQuery = 1u;
            glQueryType = GL_SAMPLES_PASSED;
        }
        else if(params.queryType == IQueryPool::EQT_TIMESTAMP)
        {
            glQueriesPerQuery = 1u;
            glQueryType = GL_TIMESTAMP;
        }
        else
        {
            // TODO: Add ARB_pipeline_statistics support: https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_pipeline_statistics_query.txt
            assert(false && "QueryType is not supported.");
            return nullptr;
        }

        const uint32_t actualQueryCount = glQueriesPerQuery * params.queryCount;

        for (auto& q : (*m_queues))
        {
            auto openglQueue = static_cast<QueueType*>(q->getUnderlyingQueue());
            core::smart_refctd_dynamic_array<GLuint> & outQueriesToFill = queries[openglQueue->getCtxId()];
            outQueriesToFill = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<GLuint>>(actualQueryCount);
            openglQueue->createQueries(outQueriesToFill, glQueryType, actualQueryCount);
        }

        return core::make_smart_refctd_ptr<COpenGLQueryPool>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), queries, glQueriesPerQuery, std::move(params));
    }

    bool getQueryPoolResults(IQueryPool* queryPool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void * pData, uint64_t stride, core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags) override
    {
        const COpenGLQueryPool* qp = IBackendObject::device_compatibility_cast<const COpenGLQueryPool*>(queryPool, this);
        auto queryPoolQueriesCount = qp->getCreationParameters().queryCount;

        if(pData != nullptr && qp != nullptr)
        {
            IQueryPool::E_QUERY_TYPE queryType = qp->getCreationParameters().queryType;
            bool use64Version = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_64_BIT);
            bool availabilityFlag = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WITH_AVAILABILITY_BIT);
            bool waitForAllResults = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_WAIT_BIT);
            bool partialResults = flags.hasFlags(IQueryPool::E_QUERY_RESULTS_FLAGS::EQRF_PARTIAL_BIT);

            assert(queryType == IQueryPool::E_QUERY_TYPE::EQT_OCCLUSION || queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP);

            if(firstQuery + queryCount > queryPoolQueriesCount)
            {
                assert(false && "The sum of firstQuery and queryCount must be less than or equal to the number of queries in queryPool");
                return false;
            }
            if(partialResults && queryType == IQueryPool::E_QUERY_TYPE::EQT_TIMESTAMP) {
                assert(false && "QUERY_RESULT_PARTIAL_BIT must not be used if the pool�s queryType is QUERY_TYPE_TIMESTAMP.");
                return false;
            }

            size_t currentDataPtrOffset = 0;
            const uint32_t glQueriesPerQuery = qp->getGLQueriesPerQuery();
            const size_t queryElementDataSize = (use64Version) ? sizeof(GLuint64) : sizeof(GLuint); // each query might write to multiple values/elements
            const size_t eachQueryDataSize = queryElementDataSize * glQueriesPerQuery;
            const size_t eachQueryWithAvailabilityDataSize = (availabilityFlag) ? queryElementDataSize + eachQueryDataSize : eachQueryDataSize;

            assert(stride >= eachQueryWithAvailabilityDataSize);
            assert(stride && core::is_aligned_to(stride, eachQueryWithAvailabilityDataSize)); // stride must be aligned to each query data size considering the specified flags
            assert(dataSize >= (queryCount * stride)); // dataSize is not enough for "queryCount" queries and specified stride
            assert(dataSize >= (queryCount * eachQueryWithAvailabilityDataSize)); // dataSize is not enough for "queryCount" queries with considering the specified flags

            auto getQueryObject = [&](GLuint queryId, GLenum pname, void* pData, uint32_t queueIdx) -> void 
            {
                for (auto& q : (*m_queues))
                {
                    auto openglQueue = static_cast<QueueType*>(q->getUnderlyingQueue());
                    if(queueIdx == openglQueue->getCtxId())
                    {
                        openglQueue->getQueryResult(queryId, pname, pData, use64Version);
                        break;
                    }
                }
            }; 
            auto getQueryAvailablity = [&](GLuint queryId, uint32_t queueIdx) -> bool 
            {
                GLuint64 ret = 0;
                getQueryObject(queryId, GL_QUERY_RESULT_AVAILABLE, &ret, queueIdx);
                return (ret == GL_TRUE);
            };
            auto writeValueToData = [&](void* pData, const uint64_t value)
            {
                if(use64Version)
                {
                    GLuint64* dataPtr = reinterpret_cast<GLuint64*>(pData);
                    *dataPtr = value;
                }
                else
                {
                    GLuint* dataPtr = reinterpret_cast<GLuint*>(pData);
                    *dataPtr = static_cast<uint32_t>(value);
                }
            };

            // iterate on each query
            for(uint32_t i = 0; i < queryCount; ++i)
            {
                if(currentDataPtrOffset >= dataSize)
                {
                    assert(false);
                    break;
                }

                uint8_t* pQueryData = reinterpret_cast<uint8_t*>(pData) + currentDataPtrOffset;
                uint8_t* pAvailabilityData = pQueryData + eachQueryDataSize; // Write Availability to this value if flag specified

                // iterate on each gl query (we may have multiple gl queries per query like pipelinestatistics query type)
                const uint32_t queryIndex = i + firstQuery;
                const uint32_t glQueryBegin = queryIndex * glQueriesPerQuery;
                bool allGlQueriesAvailable = true;
                for(uint32_t q = 0; q < glQueriesPerQuery; ++q)
                {
                    uint8_t* pSubQueryData = pQueryData + q * queryElementDataSize;
                    const uint32_t queryIdx = glQueryBegin + q;
                    const uint32_t lastQueueToUse = qp->getLastQueueToUseForQuery(queryIdx);
                    GLuint query = qp->getQueryAt(lastQueueToUse, queryIdx);

                    if(query == GL_NONE)
                        continue;

                    GLenum pname;

                    if(waitForAllResults)
                    {
                        // Has WAIT_BIT -> Get Result with Wait (GL_QUERY_RESULT) + don't getQueryAvailability (if availability flag is set it will report true)
                        pname = GL_QUERY_RESULT;
                    }
                    else if(partialResults)
                    {
                        // Has PARTIAL_BIT but no WAIT_BIT -> (read vk spec) -> result value between zero and the final result value
                        // No PARTIAL queries for GL -> GL_QUERY_RESULT_NO_WAIT best match
                        // TODO(Erfan): Maybe set the values to 0 before query so it's consistent with vulkan spec? (what to do about the cmd version where we have to upload 0's to buffer)
                        pname = GL_QUERY_RESULT_NO_WAIT;
                    }
                    else if(availabilityFlag)
                    {
                        // Only Availablity -> Get Results with NoWait + get Query Availability
                        pname = GL_QUERY_RESULT_NO_WAIT;
                    }
                    else
                    {
                        // No Flags -> GL_QUERY_RESULT_NO_WAIT
                        pname = GL_QUERY_RESULT_NO_WAIT;
                    }
                            
                    if(availabilityFlag)
                        allGlQueriesAvailable &= getQueryAvailablity(query, lastQueueToUse);
                    getQueryObject(query, pname, pSubQueryData, lastQueueToUse);
                }

                if(availabilityFlag)
                {
                    if(waitForAllResults)
                        writeValueToData(pAvailabilityData, 1ull);
                    else
                        writeValueToData(pAvailabilityData, (allGlQueriesAvailable) ? 1ull : 0ull);
                }

                currentDataPtrOffset += stride;
            }
        }

        return true;
    }

    // TODO: remove from the engine, not thread safe (access to queues must be synchronized externally)
    void waitIdle() override
    {
        // TODO: glFinish affects only the current context... you'd have to post a request for a glFinish for every single queue and swapchain as well.
        SRequestWaitIdle params;
        auto& req = m_threadHandler.request(std::move(params));
        m_threadHandler.template waitForRequestCompletion<SRequestWaitIdle>(req);
    }

    void destroyFramebuffer(COpenGLFramebuffer::hash_t fbohash) override final
    {
        for (auto& q : (*m_queues))
        {
            static_cast<QueueType*>(q->getUnderlyingQueue())->destroyFramebuffer(fbohash);
        }
    }
    void destroyPipeline(COpenGLRenderpassIndependentPipeline* pipeline) override final
    {
        for (auto& q : (*m_queues))
        {
            static_cast<QueueType*>(q->getUnderlyingQueue())->destroyPipeline(pipeline);
        }
    }
    void destroyTexture(GLuint img) override final
    {
        destroyGlObjects<ERT_TEXTURE_DESTROY>(1u, &img);
    }
    void destroyBuffer(GLuint buf) override final
    {
        destroyGlObjects<ERT_BUFFER_DESTROY>(1u, &buf);
    }
    void destroySampler(GLuint s) override final
    {
        destroyGlObjects<ERT_SAMPLER_DESTROY>(1u, &s);
    }
    void destroySpecializedShaders(core::smart_refctd_dynamic_array<IOpenGLPipelineBase::SShaderProgram>&& programs) override final
    {
        constexpr auto MaxCount = COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT*MaxGlNamesForSingleObject;

        const auto count = programs->size();
        assert(count<=MaxCount);

        SRequest_Destroy<ERT_PROGRAM_DESTROY> params;
        params.count = 0u;
        for (auto i=0u; i<count; i++)
        {
            const auto glname = programs->operator[](i).GLname;
            if (glname)
                params.glnames[params.count++] = glname;
        }

        auto& req = m_threadHandler.request(std::move(params));
        // dont need to wait on this
    }
    void destroySync(GLsync sync) override final
    {
        SRequestSyncDestroy req_params;
        req_params.glsync = sync;
        auto& req = m_threadHandler.request(std::move(req_params));
        //dont need to wait on this
    }
    void setObjectDebugName(GLenum id, GLuint object, GLsizei len, const GLchar* label) override
    {
        //any other object type having name set by device request is something unexcpected
        assert(id == GL_BUFFER || id == GL_SAMPLER || id == GL_TEXTURE);
        assert(len <= IBackendObject::MAX_DEBUG_NAME_LENGTH);
#ifdef _NBL_DEBUG
        assert(len == strlen(label));
#endif

        SRequestSetDebugName req_params{ id, object, len };
        strcpy(req_params.label, label);

        auto& req = m_threadHandler.request(std::move(req_params));
        m_threadHandler.template waitForRequestCompletion<SRequestSetDebugName>(req);
    }

    void destroyQueryPool(COpenGLQueryPool* qp) override final
    {
        for (auto& q : (*m_queues))
        {
            auto openglQueue = static_cast<QueueType*>(q->getUnderlyingQueue());
            core::smart_refctd_dynamic_array<GLuint> queriesToDestroy = qp->getQueriesForQueueIdx(openglQueue->getCtxId());
            openglQueue->destroyQueries(queriesToDestroy);
        }
    }

    int getEGLAPI() override final
    {
        return QueueType_::FunctionTableType::EGL_API_TYPE;
    }

    const void* getNativeHandle() const override { return &m_threadHandler.glctx; }

    EGLConfig getEglConfig() override
    {
        return m_config;
    }

    EGLContext getEglContext() override
    {
        return m_threadHandler.glctx.ctx;
    }

    const FeaturesType* getGlFeatures() override
    {
        return m_glfeatures;
    }

    std::pair<EGLint, EGLint> getGlVersion() override
    {
        return m_gl_ver;
    }

    void bindMasterContext()
    {
        SRequestMakeCurrent req_params;
        req_params.bind = true;
        auto& req = m_threadHandler.request(std::move(req_params));
        //m_masterContextCallsInvoked++; should we?
        m_threadHandler.template waitForRequestCompletion<SRequestMakeCurrent>(req);
    }
    void unbindMasterContext()
    {
        SRequestMakeCurrent req_params;
        req_params.bind = false;
        auto& req = m_threadHandler.request(std::move(req_params));
        //m_masterContextCallsInvoked++; should we?
        m_threadHandler.template waitForRequestCompletion<SRequestMakeCurrent>(req);
    }

protected:
    inline system::logger_opt_ptr getLogger() const {return m_physicalDevice->getDebugCallback()->getLogger();}

    bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _output) override final
    {
        for (uint32_t i = 0u; i < _count; ++i)
            _output[i] = core::make_smart_refctd_ptr<COpenGLCommandBuffer>(
                core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this),
                _level, core::smart_refctd_ptr<IGPUCommandPool>(_cmdPool),
                core::smart_refctd_ptr<system::ILogger>(getLogger().get()),
                m_glfeatures
            );
        return true;
    }
    bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) override final
    {
        return false; // not sure if we even need this method at all...
    }
    core::smart_refctd_ptr<IGPUFramebuffer> createFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) override final
    {
        // now supporting only single subpass and no input nor resolve attachments
        // obvs preserve attachments are ignored as well
        if (params.renderpass->getCreationParameters().subpassCount != 1u)
            return nullptr;

        return core::make_smart_refctd_ptr<COpenGLFramebuffer>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(params));
    }
    core::smart_refctd_ptr<IGPUSpecializedShader> createSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt = nullptr) override final
    {
        const COpenGLShader* glUnspec = IBackendObject::device_compatibility_cast<const COpenGLShader*>(_unspecialized, this);

        const std::string& EP = _specInfo.entryPoint;
        const asset::IShader::E_SHADER_STAGE stage = _unspecialized->getStage();

        core::smart_refctd_ptr<asset::ICPUBuffer> spirv;
        if (glUnspec->containsGLSL())
        {
            auto begin = reinterpret_cast<const char*>(glUnspec->getSPVorGLSL()->getPointer());
            auto end = begin + glUnspec->getSPVorGLSL()->getSize();
            std::string glsl(begin,end);
            asset::IShader::insertAfterVersionAndPragmaShaderStage(glsl,std::ostringstream()<<COpenGLShader::k_openGL2VulkanExtensionMap); // TODO: remove this eventually
            asset::IShader::insertDefines(glsl,m_physicalDevice->getExtraGLSLDefines());
            auto glslShader_woIncludes = m_physicalDevice->getGLSLCompiler()->resolveIncludeDirectives(glsl.c_str(), stage, glUnspec->getFilepathHint().c_str(), 4u, getLogger());
            spirv = m_physicalDevice->getGLSLCompiler()->compileSPIRVFromGLSL(
                reinterpret_cast<const char*>(glslShader_woIncludes->getSPVorGLSL()->getPointer()),
                stage,
                EP.c_str(),
                glUnspec->getFilepathHint().c_str(),
                true,
                nullptr,
                getLogger(),
                m_physicalDevice->getLimits().spirvVersion
            );

            if (!spirv)
                return nullptr;
        }
        else
        {
            spirv = glUnspec->getSPVorGLSL_refctd();
        }

        if (_spvopt)                                                      
            spirv = _spvopt->optimize(spirv.get(),getLogger());

        if (!spirv)
            return nullptr;

        auto spvCPUShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirv), stage, std::string(_unspecialized->getFilepathHint()));

        asset::CShaderIntrospector::SIntrospectionParams introspectionParams{_specInfo.entryPoint.c_str(),m_physicalDevice->getExtraGLSLDefines()};
        asset::CShaderIntrospector introspector(m_physicalDevice->getGLSLCompiler()); // TODO: shouldn't the introspection be cached for all calls to `createSpecializedShader` (or somehow embedded into the OpenGL pipeline cache?)
        const asset::CIntrospectionData* introspection = introspector.introspect(spvCPUShader.get(), introspectionParams);
        if (!introspection)
        {
            _NBL_DEBUG_BREAK_IF(true);
            getLogger().log("Unable to introspect the SPIR-V shader to extract information about bindings and push constants. Creation failed.", system::ILogger::ELL_ERROR);
            return nullptr;
        }

        core::vector<COpenGLSpecializedShader::SUniform> uniformList;
        if (!COpenGLSpecializedShader::getUniformsFromPushConstants(&uniformList,introspection,getLogger().get()))
        {
            _NBL_DEBUG_BREAK_IF(true);
            getLogger().log("Attempted to create OpenGL GPU specialized shader from SPIR-V without debug info - unable to set push constants. Creation failed.", system::ILogger::ELL_ERROR);
            return nullptr;
        }

        return core::make_smart_refctd_ptr<COpenGLSpecializedShader>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), m_glfeatures->ShaderLanguageVersion, spvCPUShader->getSPVorGLSL(), _specInfo, std::move(uniformList), stage);
    }
    core::smart_refctd_ptr<IGPUBufferView> createBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) override final
    {
        SRequestBufferViewCreate req_params;
        req_params.buffer = core::smart_refctd_ptr<IGPUBuffer>(_underlying);
        req_params.format = _fmt;
        req_params.offset = _offset;
        req_params.size = _size;
        core::smart_refctd_ptr<IGPUBufferView> retval;
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestBufferViewCreate>(req);
        return retval;
    }
    core::smart_refctd_ptr<IGPUImageView> createImageView_impl(IGPUImageView::SCreationParams&& params) override final
    {
        if (!IGPUImageView::validateCreationParameters(params))
            return nullptr;
        if constexpr (IsGLES)
        {
            if (params.viewType == IGPUImageView::ET_1D || params.viewType == IGPUImageView::ET_1D_ARRAY)
                return nullptr;
            if (params.viewType == IGPUImageView::ET_CUBE_MAP_ARRAY && m_glfeatures->Version < 320 && !m_glfeatures->isFeatureAvailable(COpenGLFeatureMap::NBL_OES_texture_cube_map_array))
                return nullptr;
        }

        core::smart_refctd_ptr<IGPUImageView> retval;

        SRequestImageViewCreate req_params;
        req_params.params = std::move(params);
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestImageViewCreate>(req);

        return retval;
    }
    core::smart_refctd_ptr<IGPUDescriptorSet> createDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, const uint32_t* descriptorStorageOffsets) override final
    {
        return core::make_smart_refctd_ptr<COpenGLDescriptorSet>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(layout), core::smart_refctd_ptr<IDescriptorPool>(pool), descriptorStorageOffsets);
    }
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override final
    {
        return core::make_smart_refctd_ptr<IGPUDescriptorSetLayout>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), _begin, _end);//there's no COpenGLDescriptorSetLayout (no need for such)
    }
    core::smart_refctd_ptr<IGPUAccelerationStructure> createAccelerationStructure_impl(IGPUAccelerationStructure::SCreationParams&& params)
    {
        assert(false && "AccelerationStructures not supported.");
        return nullptr;
    }
    core::smart_refctd_ptr<IGPUPipelineLayout> createPipelineLayout_impl(
        const asset::SPushConstantRange* const _pcRangesBegin, const asset::SPushConstantRange* const _pcRangesEnd,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
    ) override final
    {
        return core::make_smart_refctd_ptr<COpenGLPipelineLayout>(
            core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this),
            _pcRangesBegin, _pcRangesEnd,
            std::move(_layout0), std::move(_layout1),
            std::move(_layout2), std::move(_layout3)
        );
    }
    core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline_impl(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        core::smart_refctd_ptr<IGPUSpecializedShader>&& _shader
    ) override final
    {
        core::smart_refctd_ptr<IGPUComputePipeline> retval;

        IGPUComputePipeline::SCreationParams params;
        params.layout = std::move(_layout);
        params.shader = std::move(_shader);
        SRequestComputePipelineCreate req_params;
        req_params.params = &params;
        req_params.count = 1u;
        req_params.pipelineCache = _pipelineCache;
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestComputePipelineCreate>(req);

        return retval;
    }
    bool createComputePipelines_impl(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPUComputePipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPUComputePipeline>* output
    ) override final
    {
        SRequestComputePipelineCreate req_params;
        req_params.params = createInfos.begin();
        req_params.count = createInfos.size();
        req_params.pipelineCache = pipelineCache;
        auto& req = m_threadHandler.request(std::move(req_params), output);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestComputePipelineCreate>(req);

        return true;
    }
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline_impl(
        IGPUPipelineCache* _pipelineCache,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& _layout,
        IGPUSpecializedShader* const* _shaders, IGPUSpecializedShader* const* _shadersEnd,
        const asset::SVertexInputParams& _vertexInputParams,
        const asset::SBlendParams& _blendParams,
        const asset::SPrimitiveAssemblyParams& _primAsmParams,
        const asset::SRasterizationParams& _rasterParams
    ) override final
    {
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> retval;

        IGPURenderpassIndependentPipeline::SCreationParams params;
        params.blend = _blendParams;
        params.primitiveAssembly = _primAsmParams;
        params.rasterization = _rasterParams;
        params.vertexInput = _vertexInputParams;
        params.layout = std::move(_layout);
        for (auto* s = _shaders; s != _shadersEnd; ++s)
        {
            uint32_t ix = core::findLSB<uint32_t>((*s)->getStage());
            params.shaders[ix] = core::smart_refctd_ptr<const IGPUSpecializedShader>(*s);
        }

        SRequestRenderpassIndependentPipelineCreate req_params;
        req_params.params = &params;
        req_params.count = 1u;
        req_params.pipelineCache = _pipelineCache;
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestRenderpassIndependentPipelineCreate>(req);

        return retval;
    }
    bool createRenderpassIndependentPipelines_impl(
        IGPUPipelineCache* pipelineCache,
        core::SRange<const IGPURenderpassIndependentPipeline::SCreationParams> createInfos,
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* output
    ) override final
    {
        SRequestRenderpassIndependentPipelineCreate req_params;
        req_params.params = createInfos.begin();
        req_params.count = createInfos.size();
        req_params.pipelineCache = pipelineCache;
        auto& req = m_threadHandler.request(std::move(req_params), output);
        m_masterContextCallsInvoked++;
        m_threadHandler.template waitForRequestCompletion<SRequestRenderpassIndependentPipelineCreate>(req);

        return true;
    }
    core::smart_refctd_ptr<IGPUGraphicsPipeline> createGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params) override final
    {
        return core::make_smart_refctd_ptr<IGPUGraphicsPipeline>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(params)); // theres no COpenGLGraphicsPipeline (no need for such)
    }
    bool createGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) override final
    {
        uint32_t i = 0u;
        for (const auto& ci : params)
        {
            if (!(output[i++] = createGraphicsPipeline(pipelineCache, IGPUGraphicsPipeline::SCreationParams(ci))))
                return false;
        }
        return true;
    }

private:
    renderdoc_api_t* m_rdoc_api;
    CThreadHandler<FunctionTableType> m_threadHandler;
    const FeaturesType* m_glfeatures;
    EGLConfig m_config;
    std::pair<EGLint, EGLint> m_gl_ver;

    COpenGLDebugCallback* m_dbgCb;
};

}


#include "nbl/video/COpenGL_Queue.h" 

namespace nbl::video
{

using COpenGLLogicalDevice = COpenGL_LogicalDevice<COpenGLQueue>;
using COpenGLESLogicalDevice = COpenGL_LogicalDevice<COpenGLESQueue>;

}

#endif