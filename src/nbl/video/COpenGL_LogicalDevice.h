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

template <typename QueueType_, typename SwapchainType_>
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
    using SwapchainType = SwapchainType_;
    using FunctionTableType = typename QueueType::FunctionTableType;
    using FeaturesType = COpenGLFeatureMap;

    static_assert(std::is_same_v<typename QueueType::FunctionTableType, typename SwapchainType::FunctionTableType>, "QueueType and SwapchainType come from 2 different backends!");

    COpenGL_LogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, IPhysicalDevice* physicalDevice, renderdoc_api_t* rdoc, const SCreationParams& params, const egl::CEGL* _egl, const FeaturesType* _features, EGLConfig config, EGLint major, EGLint minor) :
        IOpenGL_LogicalDevice(std::move(api),physicalDevice,params,_egl),
        m_rdoc_api(rdoc),
        m_threadHandler(
            this,_egl,_features,
            getTotalQueueCount(params),
            createWindowlessGLContext(FunctionTableType::EGL_API_TYPE,_egl,major,minor,config),
            static_cast<COpenGLDebugCallback*>(physicalDevice->getDebugCallback())
        ),
        m_glfeatures(_features),
        m_config(config),
        m_gl_ver(major, minor)
    {
        EGLContext master_ctx = m_threadHandler.getContext();

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

                SGLContext glctx = createWindowlessGLContext(FunctionTableType::EGL_API_TYPE, _egl, major, minor, config, master_ctx);

                const uint32_t ix = offset + j;
                const uint32_t ctxid = 1u + ix; // +1 because one ctx is here, in logical device (consider if it means we have to have another spec shader GL name for it, probably not) -- [TODO]
                
                (*m_queues)[ix] = new CThreadSafeGPUQueueAdapter
                (
                    this,
                    new QueueType(this, rdoc, _egl, m_glfeatures, ctxid, glctx.ctx,
                        glctx.pbuffer, famIx, flags, priority,
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
            }
        }

        m_threadHandler.start();
        m_threadHandler.waitForInitComplete();
    }


    core::smart_refctd_ptr<IGPUImage> createGPUImageOnDedMem(IGPUImage::SCreationParams&& params, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs) override final
    {
        if (!asset::IImage::validateCreationParameters(params))
            return nullptr;
        if constexpr (IsGLES)
        {
            if (params.type == IGPUImage::ET_1D)
                return nullptr;
        }

        core::smart_refctd_ptr<IGPUImage> retval;

        SRequestImageCreate req_params;
        req_params.params = std::move(params);
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_threadHandler.template waitForRequestCompletion<SRequestImageCreate>(req);

        return retval;
    }


    core::smart_refctd_ptr<IGPUSampler> createGPUSampler(const IGPUSampler::SParams& _params) override final
    {
        core::smart_refctd_ptr<IGPUSampler> retval;

        SRequestSamplerCreate req_params;
        req_params.params = _params;
        req_params.is_gles = IsGLES;
        auto& req = m_threadHandler.template request<SRequestSamplerCreate>(std::move(req_params), &retval);
        m_threadHandler.template waitForRequestCompletion<SRequestSamplerCreate>(req);

        return retval;
    }

    core::smart_refctd_ptr<IGPUShader> createGPUShader(core::smart_refctd_ptr<asset::ICPUShader>&& cpushader) override final
    {
        auto source = cpushader->getSPVorGLSL();
        auto clone = core::smart_refctd_ptr_static_cast<asset::ICPUBuffer>(source->clone(1u));
        if (cpushader->containsGLSL())
            return core::make_smart_refctd_ptr<COpenGLShader>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(clone), IGPUShader::buffer_contains_glsl);
        else
            return core::make_smart_refctd_ptr<COpenGLShader>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(clone));
    }

    core::smart_refctd_ptr<IGPURenderpass> createGPURenderpass(const IGPURenderpass::SCreationParams& params) override final
    {
        return core::make_smart_refctd_ptr<COpenGLRenderpass>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), params);
    }

    core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) override final
    {
        if ((params.presentMode == ISurface::EPM_MAILBOX) || (params.presentMode == ISurface::EPM_UNKNOWN))
            return nullptr;

        IGPUImage::SCreationParams imgci;
        imgci.arrayLayers = params.arrayLayers;
        imgci.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0);
        imgci.format = params.surfaceFormat.format;
        imgci.mipLevels = 1u;
        imgci.queueFamilyIndexCount = params.queueFamilyIndexCount;
        imgci.queueFamilyIndices = params.queueFamilyIndices;
        imgci.samples = asset::IImage::ESCF_1_BIT;
        imgci.type = asset::IImage::ET_2D;
        imgci.extent = asset::VkExtent3D{ params.width, params.height, 1u };

        IDriverMemoryBacked::SDriverMemoryRequirements mreqs;
        mreqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
        mreqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
        auto images = core::make_refctd_dynamic_array<typename SwapchainType::ImagesArrayType>(params.minImageCount);
        for (auto& img_dst : (*images))
        {
            img_dst = createGPUImageOnDedMem(IGPUImage::SCreationParams(imgci), mreqs);
            if (!img_dst)
                return nullptr;
        }

        EGLContext master_ctx = m_threadHandler.getContext();
        EGLConfig fbconfig = m_config;
        auto glver = m_gl_ver;

        // master context must not be current while creating a context with whom it will be sharing
        unbindMasterContext();
        EGLContext ctx = createGLContext(FunctionTableType::EGL_API_TYPE, m_egl, glver.first, glver.second, fbconfig, master_ctx);
        auto sc = SwapchainType::create(std::move(params),core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this),m_egl,std::move(images),m_glfeatures,ctx,fbconfig,static_cast<COpenGLDebugCallback*>(m_physicalDevice->getDebugCallback()));
        if (!sc)
            return nullptr;
        // wait until swapchain's internal thread finish context creation
        sc->waitForInitComplete();
        // make master context (in logical device internal thread) again
        bindMasterContext();

        return sc;
    }

    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags) override
    {
        return core::make_smart_refctd_ptr<COpenGLCommandPool>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), flags, _familyIx);
    }

    core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) override
    {
        return core::make_smart_refctd_ptr<IDescriptorPool>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this),maxSets);
    }

    core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IGPUBuffer::SCreationParams& unused, const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false) override
    {
        SRequestBufferCreate params;
        params.mreqs = initialMreqs;
        params.canModifySubdata = canModifySubData;
        core::smart_refctd_ptr<IGPUBuffer> output;
        auto& req = m_threadHandler.request(std::move(params), &output);
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

    void resetFences(uint32_t _count, IGPUFence*const * _fences) override final
    {
        for (uint32_t i = 0u; i < _count; ++i)
            static_cast<COpenGLFence*>(_fences[i])->reset();
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

    void updateDescriptorSets(uint32_t descriptorWriteCount, const IGPUDescriptorSet::SWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const IGPUDescriptorSet::SCopyDescriptorSet* pDescriptorCopies) override final
    {
        for (uint32_t i = 0u; i < descriptorWriteCount; i++)
            static_cast<COpenGLDescriptorSet*>(pDescriptorWrites[i].dstSet)->writeDescriptorSet(pDescriptorWrites[i]);
        for (uint32_t i = 0u; i < descriptorCopyCount; i++)
            static_cast<COpenGLDescriptorSet*>(pDescriptorCopies[i].dstSet)->copyDescriptorSet(pDescriptorCopies[i]);
    }

    void flushMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) override final
    {
        SRequestFlushMappedMemoryRanges req_params{ ranges };
        auto& req = m_threadHandler.request(std::move(req_params));
        m_threadHandler.template waitForRequestCompletion<SRequestFlushMappedMemoryRanges>(req);
    }

    void invalidateMappedMemoryRanges(core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> ranges) override final
    {
        SRequestInvalidateMappedMemoryRanges req_params{ ranges };
        auto& req = m_threadHandler.request(std::move(req_params));
    }

    void* mapMemory(const IDriverMemoryAllocation::MappedMemoryRange& memory, IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG access = IDriverMemoryAllocation::EMCAF_READ_AND_WRITE) override final
    {
        assert(access != IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
        assert(!memory.memory->isCurrentlyMapped());

        auto* buf = static_cast<COpenGLBuffer*>(memory.memory);
        const GLbitfield storageFlags = buf->getOpenGLStorageFlags();

        GLbitfield flags = GL_MAP_PERSISTENT_BIT | ((access & IDriverMemoryAllocation::EMCAF_READ) ? GL_MAP_READ_BIT : 0);
        if (storageFlags & GL_MAP_COHERENT_BIT)
            flags |= GL_MAP_COHERENT_BIT | ((access & IDriverMemoryAllocation::EMCAF_WRITE) ? GL_MAP_WRITE_BIT : 0);
        else if (access & IDriverMemoryAllocation::EMCAF_WRITE)
            flags |= GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_WRITE_BIT;

        SRequestMapBufferRange req_params;
        req_params.buf = core::smart_refctd_ptr<IDriverMemoryAllocation>(memory.memory);
        req_params.offset = memory.offset;
        req_params.size = memory.length;
        req_params.flags = flags;

        void* retval = nullptr;
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_threadHandler.template waitForRequestCompletion<SRequestMapBufferRange>(req);

        core::bitflag<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG> actualAccess = static_cast< IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(0);
        if (flags & GL_MAP_READ_BIT)
            actualAccess |= IDriverMemoryAllocation::EMCAF_READ;
        if (flags & GL_MAP_WRITE_BIT)
            actualAccess |= IDriverMemoryAllocation::EMCAF_WRITE;
        if (retval)
            post_mapMemory(memory.memory, retval, memory.range, actualAccess.value);

        return retval;
    }

    void unmapMemory(IDriverMemoryAllocation* memory) override final
    {
        assert(memory->isCurrentlyMapped());

        SRequestUnmapBuffer req_params;
        req_params.buf = core::smart_refctd_ptr<IDriverMemoryAllocation>(memory);

        auto& req = m_threadHandler.request(std::move(req_params));

        post_unmapMemory(memory);
    }

    // TODO: remove from the engine, not thread safe (access to queues must be synchronized externally)
    void waitIdle() override
    {
        // TODO: I think glFinish affects only the current context... you'd have to post a request for a glFinish for every single queue and swapchain as well.
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

protected:
    void bindMasterContext()
    {
        SRequestMakeCurrent req_params;
        req_params.bind = true;
        auto& req = m_threadHandler.request(std::move(req_params));
        m_threadHandler.template waitForRequestCompletion<SRequestMakeCurrent>(req);
    }
    void unbindMasterContext()
    {
        SRequestMakeCurrent req_params;
        req_params.bind = false;
        auto& req = m_threadHandler.request(std::move(req_params));
        m_threadHandler.template waitForRequestCompletion<SRequestMakeCurrent>(req);
    }

    inline system::logger_opt_ptr getLogger() const {return m_physicalDevice->getDebugCallback()->getLogger();}

    bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _output) override final
    {
        for (uint32_t i = 0u; i < _count; ++i)
            _output[i] = core::make_smart_refctd_ptr<COpenGLCommandBuffer>(
                core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this),
                _level, _cmdPool,
                core::smart_refctd_ptr<system::ILogger>(getLogger().get()),
                m_glfeatures
            );
        return true;
    }
    bool freeCommandBuffers_impl(IGPUCommandBuffer** _cmdbufs, uint32_t _count) override final
    {
        return false; // not sure if we even need this method at all...
    }
    core::smart_refctd_ptr<IGPUFramebuffer> createGPUFramebuffer_impl(IGPUFramebuffer::SCreationParams&& params) override final
    {
        // now supporting only single subpass and no input nor resolve attachments
        // obvs preserve attachments are ignored as well
        if (params.renderpass->getCreationParameters().subpassCount != 1u)
            return nullptr;

        return core::make_smart_refctd_ptr<COpenGLFramebuffer>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(params));
    }
    core::smart_refctd_ptr<IGPUSpecializedShader> createGPUSpecializedShader_impl(const IGPUShader* _unspecialized, const asset::ISpecializedShader::SInfo& _specInfo, const asset::ISPIRVOptimizer* _spvopt = nullptr) override final
    {
        const COpenGLShader* glUnspec = static_cast<const COpenGLShader*>(_unspecialized);

        const std::string& EP = _specInfo.entryPoint;
        const asset::ISpecializedShader::E_SHADER_STAGE stage = _specInfo.shaderStage;

        core::smart_refctd_ptr<asset::ICPUBuffer> spirv;
        if (glUnspec->containsGLSL())
        {
            auto begin = reinterpret_cast<const char*>(glUnspec->getSPVorGLSL()->getPointer());
            auto end = begin + glUnspec->getSPVorGLSL()->getSize();
            std::string glsl(begin,end);
            asset::IShader::insertAfterVersionAndPragmaShaderStage(glsl,std::ostringstream()<<COpenGLShader::k_openGL2VulkanExtensionMap); // TODO: remove this eventually
            asset::IShader::insertDefines(glsl,m_physicalDevice->getExtraGLSLDefines());
            auto glslShader_woIncludes = m_physicalDevice->getGLSLCompiler()->resolveIncludeDirectives(glsl.c_str(), stage, _specInfo.m_filePathHint.string().c_str(), 4u, getLogger());
            spirv = m_physicalDevice->getGLSLCompiler()->compileSPIRVFromGLSL(
                reinterpret_cast<const char*>(glslShader_woIncludes->getSPVorGLSL()->getPointer()),
                stage,
                EP.c_str(),
                _specInfo.m_filePathHint.string().c_str(),
                true,
                nullptr,
                getLogger()
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

        core::smart_refctd_ptr<asset::ICPUShader> spvCPUShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirv));

        asset::CShaderIntrospector::SIntrospectionParamsOld introspectionParams{_specInfo.entryPoint.c_str(),m_physicalDevice->getExtraGLSLDefines(),_specInfo.shaderStage,_specInfo.m_filePathHint};
        asset::CShaderIntrospector introspector(m_physicalDevice->getGLSLCompiler()); // TODO: shouldn't the introspection be cached for all calls to `createGPUSpecializedShader` (or somehow embedded into the OpenGL pipeline cache?)
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

        return core::make_smart_refctd_ptr<COpenGLSpecializedShader>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), m_glfeatures->ShaderLanguageVersion, spvCPUShader->getSPVorGLSL(), _specInfo, std::move(uniformList));
    }
    core::smart_refctd_ptr<IGPUBufferView> createGPUBufferView_impl(IGPUBuffer* _underlying, asset::E_FORMAT _fmt, size_t _offset = 0ull, size_t _size = IGPUBufferView::whole_buffer) override final
    {
        SRequestBufferViewCreate req_params;
        req_params.buffer = core::smart_refctd_ptr<IGPUBuffer>(_underlying);
        req_params.format = _fmt;
        req_params.offset = _offset;
        req_params.size = _size;
        core::smart_refctd_ptr<IGPUBufferView> retval;
        auto& req = m_threadHandler.request(std::move(req_params), &retval);
        m_threadHandler.template waitForRequestCompletion<SRequestBufferViewCreate>(req);
        return retval;
    }
    core::smart_refctd_ptr<IGPUImageView> createGPUImageView_impl(IGPUImageView::SCreationParams&& params) override final
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
        m_threadHandler.template waitForRequestCompletion<SRequestImageViewCreate>(req);

        return retval;
    }
    core::smart_refctd_ptr<IGPUDescriptorSet> createGPUDescriptorSet_impl(IDescriptorPool* pool, core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout) override final
    {
        // ignoring descriptor pool
        return core::make_smart_refctd_ptr<COpenGLDescriptorSet>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(layout));
    }
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override final
    {
        return core::make_smart_refctd_ptr<IGPUDescriptorSetLayout>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), _begin, _end);//there's no COpenGLDescriptorSetLayout (no need for such)
    }
    core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout_impl(
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
    core::smart_refctd_ptr<IGPUComputePipeline> createGPUComputePipeline_impl(
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
        m_threadHandler.template waitForRequestCompletion<SRequestComputePipelineCreate>(req);

        return retval;
    }
    bool createGPUComputePipelines_impl(
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
        m_threadHandler.template waitForRequestCompletion<SRequestComputePipelineCreate>(req);

        return true;
    }
    core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createGPURenderpassIndependentPipeline_impl(
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
        m_threadHandler.template waitForRequestCompletion<SRequestRenderpassIndependentPipelineCreate>(req);

        return retval;
    }
    bool createGPURenderpassIndependentPipelines_impl(
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
        m_threadHandler.template waitForRequestCompletion<SRequestRenderpassIndependentPipelineCreate>(req);

        return true;
    }
    core::smart_refctd_ptr<IGPUGraphicsPipeline> createGPUGraphicsPipeline_impl(IGPUPipelineCache* pipelineCache, IGPUGraphicsPipeline::SCreationParams&& params) override final
    {
        return core::make_smart_refctd_ptr<IGPUGraphicsPipeline>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(this), std::move(params)); // theres no COpenGLGraphicsPipeline (no need for such)
    }
    bool createGPUGraphicsPipelines_impl(IGPUPipelineCache* pipelineCache, core::SRange<const IGPUGraphicsPipeline::SCreationParams> params, core::smart_refctd_ptr<IGPUGraphicsPipeline>* output) override final
    {
        uint32_t i = 0u;
        for (const auto& ci : params)
        {
            if (!(output[i++] = createGPUGraphicsPipeline(pipelineCache, IGPUGraphicsPipeline::SCreationParams(ci))))
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

using COpenGLLogicalDevice = COpenGL_LogicalDevice<COpenGLQueue,COpenGLSwapchain>;
using COpenGLESLogicalDevice = COpenGL_LogicalDevice<COpenGLESQueue,COpenGLESSwapchain>;

}

#endif