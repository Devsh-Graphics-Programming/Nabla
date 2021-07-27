#ifndef __NBL_C_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IOpenGL_LogicalDevice.h"

#include "nbl/video/COpenGLFramebuffer.h"
#include "nbl/video/COpenGLRenderpass.h"
#include "nbl/video/COpenGLDescriptorSet.h"
#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/COpenGLEvent.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/debug/debug.h"
#include "nbl/system/ILogger.h"

#include <chrono>

namespace nbl {
namespace video
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
        std::copy(names, names + count, params.glnames);

        auto& req = m_threadHandler.request(std::move(params));
        // dont need to wait on this
        //m_threadHandler.waitForRequestCompletion<req_params_t>(req);
        return req;
    }

    static inline constexpr bool IsGLES = (QueueType_::FunctionTableType::EGL_API_TYPE == EGL_OPENGL_ES_API);
    static_assert(QueueType_::FunctionTableType::EGL_API_TYPE == EGL_OPENGL_API || QueueType_::FunctionTableType::EGL_API_TYPE == EGL_OPENGL_ES_API);

    static uint32_t getTotalQueueCount(const SCreationParams& params)
    {
        uint32_t count = 0u;
        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
            count += params.queueCreateInfos[i].count;
        return count;
    }

public:
    using QueueType = QueueType_;
    using SwapchainType = SwapchainType_;
    using FunctionTableType = typename QueueType::FunctionTableType;
    using FeaturesType = COpenGLFeatureMap;

    static_assert(std::is_same_v<typename QueueType::FunctionTableType, typename SwapchainType::FunctionTableType>, "QueueType and SwapchainType come from 2 different backends!");

    COpenGL_LogicalDevice(const egl::CEGL* _egl, E_API_TYPE api_type, FeaturesType* _features, EGLConfig config, EGLint major, EGLint minor, const SCreationParams& params, SDebugCallback* _dbgCb, core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc, core::smart_refctd_ptr<system::ILogger>&& logger) :
        IOpenGL_LogicalDevice(_egl, api_type, params, std::move(s), std::move(glslc), std::move(logger)),
        m_threadHandler(this, _egl, _features, getTotalQueueCount(params), createWindowlessGLContext(FunctionTableType::EGL_API_TYPE, _egl, major, minor, config), _dbgCb, core::smart_refctd_ptr(m_logger)),
        m_glfeatures(_features),
        m_config(config),
        m_gl_ver(major, minor)
    {
        EGLContext master_ctx = m_threadHandler.getContext();

        uint32_t totalQCount = getTotalQueueCount(params);
        assert(totalQCount <= MaxQueueCount);

        for (uint32_t i = 0u; i < params.queueParamsCount; ++i)
        {
            const auto& qci = params.queueCreateInfos[i];
            const uint32_t famIx = qci.familyIndex;
            const uint32_t offset = (*m_offsets)[famIx];
            const auto flags = qci.flags;

            for (uint32_t j = 0u; j < qci.count; ++j)
            {
                const float priority = qci.priorities[j];

                SGLContext glctx = createWindowlessGLContext(FunctionTableType::EGL_API_TYPE, _egl, major, minor, config, master_ctx);

                const uint32_t ix = offset + j;
                const uint32_t ctxid = 1u + ix; // +1 because one ctx is here, in logical device (consider if it means we have to have another spec shader GL name for it, probably not) -- [TODO]
                (*m_queues)[ix] = core::make_smart_refctd_ptr<QueueType>(this, this, _egl, _features, ctxid, glctx.ctx, glctx.pbuffer, famIx, flags, priority, _dbgCb, core::smart_refctd_ptr(m_logger));
            }
        }

        m_threadHandler.start();

        constexpr size_t GLSLcnt = std::extent<decltype(FeaturesType::m_GLSLExtensions)>::value;
        if (!m_supportedGLSLExtsNames)
        {
            size_t cnt = 0ull;
            for (size_t i = 0ull; i < GLSLcnt; ++i)
                cnt += _features->isFeatureAvailable(_features->m_GLSLExtensions[i]);
            if (_features->runningInRenderDoc)
                ++cnt;
            m_supportedGLSLExtsNames = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<std::string>>(cnt);
            size_t i = 0ull;
            for (size_t j = 0ull; j < GLSLcnt; ++j)
                if (_features->isFeatureAvailable(_features->m_GLSLExtensions[j]))
                    (*m_supportedGLSLExtsNames)[i++] = _features->OpenGLFeatureStrings[_features->m_GLSLExtensions[j]];
            if (_features->runningInRenderDoc)
                (*m_supportedGLSLExtsNames)[i] = _features->RUNNING_IN_RENDERDOC_EXTENSION_NAME;
        }

        initDefaultDownloadBuffer();
        initDefaultUploadBuffer();
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
            return core::make_smart_refctd_ptr<COpenGLShader>(this, std::move(clone), IGPUShader::buffer_contains_glsl);
        else
            return core::make_smart_refctd_ptr<COpenGLShader>(this, std::move(clone));
    }

    core::smart_refctd_ptr<IGPURenderpass> createGPURenderpass(const IGPURenderpass::SCreationParams& params) override final
    {
        return core::make_smart_refctd_ptr<COpenGLRenderpass>(this, params);
    }

    core::smart_refctd_ptr<ISwapchain> createSwapchain(ISwapchain::SCreationParams&& params) override final
    {
        IGPUImage::SCreationParams imgci;
        imgci.arrayLayers = params.arrayLayers;
        imgci.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0);
        imgci.format = params.surfaceFormat.format;
        imgci.mipLevels = 1u;
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
        auto sc = SwapchainType::create(std::move(params), this, m_egl, std::move(images), m_glfeatures, ctx, fbconfig, m_dbgCb, core::smart_refctd_ptr(m_logger));
        if (!sc)
            return nullptr;
        // wait until swapchain's internal thread finish context creation
        sc->waitForContextCreation();
        // make master context (in logical device internal thread) again
        bindMasterContext();

        return sc;
    }

    core::smart_refctd_ptr<IGPUCommandPool> createCommandPool(uint32_t _familyIx, IGPUCommandPool::E_CREATE_FLAGS flags) override
    {
        return core::make_smart_refctd_ptr<COpenGLCommandPool>(this, flags, _familyIx);
    }

    core::smart_refctd_ptr<IDescriptorPool> createDescriptorPool(IDescriptorPool::E_CREATE_FLAGS flags, uint32_t maxSets, uint32_t poolSizeCount, const IDescriptorPool::SDescriptorPoolSize* poolSizes) override
    {
        return core::make_smart_refctd_ptr<IDescriptorPool>(this);
    }

    core::smart_refctd_ptr<IGPUBuffer> createGPUBufferOnDedMem(const IDriverMemoryBacked::SDriverMemoryRequirements& initialMreqs, const bool canModifySubData = false) override
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
        return core::make_smart_refctd_ptr<COpenGLSemaphore>(this);
    }

    core::smart_refctd_ptr<IGPUEvent> createEvent(IGPUEvent::E_CREATE_FLAGS flags) override
    {
        return core::make_smart_refctd_ptr<COpenGLEvent>(this, flags);
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
        return core::make_smart_refctd_ptr<COpenGLFence>(this);
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

    void resetFences(uint32_t _count, IGPUFence** _fences) override final
    {
        for (uint32_t i = 0u; i < _count; ++i)
            static_cast<COpenGLFence*>(_fences[i])->reset();
    }

    IGPUFence::E_STATUS waitForFences(uint32_t _count, IGPUFence** _fences, bool _waitAll, uint64_t _timeout) override final
    {
        SRequestWaitForFences params{ core::SRange<IGPUFence*>(_fences, _fences + _count) , _waitAll, _timeout };
        IGPUFence::E_STATUS retval;
        auto& req = m_threadHandler.request(std::move(params), &retval);
        m_threadHandler.template waitForRequestCompletion<SRequestWaitForFences>(req);

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
        m_threadHandler.template waitForRequestCompletion<SRequestInvalidateMappedMemoryRanges>(req);
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

        std::underlying_type_t<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG> actualAccess = 0;
        if (flags & GL_MAP_READ_BIT)
            actualAccess |= IDriverMemoryAllocation::EMCAF_READ;
        if (flags & GL_MAP_WRITE_BIT)
            actualAccess |= IDriverMemoryAllocation::EMCAF_WRITE;
        if (retval)
            post_mapMemory(memory.memory, retval, memory.range, static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(actualAccess));

        return retval;
    }

    void unmapMemory(IDriverMemoryAllocation* memory) override final
    {
        assert(memory->isCurrentlyMapped());

        SRequestUnmapBuffer req_params;
        req_params.buf = core::smart_refctd_ptr<IDriverMemoryAllocation>(memory);

        auto& req = m_threadHandler.request(std::move(req_params));
        m_threadHandler.template waitForRequestCompletion<SRequestUnmapBuffer>(req);

        post_unmapMemory(memory);
    }

    void waitIdle() override
    {
        SRequestWaitIdle params;
        auto& req = m_threadHandler.request(std::move(params));
        m_threadHandler.template waitForRequestCompletion<SRequestWaitIdle>(req);
    }

    void destroyFramebuffer(COpenGLFramebuffer::hash_t fbohash) override final
    {
        for (auto& q : (*m_queues))
        {
            static_cast<QueueType*>(q.get())->destroyFramebuffer(fbohash);
        }
    }
    void destroyPipeline(COpenGLRenderpassIndependentPipeline* pipeline) override final
    {
        for (auto& q : (*m_queues))
        {
            static_cast<QueueType*>(q.get())->destroyPipeline(pipeline);
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
    void destroySpecializedShader(uint32_t count, const GLuint* programs) override final
    {
        auto& req = destroyGlObjects<ERT_PROGRAM_DESTROY>(count, programs);
        // actually wait for this to complete because `programs` is most likely stack array or something owned exclusively by the object (which is being destroyed)
        m_threadHandler.template waitForRequestCompletion<SRequest_Destroy<ERT_PROGRAM_DESTROY>>(req);
    }
    void destroySync(GLsync sync) override final
    {
        SRequestSyncDestroy req_params;
        req_params.glsync = sync;
        auto& req = m_threadHandler.request(std::move(req_params));
        //dont need to wait on this
        //m_threadHandler.template waitForRequestCompletion<SRequestSyncDestroy>(req);
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

    bool createCommandBuffers_impl(IGPUCommandPool* _cmdPool, IGPUCommandBuffer::E_LEVEL _level, uint32_t _count, core::smart_refctd_ptr<IGPUCommandBuffer>* _output) override final
    {
        for (uint32_t i = 0u; i < _count; ++i)
            _output[i] = core::make_smart_refctd_ptr<COpenGLCommandBuffer>(this, _level, _cmdPool);
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

        return core::make_smart_refctd_ptr<COpenGLFramebuffer>(this, std::move(params));
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
            std::string glsl(begin, end);
            COpenGLShader::insertGLtoVKextensionsMapping(glsl, getSupportedGLSLExtensions().get());
            auto glslShader_woIncludes = m_GLSLCompiler->resolveIncludeDirectives(glsl.c_str(), stage, _specInfo.m_filePathHint.c_str());
            {
                auto fl = fopen("shader.glsl", "w");
                fwrite(glsl.c_str(), 1, glsl.size(), fl);
                fclose(fl);
            }
            spirv = m_GLSLCompiler->compileSPIRVFromGLSL(
                reinterpret_cast<const char*>(glslShader_woIncludes->getSPVorGLSL()->getPointer()),
                stage,
                EP.c_str(),
                _specInfo.m_filePathHint.c_str()
            );

            if (!spirv)
                return nullptr;
        }
        else
        {
            spirv = glUnspec->getSPVorGLSL_refctd();
        }

        if (_spvopt)                                                      
            spirv = _spvopt->optimize(spirv.get(), system::logger_opt_ptr(m_logger.get()));

        if (!spirv)
            return nullptr;

        core::smart_refctd_ptr<asset::ICPUShader> spvCPUShader = core::make_smart_refctd_ptr<asset::ICPUShader>(std::move(spirv));

        asset::CShaderIntrospector::SIntrospectionParams introspectionParams{ _specInfo.shaderStage, _specInfo.entryPoint, getSupportedGLSLExtensions(), _specInfo.m_filePathHint };
        asset::CShaderIntrospector introspector(m_GLSLCompiler.get()); // TODO: shouldn't the introspection be cached for all calls to `createGPUSpecializedShader` (or somehow embedded into the OpenGL pipeline cache?)
        const asset::CIntrospectionData* introspection = introspector.introspect(spvCPUShader.get(), introspectionParams);
        if (!introspection)
        {
            _NBL_DEBUG_BREAK_IF(true);
            m_logger->log("Unable to introspect the SPIR-V shader to extract information about bindings and push constants. Creation failed.", system::ILogger::ELL_ERROR);
            return nullptr;
        }

        core::vector<COpenGLSpecializedShader::SUniform> uniformList;
        if (!COpenGLSpecializedShader::getUniformsFromPushConstants(&uniformList, introspection, m_logger.get()))
        {
            _NBL_DEBUG_BREAK_IF(true);
            m_logger->log("Attempted to create OpenGL GPU specialized shader from SPIR-V without debug info - unable to set push constants. Creation failed.", system::ILogger::ELL_ERROR);
            return nullptr;
        }

        return core::make_smart_refctd_ptr<COpenGLSpecializedShader>(this, m_glfeatures->ShaderLanguageVersion, spvCPUShader->getSPVorGLSL(), _specInfo, std::move(uniformList));
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
        return core::make_smart_refctd_ptr<COpenGLDescriptorSet>(this, std::move(layout));
    }
    core::smart_refctd_ptr<IGPUDescriptorSetLayout> createGPUDescriptorSetLayout_impl(const IGPUDescriptorSetLayout::SBinding* _begin, const IGPUDescriptorSetLayout::SBinding* _end) override final
    {
        return core::make_smart_refctd_ptr<IGPUDescriptorSetLayout>(this, _begin, _end);//there's no COpenGLDescriptorSetLayout (no need for such)
    }
    core::smart_refctd_ptr<IGPUPipelineLayout> createGPUPipelineLayout_impl(
        const asset::SPushConstantRange* const _pcRangesBegin, const asset::SPushConstantRange* const _pcRangesEnd,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
    ) override final
    {
        return core::make_smart_refctd_ptr<COpenGLPipelineLayout>(
            this,
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
        IGPUSpecializedShader** _shaders, IGPUSpecializedShader** _shadersEnd,
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
            params.shaders[ix] = core::smart_refctd_ptr<IGPUSpecializedShader>(*s);
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
        return core::make_smart_refctd_ptr<IGPUGraphicsPipeline>(this, std::move(params)); // theres no COpenGLGraphicsPipeline (no need for such)
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
    CThreadHandler<FunctionTableType> m_threadHandler;
    FeaturesType* m_glfeatures;
    EGLConfig m_config;
    std::pair<EGLint, EGLint> m_gl_ver;

    SDebugCallback* m_dbgCb;
};

}
}

#endif