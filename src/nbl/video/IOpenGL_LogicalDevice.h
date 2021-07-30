#ifndef __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include <variant>

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/CEGL.h"
#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/ILogger.h"
#include "nbl/video/COpenGLComputePipeline.h"
#include "nbl/video/COpenGLRenderpassIndependentPipeline.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/ISwapchain.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/utils/ISPIRVOptimizer.h"
#include "COpenGLBuffer.h"
#include "nbl/video/COpenGLBufferView.h"
#include "nbl/video/COpenGLImage.h"
#include "nbl/video/COpenGLImageView.h"
#include "nbl/video/COpenGLFramebuffer.h"
#include "COpenGLSync.h"
#include "COpenGLSpecializedShader.h"
#include "nbl/video/COpenGLSampler.h"
#include "nbl/video/COpenGLPipelineCache.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLDebug.h"

#ifndef EGL_CONTEXT_OPENGL_NO_ERROR_KHR
#	define EGL_CONTEXT_OPENGL_NO_ERROR_KHR 0x31B3
#endif

namespace nbl {
namespace video
{

namespace impl
{
    class IOpenGL_LogicalDeviceBase
    {
    public:
        static inline constexpr uint32_t MaxQueueCount = 8u;
        static inline constexpr uint32_t MaxGlNamesForSingleObject = MaxQueueCount + 1u;

        /*
        template <typename CRTP>
        struct SRequestBase
        {
            static size_t neededMemorySize(const CRTP&) { return 0ull; }
            static void copyContentsToOwnedMemory(CRTP&, void*) {}
        };
        */

        enum E_REQUEST_TYPE
        {
            // GL pipelines and vaos are kept, created and destroyed in COpenGL_Queue internal thread
            ERT_BUFFER_DESTROY,
            ERT_TEXTURE_DESTROY,
            ERT_SWAPCHAIN_DESTROY,
            ERT_SYNC_DESTROY,
            ERT_SAMPLER_DESTROY,
            //ERT_GRAPHICS_PIPELINE_DESTROY,
            ERT_PROGRAM_DESTROY,

            ERT_BUFFER_CREATE,
            ERT_BUFFER_VIEW_CREATE,
            ERT_IMAGE_CREATE,
            ERT_IMAGE_VIEW_CREATE,
            ERT_SWAPCHAIN_CREATE,
            ERT_EVENT_CREATE,
            ERT_FENCE_CREATE,
            ERT_SAMPLER_CREATE,
            ERT_RENDERPASS_INDEPENDENT_PIPELINE_CREATE,
            ERT_COMPUTE_PIPELINE_CREATE,
            //ERT_GRAPHICS_PIPELINE_CREATE,

            // non-create requests
            ERT_GET_EVENT_STATUS,
            ERT_RESET_EVENT,
            ERT_SET_EVENT,
            ERT_RESET_FENCES,
            ERT_WAIT_FOR_FENCES,
            ERT_GET_FENCE_STATUS,
            ERT_FLUSH_MAPPED_MEMORY_RANGES,
            ERT_INVALIDATE_MAPPED_MEMORY_RANGES,
            ERT_MAP_BUFFER_RANGE,
            ERT_UNMAP_BUFFER,
            //BIND_BUFFER_MEMORY,

            ERT_CTX_MAKE_CURRENT,

            ERT_WAIT_IDLE
        };

        constexpr static inline bool isDestroyRequest(E_REQUEST_TYPE rt)
        {
            return (rt < ERT_BUFFER_CREATE);
        }
        constexpr static inline bool isCreationRequest(E_REQUEST_TYPE rt)
        {
            return !isDestroyRequest(rt) && (rt < ERT_GET_EVENT_STATUS);
        }

        template <E_REQUEST_TYPE rt>
        struct SRequest_Destroy
        {
            static_assert(isDestroyRequest(rt));
            static inline constexpr E_REQUEST_TYPE type = rt;
            using retval_t = void;
            
            GLuint glnames[MaxGlNamesForSingleObject];
            uint32_t count;
        };
        struct SRequestSyncDestroy
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SYNC_DESTROY;
            using retval_t = void;

            GLsync glsync;
        };
        struct SRequestEventCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_EVENT_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUEvent>;
        };
        struct SRequestFenceCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_FENCE_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUFence>;

            IGPUFence::E_CREATE_FLAGS flags;
        };
        struct SRequestBufferCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_BUFFER_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUBuffer>;
            IDriverMemoryBacked::SDriverMemoryRequirements mreqs;
            bool canModifySubdata;
        };
        struct SRequestBufferViewCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_BUFFER_VIEW_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUBufferView>;
            core::smart_refctd_ptr<IGPUBuffer> buffer;
            asset::E_FORMAT format;
            size_t offset;
            size_t size;
        };
        struct SRequestImageCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_IMAGE_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUImage>;
            IGPUImage::SCreationParams params;
        };
        struct SRequestImageViewCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_IMAGE_VIEW_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUImageView>;
            IGPUImageView::SCreationParams params;
        };
        struct SRequestSamplerCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SAMPLER_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUSampler>;
            IGPUSampler::SParams params;
            bool is_gles;
        };
        struct SRequestRenderpassIndependentPipelineCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_RENDERPASS_INDEPENDENT_PIPELINE_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>;
            const IGPURenderpassIndependentPipeline::SCreationParams* params;
            uint32_t count;
            IGPUPipelineCache* pipelineCache;
        };
        struct SRequestComputePipelineCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_COMPUTE_PIPELINE_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUComputePipeline>;
            const IGPUComputePipeline::SCreationParams* params;
            uint32_t count;
            IGPUPipelineCache* pipelineCache;
        };

        //
        // Non-create requests:
        //
        struct SRequestGetEventStatus
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_GET_EVENT_STATUS;
            using retval_t = IGPUEvent::E_STATUS;
            const IGPUEvent* event;
        };
        struct SRequestResetEvent
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_RESET_EVENT;
            using retval_t = IGPUEvent::E_STATUS;
            IGPUEvent* event;
        };
        struct SRequestSetEvent
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SET_EVENT;
            using retval_t = IGPUEvent::E_STATUS;
            IGPUEvent* event;
        };
        struct SRequestResetFences
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_RESET_FENCES;
            using retval_t = void;
            core::SRange<core::smart_refctd_ptr<IGPUFence>> fences = { nullptr, nullptr };
        };
        struct SRequestWaitForFences
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_WAIT_FOR_FENCES;
            using retval_t = IGPUFence::E_STATUS;
            core::SRange<IGPUFence*> fences = { nullptr, nullptr };
            bool waitForAll;
            uint64_t timeout;
        };
        struct SRequestGetFenceStatus
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_GET_FENCE_STATUS;
            using retval_t = IGPUFence::E_STATUS;
            IGPUFence* fence;
        };
        struct SRequestFlushMappedMemoryRanges
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_FLUSH_MAPPED_MEMORY_RANGES;
            using retval_t = void;
            core::SRange<const IDriverMemoryAllocation::MappedMemoryRange> memoryRanges = { nullptr, nullptr };
        };
        struct SRequestInvalidateMappedMemoryRanges
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_INVALIDATE_MAPPED_MEMORY_RANGES;
            using retval_t = void;
            core::SRange<const IDriverMemoryAllocation::MappedMemoryRange> memoryRanges = { nullptr, nullptr };
        };
        struct SRequestMapBufferRange
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_MAP_BUFFER_RANGE;
            using retval_t = void*;

            core::smart_refctd_ptr<IDriverMemoryAllocation> buf;
            GLintptr offset;
            GLsizeiptr size;
            GLbitfield flags;
        };
        struct SRequestUnmapBuffer
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_UNMAP_BUFFER;
            using retval_t = void;

            core::smart_refctd_ptr<IDriverMemoryAllocation> buf;
        };
        struct SRequestMakeCurrent
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_CTX_MAKE_CURRENT;
            using retval_t = void;
            bool bind = true; // bind/unbind context flag
        };

        struct SRequestWaitIdle
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_WAIT_IDLE;
            using retval_t = void;
        };
    };

/*
    template <>
    size_t IOpenGL_LogicalDeviceBase::SRequestBase<IOpenGL_LogicalDeviceBase::SRequestResetFences>::neededMemorySize(const SRequestResetFences& x)
    {
        return x.fences.size() * sizeof(core::smart_refctd_ptr<IGPUFence>);
    }
    template <>
    void IOpenGL_LogicalDeviceBase::SRequestBase<IOpenGL_LogicalDeviceBase::SRequestResetFences>::copyContentsToOwnedMemory(SRequestResetFences& x, void* mem)
    {

    }

    template <>
    size_t IOpenGL_LogicalDeviceBase::SRequestBase<IOpenGL_LogicalDeviceBase::SRequestFlushMappedMemoryRanges>::neededMemorySize(const SRequestFlushMappedMemoryRanges& x)
    { 
        return x.memoryRanges.size() * sizeof(IDriverMemoryAllocation::MappedMemoryRange);
    }
    template <>
    void IOpenGL_LogicalDeviceBase::SRequestBase<IOpenGL_LogicalDeviceBase::SRequestFlushMappedMemoryRanges>::copyContentsToOwnedMemory(SRequestFlushMappedMemoryRanges& x, void* mem)
    {

    }

    template <>
    size_t IOpenGL_LogicalDeviceBase::SRequestBase<IOpenGL_LogicalDeviceBase::SRequestInvalidateMappedMemoryRanges>::neededMemorySize(const SRequestInvalidateMappedMemoryRanges& x)
    {
        return x.memoryRanges.size() * sizeof(IDriverMemoryAllocation::MappedMemoryRange);
    }
    template <>
    void IOpenGL_LogicalDeviceBase::SRequestBase<IOpenGL_LogicalDeviceBase::SRequestInvalidateMappedMemoryRanges>::copyContentsToOwnedMemory(SRequestInvalidateMappedMemoryRanges& x, void* mem)
    {

    }
*/
}

// Common base for GL and GLES logical devices
// All COpenGL* objects (buffers, images, views...) will keep pointer of this type
// Implementation of both GL and GLES is the same code (see COpenGL_LogicalDevice) thanks to IOpenGL_FunctionTable abstraction layer
class IOpenGL_LogicalDevice : public ILogicalDevice, protected impl::IOpenGL_LogicalDeviceBase
{
protected:
    system::logger_opt_smart_ptr m_logger;
    struct SGLContext
    {
        EGLContext ctx = EGL_NO_CONTEXT;
        EGLSurface pbuffer = EGL_NO_SURFACE;
    };

    static EGLContext createGLContext(EGLenum apiType, const egl::CEGL* egl, EGLint major, EGLint minor, EGLConfig config, EGLContext master = EGL_NO_CONTEXT)
    {
        egl->call.peglBindAPI(apiType);

        const EGLint ctx_attributes[] = {
            EGL_CONTEXT_MAJOR_VERSION, major,
            EGL_CONTEXT_MINOR_VERSION, minor,
// ANGLE validation is bugged and produces false positives, this flag turns off validation (glGetError wont ever return non-zero value then)
#ifdef _NBL_PLATFORM_ANDROID_
            EGL_CONTEXT_OPENGL_NO_ERROR_KHR, EGL_TRUE,
#endif
// ANGLE does not support debug contexts
#if defined(_NBL_DEBUG) && !defined(_NBL_PLATFORM_ANDROID_)
            EGL_CONTEXT_OPENGL_DEBUG, EGL_TRUE,
#endif
            //EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, // core profile is default setting

            EGL_NONE
        };

        EGLContext ctx = egl->call.peglCreateContext(egl->display, config, master, ctx_attributes);
        assert(ctx != EGL_NO_CONTEXT);

        return ctx;
    }
    static SGLContext createWindowlessGLContext(EGLenum apiType, const egl::CEGL* egl, EGLint major, EGLint minor, EGLConfig config, EGLContext master = EGL_NO_CONTEXT)
    {
        SGLContext retval;

        retval.ctx = createGLContext(apiType, egl, major, minor, config, master);

        // why not 1x1?
        const EGLint pbuffer_attributes[] = {
            EGL_WIDTH, 128,
            EGL_HEIGHT, 128,

            EGL_NONE
        };
        retval.pbuffer = egl->call.peglCreatePbufferSurface(egl->display, config, pbuffer_attributes);
        assert(retval.pbuffer != EGL_NO_SURFACE);

        return retval;
    }

    struct SRequest : public system::impl::IAsyncQueueDispatcherBase::request_base_t
    {
        using params_variant_t = std::variant<
            SRequestEventCreate,
            SRequestFenceCreate,
            SRequestBufferCreate,
            SRequestBufferViewCreate,
            SRequestImageCreate,
            SRequestImageViewCreate,
            SRequestSamplerCreate,
            SRequestRenderpassIndependentPipelineCreate,
            SRequestComputePipelineCreate,

            SRequest_Destroy<ERT_BUFFER_DESTROY>,
            SRequest_Destroy<ERT_TEXTURE_DESTROY>,
            SRequest_Destroy<ERT_SWAPCHAIN_DESTROY>,
            SRequest_Destroy<ERT_SAMPLER_DESTROY>,
            SRequest_Destroy<ERT_PROGRAM_DESTROY>,
            SRequestSyncDestroy,

            SRequestGetEventStatus,
            SRequestResetEvent,
            SRequestSetEvent,
            SRequestResetFences,
            SRequestWaitForFences,
            SRequestGetFenceStatus,
            SRequestFlushMappedMemoryRanges,
            SRequestInvalidateMappedMemoryRanges,
            SRequestMapBufferRange,
            SRequestUnmapBuffer,

            SRequestMakeCurrent,

            SRequestWaitIdle
        >;

        E_REQUEST_TYPE type;
        params_variant_t params_variant;

        // cast to `RequestParams::retval_t*`
        void* pretval;
    };

    template <typename FunctionTableType>
    class CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler<FunctionTableType>, SRequest, 256u, FunctionTableType>
    {
        using base_t = system::IAsyncQueueDispatcher<CThreadHandler<FunctionTableType>, SRequest, 256u, FunctionTableType>;
        friend base_t;
        using FeaturesType = typename FunctionTableType::features_t;
        system::logger_opt_smart_ptr&& m_logger;
    public:
        CThreadHandler(IOpenGL_LogicalDevice* dev,
            const egl::CEGL* _egl,
            FeaturesType* _features,
            uint32_t _qcount,
            const SGLContext& glctx,
            SDebugCallback* _dbgCb,
            system::logger_opt_smart_ptr&& logger) :
            m_queueCount(_qcount),
            egl(_egl),
            thisCtx(glctx.ctx), pbuffer(glctx.pbuffer),
            features(_features),
            device(dev),
            m_dbgCb(_dbgCb),
            m_logger(std::move(logger))
        {
        }

        EGLContext getContext() const
        {
            return thisCtx;
        }

        EGLSurface getSurface() const
        {
            return pbuffer;
        }

        uint32_t getQueueCount() const
        {
            return m_queueCount;
        }

        template <typename RequestParams>
        void waitForRequestCompletion(SRequest& req)
        {
            auto lk = req.wait();

            // clear params, just to make sure no refctd ptr is holding an object longer than it needs to
            std::get<RequestParams>(req.params_variant) = RequestParams{};
        }

        void init(FunctionTableType* state_ptr)
        {
            egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

            EGLBoolean mcres = egl->call.peglMakeCurrent(egl->display, pbuffer, pbuffer, thisCtx);
            assert(mcres == EGL_TRUE);

            new (state_ptr) FunctionTableType(egl, features, system::logger_opt_smart_ptr(m_logger));
            auto* gl = state_ptr;
            if (m_dbgCb)
                gl->extGlDebugMessageCallback(&opengl_debug_callback, m_dbgCb);
        }

        // RequestParams must be one of request parameter structs
        template <typename RequestParams>
        void request_impl(SRequest& req, RequestParams&& params, typename RequestParams::retval_t* pretval = nullptr)
        {
            req.type = params.type;
            req.params_variant = std::move(params);
            if constexpr (!std::is_void_v<typename RequestParams::retval_t>)
            {
                assert(pretval);
                req.pretval = pretval;
            }
            else
            {
                req.pretval = nullptr;
            }
        }

        void process_request(SRequest& req, FunctionTableType& _gl)
        {
            IOpenGL_FunctionTable& gl = static_cast<IOpenGL_FunctionTable&>(_gl);
            switch (req.type)
            {
            case ERT_BUFFER_DESTROY:
            {
                auto& p = std::get<SRequest_Destroy<ERT_BUFFER_DESTROY>>(req.params_variant);
                gl.glBuffer.pglDeleteBuffers(p.count, p.glnames);
            }
                break;
            case ERT_TEXTURE_DESTROY:
            {
                auto& p = std::get<SRequest_Destroy<ERT_TEXTURE_DESTROY>>(req.params_variant);
                gl.glTexture.pglDeleteTextures(p.count, p.glnames);
            }
                break;
            case ERT_SYNC_DESTROY:
            {
                auto& p = std::get<SRequestSyncDestroy>(req.params_variant);
                gl.glSync.pglDeleteSync(p.glsync);
            }
                break;
            case ERT_SAMPLER_DESTROY:
            {
                auto& p = std::get<SRequest_Destroy<ERT_SAMPLER_DESTROY>>(req.params_variant);
                gl.glTexture.pglDeleteSamplers(p.count, p.glnames);
            }
                break;
            case ERT_PROGRAM_DESTROY:
            {
                auto& p = std::get<SRequest_Destroy<ERT_PROGRAM_DESTROY>>(req.params_variant);
                assert(p.count == 1u);
                gl.glShader.pglDeleteProgram(p.glnames[0]);
            }
                break;

            case ERT_BUFFER_CREATE:
            {
                auto& p = std::get<SRequestBufferCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUBuffer>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUBuffer>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLBuffer>(device, &gl, p.mreqs, p.canModifySubdata);
            }
                break;
            case ERT_BUFFER_VIEW_CREATE:
            {
                auto& p = std::get<SRequestBufferViewCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUBufferView>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUBufferView>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLBufferView>(device, &gl, std::move(p.buffer), p.format, p.offset, p.size);
            }
                break;
            case ERT_IMAGE_CREATE:
            {
                auto& p = std::get<SRequestImageCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUImage>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUImage>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLImage>(device, &gl, std::move(p.params));
            }
                break;
            case ERT_IMAGE_VIEW_CREATE:
            {
                auto& p = std::get<SRequestImageViewCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUImageView>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUImageView>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLImageView>(device, &gl, std::move(p.params));
            }
                break;
            case ERT_SAMPLER_CREATE:
            {
                auto& p = std::get<SRequestSamplerCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUSampler>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUSampler>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLSampler>(device, &gl, p.params);
            }
                break;
            case ERT_RENDERPASS_INDEPENDENT_PIPELINE_CREATE:
            {
                auto& p = std::get<SRequestRenderpassIndependentPipelineCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>*>(req.pretval);
                for (uint32_t i = 0u; i < p.count; ++i)
                    pretval[i] = createRenderpassIndependentPipeline(gl, p.params[i], p.pipelineCache);
            }
                break;
            case ERT_COMPUTE_PIPELINE_CREATE:
            {
                auto& p = std::get<SRequestComputePipelineCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUComputePipeline>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUComputePipeline>*>(req.pretval);
                for (uint32_t i = 0u; i < p.count; ++i)
                    pretval[i] = createComputePipeline(gl, p.params[i], p.pipelineCache);
            }
                break;
            case ERT_FENCE_CREATE:
            {
                auto& p = std::get<SRequestFenceCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUFence>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUFence>*>(req.pretval);
                if (p.flags & IGPUFence::ECF_SIGNALED_BIT)
                    pretval[0] = core::make_smart_refctd_ptr<COpenGLFence>(device, device, &gl);
                else
                    pretval[0] = core::make_smart_refctd_ptr<COpenGLFence>(device);
            }
                break;

            case ERT_FLUSH_MAPPED_MEMORY_RANGES:
            {
                auto& p = std::get<SRequestFlushMappedMemoryRanges>(req.params_variant);
                for (auto mrng : p.memoryRanges)
                    gl.extGlFlushMappedNamedBufferRange(static_cast<COpenGLBuffer*>(mrng.memory)->getOpenGLName(), mrng.offset, mrng.length);
            }
                break;
            case ERT_INVALIDATE_MAPPED_MEMORY_RANGES:
            {
                gl.glSync.pglMemoryBarrier(gl.CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            }
                break;
            case ERT_MAP_BUFFER_RANGE:
            {
                auto& p = std::get<SRequestMapBufferRange>(req.params_variant);

                void** pretval = reinterpret_cast<void**>(req.pretval);
                pretval[0] = gl.extGlMapNamedBufferRange(static_cast<COpenGLBuffer*>(p.buf.get())->getOpenGLName(), p.offset, p.size, p.flags);
            }
                break;
            case ERT_UNMAP_BUFFER:
            {
                auto& p = std::get<SRequestUnmapBuffer>(req.params_variant);

                gl.extGlUnmapNamedBuffer(static_cast<COpenGLBuffer*>(p.buf.get())->getOpenGLName());
            }
                break;
            case ERT_GET_FENCE_STATUS:
            {
                auto& p = std::get<SRequestGetFenceStatus>(req.params_variant);
                auto* glfence = static_cast<COpenGLFence*>(p.fence);
                IGPUFence::E_STATUS* retval = reinterpret_cast<IGPUFence::E_STATUS*>(req.pretval);

                retval[0] = glfence->getStatus(&gl);
            }
                break;
            case ERT_WAIT_FOR_FENCES:
            {
                auto& p = std::get<SRequestWaitForFences>(req.params_variant);
                uint32_t _count = p.fences.size();
                IGPUFence** _fences = p.fences.begin();
                bool _waitAll = p.waitForAll;
                uint64_t _timeout = p.timeout;

                IGPUFence::E_STATUS* retval = reinterpret_cast<IGPUFence::E_STATUS*>(req.pretval);
                retval[0] = waitForFences(gl, _count, _fences, _waitAll, _timeout);
            }
                break;
            case ERT_CTX_MAKE_CURRENT:
            {
                auto& p = std::get<SRequestMakeCurrent>(req.params_variant);
                EGLBoolean mcres = EGL_FALSE;
                if (p.bind)
                    mcres = egl->call.peglMakeCurrent(egl->display, pbuffer, pbuffer, thisCtx);
                else
                    mcres = egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                assert(mcres);
            }
                break;
            case ERT_WAIT_IDLE:
            {
                gl.glGeneral.pglFinish();
            }
                break;
            default: 
                break;
            }
            gl.glGeneral.pglFlush();
            // created GL object must be in fact ready when request gets reported as ready
            // @matt - needed?
            if (isCreationRequest(req.type))
                gl.glGeneral.pglFinish();
        }

        void exit(FunctionTableType* gl)
        {
            gl->glGeneral.pglFinish();
            gl->~FunctionTableType();

            egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // detach ctx from thread
            egl->call.peglDestroyContext(egl->display, thisCtx);
            egl->call.peglDestroySurface(egl->display, pbuffer);
        }

    private:
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline(IOpenGL_FunctionTable& gl, const IGPURenderpassIndependentPipeline::SCreationParams& params, IGPUPipelineCache* _pipelineCache)
        {
            //_parent parameter is ignored

            using GLPpln = COpenGLRenderpassIndependentPipeline;

            IGPUSpecializedShader* shaders_array[IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT]{};
            uint32_t shaderCount = 0u;
            for (uint32_t i = 0u; i < IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++i)
                if (params.shaders[i])
                    shaders_array[shaderCount++] = params.shaders[i].get();

            auto shaders = core::SRange<IGPUSpecializedShader*>(shaders_array, shaders_array+shaderCount);
            auto vsIsPresent = [&shaders] {
                return std::find_if(shaders.begin(), shaders.end(), [](IGPUSpecializedShader* shdr) {return shdr->getStage() == asset::ISpecializedShader::ESS_VERTEX; }) != shaders.end();
            };

            asset::ISpecializedShader::E_SHADER_STAGE lastVertexLikeStage = asset::ISpecializedShader::ESS_VERTEX;
            for (uint32_t i = 0u; i < shaders.size(); ++i)
            {
                auto stage = shaders.begin()[shaders.size()-1u-i]->getStage();
                if (stage != asset::ISpecializedShader::ESS_FRAGMENT)
                {
                    lastVertexLikeStage = stage;
                    break;
                }
            }

            auto layout = params.layout;
            if (!layout || !vsIsPresent())
                return nullptr;

            GLuint GLnames[COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT]{};
            COpenGLSpecializedShader::SProgramBinary binaries[COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT];

            bool needClipControlWorkaround = false;
            if (gl.isGLES())
            {
                needClipControlWorkaround = !gl.getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_clip_control);
            }

            COpenGLPipelineCache* cache = static_cast<COpenGLPipelineCache*>(_pipelineCache);
            COpenGLPipelineLayout* gllayout = static_cast<COpenGLPipelineLayout*>(layout.get());
            for (auto shdr = shaders.begin(); shdr != shaders.end(); ++shdr)
            {
                COpenGLSpecializedShader* glshdr = static_cast<COpenGLSpecializedShader*>(*shdr);

                auto stage = glshdr->getStage();
                uint32_t ix = core::findLSB<uint32_t>(stage);
                assert(ix < COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT);

                COpenGLPipelineCache::SCacheKey key{ glshdr->getSpirvHash(), glshdr->getSpecializationInfo(), core::smart_refctd_ptr_static_cast<COpenGLPipelineLayout>(layout) };
                auto bin = cache ? cache->find(key) : COpenGLSpecializedShader::SProgramBinary{ 0,nullptr };
                if (bin.binary)
                {
                    const GLuint GLname = gl.glShader.pglCreateProgram();
                    gl.glShader.pglProgramBinary(GLname, bin.format, bin.binary->data(), bin.binary->size());
                    GLnames[ix] = GLname;
                    binaries[ix] = bin;

                    continue;
                }
                std::tie(GLnames[ix], bin) = glshdr->compile(&gl, needClipControlWorkaround && (stage == lastVertexLikeStage), gllayout, cache ? cache->findParsedSpirv(key.hash) : nullptr);
                binaries[ix] = bin;

                if (cache)
                {
                    cache->insertParsedSpirv(key.hash, glshdr->getSpirv());

                    COpenGLPipelineCache::SCacheVal val{ std::move(bin) };
                    cache->insert(std::move(key), std::move(val));
                }
            }

            return core::make_smart_refctd_ptr<COpenGLRenderpassIndependentPipeline>(
                device, device, &gl,
                std::move(layout),
                shaders.begin(), shaders.end(),
                params.vertexInput, params.blend, params.primitiveAssembly, params.rasterization,
                getNameCountForSingleEngineObject(), 0u, GLnames, binaries
            );
        }
        core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline(IOpenGL_FunctionTable& gl, const IGPUComputePipeline::SCreationParams& params, IGPUPipelineCache* _pipelineCache)
        {
            if (!params.layout || !params.shader)
                return nullptr;
            if (params.shader->getStage() != asset::ISpecializedShader::ESS_COMPUTE)
                return nullptr;

            GLuint GLname = 0u;
            COpenGLSpecializedShader::SProgramBinary binary;
            COpenGLPipelineCache* cache = static_cast<COpenGLPipelineCache*>(_pipelineCache);
            auto layout = core::smart_refctd_ptr_static_cast<COpenGLPipelineLayout>(params.layout);
            auto glshdr = core::smart_refctd_ptr_static_cast<COpenGLSpecializedShader>(params.shader);

            COpenGLPipelineCache::SCacheKey key{ glshdr->getSpirvHash(), glshdr->getSpecializationInfo(), layout };
            auto bin = cache ? cache->find(key) : COpenGLSpecializedShader::SProgramBinary{ 0,nullptr };
            if (bin.binary)
            {
                const GLuint GLshader = gl.glShader.pglCreateProgram();
                gl.glShader.pglProgramBinary(GLname, bin.format, bin.binary->data(), bin.binary->size());
                GLname = GLshader;
                binary = bin;
            }
            else
            {
                std::tie(GLname, bin) = glshdr->compile(&gl, false, layout.get(), cache ? cache->findParsedSpirv(key.hash) : nullptr);
                binary = bin;

                if (cache)
                {
                    cache->insertParsedSpirv(key.hash, glshdr->getSpirv());

                    COpenGLPipelineCache::SCacheVal val{ std::move(bin) };
                    cache->insert(std::move(key), std::move(val));
                }
            }

            return core::make_smart_refctd_ptr<COpenGLComputePipeline>(device, device, &gl, core::smart_refctd_ptr<IGPUPipelineLayout>(layout.get()), core::smart_refctd_ptr<IGPUSpecializedShader>(glshdr.get()), getNameCountForSingleEngineObject(), 0u, GLname, binary);
        }
        IGPUFence::E_STATUS waitForFences(IOpenGL_FunctionTable& gl, uint32_t _count, IGPUFence** _fences, bool _waitAll, uint64_t _timeout)
        {
            if (_waitAll)
            {
                using clock_t = std::chrono::high_resolution_clock;

                auto start = clock_t::time_point();
                for (uint32_t i = 0u; i < _count; ++i)
                {
                    COpenGLFence* fence = static_cast<COpenGLFence*>(_fences[i]);
                    IGPUFence::E_STATUS status;
                    
                    uint64_t timeout = _timeout;
                    if (start == clock_t::time_point())
                        start = clock_t::now();
                    else
                    {
                        const uint64_t dt = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t::now() - start).count();
                        if (dt >= timeout)
                            return IGPUFence::ES_TIMEOUT;
                        timeout -= dt;
                    }

                    status = fence->wait(&gl, timeout);
                    if (status != IGPUFence::ES_SUCCESS)
                        return status;
                }
                return IGPUFence::ES_SUCCESS;
            }
            else
            {
                for (uint32_t i = 0u; i < _count; ++i)
                {
                    COpenGLFence* fence = static_cast<COpenGLFence*>(_fences[i]);
                    return fence->wait(&gl, _timeout);
                }
            }
        }

        // currently used by shader programs only
        // they theoretically can be shared between contexts however, because uniforms state is program's state, we cant use that
        // because they are shareable, all GL names cane be created in the same thread at once though
        uint32_t getNameCountForSingleEngineObject() const
        {
            return m_queueCount + 1u; // +1 because of this context (the one in logical device), probably not needed though
        }

        uint32_t m_queueCount;

        const egl::CEGL* egl;
        EGLContext thisCtx;
        EGLSurface pbuffer;
        FeaturesType* features;

        IOpenGL_LogicalDevice* device;
        SDebugCallback* m_dbgCb;
    };

protected:
    const egl::CEGL* m_egl;
    core::smart_refctd_dynamic_array<std::string> m_supportedGLSLExtsNames;

public:
    IOpenGL_LogicalDevice(const egl::CEGL* _egl,
        E_API_TYPE api_type, 
        const SCreationParams& params, 
        core::smart_refctd_ptr<system::ISystem>&& s,
        core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc,
        system::logger_opt_smart_ptr&& logger) : ILogicalDevice(api_type, params, std::move(s), std::move(glslc)), m_egl(_egl), m_logger(std::move(logger))
    {

    }

    const core::smart_refctd_dynamic_array<std::string> getSupportedGLSLExtensions() const override
    {
        return m_supportedGLSLExtsNames;
    }

    virtual void destroyFramebuffer(COpenGLFramebuffer::hash_t fbohash) = 0;
    virtual void destroyPipeline(COpenGLRenderpassIndependentPipeline* pipeline) = 0;
    virtual void destroyTexture(GLuint img) = 0;
    virtual void destroyBuffer(GLuint buf) = 0;
    virtual void destroySampler(GLuint s) = 0;
    virtual void destroySpecializedShader(uint32_t count, const GLuint* programs) = 0;
    virtual void destroySync(GLsync sync) = 0;
};

}
}

#endif
