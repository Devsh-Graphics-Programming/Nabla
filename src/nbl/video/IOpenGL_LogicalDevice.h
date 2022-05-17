#ifndef __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/ILogger.h"

#include <variant>

#include "nbl/asset/utils/ISPIRVOptimizer.h"

#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/ISwapchain.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/debug/COpenGLDebugCallback.h"

#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/CEGL.h"
#include "nbl/video/COpenGLComputePipeline.h"
#include "nbl/video/COpenGLRenderpassIndependentPipeline.h"
#include "nbl/video/COpenGLBuffer.h"
#include "nbl/video/COpenGLBufferView.h"
#include "nbl/video/COpenGLImage.h"
#include "nbl/video/COpenGLImageView.h"
#include "nbl/video/COpenGLFramebuffer.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/video/COpenGLSpecializedShader.h"
#include "nbl/video/COpenGLSampler.h"
#include "nbl/video/COpenGLPipelineCache.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLQueryPool.h"

#ifndef EGL_CONTEXT_OPENGL_NO_ERROR_KHR
#	define EGL_CONTEXT_OPENGL_NO_ERROR_KHR 0x31B3
#endif

namespace nbl::video
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
        
        //! Request Type
        //! Cross-context Sync Sensitive Request Types:
        //!     some requests need to sync between contexts such as creation requests
        //!     for such requests ensure `incrementProgressionSync` gets called on your request, by direct function call or making it a creation request (see isCreationRequest)
        //!     [WARNING] otherwise you may encounter a deadlock waiting for master context
        enum E_REQUEST_TYPE : uint8_t
        {
            // GL pipelines and vaos are kept, created and destroyed in COpenGL_Queue internal thread
            ERT_BUFFER_DESTROY,
            ERT_TEXTURE_DESTROY,
            ERT_SYNC_DESTROY,
            ERT_SAMPLER_DESTROY,
            //ERT_GRAPHICS_PIPELINE_DESTROY,
            ERT_PROGRAM_DESTROY,

            //! create requests
            //! creation requests are cross-context sync sensitive (see description above
            ERT_BUFFER_CREATE,
            ERT_BUFFER_CREATE2,
            ERT_BUFFER_VIEW_CREATE,
            ERT_IMAGE_CREATE,
            ERT_IMAGE_CREATE2,
            ERT_IMAGE_VIEW_CREATE,
            ERT_FENCE_CREATE,
            ERT_SAMPLER_CREATE,
            ERT_ALLOCATE,
            ERT_RENDERPASS_INDEPENDENT_PIPELINE_CREATE,
            ERT_COMPUTE_PIPELINE_CREATE,
            //ERT_GRAPHICS_PIPELINE_CREATE,

            // non-create requests
            ERT_WAIT_FOR_FENCES,
            ERT_GET_FENCE_STATUS,
            ERT_FLUSH_MAPPED_MEMORY_RANGES,
            ERT_INVALIDATE_MAPPED_MEMORY_RANGES,
            ERT_MAP_BUFFER_RANGE,
            ERT_UNMAP_BUFFER,
            
            ERT_SET_DEBUG_NAME,
            ERT_GET_QUERY_POOL_RESULTS,

            ERT_CTX_MAKE_CURRENT,

            ERT_WAIT_IDLE,

            ERT_INVALID
        };

        constexpr static inline bool isDestroyRequest(E_REQUEST_TYPE rt)
        {
            return (rt < ERT_BUFFER_CREATE);
        }
        constexpr static inline bool isCreationRequest(E_REQUEST_TYPE rt)
        {
            return !isDestroyRequest(rt) && rt<=ERT_COMPUTE_PIPELINE_CREATE;
        }
        constexpr static inline bool isWaitlessRequest(E_REQUEST_TYPE rt)
        {
            // we could actually make the creation waitless too, if we were careful
            // TODO: if we actually copied the range parameter we wouldn't have to wait on ERT_FLUSH_MAPPED_MEMORY_RANGES
            return isDestroyRequest(rt) || rt == ERT_UNMAP_BUFFER;
        }

        template <E_REQUEST_TYPE rt>
        struct SRequest_Destroy
        {
            static_assert(isDestroyRequest(rt));
            static inline constexpr E_REQUEST_TYPE type = rt;
            using retval_t = void;
            
            GLuint glnames[COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT*MaxGlNamesForSingleObject];
            uint32_t count;
        };
        struct SRequestSyncDestroy
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SYNC_DESTROY;
            using retval_t = void;

            GLsync glsync;
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
            IGPUBuffer::SCachedCreationParams cachedCreationParams;
        };
        struct SRequestBufferCreate2
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_BUFFER_CREATE2;
            using retval_t = core::smart_refctd_ptr<IGPUBuffer>;
            IGPUBuffer::SCachedCreationParams creationParams;
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
        struct SRequestImageCreate2
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_IMAGE_CREATE2;
            using retval_t = core::smart_refctd_ptr<IGPUImage>;
            uint32_t deviceLocalMemoryTypeBits;
            IGPUImage::SCreationParams creationParams;
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
        struct SRequestWaitForFences
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_WAIT_FOR_FENCES;
            using retval_t = IGPUFence::E_STATUS;
            using clock_t = std::chrono::steady_clock;
            core::SRange<IGPUFence*const,IGPUFence*const*,IGPUFence*const*> fences = { nullptr, nullptr };
            clock_t::time_point timeoutPoint;
            bool waitForAll;
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
        struct SRequestAllocate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_ALLOCATE;
            using retval_t = IDriverMemoryAllocator::SMemoryOffset;
            IOpenGLMemoryAllocation* dedicationAsAllocation = nullptr;
            core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> memoryAllocateFlags;
            core::bitflag<IDriverMemoryAllocation::E_MEMORY_PROPERTY_FLAGS> memoryPropertyFlags;
        };
        struct SRequestSetDebugName
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SET_DEBUG_NAME;
            using retval_t = void;

            GLenum id;
            GLuint object;
            GLsizei len;
            char label[IBackendObject::MAX_DEBUG_NAME_LENGTH+1U];
        };
        struct SRequestGetQueryPoolResults
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_GET_QUERY_POOL_RESULTS;
            using retval_t = void;
            core::smart_refctd_ptr<const IQueryPool> queryPool;
            uint32_t firstQuery;
            uint32_t queryCount;
            size_t dataSize;
            void * pData;
            uint64_t stride;
            core::bitflag<IQueryPool::E_QUERY_RESULTS_FLAGS> flags;
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
    static auto createWindowlessGLContext(EGLenum apiType, const egl::CEGL* egl, EGLint major, EGLint minor, EGLConfig config, EGLContext master = EGL_NO_CONTEXT)
    {
        egl::CEGL::Context retval;

        retval.ctx = createGLContext(apiType, egl, major, minor, config, master);

        // why not 1x1?
        const EGLint pbuffer_attributes[] = {
            EGL_WIDTH, 128,
            EGL_HEIGHT, 128,

            EGL_NONE
        };
        retval.surface = egl->call.peglCreatePbufferSurface(egl->display, config, pbuffer_attributes);
        assert(retval.surface != EGL_NO_SURFACE);

        return retval;
    }

    struct SRequest : public system::impl::IAsyncQueueDispatcherBase::request_base_t
    {
        using params_variant_t = std::variant<
            SRequestFenceCreate,
            SRequestAllocate,
            SRequestBufferCreate,
            SRequestBufferCreate2,
            SRequestBufferViewCreate,
            SRequestImageCreate,
            SRequestImageCreate2,
            SRequestImageViewCreate,
            SRequestSamplerCreate,
            SRequestRenderpassIndependentPipelineCreate,
            SRequestComputePipelineCreate,

            SRequest_Destroy<ERT_BUFFER_DESTROY>,
            SRequest_Destroy<ERT_TEXTURE_DESTROY>,
            SRequest_Destroy<ERT_SAMPLER_DESTROY>,
            SRequest_Destroy<ERT_PROGRAM_DESTROY>,
            SRequestSyncDestroy,

            SRequestWaitForFences,
            SRequestGetFenceStatus,
            SRequestFlushMappedMemoryRanges,
            SRequestInvalidateMappedMemoryRanges,
            SRequestMapBufferRange,
            SRequestUnmapBuffer,
            SRequestGetQueryPoolResults,

            SRequestSetDebugName,

            SRequestMakeCurrent,

            SRequestWaitIdle
        >;

        // lock when overwriting the request
        void reset()
        {
            if (isWaitlessRequest(type))
            {
                uint32_t expected = ES_READY;
                while (!state.compare_exchange_strong(expected,ES_RECORDING))
                {
                    state.wait(expected);
                    expected = ES_READY;
                }
                assert(expected==ES_READY);
            }
            else
                system::impl::IAsyncQueueDispatcherBase::request_base_t::reset();
        }

        params_variant_t params_variant;
        E_REQUEST_TYPE type = ERT_INVALID;

        // cast to `RequestParams::retval_t*`
        void* pretval;
    };

    template <typename FunctionTableType>
    class CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler<FunctionTableType>, SRequest, 256u, FunctionTableType>
    {
        using base_t = system::IAsyncQueueDispatcher<CThreadHandler<FunctionTableType>, SRequest, 256u, FunctionTableType>;
        friend base_t;
        using FeaturesType = typename FunctionTableType::features_t;
    public:
        CThreadHandler(
            IOpenGL_LogicalDevice* dev,
            std::atomic<GLsync>* const _masterContextSync,
            std::atomic<uint64_t>* const _masterContextCallsReturned,
            const egl::CEGL* _egl,
            const FeaturesType* _features,
            uint32_t _qcount,
            const egl::CEGL::Context& _glctx,
            const COpenGLDebugCallback* _dbgCb) :
            m_queueCount(_qcount),
            masterContextSync(_masterContextSync),
            masterContextCallsReturned(_masterContextCallsReturned),
            egl(_egl),
            glctx(_glctx),
            features(_features),
            device(dev),
            m_dbgCb(_dbgCb)
        {
        }

        uint32_t getQueueCount() const
        {
            return m_queueCount;
        }

        template <typename RequestParams>
        void waitForRequestCompletion(SRequest& req)
        {
            assert(!isWaitlessRequest(req.type));
            req.wait_ready();

            // clear params, just to make sure no refctd ptr is holding an object longer than it needs to
            std::get<RequestParams>(req.params_variant) = RequestParams{};

            req.discard_storage();
        }

        void init(FunctionTableType* state_ptr)
        {
            egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

            EGLBoolean mcres = egl->call.peglMakeCurrent(egl->display, glctx.surface, glctx.surface, glctx.ctx);
            assert(mcres == EGL_TRUE);

            auto logger = m_dbgCb->getLogger();
            new (state_ptr) FunctionTableType(egl,features,core::smart_refctd_ptr<system::ILogger>(logger));

            auto* gl = state_ptr;
            if (logger)
            {
                const char* vendor = reinterpret_cast<const char*>(gl->glGeneral.pglGetString(GL_VENDOR));
                const char* renderer = reinterpret_cast<const char*>(gl->glGeneral.pglGetString(GL_RENDERER));
                const char* version = reinterpret_cast<const char*>(gl->glGeneral.pglGetString(GL_VERSION));
                if constexpr (FunctionTableType::EGL_API_TYPE==EGL_OPENGL_API)
                    logger->log("Created OpenGL Logical Device. Vendor: %s Renderer: %s Version: %s",system::ILogger::ELL_INFO,vendor,renderer,version);
                else if (FunctionTableType::EGL_API_TYPE==EGL_OPENGL_ES_API)
                    logger->log("Created OpenGL ES Logical Device. Vendor: %s Renderer: %s Version: %s",system::ILogger::ELL_INFO,vendor,renderer,version);
            }

            #ifdef _NBL_DEBUG
            gl->glGeneral.pglEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            // TODO: debug message control (to exclude callback spam)
            #endif
            if (m_dbgCb)
                gl->extGlDebugMessageCallback(m_dbgCb->m_callback,m_dbgCb);

            gl->glGeneral.pglFinish();
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
                for (uint32_t i = 0u; i < p.count; ++i) 
                    gl.glShader.pglDeleteProgram(p.glnames[i]);
            }
                break;

            case ERT_BUFFER_CREATE:
            {
                auto& p = std::get<SRequestBufferCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUBuffer>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUBuffer>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLBuffer>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl, p.mreqs, p.cachedCreationParams);
            }
                break;
            case ERT_BUFFER_CREATE2:
            {
                auto& p = std::get<SRequestBufferCreate2>(req.params_variant);
                core::smart_refctd_ptr<IGPUBuffer>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUBuffer>*>(req.pretval);

                IDriverMemoryBacked::SDriverMemoryRequirements2 mreqs;
                mreqs.size = p.creationParams.declaredSize;
                mreqs.memoryTypeBits = 0xffffffffu;
                mreqs.alignmentLog2 = 0u; // TODO(Erfan) Alignment previously was 0u in getXXXMemoryRequirementsOnDedMem(). what to set here? get minXXXOffsetAlignment from physical device and deduce from usage?
                mreqs.prefersDedicatedAllocation = true;
                mreqs.requiresDedicatedAllocation = true;

                GLuint bufferName;
                gl.extGlCreateBuffers(1,&bufferName);
                if (bufferName!=0)
                {
                    pretval[0] = core::make_smart_refctd_ptr<COpenGLBuffer>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), mreqs, p.creationParams, bufferName);
                }
                else
                {
                    pretval[0] = nullptr;
                }
            }
                break;
            case ERT_BUFFER_VIEW_CREATE:
            {
                auto& p = std::get<SRequestBufferViewCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUBufferView>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUBufferView>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLBufferView>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl, std::move(p.buffer), p.format, p.offset, p.size);
            }
                break;
            case ERT_IMAGE_CREATE:
            {
                auto& p = std::get<SRequestImageCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUImage>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUImage>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLImage>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl, std::move(p.params));
            }
                break;
            case ERT_IMAGE_CREATE2:
            {
                auto& p = std::get<SRequestImageCreate2>(req.params_variant);
                core::smart_refctd_ptr<IGPUImage>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUImage>*>(req.pretval);

                GLenum internalFormat = getSizedOpenGLFormatFromOurFormat(&gl, p.creationParams.format);
                GLenum target;
                GLuint name;

                IDriverMemoryBacked::SDriverMemoryRequirements2 mreqs;
                mreqs.size = 0u; // TODO(Erfan) some approx of image size -> considering dimensions, mipLevels, arrayLayers, samples, minImageGranularity and texelBlockInfo (see ImageUploadUtility)
                mreqs.memoryTypeBits = p.deviceLocalMemoryTypeBits;
                mreqs.alignmentLog2 = 0u; // TODO(Erfan) Alignment previously was 0u in getXXXMemoryRequirementsOnDedMem(). what to set here? get minXXXOffsetAlignment from physical device and deduce from usage?
                mreqs.prefersDedicatedAllocation = true;
                mreqs.requiresDedicatedAllocation = true;

                GLsizei samples = p.creationParams.samples;
                switch (p.creationParams.type)
                {
                    case IGPUImage::ET_1D:
                        target = GL_TEXTURE_1D_ARRAY;
                        gl.extGlCreateTextures(target, 1, &name);
                        break;
                    case IGPUImage::ET_2D:
                        if (p.creationParams.flags & asset::IImage::ECF_CUBE_COMPATIBLE_BIT)
                            target = GL_TEXTURE_CUBE_MAP_ARRAY;
                        else
                            target = samples>1 ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY : GL_TEXTURE_2D_ARRAY;
                        gl.extGlCreateTextures(target, 1, &name);
                        break;
                    case IGPUImage::ET_3D:
                        target = GL_TEXTURE_3D;
                        gl.extGlCreateTextures(target, 1, &name);
                        break;
                    default:
                        assert(false);
                        break;
                }
                pretval[0] = core::make_smart_refctd_ptr<COpenGLImage>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), std::move(mreqs), std::move(p.creationParams), internalFormat, target, name);
            }
                break;
            case ERT_IMAGE_VIEW_CREATE:
            {
                auto& p = std::get<SRequestImageViewCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUImageView>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUImageView>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLImageView>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl, std::move(p.params));
            }
                break;
            case ERT_SAMPLER_CREATE:
            {
                auto& p = std::get<SRequestSamplerCreate>(req.params_variant);
                core::smart_refctd_ptr<IGPUSampler>* pretval = reinterpret_cast<core::smart_refctd_ptr<IGPUSampler>*>(req.pretval);
                pretval[0] = core::make_smart_refctd_ptr<COpenGLSampler>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl, p.params);
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
                    pretval[0] = core::make_smart_refctd_ptr<COpenGLFence>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl);
                else
                    pretval[0] = core::make_smart_refctd_ptr<COpenGLFence>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device));
                // only fence create should flush, nothing else needs to do flush or wait idle
                gl.glGeneral.pglFlush();
            }
                break;
            case ERT_FLUSH_MAPPED_MEMORY_RANGES:
            {
                auto& p = std::get<SRequestFlushMappedMemoryRanges>(req.params_variant);
                for (auto mrng : p.memoryRanges)
                    gl.extGlFlushMappedNamedBufferRange(static_cast<COpenGLBuffer*>(mrng.memory)->getOpenGLName(), mrng.offset, mrng.length);
                // unfortunately I have to make every other context wait on the master to ensure the flush completes
                incrementProgressionSync(_gl);
            }
                break;
            case ERT_INVALIDATE_MAPPED_MEMORY_RANGES:
            {
                // master context doesn't need to `glWaitSync` because under Vulkan API rules, the user must do a CPU wait before trying to access the pointer, ergo completion guaranteed by the time we call this
                gl.glSync.pglMemoryBarrier(gl.CLIENT_MAPPED_BUFFER_BARRIER_BIT);
            }
                break;
            case ERT_MAP_BUFFER_RANGE:
            {
                auto& p = std::get<SRequestMapBufferRange>(req.params_variant);

                void** pretval = reinterpret_cast<void**>(req.pretval);
                pretval[0] = gl.extGlMapNamedBufferRange(static_cast<COpenGLBuffer*>(p.buf.get())->getOpenGLName(), p.offset, p.size, p.flags);
                // unfortunately I have to make every other context wait on the master to ensure the mapping completes
                incrementProgressionSync(_gl);
            }
                break;
            case ERT_UNMAP_BUFFER:
            {
                auto& p = std::get<SRequestUnmapBuffer>(req.params_variant);

                gl.extGlUnmapNamedBuffer(static_cast<COpenGLBuffer*>(p.buf.get())->getOpenGLName());
                // unfortunately I have to make every other context wait on the master to ensure the unmapping completes
                incrementProgressionSync(_gl);
            }
                break;
            case ERT_ALLOCATE:
            {
                auto& p = std::get<SRequestAllocate>(req.params_variant);
                IDriverMemoryAllocator::SMemoryOffset* pretval = reinterpret_cast<IDriverMemoryAllocator::SMemoryOffset*>(req.pretval);
                IDriverMemoryAllocator::SMemoryOffset& retval = *pretval;
                if(p.dedicationAsAllocation)
                {
                    p.dedicationAsAllocation->initMemory(&gl, p.memoryAllocateFlags, p.memoryPropertyFlags);
                    retval.memory = core::smart_refctd_ptr<IDriverMemoryAllocation>(p.dedicationAsAllocation);
                    retval.offset = 0ull;
                }
                else
                {
                    retval.memory = nullptr;
                    retval.offset = IDriverMemoryAllocator::InvalidMemoryOffset;
                }
            }
                break;
            case ERT_GET_FENCE_STATUS:
            {
                auto& p = std::get<SRequestGetFenceStatus>(req.params_variant);
                auto* glfence = IBackendObject::device_compatibility_cast<COpenGLFence*>(p.fence, device);
                IGPUFence::E_STATUS* retval = reinterpret_cast<IGPUFence::E_STATUS*>(req.pretval);

                retval[0] = glfence->getStatus(&gl);
            }
                break;
            case ERT_WAIT_FOR_FENCES:
            {
                auto& p = std::get<SRequestWaitForFences>(req.params_variant);
                uint32_t _count = p.fences.size();
                IGPUFence*const *const _fences = p.fences.begin();

                *reinterpret_cast<IGPUFence::E_STATUS*>(req.pretval) = waitForFences(gl, _count, _fences, p.waitForAll, p.timeoutPoint);
            }
                break;
            case ERT_SET_DEBUG_NAME:
            {
                auto& p = std::get<SRequestSetDebugName>(req.params_variant);

                if (p.len)
                    gl.extGlObjectLabel(p.id, p.object, p.len, p.label);
                else
                    gl.extGlObjectLabel(p.id, p.object, 0u, nullptr); // remove debug name
            }
                break;
            case ERT_CTX_MAKE_CURRENT:
            {
                auto& p = std::get<SRequestMakeCurrent>(req.params_variant);
                EGLBoolean mcres = EGL_FALSE;
                if (p.bind)
                    mcres = egl->call.peglMakeCurrent(egl->display, glctx.surface, glctx.surface, glctx.ctx);
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
            // Nvidia's OpenGL nastily gaslights the user with plain wrong errors, i.e. about invalid offsets and sizes when doing buffer2buffer copies
            // there's nothing in the spec saying that I must flush after creating a buffer with ARB_buffer_storage on another context/thread in the sharelist
            // but all this undocumented goodness has finally reared its head
            // OpenGL spec is worse and looser than Vulkan, because we use DSA this affects us, if we didnt it wouldn't.
            // Flushing is a particular PITA because its a not a thing that synchronises with the CPU.
            // TODO: One could also want object creation to optionally only sync with a queue submission and not CPU (so a semaphore).
            if (req.type!=ERT_FENCE_CREATE && isCreationRequest(req.type))
                incrementProgressionSync(_gl);
        }

        void exit(FunctionTableType* gl)
        {
            gl->glGeneral.pglFinish();
            gl->~FunctionTableType();

            egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // detach ctx from thread
            egl->call.peglDestroyContext(egl->display, glctx.ctx);
            egl->call.peglDestroySurface(egl->display, glctx.surface);
        }

        egl::CEGL::Context glctx;
    private:
        core::smart_refctd_ptr<IGPURenderpassIndependentPipeline> createRenderpassIndependentPipeline(IOpenGL_FunctionTable& gl, const IGPURenderpassIndependentPipeline::SCreationParams& params, IGPUPipelineCache* _pipelineCache)
        {
            //_parent parameter is ignored

            using GLPpln = COpenGLRenderpassIndependentPipeline;

            const IGPUSpecializedShader* shaders_array[IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT]{};
            uint32_t shaderCount = 0u;
            for (uint32_t i = 0u; i < IGPURenderpassIndependentPipeline::SHADER_STAGE_COUNT; ++i)
                if (params.shaders[i])
                    shaders_array[shaderCount++] = params.shaders[i].get();

            auto shaders = core::SRange<const IGPUSpecializedShader*>(shaders_array, shaders_array+shaderCount);
            auto vsIsPresent = [&shaders]() -> bool {
                return std::find_if(shaders.begin(), shaders.end(), [](const IGPUSpecializedShader* shdr) {return shdr->getStage() == asset::IShader::ESS_VERTEX; }) != shaders.end();
            };

            asset::IShader::E_SHADER_STAGE lastVertexLikeStage = asset::IShader::ESS_VERTEX;
            for (uint32_t i = 0u; i < shaders.size(); ++i)
            {
                auto stage = shaders.begin()[shaders.size()-1u-i]->getStage();
                if (stage != asset::IShader::ESS_FRAGMENT)
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

            COpenGLPipelineCache* cache = IBackendObject::device_compatibility_cast<COpenGLPipelineCache*>(_pipelineCache, device);
            COpenGLPipelineLayout* gllayout = IBackendObject::device_compatibility_cast<COpenGLPipelineLayout*>(layout.get(), device);
            for (auto shdr = shaders.begin(); shdr != shaders.end(); ++shdr)
            {
                const auto* glshdr = IBackendObject::device_compatibility_cast<const COpenGLSpecializedShader*>(*shdr, device);

                auto stage = glshdr->getStage();
                uint32_t ix = core::findLSB<uint32_t>(stage);
                assert(ix < COpenGLRenderpassIndependentPipeline::SHADER_STAGE_COUNT);

                COpenGLPipelineCache::SCacheKey key{ glshdr->getSpirvHash(), glshdr->getSpecializationInfo(), core::smart_refctd_ptr_static_cast<COpenGLPipelineLayout>(layout), stage };
                auto bin = cache ? cache->find(key) : COpenGLSpecializedShader::SProgramBinary{ 0,nullptr };
                if (bin.binary)
                {
                    const char* dbgnm = glshdr->getObjectDebugName();

                    const GLuint GLname = gl.glShader.pglCreateProgram();
                    gl.glShader.pglProgramBinary(GLname, bin.format, bin.binary->data(), bin.binary->size());
                    if (dbgnm[0])
                        gl.extGlObjectLabel(GL_PROGRAM, GLname, strlen(dbgnm), dbgnm);
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

            auto raster = params.rasterization;
            if (gl.isGLES() && !gl.getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_clip_control))
            {
                if (raster.faceCullingMode == asset::EFCM_BACK_BIT)
                    raster.faceCullingMode = asset::EFCM_FRONT_BIT;
                else if (raster.faceCullingMode == asset::EFCM_FRONT_BIT)
                    raster.faceCullingMode = asset::EFCM_BACK_BIT;
            }

            return core::make_smart_refctd_ptr<COpenGLRenderpassIndependentPipeline>(
                core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl,
                std::move(layout),
                shaders.begin(), shaders.end(),
                params.vertexInput, params.blend, params.primitiveAssembly, raster,
                getNameCountForSingleEngineObject(), 0u, GLnames, binaries
            );
        }
        core::smart_refctd_ptr<IGPUComputePipeline> createComputePipeline(IOpenGL_FunctionTable& gl, const IGPUComputePipeline::SCreationParams& params, IGPUPipelineCache* _pipelineCache)
        {
            if (!params.layout || !params.shader)
                return nullptr;
            if (params.shader->getStage() != asset::IShader::ESS_COMPUTE)
                return nullptr;

            GLuint GLname = 0u;
            COpenGLSpecializedShader::SProgramBinary binary;
            COpenGLPipelineCache* cache = static_cast<COpenGLPipelineCache*>(_pipelineCache);
            auto layout = core::smart_refctd_ptr_static_cast<COpenGLPipelineLayout>(params.layout);
            auto glshdr = core::smart_refctd_ptr_static_cast<COpenGLSpecializedShader>(params.shader);

            COpenGLPipelineCache::SCacheKey key{ glshdr->getSpirvHash(), glshdr->getSpecializationInfo(), layout, asset::IShader::ESS_COMPUTE };
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

            return core::make_smart_refctd_ptr<COpenGLComputePipeline>(core::smart_refctd_ptr<IOpenGL_LogicalDevice>(device), &gl, core::smart_refctd_ptr<IGPUPipelineLayout>(layout.get()), core::smart_refctd_ptr<IGPUSpecializedShader>(glshdr.get()), getNameCountForSingleEngineObject(), 0u, GLname, binary);
        }
        IGPUFence::E_STATUS waitForFences(IOpenGL_FunctionTable& gl, uint32_t _count, IGPUFence*const *const _fences, bool _waitAll, const std::chrono::steady_clock::time_point& timeoutPoint)
        {
            const auto start = SRequestWaitForFences::clock_t::now();
            
            assert(_count!=0u);

            // want ~1/4 us on second try when not waiting for all
            constexpr uint64_t pollingFirstTimeout = 256ull>>1u;
            // poll once with zero timeout if have multiple fences to wait on
            uint64_t timeout = 0ull;
            if (_count==1u) // if we're only waiting for one fence, we can skip all the shenanigans
            {
                _waitAll = true;
                timeout = 0xdeadbeefBADC0FFEull;
            }
            for (bool notFirstRun=false; true; notFirstRun=true)
            {
                for (uint32_t i=0u; i<_count; )
                {
                    const auto now = SRequestWaitForFences::clock_t::now();
                    if (timeout)
                    {
                        if(notFirstRun && now>=timeoutPoint)
                            return IGPUFence::ES_TIMEOUT;
                        else if (_waitAll) // all fences have to get signalled anyway so no use round robining
                        {
                            if (timeoutPoint>now)
                                timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(timeoutPoint-now).count();
                            else
                                timeout = 0ull;
                        }
                        else if (i==0u) // if we're only looking for one to succeed then poll with increasing timeouts until deadline
                            timeout <<= 1u;
                    }
                    const auto result = static_cast<COpenGLFence*>(_fences[i])->wait(&gl,timeout);
                    switch (result)
                    {
                        case IGPUFence::ES_SUCCESS:
                            if (!_waitAll)
                                return result;
                            break;
                        case IGPUFence::ES_TIMEOUT:
                        case IGPUFence::ES_NOT_READY:
                            if (_waitAll) // keep polling this fence until success or overall timeout
                            {
                                if (!notFirstRun)
                                {
                                    timeout = 0x45u; // to make it start computing and using timeouts
                                    notFirstRun = true;
                                }
                                continue;
                            }
                            break;
                        case IGPUFence::ES_ERROR:
                            return result;
                            break;
                    }
                    i++;
                }
                if (_waitAll)
                    return IGPUFence::ES_SUCCESS;
                else if (!timeout)
                    timeout = pollingFirstTimeout;
            }
            // everything below this line is just to make the compiler shut up
            assert(false);
            return IGPUFence::ES_ERROR;
        }

        // currently used by shader programs only
        // they theoretically can be shared between contexts however, because uniforms state is program's state, we cant use that
        // because they are shareable, all GL names can be created in the same thread at once though
        uint32_t getNameCountForSingleEngineObject() const
        {
            return m_queueCount + 1u; // +1 because of this context (the one in logical device), probably not needed though
        }

        // section 5.3 of OpenGL 4.6 spec requires us to make the other contexts wait on the master context
        void incrementProgressionSync(FunctionTableType& _gl)
        {
            GLsync sync = _gl.glSync.pglFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
            _gl.glGeneral.pglFlush(); // cause its a cross context GLsync
            GLsync oldSync = masterContextSync->exchange(sync);
            masterContextCallsReturned->operator++();
            if (oldSync)
                _gl.glSync.pglDeleteSync(oldSync);
        }

        const uint32_t m_queueCount;
        std::atomic<GLsync>* const masterContextSync;
        std::atomic<uint64_t>* const masterContextCallsReturned;

        const egl::CEGL* egl;
        const FeaturesType* features;

        IOpenGL_LogicalDevice* device;
        const COpenGLDebugCallback* m_dbgCb;
    };

protected:
    std::atomic<uint64_t> m_masterContextCallsInvoked; // increment from calling thread after submitting request
    std::atomic<uint64_t> m_masterContextCallsReturned; // increment at the end of request process
    std::atomic<GLsync> m_masterContextSync; // swapped before `m_masterContextCallsReturned` is incremented
    const egl::CEGL* m_egl;

public:
    IOpenGL_LogicalDevice(core::smart_refctd_ptr<IAPIConnection>&& api, IPhysicalDevice* physicalDevice, const SCreationParams& params, const egl::CEGL* _egl)
        : ILogicalDevice(std::move(api),physicalDevice,params), m_masterContextCallsInvoked(0u), m_masterContextCallsReturned(0u), m_masterContextSync(nullptr), m_egl(_egl) {}

    template <typename FunctionTableType>
    inline uint64_t waitOnMasterContext(FunctionTableType& _gl, const uint64_t waitedCallsSoFar)
    {
        const uint64_t invokedSoFar = m_masterContextCallsInvoked.load();
        assert(invokedSoFar>=waitedCallsSoFar); // something went very wrong with causality
        if (invokedSoFar==waitedCallsSoFar)
            return waitedCallsSoFar;
        uint64_t returnedSoFar;
        while ((returnedSoFar=m_masterContextCallsReturned.load())<invokedSoFar) {} // waiting on address deadlocks
        _gl.glSync.pglWaitSync(m_masterContextSync.load(),0,GL_TIMEOUT_IGNORED);
        return returnedSoFar;
    }

    virtual void destroyFramebuffer(COpenGLFramebuffer::hash_t fbohash) = 0;
    virtual void destroyPipeline(COpenGLRenderpassIndependentPipeline* pipeline) = 0;
    virtual void destroyTexture(GLuint img) = 0;
    virtual void destroyBuffer(GLuint buf) = 0;
    virtual void destroySampler(GLuint s) = 0;
    virtual void destroySpecializedShaders(core::smart_refctd_dynamic_array<IOpenGLPipelineBase::SShaderProgram>&& programs) = 0;
    virtual void destroySync(GLsync sync) = 0;
    virtual void setObjectDebugName(GLenum id, GLuint object, GLsizei len, const GLchar* label) = 0;
    virtual void destroyQueryPool(COpenGLQueryPool* qp) = 0;
};

}

#endif
