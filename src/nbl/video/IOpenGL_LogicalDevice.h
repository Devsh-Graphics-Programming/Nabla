#ifndef __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_I_OPENGL__LOGICAL_DEVICE_H_INCLUDED__

#include <variant>

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/CEGL.h"
#include "nbl/system/IThreadHandler.h"
#include "nbl/video/IGPUFramebuffer.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPUSampler.h"
#include "nbl/video/IGPURenderpassIndependentPipeline.h"
#include "nbl/video/IGPUSpecializedShader.h"
#include "nbl/video/ISwapchain.h"
#include "nbl/video/IGPUShader.h"
#include "nbl/asset/ISPIRVOptimizer.h"

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
            ERT_FRAMEBUFFER_DESTROY,
            ERT_SWAPCHAIN_DESTROY,
            ERT_SYNC_DESTROY,
            ERT_SAMPLER_DESTROY,
            //ERT_GRAPHICS_PIPELINE_DESTROY,
            ERT_PROGRAM_DESTROY,

            ERT_BUFFER_CREATE,
            ERT_BUFFER_VIEW_CREATE,
            ERT_IMAGE_CREATE,
            ERT_IMAGE_VIEW_CREATE,
            ERT_FRAMEBUFFER_CREATE,
            ERT_SWAPCHAIN_CREATE,
            ERT_SEMAPHORE_CREATE,
            ERT_EVENT_CREATE,
            ERT_FENCE_CREATE,
            ERT_SAMPLER_CREATE,
            ERT_RENDERPASS_INDEPENDENT_PIPELINE_CREATE,
            //ERT_GRAPHICS_PIPELINE_CREATE,
            ERT_SPECIALIZED_SHADER_CREATE,

            // non-create requests
            ERT_GET_EVENT_STATUS,
            ERT_RESET_EVENT,
            ERT_SET_EVENT,
            ERT_RESET_FENCES,
            ERT_WAIT_FOR_FENCES,
            ERT_FLUSH_MAPPED_MEMORY_RANGES,
            ERT_INVALIDATE_MAPPED_MEMORY_RANGES,
            //BIND_BUFFER_MEMORY
        };

        constexpr static inline bool isDestroyRequest(E_REQUEST_TYPE rt)
        {
            return (rt < ERT_BUFFER_CREATE);
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
        struct SRequestSemaphoreCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SEMAPHORE_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUSemaphore>;
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
        };
        struct SRequestFramebufferCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_FRAMEBUFFER_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUFramebuffer>;
            IGPUFramebuffer::SCreationParams params;
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
            IGPUBuffer* buffer;
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
        };
        struct SRequestRenderpassIndependentPipelineCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_RENDERPASS_INDEPENDENT_PIPELINE_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPURenderpassIndependentPipeline>;
            IGPURenderpassIndependentPipeline::SCreationParams params;
        };
        struct SRequestSwapchainCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SWAPCHAIN_CREATE;
            using retval_t = core::smart_refctd_ptr<ISwapchain>;
            ISwapchain::SCreationParams params;
        };
        struct SRequestSpecializedShaderCreate
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_SPECIALIZED_SHADER_CREATE;
            using retval_t = core::smart_refctd_ptr<IGPUSpecializedShader>;
            const IGPUShader* unspecialized;
            const asset::ISpecializedShader::SInfo& specInfo;
            const asset::ISPIRVOptimizer* spvopt;
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
            core::SRange<core::smart_refctd_ptr<IGPUFence>> fences;
        };
        struct SRequestWaitForFences
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_WAIT_FOR_FENCES;
            using retval_t = IGPUFence::E_STATUS;
            core::SRange<IGPUFence*> fences;
            bool waitForAll;
            uint64_t timeout;
        };
        struct SRequestFlushMappedMemoryRanges
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_FLUSH_MAPPED_MEMORY_RANGES;
            using retval_t = void;
            core::SRange<const IDriverMemoryAllocation::MappedMemoryRange> memoryRanges;
        };
        struct SRequestInvalidateMappedMemoryRanges
        {
            static inline constexpr E_REQUEST_TYPE type = ERT_INVALIDATE_MAPPED_MEMORY_RANGES;
            using retval_t = void;
            core::SRange<const IDriverMemoryAllocation::MappedMemoryRange> memoryRanges;
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
// All COpenGL* objects (buffers, images, views...) will keep pointer of this type (just to be able to request destruction in destructor though)
// Implementation of both GL and GLES is the same code (see COpenGL_LogicalDevice) thanks to IOpenGL_FunctionTable abstraction layer
class IOpenGL_LogicalDevice : public ILogicalDevice, protected impl::IOpenGL_LogicalDeviceBase
{
    struct SRequest
    {
        using params_variant_t = std::variant<
            SRequestSemaphoreCreate,
            SRequestEventCreate,
            SRequestFenceCreate,
            SRequestFramebufferCreate,
            SRequestBufferCreate,
            SRequestBufferViewCreate,
            SRequestImageCreate,
            SRequestImageViewCreate,
            SRequestSamplerCreate,
            SRequestRenderpassIndependentPipelineCreate,
            SRequestSwapchainCreate,
            SRequestSpecializedShaderCreate,

            SRequest_Destroy<ERT_BUFFER_DESTROY>,
            SRequest_Destroy<ERT_TEXTURE_DESTROY>,
            SRequest_Destroy<ERT_FRAMEBUFFER_DESTROY>,
            SRequest_Destroy<ERT_SWAPCHAIN_DESTROY>,
            SRequest_Destroy<ERT_SYNC_DESTROY>,
            SRequest_Destroy<ERT_SAMPLER_DESTROY>,
            SRequest_Destroy<ERT_PROGRAM_DESTROY>,

            SRequestGetEventStatus,
            SRequestResetEvent,
            SRequestSetEvent,
            SRequestResetFences,
            SRequestWaitForFences,
            SRequestFlushMappedMemoryRanges,
            SRequestInvalidateMappedMemoryRanges
        >;

        E_REQUEST_TYPE type;
        params_variant_t params_variant;

        // cast to `RequestParams::retval_t*`
        void* pretval;
        // wait on this for result to be ready
        std::condition_variable cvar;
        bool ready = false;
    };

    template <typename FunctionTableType>
    class CThreadHandler : public system::IThreadHandler<FunctionTableType>
    {
        using FeaturesType = typename FunctionTableType::features_t;

        constexpr static inline uint32_t MaxRequestCount = 256u;
        constexpr static inline uint32_t CircularBufMask = MaxRequestCount - 1u;

        SRequest request_pool[MaxRequestCount];
        uint32_t cb_begin = 0u;
        uint32_t cb_end = 0u;

    public:
        CThreadHandler(const egl::CEGL* _egl, FeaturesType* _features, EGLConfig _config, EGLint _major, EGLint _minor) :
            egl(_egl),
            config(_config),
            major(_major), minor(_minor),
            thisCtx(EGL_NO_CONTEXT), pbuffer(EGL_NO_SURFACE),
            features(_features)
        {

        }

        // T must be one of request parameter structs
        template <typename RequestParams>
        SRequest& request(const RequestParams& params, typename RequestParams::retval_t* pretval = nullptr)
        {
            auto raii_handler = createRAIIDispatchHandler();

            const uint32_t r_id = cb_end;
            cb_end = (cb_end + 1u) & CircularBufMask;

            SRequest& req = request_pool[r_id];
            req.type = params.type;
            req.params_variant = params;
            req.ready = false;
            if constexpr (!std::is_void_v<typename RequestParams::retval_t>)
            {
                assert(pretval);
                req.pretval = pretval;
            }
            else
            {
                req.pretval = nullptr;
            }

            return req;
        }

        EGLContext getContext()
        {
            auto lk = createLock();

            return thisCtx;
        }

        template <typename RequestParams>
        void waitForRequestCompletion(SRequest& req)
        {
            auto lk = createLock();
            req.cvar.wait(lk, [&req]() -> bool { return req.ready; });

            // clear params, just to make sure no refctd ptr is holding an object longer than it needs to
            std::get<RequestParams>(req.params_variant) = RequestParams{};
        }

    protected:
        FunctionTableType init() override
        {
            egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

            const EGLint ctx_attributes[] = {
                EGL_CONTEXT_MAJOR_VERSION, major,
                EGL_CONTEXT_MINOR_VERSION, minor,
                EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

                EGL_NONE
            };

            thisCtx = egl->call.peglCreateContext(egl->display, config, EGL_NO_CONTEXT, ctx_attributes);

            // why not 1x1?
            const EGLint pbuffer_attributes[] = {
                EGL_WIDTH, 128,
                EGL_HEIGHT, 128,

                EGL_NONE
            };
            pbuffer = egl->call.peglCreatePbufferSurface(egl->display, config, pbuffer_attributes);

            egl->call.peglMakeCurrent(egl->display, pbuffer, pbuffer, thisCtx);

            return FunctionTableType(&egl->call, features);
        }

        bool wakeupPredicate() const override final { return (cb_begin!=cb_end) || base_t::wakeupPredicate(); }
        bool continuePredicate() const override final { return (cb_begin!=cb_end) && base_t::continuePredicate(); }

        void work(lock_t& lock, FunctionTableType& _gl) override
        {
            SRequest& req = request_pool[cb_begin];
            cb_begin = (cb_begin + 1u) & CircularBufMask;
            
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
            case ERT_FRAMEBUFFER_DESTROY:
            {
                auto& p = std::get<SRequest_Destroy<ERT_FRAMEBUFFER_DESTROY>>(req.params_variant);
                gl.glFramebuffer.pglDeleteFramebuffers(p.count, p.glnames);
            }
                break;
            case ERT_SYNC_DESTROY:
            {
                auto& p = std::get<SRequest_Destroy<ERT_SYNC_DESTROY>>(req.params_variant);
                assert(p.count == 1u);
                gl.glSync.pglDeleteSync(p.glnames[0]);
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
            case ERT_FLUSH_MAPPED_MEMORY_RANGES:
            {
                auto& p = std::get<SRequestFlushMappedMemoryRanges>(req.params_variant);
                for (auto mrng : p.memoryRanges)
                    gl.extGlFlushMappedNamedBufferRange(static_cast<COpenGLBuffer*>(mrng.memory)->getOpenGLName(), mrng.offset, mrng.length);
            }
                break;
            case ERT_INVALIDATE_MAPPED_MEMORY_RANGES:
            {
                gl.glSync.pglMemoryBarrier(gl.CLIENT_MAPPED_BUFFER_BARRIER_BIT); // i think there's no point in calling it number_of_mem_ranges times?
            }
                break;
            }

            req.ready = true;
            // moving unlock before the switch (but after cb_begin increment) probably wouldnt hurt
            lock.unlock(); // unlock so that notified thread wont immidiately block again
            req.cvar.notify_all(); //notify_one() would do as well, but lets call notify_all() in case something went horribly wrong (theoretically not possible) and multiple threads are waiting for single request
            lock.lock(); // reacquire (must be locked at the exit of this function -- see system::IThreadHandler docs)
        }

        void exit(FunctionTableType& gl) override
        {
            egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // detach ctx from thread
            egl->call.peglDestroyContext(egl->display, thisCtx);
            egl->call.peglDestroySurface(egl->display, pbuffer);
        }

    private:
        const egl::CEGL* egl;
        EGLConfig config;
        EGLint major, minor;
        EGLContext thisCtx;
        EGLSurface pbuffer;
        FeaturesType* features;
    };

public:
    IOpenGL_LogicalDevice(const egl::CEGL* _egl, EGLConfig config, EGLint major, EGLint minor, const SCreationParams& params) : ILogicalDevice(params)
    {

    }
};

}
}

#endif
