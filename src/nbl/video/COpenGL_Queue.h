#ifndef __NBL_C_OPENGL__QUEUE_H_INCLUDED__
#define __NBL_C_OPENGL__QUEUE_H_INCLUDED__

#include <variant>
#include "nbl/video/IGPUQueue.h"
#include "nbl/video/COpenGLSemaphore.h"
#include "nbl/video/COpenGLFence.h"
#include "nbl/video/COpenGLSync.h"
#include "nbl/video/COpenGLFramebuffer.h"
#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/video/CEGL.h"
#include "nbl/video/IOpenGL_FunctionTable.h"
#include "nbl/video/SOpenGLContextLocalCache.h"
#include "nbl/video/COpenGLCommandBuffer.h"
#include "nbl/video/COpenGL_Swapchain.h"
#include "nbl/video/COpenGLCommon.h"
#include "nbl/core/alloc/GeneralpurposeAddressAllocator.h"
#include "nbl/core/containers/CMemoryPool.h"
#include "nbl/video/utilities/renderdoc.h"

//#define DEBUGGING_BAW

#ifdef DEBUGGING_BAW
#include "renderdoc_app.h"

extern RENDERDOC_API_1_1_2* g_rdoc_api;
extern volatile bool g_rdoc_start_capture;
#endif

namespace nbl::video
{

template <typename FunctionTableType_>
class COpenGL_Queue final : public IGPUQueue
{
    public:
        using FunctionTableType = FunctionTableType_;
        using FeaturesType = typename FunctionTableType::features_t;

    private:
        static inline constexpr bool IsGLES = (FunctionTableType::EGL_API_TYPE == EGL_OPENGL_ES_API);

        struct ThreadInternalStateType
        {
            ThreadInternalStateType(const egl::CEGL* egl, const FeaturesType* features, system::logger_opt_smart_ptr&& logger) : gl(egl,features,std::move(logger)), ctxlocal(&gl) {}

            FunctionTableType gl;
            SOpenGLContextLocalCache ctxlocal;
        };

        enum E_REQUEST_TYPE
        {
            ERT_SUBMIT,
            ERT_DESTROY_FRAMEBUFFER,
            ERT_DESTROY_PIPELINE,
            ERT_BEGIN_CAPTURE,
            ERT_END_CAPTURE
        };
        template <E_REQUEST_TYPE ERT>
        struct SRequestParamsBase
        {
            static inline constexpr E_REQUEST_TYPE type = ERT;
        };
        struct SRequestParams_Submit : SRequestParamsBase<ERT_SUBMIT>
        {
            uint32_t waitSemaphoreCount = 0u;
            core::smart_refctd_ptr<IGPUSemaphore>* pWaitSemaphores = nullptr;
            const asset::E_PIPELINE_STAGE_FLAGS* pWaitDstStageMask = nullptr;
            uint32_t signalSemaphoreCount = 0u;
            core::smart_refctd_ptr<IGPUSemaphore>* pSignalSemaphores = nullptr;
            uint32_t commandBufferCount = 0u;
            core::smart_refctd_ptr<IGPUCommandBuffer>* commandBuffers = nullptr;

            core::smart_refctd_ptr<COpenGLSync> syncToInit;
        };
        struct SRequestParams_DestroyFramebuffer : SRequestParamsBase<ERT_DESTROY_FRAMEBUFFER>
        {
            SOpenGLState::SFBOHash fbo_hash;
        };
        struct SRequestParams_DestroyPipeline : SRequestParamsBase<ERT_DESTROY_PIPELINE>
        {
            SOpenGLState::SGraphicsPipelineHash hash;
        };
        using SRequestParams_BeginCapture = SRequestParamsBase<ERT_BEGIN_CAPTURE>;
        using SRequestParams_EndCapture = SRequestParamsBase<ERT_END_CAPTURE>;
        struct SRequest : public system::impl::IAsyncQueueDispatcherBase::request_base_t 
        {
            E_REQUEST_TYPE type;

            std::variant<
                SRequestParams_Submit, 
                SRequestParams_DestroyFramebuffer, 
                SRequestParams_DestroyPipeline,
                SRequestParams_BeginCapture,
                SRequestParams_EndCapture
            > params;
        };

        struct CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u, ThreadInternalStateType>
        {
        public:
            using base_t = system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u, ThreadInternalStateType>;
            friend base_t;

            CThreadHandler(const egl::CEGL* _egl, renderdoc_api_t* rdoc, IOpenGL_LogicalDevice* dev, const FeaturesType* _features, EGLContext _ctx, EGLSurface _pbuf, uint32_t _ctxid, COpenGLDebugCallback* _dbgCb) :
                m_rdoc_api(rdoc),
                egl(_egl),
                m_device(dev),
                thisCtx(_ctx), pbuffer(_pbuf),
                features(_features),
                m_ctxid(_ctxid),
                m_dbgCb(_dbgCb)
            {
                this->start();
            }

            template <typename RequestParams>
            void waitForRequestCompletion(SRequest& req)
            {
                req.wait_ready();

                // clear params, just to make sure no refctd ptr is holding an object longer than it needs to
                std::get<RequestParams>(req.params) = RequestParams{};

                req.discard_storage();
            }

            void init(ThreadInternalStateType* state_ptr)
            {
                egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

                EGLBoolean mcres = EGL_FALSE;
                while (mcres!=EGL_TRUE)
                {
                    mcres = egl->call.peglMakeCurrent(egl->display,pbuffer,pbuffer,thisCtx);
                    /*
                    * I think Queue context creation has a timing bug
                    * Debug build breaks, context can never be made current
                    * looks like auxillary context will not make itself current
                    * until something happens on main thread/main context ?
                    */
                    //_NBL_DEBUG_BREAK_IF(mcres!=EGL_TRUE);
                }

                new (state_ptr) ThreadInternalStateType(egl,features,core::smart_refctd_ptr<system::ILogger>(m_dbgCb->getLogger()));
                auto& gl = state_ptr->gl;
                auto& ctxlocal = state_ptr->ctxlocal;
                
                #ifdef _NBL_DEBUG
                gl.glGeneral.pglEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
                // TODO: debug message control (to exclude callback spam)
                #endif
                if (m_dbgCb)
                    gl.extGlDebugMessageCallback(m_dbgCb->m_callback,m_dbgCb);

                // defaults once set and not tracked by engine (should never change)
                gl.glGeneral.pglEnable(GL_FRAMEBUFFER_SRGB);
                gl.glFragment.pglDepthRangef(1.f, 0.f);
                if constexpr (IsGLES)
                {
                    if (gl.getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_clip_control)) // if not supported, modifications to spir-v will be applied to emulate this
                        gl.extGlClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
                }
                else
                {
                    // on desktop GL clip control is assumed to be always supported
                    gl.extGlClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
                }

                if constexpr (!IsGLES)
                {
                    gl.glGeneral.pglEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
                }

                // default values tracked by engine
                ctxlocal.nextState.rasterParams.multisampleEnable = 0;
                ctxlocal.nextState.rasterParams.depthFunc = GL_GEQUAL;
                ctxlocal.nextState.rasterParams.frontFace = GL_CCW;
            }

            template <typename RequestParams>
            void request_impl(SRequest& req, RequestParams&& params)
            {
                req.type = params.type;
                req.params = std::move(params);
            }

            void process_request(SRequest& req, ThreadInternalStateType& _state)
            {
                auto& gl = _state.gl;
                auto& ctxlocal = _state.ctxlocal;

                switch (req.type)
                {
                case ERT_SUBMIT:
                {
                    auto& submit = std::get<SRequestParams_Submit>(req.params);
                    // wait semaphores
                    GLbitfield barrierBits = 0;
                    for (uint32_t i = 0; i < submit.waitSemaphoreCount; ++i)
                    {
                        barrierBits |= SOpenGLBarrierHelper(gl.features).pipelineStageFlagsToMemoryBarrierBits(asset::EPSF_BOTTOM_OF_PIPE_BIT,submit.pWaitDstStageMask[i]);
                    }

#ifdef DEBUGGING_BAW
                    bool rdc_capturing = false;
                    if (g_rdoc_api && g_rdoc_start_capture)
                    {
                        g_rdoc_api->StartFrameCapture(NULL, NULL);
                        rdc_capturing = g_rdoc_start_capture;
                        g_rdoc_start_capture = false;
                    }
#endif

                    for (uint32_t i = 0; i < submit.waitSemaphoreCount; ++i)
                    {
                        IGPUSemaphore* sem = submit.pWaitSemaphores[i].get();
                        COpenGLSemaphore* glsem = static_cast<COpenGLSemaphore*>(sem);
                        glsem->wait(&gl);
                    }

                    if (barrierBits)
                        gl.glSync.pglMemoryBarrier(barrierBits);

                    for (uint32_t i = 0u; i < submit.commandBufferCount; ++i)
                    {
                        // reset initial state to default before cmdbuf execution (in Vulkan command buffers are independent of each other)
                        ctxlocal.nextState = SOpenGLState();
                        auto* cmdbuf = static_cast<COpenGLCommandBuffer*>(submit.commandBuffers[i].get());
                        cmdbuf->executeAll(&gl, &_state.ctxlocal, m_ctxid);
                    }

                    if (submit.syncToInit)
                    {
                        submit.syncToInit->init(m_device, &gl);
                    }
                    else // need to flush, otherwise OpenGL goes gaslighting the user with wrong error messages
                        gl.glGeneral.pglFlush();

#ifdef DEBUGGING_BAW
                    if (g_rdoc_api && rdc_capturing)
                    	g_rdoc_api->EndFrameCapture(NULL, NULL);
#endif
                }
                break;
                case ERT_DESTROY_FRAMEBUFFER:
                {
                    auto& p = std::get<SRequestParams_DestroyFramebuffer>(req.params);
                    auto fbo_hash = p.fbo_hash;
                    _state.ctxlocal.removeFBOEntry(&gl, fbo_hash);
                }
                break;
                case ERT_DESTROY_PIPELINE:
                {
                    auto& p = std::get<SRequestParams_DestroyPipeline>(req.params);
                    auto hash = p.hash;
                    _state.ctxlocal.removePipelineEntry(&gl, hash);
                }
                break;
                case ERT_BEGIN_CAPTURE:
                    m_rdoc_api->StartFrameCapture(NULL, NULL);
                break;
                case ERT_END_CAPTURE:
                    m_rdoc_api->EndFrameCapture(NULL, NULL);
                break;
                }
            }

            void exit(ThreadInternalStateType* state_ptr)
            {
                state_ptr->gl.glGeneral.pglFinish();
                state_ptr->~ThreadInternalStateType();

                egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // detach ctx from thread
                egl->call.peglDestroyContext(egl->display, thisCtx);
                egl->call.peglDestroySurface(egl->display, pbuffer);
            }

            renderdoc_api_t* m_rdoc_api;
        private:
            const egl::CEGL* egl;
            IOpenGL_LogicalDevice* m_device;
            EGLContext thisCtx;
            EGLSurface pbuffer;
            const FeaturesType* features;
            uint32_t m_ctxid;
            COpenGLDebugCallback* m_dbgCb;
        };

    public:
        COpenGL_Queue(
            IOpenGL_LogicalDevice* gldev,
            renderdoc_api_t* rdoc,
            const egl::CEGL* _egl,
            const FeaturesType* _features,
            uint32_t _ctxid,
            EGLContext _ctx,
            EGLSurface _surface,
            uint32_t _famIx,
            E_CREATE_FLAGS _flags,
            float _priority,
            COpenGLDebugCallback* _dbgCb
        ) : IGPUQueue(gldev,_famIx,_flags,_priority),
            threadHandler(_egl,rdoc,gldev,_features,_ctx,_surface,_ctxid,_dbgCb),
            m_mempool(128u,1u,512u,sizeof(void*)),
            m_ctxid(_ctxid)
        {
        }

        void waitForInitComplete()
        {
            threadHandler.waitForInitComplete();
        }

        bool submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override
        {
            if (!IGPUQueue::submit(_count, _submits, _fence))
                return false;

            core::smart_refctd_ptr<COpenGLSync> sync;
            for (uint32_t i = 0u; i < _count; ++i)
            {
                if (_submits[i].commandBufferCount == 0u)
                    continue;
                if (_submits[i].signalSemaphoreCount || (i == _count-1u && _fence))
                {
                    sync = core::make_smart_refctd_ptr<COpenGLSync>();
                }

                SRequestParams_Submit params;
                const SSubmitInfo& submit = _submits[i];
                params.waitSemaphoreCount = submit.waitSemaphoreCount;
                params.signalSemaphoreCount = submit.signalSemaphoreCount;
                params.commandBufferCount = submit.commandBufferCount;
                for (uint32_t i = 0u; i < submit.signalSemaphoreCount; ++i)
                {
                    COpenGLSemaphore* sem = static_cast<COpenGLSemaphore*>(submit.pSignalSemaphores[i]);
                    sem->associateGLSync(core::smart_refctd_ptr(sync));
                }
                core::smart_refctd_ptr<IGPUSemaphore>* waitSems = nullptr;
                asset::E_PIPELINE_STAGE_FLAGS* dstStageMask = nullptr;
                if (submit.waitSemaphoreCount)
                {
                    waitSems = params.pWaitSemaphores = m_mempool.emplace_n<core::smart_refctd_ptr<IGPUSemaphore>>(submit.waitSemaphoreCount);
                    params.pWaitDstStageMask = dstStageMask = m_mempool.emplace_n<asset::E_PIPELINE_STAGE_FLAGS>(submit.waitSemaphoreCount);
                }
                core::smart_refctd_ptr<IGPUSemaphore>* signalSems = nullptr;
                if (submit.signalSemaphoreCount)
                    signalSems = params.pSignalSemaphores = m_mempool.emplace_n<core::smart_refctd_ptr<IGPUSemaphore>>(submit.signalSemaphoreCount);
                auto* cmdBufs = params.commandBuffers = m_mempool.emplace_n<core::smart_refctd_ptr<IGPUCommandBuffer>>(submit.commandBufferCount);
                for (uint32_t j = 0u; j < submit.waitSemaphoreCount; ++j)
                {
                    params.pWaitSemaphores[j] = core::smart_refctd_ptr<IGPUSemaphore>(submit.pWaitSemaphores[j]);
                    dstStageMask[j] = submit.pWaitDstStageMask[j];
                }
                for (uint32_t j = 0u; j < submit.signalSemaphoreCount; ++j)
                    params.pSignalSemaphores[j] = core::smart_refctd_ptr<IGPUSemaphore>(submit.pSignalSemaphores[j]);
                for (uint32_t j = 0u; j < submit.commandBufferCount; ++j)
                    params.commandBuffers[j] = core::smart_refctd_ptr<IGPUCommandBuffer>(submit.commandBuffers[j]);

                params.syncToInit = sync;

                auto& req = threadHandler.request(std::move(params));
                // TODO: Copy all the data to the request and dont wait for the request to finish, then mark this request type as `isWaitlessRequest`
                threadHandler.template waitForRequestCompletion<SRequestParams_Submit>(req);

                if (waitSems)
                    m_mempool.free_n<core::smart_refctd_ptr<IGPUSemaphore>>(waitSems, submit.waitSemaphoreCount);
                if (dstStageMask)
                    m_mempool.free_n<asset::E_PIPELINE_STAGE_FLAGS>(dstStageMask, submit.waitSemaphoreCount);
                if (signalSems)
                    m_mempool.free_n<core::smart_refctd_ptr<IGPUSemaphore>>(signalSems, submit.signalSemaphoreCount);
                m_mempool.free_n<core::smart_refctd_ptr<IGPUCommandBuffer>>(cmdBufs, submit.commandBufferCount);
            }

            if (_fence)
            {
                COpenGLFence* glfence = static_cast<COpenGLFence*>(_fence);
                glfence->associateGLSync(std::move(sync)); // associate sync used for signal semaphores in last submit
            }

            return true;
        }

        bool present(const SPresentInfo& info) override
        {
            for (uint32_t i = 0u; i < info.waitSemaphoreCount; ++i)
            {
                if (m_originDevice != info.waitSemaphores[i]->getOriginDevice())
                    return false;
            }

            for (uint32_t i = 0u; i < info.swapchainCount; ++i)
            {
                if (m_originDevice != info.swapchains[i]->getOriginDevice())
                    return false;
            }

            using swapchain_t = COpenGL_Swapchain<FunctionTableType_>;
            bool retval = true;
            for (uint32_t i = 0u; i < info.swapchainCount; ++i)
            {
                swapchain_t* sc = static_cast<swapchain_t*>(info.swapchains[i]);
                const uint32_t imgix = info.imgIndices[i];
                retval &= sc->present(imgix, info.waitSemaphoreCount, info.waitSemaphores);
            }

            return retval;
        }

        void destroyFramebuffer(COpenGLFramebuffer::hash_t fbohash)
        {
            SRequestParams_DestroyFramebuffer params;
            params.fbo_hash = fbohash;

            auto& req = threadHandler.request(std::move(params));
            // TODO: Use a special form of request/IAsyncQueueDispatcher that lets us specify that certain requests wont be waited for and can be transitioned straight into ES_INITIAL
            // TODO: basically implement `isWaitlessRequest` for this thread handler (like the LogicalDevice queue)
            threadHandler.template waitForRequestCompletion<SRequestParams_DestroyFramebuffer>(req);
        }

        void destroyPipeline(COpenGLRenderpassIndependentPipeline* pipeline)
        {
            SRequestParams_DestroyPipeline params;
            params.hash = pipeline->getPipelineHash(m_ctxid);

            auto& req = threadHandler.request(std::move(params));
            // TODO: Use a special form of request/IAsyncQueueDispatcher that lets us specify that certain requests wont be waited for and can be transitioned straight into ES_INITIAL
            // TODO: basically implement `isWaitlessRequest` for this thread handler (like the LogicalDevice queue)
            threadHandler.template waitForRequestCompletion<SRequestParams_DestroyPipeline>(req);
        }

        bool startCapture() override
        {
            if (!threadHandler.m_rdoc_api)
                return false;

            SRequestParams_BeginCapture p;

            threadHandler.request(std::move(p));

            return true;
        }
        bool endCapture() override
        {
            if (!threadHandler.m_rdoc_api)
                return false;

            SRequestParams_EndCapture p;

            threadHandler.request(std::move(p));

            return true;
        }

    protected:
        ~COpenGL_Queue()
        {
        }

    private:
        CThreadHandler threadHandler;
        using memory_pool_t = core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>,core::default_aligned_allocator,uint32_t>;
        memory_pool_t m_mempool;
        uint32_t m_ctxid;
};

}


#include "nbl/video/COpenGLFunctionTable.h"
#include "nbl/video/COpenGLESFunctionTable.h"

namespace nbl::video
{

using COpenGLQueue = COpenGL_Queue<COpenGLFunctionTable>;
using COpenGLESQueue = COpenGL_Queue<COpenGLESFunctionTable>;

}

#endif