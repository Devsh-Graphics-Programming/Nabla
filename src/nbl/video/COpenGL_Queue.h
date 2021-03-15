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

namespace nbl {
namespace video
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
            ThreadInternalStateType(const egl::CEGL* egl, FeaturesType* features) : gl(egl, features), ctxlocal(&gl) {}

            FunctionTableType gl;
            SOpenGLContextLocalCache ctxlocal;
        };

        enum E_REQUEST_TYPE
        {
            ERT_SUBMIT,
            ERT_SIGNAL_FENCE,
            ERT_DESTROY_FRAMEBUFFER,
            ERT_DESTROY_PIPELINE
        };
        template <E_REQUEST_TYPE ERT>
        struct SRequestParamsBase
        {
            static inline constexpr E_REQUEST_TYPE type = ERT;
        };
        struct SRequestParams_Submit : SRequestParamsBase<ERT_SUBMIT>
        {
            uint32_t waitSemaphoreCount;
            core::smart_refctd_ptr<IGPUSemaphore>* pWaitSemaphores;
            const asset::E_PIPELINE_STAGE_FLAGS* pWaitDstStageMask;
            uint32_t signalSemaphoreCount;
            core::smart_refctd_ptr<IGPUSemaphore>* pSignalSemaphores;
            uint32_t commandBufferCount;
            core::smart_refctd_ptr<IGPUPrimaryCommandBuffer>* commandBuffers;
        };
        struct SRequestParams_Fence : SRequestParamsBase<ERT_SIGNAL_FENCE>
        {
            core::smart_refctd_ptr<COpenGLFence> fence;
        };
        struct SRequestParams_DestroyFramebuffer : SRequestParamsBase<ERT_DESTROY_FRAMEBUFFER>
        {
            SOpenGLState::SFBOHash fbo_hash;
        };
        struct SRequestParams_DestroyPipeline : SRequestParamsBase<ERT_DESTROY_PIPELINE>
        {
            SOpenGLState::SGraphicsPipelineHash hash;
        };
        struct SRequest : public system::impl::IAsyncQueueDispatcherBase::request_base_t 
        {
            E_REQUEST_TYPE type;

            std::variant<SRequestParams_Submit, SRequestParams_Fence, SRequestParams_DestroyFramebuffer, SRequestParams_DestroyPipeline> params;
        };

        struct CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u, ThreadInternalStateType>
        {
        public:
            CThreadHandler(const egl::CEGL* _egl, IOpenGL_LogicalDevice* dev, FeaturesType* _features, EGLContext _master, EGLConfig _config, EGLint _major, EGLint _minor, uint32_t _ctxid) :
                egl(_egl),
                m_device(dev),
                masterCtx(_master), config(_config),
                major(_major), minor(_minor),
                thisCtx(EGL_NO_CONTEXT), pbuffer(EGL_NO_SURFACE),
                features(_features),
                m_ctxid(_ctxid)
            {

            }

            using base_t = system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u, ThreadInternalStateType>;
            friend base_t;

            void init(ThreadInternalStateType* state_ptr)
            {
                egl->call.peglBindAPI(FunctionTableType::EGL_API_TYPE);

                const EGLint ctx_attributes[] = {
                    EGL_CONTEXT_MAJOR_VERSION, major,
                    EGL_CONTEXT_MINOR_VERSION, minor,
                    //EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, // core profile is default setting

                    EGL_NONE
                };

                thisCtx = egl->call.peglCreateContext(egl->display, config, masterCtx, ctx_attributes);

                // why not 1x1?
                const EGLint pbuffer_attributes[] = {
                    EGL_WIDTH, 128,
                    EGL_HEIGHT, 128,

                    EGL_NONE
                };
                pbuffer = egl->call.peglCreatePbufferSurface(egl->display, config, pbuffer_attributes);

                egl->call.peglMakeCurrent(egl->display, pbuffer, pbuffer, thisCtx);

                new (state_ptr) typename ThreadInternalStateType(egl, features);
                auto& gl = state_ptr->gl;
                auto& ctxlocal = state_ptr->ctxlocal;

                // defaults once set and not tracked by engine (should never change)
                gl.glGeneral.pglEnable(IOpenGL_FunctionTable::FRAMEBUFFER_SRGB);
                gl.glFragment.pglDepthRangef(1.f, 0.f);
                if constexpr (IsGLES)
                {
                    if (gl.getFeatures()->isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_clip_control)) // if not supported, modifications to spir-v will be applied to emulate this
                        gl.extGlClipControl(IOpenGL_FunctionTable::UPPER_LEFT, IOpenGL_FunctionTable::ZERO_TO_ONE);
                }
                else
                {
                    // on desktop GL clip control is assumed to be always supported
                    gl.extGlClipControl(IOpenGL_FunctionTable::UPPER_LEFT, IOpenGL_FunctionTable::ZERO_TO_ONE);
                }

                if constexpr (!IsGLES)
                {
                    gl.glGeneral.pglEnable(IOpenGL_FunctionTable::TEXTURE_CUBE_MAP_SEAMLESS);
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
                //static_assert(std::is_same_v<ThreadInternalStateType, FunctionTableType>);
                // a cast to common base so that intellisense knows function set (can and should be removed after code gets written)
                IOpenGL_FunctionTable& gl = static_cast<IOpenGL_FunctionTable&>(_state.gl);

                switch (req.type)
                {
                case ERT_SUBMIT:
                {
                    auto& submit = std::get<SRequestParams_Submit>(req.params);

                    // wait semaphores
                    GLbitfield barrierBits = 0;
                    for (uint32_t i = 0; i < submit.waitSemaphoreCount; ++i)
                    {
                        barrierBits |= pipelineStageFlagsToMemoryBarrierBits(asset::EPSF_BOTTOM_OF_PIPE_BIT, submit.pWaitDstStageMask[i]);
                    }
                    if (barrierBits)
                        gl.glSync.pglMemoryBarrier(barrierBits);
                    for (uint32_t i = 0; i < submit.waitSemaphoreCount; ++i)
                    {
                        IGPUSemaphore* sem = submit.pWaitSemaphores[i].get();
                        COpenGLSemaphore* glsem = static_cast<COpenGLSemaphore*>(sem);
                        assert(glsem->isWaitable());
                        if (glsem->isWaitable())
                            glsem->wait(&gl);
                    }

                    for (uint32_t i = 0u; i < submit.commandBufferCount; ++i)
                    {
                        //dynamic_cast because of virtual base
                        auto* cmdbuf = dynamic_cast<COpenGLCommandBuffer*>(submit.commandBuffers[i].get());
                        cmdbuf->executeAll(&gl, &_state.ctxlocal, m_ctxid);
                    }

                    if (submit.signalSemaphoreCount)
                    {
                        auto sync = core::make_smart_refctd_ptr<COpenGLSync>(m_device, &gl);
                        for (uint32_t i = 0u; i < submit.signalSemaphoreCount; ++i)
                        {
                            IGPUSemaphore* sem = submit.pSignalSemaphores[i].get();
                            COpenGLSemaphore* glsem = static_cast<COpenGLSemaphore*>(sem);
                            glsem->signal(core::smart_refctd_ptr(sync));
                        }
                    }
                }
                break;
                case ERT_SIGNAL_FENCE:
                {
                    auto& p = std::get<SRequestParams_Fence>(req.params);
                    core::smart_refctd_ptr<COpenGLFence> fence = std::move(p.fence);
                    fence->signal(&gl);
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
                }
            }

            void exit(ThreadInternalStateType* state_ptr)
            {
                state_ptr->~ThreadInternalStateType();

                egl->call.peglMakeCurrent(egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT); // detach ctx from thread
                egl->call.peglDestroyContext(egl->display, thisCtx);
                egl->call.peglDestroySurface(egl->display, pbuffer);
            }

        private:
            const egl::CEGL* egl;
            IOpenGL_LogicalDevice* m_device;
            EGLContext masterCtx;
            EGLConfig config;
            EGLint major, minor;
            EGLContext thisCtx;
            EGLSurface pbuffer;
            FeaturesType* features;
            uint32_t m_ctxid;
        };

    public:
        COpenGL_Queue(IOpenGL_LogicalDevice* gldev, ILogicalDevice* dev, const egl::CEGL* _egl, FeaturesType* _features, uint32_t _ctxid, EGLContext _masterCtx, EGLConfig _config, EGLint _gl_major, EGLint _gl_minor, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority) :
            IGPUQueue(dev, _famIx, _flags, _priority),
            threadHandler(_egl, gldev, _features, _masterCtx, _config, _gl_major, _gl_minor, _ctxid),
            m_mempool(1u<<20,1u),
            m_ctxid(_ctxid)
        {

        }

        void submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence) override
        {
            m_mempoolMutex.lock();

            for (uint32_t i = 0u; i < _count; ++i)
            {
                if (_submits[i].commandBufferCount == 0u)
                    continue;

                SRequestParams_Submit params;
                const SSubmitInfo& submit = _submits[i];
                for (uint32_t i = 0u; i < submit.signalSemaphoreCount; ++i)
                {
                    COpenGLSemaphore* sem = static_cast<COpenGLSemaphore*>(submit.pSignalSemaphores[i]);
                    sem->setToBeSignaled();
                }
                core::smart_refctd_ptr<IGPUSemaphore>* waitSems = nullptr;
                asset::E_PIPELINE_STAGE_FLAGS* dstStageMask = nullptr;
                if (_submits[i].waitSemaphoreCount)
                {
                    waitSems = params.pWaitSemaphores = m_mempool.emplace_n<core::smart_refctd_ptr<IGPUSemaphore>>(_submits[i].waitSemaphoreCount);
                    params.pWaitDstStageMask = dstStageMask = m_mempool.emplace_n<asset::E_PIPELINE_STAGE_FLAGS>(_submits[i].waitSemaphoreCount);
                }
                core::smart_refctd_ptr<IGPUSemaphore>* signalSems = nullptr;
                if (_submits[i].signalSemaphoreCount)
                    signalSems = params.pSignalSemaphores = m_mempool.emplace_n<core::smart_refctd_ptr<IGPUSemaphore>>(_submits[i].signalSemaphoreCount);
                auto* cmdBufs = params.commandBuffers = m_mempool.emplace_n<core::smart_refctd_ptr<IGPUPrimaryCommandBuffer>>(_submits[i].commandBufferCount);
                for (uint32_t j = 0u; j < _submits[i].waitSemaphoreCount; ++j)
                    params.pWaitSemaphores[j] = core::smart_refctd_ptr<IGPUSemaphore>(_submits[i].pWaitSemaphores[j]);
                for (uint32_t j = 0u; j < _submits[i].signalSemaphoreCount; ++j)
                    params.pSignalSemaphores[j] = core::smart_refctd_ptr<IGPUSemaphore>(_submits[i].pSignalSemaphores[j]);
                for (uint32_t j = 0u; j < _submits[i].commandBufferCount; ++j)
                    params.commandBuffers[j] = core::smart_refctd_ptr<IGPUPrimaryCommandBuffer>(_submits[i].commandBuffers[j]);

                auto& req = threadHandler.request(std::move(params));
                threadHandler.waitForRequestCompletion(req);

                if (waitSems)
                    m_mempool.free_n<core::smart_refctd_ptr<IGPUSemaphore>>(waitSems, _submits[i].waitSemaphoreCount);
                if (dstStageMask)
                    m_mempool.free_n<asset::E_PIPELINE_STAGE_FLAGS>(dstStageMask, _submits[i].waitSemaphoreCount);
                if (signalSems)
                    m_mempool.free_n<core::smart_refctd_ptr<IGPUSemaphore>>(signalSems, _submits[i].signalSemaphoreCount);
                m_mempool.free_n<core::smart_refctd_ptr<IGPUPrimaryCommandBuffer>>(cmdBufs, _submits[i].commandBufferCount);
            }

            m_mempoolMutex.unlock();

            if (_fence)
            {
                SRequestParams_Fence params;
                COpenGLFence* glfence = static_cast<COpenGLFence*>(_fence);
                glfence->setToBeSignaled();
                params.fence = core::smart_refctd_ptr<COpenGLFence>(glfence);

                auto& req = threadHandler.request(std::move(params));
                // wait on completion ?
            }
        }

        bool present(const SPresentInfo& info) override
        {
            for (uint32_t i = 0u; i < info.waitSemaphoreCount; ++i)
                if (!this->isCompatibleDevicewise(info.waitSemaphores[i]))
                    return false;
            for (uint32_t i = 0u; i < info.swapchainCount; ++i)
                if (!this->isCompatibleDevicewise(info.swapchains[i]))
                    return false;

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

            threadHandler.request(std::move(params));
        }

        void destroyPipeline(COpenGLRenderpassIndependentPipeline* pipeline)
        {
            SRequestParams_DestroyPipeline params;
            params.hash = pipeline->getPipelineHash(m_ctxid);

            threadHandler.request(std::move(params));
        }

    protected:
        ~COpenGL_Queue()
        {

        }

    private:
        CThreadHandler threadHandler;
        std::mutex m_mempoolMutex;
        using memory_pool_t = core::CMemoryPool<core::GeneralpurposeAddressAllocator<uint32_t>,core::aligned_allocator>;
        memory_pool_t m_mempool;
        uint32_t m_ctxid;
};

}}

#endif