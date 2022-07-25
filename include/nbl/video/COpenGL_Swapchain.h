#ifndef __NBL_VIDEO_C_OPENGL__SWAPCHAIN_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL__SWAPCHAIN_H_INCLUDED__

#include "nbl/system/IThreadHandler.h"
#include "nbl/video/ISwapchain.h"

namespace nbl::video
{

class IOpenGL_LogicalDevice;
class COpenGLFeatureMap;
class COpenGLSync;
class COpenGLSemaphore;
class COpenGLFunctionTable;
class COpenGLESFunctionTable;

static inline constexpr uint32_t OpenGLFunctionTableSize = 6672u;
using SThreadHandlerInternalState = std::array<uint8_t, OpenGLFunctionTableSize>;

class COpenGL_SwapchainThreadHandler final : public system::IThreadHandler<COpenGL_SwapchainThreadHandler, SThreadHandlerInternalState>
{
    static inline constexpr uint32_t MaxImages = 4u;
    using base_t = system::IThreadHandler<COpenGL_SwapchainThreadHandler, SThreadHandlerInternalState>;
    friend base_t;

public:
    COpenGL_SwapchainThreadHandler(const egl::CEGL* _egl,
        IOpenGL_LogicalDevice* dev,
        const void* _window,
        ISurface::E_PRESENT_MODE presentMode,
        core::SRange<core::smart_refctd_ptr<IGPUImage>> _images,
        const COpenGLFeatureMap* _features,
        EGLContext _ctx,
        EGLConfig _config,
        COpenGLDebugCallback* _dbgCb
    );

    void requestBlit(uint32_t _imgIx, uint32_t semCount, IGPUSemaphore* const* const sems);

    core::smart_refctd_ptr<COpenGLSync> getSyncForImgIx(uint32_t imgix);

    egl::CEGL::Context glctx;

    struct SRequest
    {
        uint32_t imgIx = 0u;
        core::vector<core::smart_refctd_ptr<COpenGLSemaphore>> sems;
        uint32_t semCount = 0;
    } request;
protected:

    void init(SThreadHandlerInternalState* state_ptr);

    void work(typename base_t::lock_t& lock, typename base_t::internal_state_t& gl);

    void exit(SThreadHandlerInternalState* gl);

    bool wakeupPredicate() const { return needToBlit; }
    bool continuePredicate() const { return needToBlit; }

private:
    IOpenGL_LogicalDevice* m_device;
    uint64_t m_masterContextCallsWaited;

    const egl::CEGL* egl;
    ISurface::E_PRESENT_MODE m_presentMode;
    const COpenGLFeatureMap* features;
    core::SRange<core::smart_refctd_ptr<IGPUImage>> images;
    uint32_t fbos[MaxImages]{};
    core::smart_refctd_ptr<COpenGLSync> syncs[MaxImages];
    COpenGLDebugCallback* m_dbgCb;
    std::array<uint32_t, MaxImages> m_texViews;

    bool needToBlit = false;

    EGLBoolean m_makeCurrentRes = EGL_FALSE;
    std::condition_variable m_ctxCreatedCvar;
};

template <typename FunctionTableType_>
class NBL_API2 COpenGL_Swapchain final : public ISwapchain
{
    static inline constexpr uint32_t MaxImages = 4;
public:
    using ImagesArrayType = ISwapchain::images_array_t;
    using FunctionTableType = FunctionTableType_;

    // should be called by GL/GLES backend's impl of vkQueuePresentKHR
    inline bool present(uint32_t _imgIx, uint32_t semCount, IGPUSemaphore*const *const sems)
    {
        if (_imgIx >= m_params.minImageCount)
            return false;
        for (uint32_t i = 0u; i < semCount; ++i)
        {
            if (!this->isCompatibleDevicewise(sems[i]))
                return false;
        }
        m_threadHandler.requestBlit(_imgIx, semCount, sems);

        return true;
    }

    static core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>> create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

    E_PRESENT_RESULT present(IGPUQueue* queue, const SPresentInfo& info);

    virtual const void* getNativeHandle() const override {return &m_threadHandler.glctx;}

protected:
    // images will be created in COpenGLLogicalDevice::createSwapchain
    COpenGL_Swapchain(
        SCreationParams&& params,
        core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev,
        const egl::CEGL* _egl,
        ImagesArrayType&& images,
        const COpenGLFeatureMap* _features,
        EGLContext _ctx,
        EGLConfig _config,
        COpenGLDebugCallback* _dbgCb
    ) : ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>(dev),std::move(params),std::move(images)),
        m_threadHandler(
            _egl,dev.get(),m_params.surface->getNativeWindowHandle(),m_params.presentMode,{m_images->begin(),m_images->end()},_features,_ctx,_config,_dbgCb
        )
    {}

private:
    COpenGL_SwapchainThreadHandler m_threadHandler;
    uint32_t m_imgIx = 0u;
};

using COpenGLSwapchain = COpenGL_Swapchain<COpenGLFunctionTable>;
using COpenGLESSwapchain = COpenGL_Swapchain<COpenGLESFunctionTable>;

// TODO i don't like this, but needed for linking (figure out better way)
core::smart_refctd_ptr<COpenGLSwapchain> createGLSwapchain(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);
core::smart_refctd_ptr<COpenGLESSwapchain> createGLESSwapchain(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

}

#endif
