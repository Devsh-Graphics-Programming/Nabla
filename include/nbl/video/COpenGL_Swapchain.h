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
class COpenGL_SwapchainThreadHandler;

template <typename FunctionTableType_>
class NBL_API2 COpenGL_Swapchain final : public ISwapchain
{
    static inline constexpr uint32_t MaxImages = 4;
public:
    using ImagesArrayType = ISwapchain::images_array_t;
    using FunctionTableType = FunctionTableType_;

    static core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>> create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

    E_PRESENT_RESULT present(IGPUQueue* queue, const SPresentInfo& info);

    core::smart_refctd_ptr<IGPUImage> createImage(uint32_t imageIndex) override;

    const void* getNativeHandle() const;

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
    );

private:
    std::unique_ptr<COpenGL_SwapchainThreadHandler> m_threadHandler;
    uint32_t m_imgIx = 0u;
};

using COpenGLSwapchain = COpenGL_Swapchain<COpenGLFunctionTable>;
using COpenGLESSwapchain = COpenGL_Swapchain<COpenGLESFunctionTable>;

// TODO i don't like this, but needed for linking (figure out better way)
core::smart_refctd_ptr<COpenGLSwapchain> createGLSwapchain(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);
core::smart_refctd_ptr<COpenGLESSwapchain> createGLESSwapchain(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

}

#endif
