#ifndef __NBL_VIDEO_C_OPENGL__SWAPCHAIN_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL__SWAPCHAIN_H_INCLUDED__

#include "nbl/system/IThreadHandler.h"
#include "nbl/video/ISwapchain.h"

namespace nbl::video
{

class IOpenGL_LogicalDevice;
class COpenGLFeatureMap;
class COpenGLFunctionTable;
class COpenGLESFunctionTable;
class COpenGL_SwapchainThreadHandler;

template <typename FunctionTableType>
class COpenGL_Swapchain final : public ISwapchain
{
public:
    static core::smart_refctd_ptr<COpenGL_Swapchain<FunctionTableType>> create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

    E_PRESENT_RESULT present(IGPUQueue* queue, const SPresentInfo& info);

    core::smart_refctd_ptr<IGPUImage> createImage(const uint32_t imageIndex) override;

    const void* getNativeHandle() const;

protected:
    COpenGL_Swapchain(
        SCreationParams&& params,
        core::smart_refctd_ptr<IOpenGL_LogicalDevice>&& dev,
        uint32_t imgCount, IGPUImage::SCreationParams&& imgCreationParams,
        std::unique_ptr<COpenGL_SwapchainThreadHandler> _threadHandler
    ) : ISwapchain(core::smart_refctd_ptr<const ILogicalDevice>(dev), std::move(params), std::move(imgCreationParams), imgCount),
        m_threadHandler(std::move(_threadHandler))
    {}

private:
    std::unique_ptr<COpenGL_SwapchainThreadHandler> m_threadHandler;
    uint32_t m_imgIx = 0u;
};

using COpenGLSwapchain = COpenGL_Swapchain<COpenGLFunctionTable>;
using COpenGLESSwapchain = COpenGL_Swapchain<COpenGLESFunctionTable>;

}

#endif
