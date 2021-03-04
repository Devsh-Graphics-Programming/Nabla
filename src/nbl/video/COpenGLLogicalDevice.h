#ifndef __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/COpenGL_LogicalDevice.h"
#include "nbl/video/COpenGLQueue.h"
#include "nbl/video/COpenGLSwapchain.h"

namespace nbl {
namespace video
{

class COpenGLLogicalDevice final : public COpenGL_LogicalDevice<COpenGLQueue, COpenGLSwapchain>
{
    using base_t = COpenGL_LogicalDevice<COpenGLQueue, COpenGLSwapchain>;

public:
    COpenGLLogicalDevice(const egl::CEGL* _egl, FeaturesType* _features, EGLConfig config, EGLint major, EGLint minor, const SCreationParams& params) :
        base_t(_egl, EAT_OPENGL, _features, config, major, minor, params)
    {

    }
};

}
}

#endif
