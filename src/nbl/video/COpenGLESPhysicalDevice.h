#ifndef __NBL_C_OPENGLES_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGLES_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IOpenGL_PhysicalDeviceBase.h"
#include "nbl/video/COpenGLESLogicalDevice.h"

namespace nbl {
namespace video
{

class COpenGLESPhysicalDevice final : public IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>
{
    using base_t = IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>;

public:
    using base_t::base_t;
};

}
}

#endif