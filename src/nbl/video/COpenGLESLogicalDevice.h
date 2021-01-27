#ifndef __NBL_C_OPENGLES_LOGICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGLES_LOGICAL_DEVICE_H_INCLUDED__

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/CEGLCaller.h"

namespace nbl {
namespace video
{

class COpenGLESLogicalDevice final : public ILogicalDevice
{
public:
    COpenGLESLogicalDevice(egl::CEGLCaller* _egl, const SCreationParams& params) :
        ILogicalDevice(params)
    {
        // TODO create queues
    }

private:
    // TODO need GLES func pointers
};

}
}

#endif
