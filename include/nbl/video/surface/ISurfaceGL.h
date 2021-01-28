#ifndef __NBL_I_SURFACE_GL_H_INCLUDED__
#define __NBL_I_SURFACE_GL_H_INCLUDED__

#include "nbl/video/surface/ISurface.h"
#include <EGL/egl.h>

namespace nbl {
namespace video
{

class IPhysicalDevice;

class ISurfaceGL : public ISurface
{
public:
    inline EGLNativeWindowType getInternalObject() const { return m_winHandle; }

    bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override
    {
        // GL/GLES backends have just 1 queue family
        return (_queueFamIx == 0u);
    }

protected:
    explicit ISurfaceGL(EGLNativeWindowType w) : m_winHandle(w) {}

    EGLNativeWindowType m_winHandle;
};

}
}

#endif