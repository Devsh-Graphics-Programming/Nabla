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

    bool isSupported(const IPhysicalDevice* dev, uint32_t _queueIx) const override
    {
        // TODO
        return true;
    }

protected:
    explicit ISurfaceGL(EGLNativeWindowType w) : m_winHandle(w) {}

    EGLNativeWindowType m_winHandle;
};

}
}

#endif