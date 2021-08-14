#ifndef __NBL_I_SURFACE_GL_H_INCLUDED__
#define __NBL_I_SURFACE_GL_H_INCLUDED__

#include "nbl/video/surface/ISurface.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

class ISurfaceGL : public ISurface
{
    public:
        template<typename EGL_native_window_t>
        inline EGL_native_window_t getInternalObject() const
        {
            static_assert(sizeof(EGL_native_window_t)<=sizeof(void*));
            EGL_native_window_t retval;
            memcpy(&retval,&m_winHandle,sizeof(EGL_native_window_t));
            return retval;
        }

        bool isSupported(const IPhysicalDevice* dev, uint32_t _queueFamIx) const override
        {
            const E_API_TYPE pdev_api = dev->getAPIType();
            // GL/GLES backends have just 1 queue family
            return (_queueFamIx == 0u) && ((pdev_api == EAT_OPENGL) || (pdev_api == EAT_OPENGL_ES));
        }

    protected:
        template<video::E_API_TYPE API_TYPE, typename EGL_native_window_t>
        explicit ISurfaceGL(core::smart_refctd_ptr<video::COpenGL_Connection<API_TYPE>>&& api, const EGL_native_window_t w) : ISurface(std::move(api)), m_winHandle(reinterpret_cast<const void*>(w)) {}

        const void* m_winHandle;
};

}

#endif