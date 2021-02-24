#ifndef __NBL_C_OPENGL_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IOpenGL_PhysicalDeviceBase.h"
#include "nbl/video/COpenGLLogicalDevice.h"

namespace nbl {
namespace video
{

class COpenGLPhysicalDevice final : public IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>
{
    using base_t = IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>;

public:
    static core::smart_refctd_ptr<COpenGLPhysicalDevice> create(const egl::CEGL* _egl)
    {
		// TODO those params should be somehow sourced externally
        const EGLint
            red = 8,
            green = 8,
            blue = 8,
            alpha = 0;
        const EGLint bufsz = red + green + blue;
        const EGLint depth = 0;
        const EGLint stencil = 0;

        const EGLint egl_attributes[] = {
            EGL_RED_SIZE, red,
            EGL_GREEN_SIZE, green,
            EGL_BLUE_SIZE, blue,
            EGL_BUFFER_SIZE, bufsz,
            EGL_DEPTH_SIZE, depth,
            EGL_STENCIL_SIZE, stencil,
            EGL_ALPHA_SIZE, alpha,
            EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
            EGL_CONFORMANT, EGL_OPENGL_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            //Params.Stereobuffer
            //Params.Vsync
            EGL_SURFACE_TYPE, (EGL_WINDOW_BIT | EGL_PBUFFER_BIT),

            EGL_NONE
        };

        EGLConfig config;
        EGLint ccnt = 1;
        _egl->call.peglChooseConfig(_egl->display, egl_attributes, &config, 1, &ccnt);
        if (ccnt < 1)
            return nullptr;

        EGLint ctx_attributes[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 6,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

            EGL_NONE
        };
		EGLint& gl_major = ctx_attributes[1];
		EGLint& gl_minor = ctx_attributes[3];

        EGLContext ctx = EGL_NO_CONTEXT;
        do
        {
            ctx = _egl->call.peglCreateContext(_egl->display, config, EGL_NO_CONTEXT, ctx_attributes);
            --gl_minor;
        } while (ctx == EGL_NO_CONTEXT && gl_minor >= 3); // fail if cant create >=4.3 context
        ++gl_minor;

        if (ctx == EGL_NO_CONTEXT)
            return nullptr;

        return core::make_smart_refctd_ptr<COpenGLPhysicalDevice>(_egl, config, ctx, gl_major, gl_minor);
    }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		// TODO uncomment once GL logical device has all pure virtual methods implemented
		//return core::make_smart_refctd_ptr<COpenGLLogicalDevice>(m_egl, &m_glfeatures, m_config, m_gl_major, m_gl_minor, params);
		return nullptr;
	}

private:
    COpenGLPhysicalDevice(const egl::CEGL* _egl, EGLConfig config, EGLContext ctx, EGLint major, EGLint minor) : 
        base_t(_egl, config, ctx, major, minor)
    {

    }
};

}
}

#endif