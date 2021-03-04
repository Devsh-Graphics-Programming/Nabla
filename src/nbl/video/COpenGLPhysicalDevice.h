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
		constexpr EGLint OPENGL_MAJOR = 4;
		constexpr EGLint OPENGL_MINOR_BEST	= 6;
		constexpr EGLint OPENGL_MINOR_WORST = 3;

		auto initRes = createContext(_egl, EGL_OPENGL_API, { OPENGL_MAJOR, OPENGL_MINOR_BEST }, OPENGL_MINOR_WORST);
		if (initRes.minor < OPENGL_MINOR_WORST)
			return nullptr;

		return core::make_smart_refctd_ptr<COpenGLPhysicalDevice>(_egl, initRes.config, initRes.ctx, initRes.major, initRes.minor);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL; }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLLogicalDevice>(m_egl, &m_glfeatures, m_config, m_gl_major, m_gl_minor, params);
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