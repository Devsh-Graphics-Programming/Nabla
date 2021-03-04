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
	static core::smart_refctd_ptr<COpenGLESPhysicalDevice> create(const egl::CEGL* _egl)
	{
		constexpr EGLint OPENGL_ES_MAJOR = 3;
		constexpr EGLint OPENGL_ES_MINOR_BEST  = 2;
		constexpr EGLint OPENGL_ES_MINOR_WORST = 1;

		auto initRes = createContext(_egl, EGL_OPENGL_ES_API, { OPENGL_ES_MAJOR, OPENGL_ES_MINOR_BEST }, OPENGL_ES_MINOR_WORST);
		if (initRes.minor < OPENGL_ES_MINOR_WORST)
			return nullptr;

		return core::make_smart_refctd_ptr<COpenGLESPhysicalDevice>(_egl, initRes.config, initRes.ctx, initRes.major, initRes.minor);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL_ES; }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLESLogicalDevice>(m_egl, &m_glfeatures, m_config, m_gl_major, m_gl_minor, params);
	}

private:
	COpenGLESPhysicalDevice(const egl::CEGL* _egl, EGLConfig config, EGLContext ctx, EGLint major, EGLint minor) :
        base_t(_egl, config, ctx, major, minor)
    {

    }
};

}
}

#endif