#ifndef __NBL_C_OPENGLES_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGLES_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/CEGL.h"
#include "nbl/video/IOpenGL_PhysicalDeviceBase.h"
#include "nbl/video/COpenGL_LogicalDevice.h"

namespace nbl::video
{

class COpenGLESPhysicalDevice final : public IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>
{
    using base_t = IOpenGL_PhysicalDeviceBase<COpenGLESLogicalDevice>;

public:
	static COpenGLESPhysicalDevice* create(core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb)
	{
		constexpr EGLint OPENGL_ES_MAJOR = 3;
		constexpr EGLint OPENGL_ES_MINOR_BEST  = 2;
		constexpr EGLint OPENGL_ES_MINOR_WORST = 1;

		auto initRes = createContext(&_egl, EGL_OPENGL_ES_API, { OPENGL_ES_MAJOR, OPENGL_ES_MINOR_BEST }, OPENGL_ES_MINOR_WORST);
		if (initRes.ctx==EGL_NO_CONTEXT || initRes.minor<OPENGL_ES_MINOR_WORST) // TODO: delete context if minor is too low, right now its leaking
			return nullptr;

		return new COpenGLESPhysicalDevice(std::move(s),std::move(_egl),std::move(dbgCb),initRes.config,initRes.ctx,initRes.major,initRes.minor);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL_ES; }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLESLogicalDevice>(this,params,&m_egl,&m_glfeatures,m_config,m_gl_major,m_gl_minor);
	}

private:
	COpenGLESPhysicalDevice(
		core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb,
		EGLConfig config, EGLContext ctx, EGLint major, EGLint minor
	) : base_t(std::move(s),std::move(_egl),std::move(dbgCb), config,ctx,major,minor)
    {
    }
};

}

#endif