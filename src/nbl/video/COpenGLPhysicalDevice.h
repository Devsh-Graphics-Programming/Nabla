#ifndef __NBL_C_OPENGL_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IOpenGL_PhysicalDeviceBase.h"
#include "nbl/video/COpenGL_LogicalDevice.h"

namespace nbl::video
{

class COpenGLPhysicalDevice final : public IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>
{
    using base_t = IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>;

public:
	static COpenGLPhysicalDevice* create(IAPIConnection* api, renderdoc_api_t* rdoc, core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb)
	{
		constexpr EGLint OPENGL_MAJOR = 4;
		constexpr EGLint OPENGL_MINOR_BEST	= 6;
		constexpr EGLint OPENGL_MINOR_WORST = 3;

		auto initRes = createContext(&_egl,EGL_OPENGL_API,{OPENGL_MAJOR,OPENGL_MINOR_BEST},OPENGL_MINOR_WORST);
		if (initRes.ctx==EGL_NO_CONTEXT || initRes.minor<OPENGL_MINOR_WORST) // TODO: delete context if minor is too low, right now its leaking
			return nullptr;

		return new COpenGLPhysicalDevice(api,rdoc,std::move(s),std::move(_egl),std::move(dbgCb),initRes.config,initRes.ctx,initRes.major,initRes.minor);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL; }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLLogicalDevice>(core::smart_refctd_ptr<IAPIConnection>(m_api),this,m_rdoc_api,params,&m_egl,&m_glfeatures,m_config,m_gl_major,m_gl_minor);
	}

private:
    COpenGLPhysicalDevice(IAPIConnection* api, renderdoc_api_t* rdoc, core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb, EGLConfig config, EGLContext ctx, EGLint major, EGLint minor)
		: base_t(api, rdoc, std::move(s),std::move(_egl),std::move(dbgCb), config,ctx,major,minor)
    {
    }
};

}

#endif