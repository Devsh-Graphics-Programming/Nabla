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
	static core::smart_refctd_ptr<COpenGLPhysicalDevice> create(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc, const egl::CEGL* _egl, SDebugCallback* dbgCb, system::logger_opt_smart_ptr&& logger)
	{
		constexpr EGLint OPENGL_MAJOR = 4;
		constexpr EGLint OPENGL_MINOR_BEST	= 6;
		constexpr EGLint OPENGL_MINOR_WORST = 3;

		auto initRes = createContext(_egl, EGL_OPENGL_API, { OPENGL_MAJOR, OPENGL_MINOR_BEST }, OPENGL_MINOR_WORST);
		if (initRes.minor < OPENGL_MINOR_WORST)
			return nullptr;

		auto* pdev = new COpenGLPhysicalDevice(std::move(s), std::move(glslc), _egl, initRes.config, initRes.ctx, initRes.major, initRes.minor, dbgCb, std::move(logger));
		return core::smart_refctd_ptr<COpenGLPhysicalDevice>(pdev, core::dont_grab);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL; }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(const ILogicalDevice::SCreationParams& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLLogicalDevice>(m_egl, &m_glfeatures, m_config, m_gl_major, m_gl_minor, params, m_dbgCb, core::smart_refctd_ptr(m_system), core::smart_refctd_ptr(m_GLSLCompiler), system::logger_opt_smart_ptr(m_logger));
	}

private:
    COpenGLPhysicalDevice(core::smart_refctd_ptr<system::ISystem>&& s, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc, const egl::CEGL* _egl, EGLConfig config, EGLContext ctx, EGLint major, EGLint minor, SDebugCallback* dbgCb, system::logger_opt_smart_ptr&& logger) :
        base_t(std::move(s), std::move(glslc), _egl, config, ctx, major, minor, dbgCb, std::move(logger))
    {

    }
	_NBL_DEBUG
};

}
}

#endif