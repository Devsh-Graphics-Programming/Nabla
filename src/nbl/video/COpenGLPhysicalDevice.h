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
		if (initRes.ctx==EGL_NO_CONTEXT)
			return nullptr;

		if (initRes.minor < OPENGL_MINOR_WORST)
		{
			_egl.call.peglDestroyContext(_egl.display, initRes.ctx);
			return nullptr;
		}


		return new COpenGLPhysicalDevice(api,rdoc,std::move(s),std::move(_egl),std::move(dbgCb),initRes.config,initRes.ctx,initRes.major,initRes.minor);
	}

	E_API_TYPE getAPIType() const override { return EAT_OPENGL; }

protected:
	core::smart_refctd_ptr<ILogicalDevice> createLogicalDevice_impl(ILogicalDevice::SCreationParams&& params) final override
	{
		return core::make_smart_refctd_ptr<COpenGLLogicalDevice>(core::smart_refctd_ptr<IAPIConnection>(m_api),this,m_rdoc_api,params,&m_egl,&m_glfeatures,m_config,m_gl_major,m_gl_minor);
	}

private:
	COpenGLPhysicalDevice(IAPIConnection* api, renderdoc_api_t* rdoc, core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& dbgCb, EGLConfig config, EGLContext ctx, EGLint major, EGLint minor)
		: base_t(api, rdoc, std::move(s),std::move(_egl),std::move(dbgCb), config,ctx,major,minor)
	{
		// Set Format Usages
		for(uint32_t i = 0; i < asset::EF_COUNT; ++i)
		{
			const asset::E_FORMAT format = static_cast<asset::E_FORMAT>(i);

			// TODO: correct format reporting

			m_linearTilingUsages[format] = {};
			m_linearTilingUsages[format].sampledImage = 1;
			m_linearTilingUsages[format].storageImage = 1;
			m_linearTilingUsages[format].storageImageAtomic = 1;
			m_linearTilingUsages[format].attachment = 1;
			m_linearTilingUsages[format].attachmentBlend = 1;
			m_linearTilingUsages[format].blitSrc = 1;
			m_linearTilingUsages[format].blitDst = 1;
			m_linearTilingUsages[format].transferSrc = 1;
			m_linearTilingUsages[format].transferDst = 1;
			m_linearTilingUsages[format].log2MaxSamples = 7u;
			
			m_optimalTilingUsages[format] = {};
			m_optimalTilingUsages[format].sampledImage = 1;
			m_optimalTilingUsages[format].storageImage = 1;
			m_optimalTilingUsages[format].storageImageAtomic = 1;
			m_optimalTilingUsages[format].attachment = 1;
			m_optimalTilingUsages[format].attachmentBlend = 1;
			m_optimalTilingUsages[format].blitSrc = 1;
			m_optimalTilingUsages[format].blitDst = 1;
			m_optimalTilingUsages[format].transferSrc = 1;
			m_optimalTilingUsages[format].transferDst = 1;
			m_optimalTilingUsages[format].log2MaxSamples = 7u;

			m_bufferUsages[format] = {};
			m_bufferUsages[format].vertexAttribute = 1;
			m_bufferUsages[format].bufferView = 1;
			m_bufferUsages[format].storageBufferView = 1;
			m_bufferUsages[format].storageBufferViewAtomic = 1;
			m_bufferUsages[format].accelerationStructureVertex = 0;
		}
	}
};

}

#endif