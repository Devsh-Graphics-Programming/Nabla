#ifndef __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__

#include "nbl/video/CEGL.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/COpenGLFeatureMap.h"
#ifndef GL_GLEXT_LEGACY
#define GL_GLEXT_LEGACY 1
#endif
#include "GL/gl.h"
#undef GL_GLEXT_LEGACY
#include "GL/glext.h"

namespace nbl { 
namespace video
{

template <typename LogicalDeviceType>
class IOpenGL_PhysicalDeviceBase : public IPhysicalDevice
{
    static inline constexpr EGLint EGL_API_TYPE = LogicalDeviceType::FunctionTableType::EGL_API_TYPE;
    static inline constexpr bool IsGLES = (EGL_API_TYPE == EGL_OPENGL_ES_API);

public:
    IOpenGL_PhysicalDeviceBase(const egl::CEGL* _egl, EGLConfig _config, EGLContext ctx, EGLint _major, EGLint _minor) : 
        m_egl(_egl), config(_config), m_gl_major(_major), m_gl_minor(_minor)
    {
        // OpenGL backend emulates presence of just one queue with all capabilities (graphics, compute, transfer, ... what about sparse binding?)
        SQueueFamilyProperties qprops;
        qprops.queueFlags = EQF_GRAPHICS_BIT | EQF_COMPUTE_BIT | EQF_TRANSFER_BIT;
        qprops.queueCount = 1u;
        qprops.timestampValidBits = 64u; // ??? TODO
        qprops.minImageTransferGranularity = { 1u,1u,1u }; // ??? TODO

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(1u, qprops);

        // TODO fill m_properties and m_features (possibly should be done in derivative classes' ctors, not sure yet)


		_egl->call.peglBindAPI(EGL_API_TYPE);

		const EGLint pbuffer_attributes[] = {
			EGL_WIDTH,  1,
			EGL_HEIGHT, 1,

			EGL_NONE
		};
		EGLSurface pbuf = _egl->call.peglCreatePbufferSurface(_egl->display, config, pbuffer_attributes);

		_egl->call.peglMakeCurrent(_egl->display, pbuf, pbuf, ctx);

		auto GetString = reinterpret_cast<decltype(glGetString)*>(_egl->call.peglGetProcAddress("glGetString"));
		auto GetStringi = reinterpret_cast<PFNGLGETSTRINGIPROC>(_egl->call.peglGetProcAddress("glGetStringi"));
		auto GetIntegerv = reinterpret_cast<decltype(glGetIntegerv)*>(_egl->call.peglGetProcAddress("glGetIntegerv"));
		auto GetInteger64v = reinterpret_cast<PFNGLGETINTEGER64VPROC>(_egl->call.peglGetProcAddress("glGetInteger64v"));
		auto GetIntegeri_v = reinterpret_cast<PFNGLGETINTEGERI_VPROC>(_egl->call.peglGetProcAddress("glGetIntegeri_v"));
		auto GetFloatv = reinterpret_cast<decltype(glGetFloatv)*>(_egl->call.peglGetProcAddress("glGetFloatv"));

		// initialize features
		std::string vendor = reinterpret_cast<const char*>(GetString(GL_VENDOR));
		m_glfeatures.isIntelGPU = (vendor.find("Intel") != vendor.npos || vendor.find("INTEL") != vendor.npos);

		float ogl_ver;
		sscanf(reinterpret_cast<const char*>(GetString(GL_VERSION)), "%f", &ogl_ver);
		m_glfeatures.Version = static_cast<uint16_t>(core::round(ogl_ver * 100.0f));

		const GLubyte* shaderVersion = GetString(GL_SHADING_LANGUAGE_VERSION);
		float sl_ver;
		sscanf(reinterpret_cast<const char*>(shaderVersion), "%f", &sl_ver);
		m_glfeatures.ShaderLanguageVersion = static_cast<uint16_t>(core::round(sl_ver * 100.0f));

		//should contain space-separated OpenGL extension names
		constexpr const char* OPENGL_EXTS_ENVVAR_NAME = "_NBL_OPENGL_EXTENSIONS_LIST";//move this to some top-level header?

		const char* envvar = std::getenv(OPENGL_EXTS_ENVVAR_NAME);
		if (!envvar)
		{
			GLint extensionCount;
			GetIntegerv(GL_NUM_EXTENSIONS, &extensionCount);
			for (GLint i = 0; i < extensionCount; ++i)
			{
				const char* extensionName = reinterpret_cast<const char*>(GetStringi(GL_EXTENSIONS, i));

				for (uint32_t j = 0; j < m_glfeatures.NBL_OpenGL_Feature_Count; ++j)
				{
					if (!strcmp(OpenGLFeatureStrings[j], extensionName))
					{
						m_glfeatures.FeatureAvailable[j] = true;
						break;
					}
				}
			}
		}
		else
		{
			std::stringstream ss{ std::string(envvar) };
			std::string extname;
			extname.reserve(100);
			while (std::getline(ss, extname))
			{
				for (uint32_t j = 0; j < m_glfeatures.NBL_OpenGL_Feature_Count; ++j)
				{
					if (extname == OpenGLFeatureStrings[j])
					{
						m_glfeatures.FeatureAvailable[j] = true;
						break;
					}
				}
			}
		}

		GLint num = 0;

		GetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqUBOAlignment);
		assert(core::is_alignment(reqUBOAlignment));
		GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqSSBOAlignment);
		assert(core::is_alignment(reqSSBOAlignment));
		GetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqTBOAlignment);
		assert(core::is_alignment(reqTBOAlignment));

		GetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, reinterpret_cast<GLint64*>(&m_glfeatures.maxUBOSize));
		GetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, reinterpret_cast<GLint64*>(&m_glfeatures.maxSSBOSize));
		GetInteger64v(GL_MAX_TEXTURE_BUFFER_SIZE, reinterpret_cast<GLint64*>(&m_glfeatures.maxTBOSizeInTexels));
		m_glfeatures.maxBufferSize = std::max(m_glfeatures.maxUBOSize, m_glfeatures.maxSSBOSize);

		GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, reinterpret_cast<GLint*>(&m_glfeatures.maxUBOBindings));
		GetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, reinterpret_cast<GLint*>(&m_glfeatures.maxSSBOBindings));
		GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_glfeatures.maxTextureBindings));
		GetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_glfeatures.maxTextureBindingsCompute));
		GetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, reinterpret_cast<GLint*>(&m_glfeatures.maxImageBindings));
		GetIntegerv(GL_MAX_COLOR_ATTACHMENTS, reinterpret_cast<GLint*>(&m_glfeatures.MaxColorAttachments));

		if constexpr (!isGLES)
			GetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, &m_glfeatures.minMemoryMapAlignment);
		else
			m_glfeatures.minMemoryMapAlignment = 0;

		GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, m_glfeatures.MaxComputeWGSize);
		GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, m_glfeatures.MaxComputeWGSize + 1);
		GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, m_glfeatures.MaxComputeWGSize + 2);


		GetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &num);
		m_glfeatures.MaxArrayTextureLayers = num;

		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_texture_filter_anisotropic))
		{
			GetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &num);
			m_glfeatures.MaxAnisotropy = static_cast<uint8_t>(num);
		}


		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_ARB_geometry_shader4))
		{
			GetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &num);
			m_glfeatures.MaxGeometryVerticesOut = static_cast<uint32_t>(num);
		}

		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_texture_lod_bias))
			GetFloatv(GL_MAX_TEXTURE_LOD_BIAS_EXT, &m_glfeatures.MaxTextureLODBias);

		if constexpr (!isGLES)
		{
			GetIntegerv(GL_MAX_CLIP_DISTANCES, &num);
			m_glfeatures.MaxUserClipPlanes = static_cast<uint8_t>(num);
		}
		else
			m_glfeatures.MaxUserClipPlanes = 0;
		GetIntegerv(GL_MAX_DRAW_BUFFERS, &num);
		m_glfeatures.MaxMultipleRenderTargets = static_cast<uint8_t>(num);

		GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, m_glfeatures.DimAliasedLine);
		GetFloatv(GL_ALIASED_POINT_SIZE_RANGE, m_glfeatures.DimAliasedPoint);

		if constexpr (isGLES)
		{
			GetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE, m_glfeatures.DimSmoothedLine);
			GetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, m_glfeatures.DimSmoothedPoint);
		}
		else
		{
			m_glfeatures.DimSmoothedLine = 0;
			m_glfeatures.DimSmoothedPoint = 0;
		}

		// we dont need this any more
		_egl->call.peglMakeCurrent(_egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		_egl->call.peglDestroyContext(_egl->display, ctx);
		_egl->call.peglDestroySurface(_egl->display, pbuf);
    }

protected:
    virtual ~IOpenGL_PhysicalDeviceBase() = default;

protected:
    const egl::CEGL* m_egl;
    EGLConfig m_config;
    EGLint m_gl_major, m_gl_minor;

	COpenGLFeatureMap m_glfeatures;
};

}
}

#endif
