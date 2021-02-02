#ifndef __NBL_C_OPENGL_PHYSICAL_DEVICE_H_INCLUDED__
#define __NBL_C_OPENGL_PHYSICAL_DEVICE_H_INCLUDED__

#include "nbl/video/IOpenGL_PhysicalDeviceBase.h"
#include "nbl/video/COpenGLLogicalDevice.h"
#include "nbl/video/COpenGLFunctionTable.h"
#ifndef GL_GLEXT_LEGACY
#define GL_GLEXT_LEGACY 1
#endif
#include "GL/gl.h"
#undef GL_GLEXT_LEGACY
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include "GL/glext.h"

namespace nbl {
namespace video
{

class COpenGLPhysicalDevice final : public IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>
{
    using base_t = IOpenGL_PhysicalDeviceBase<COpenGLLogicalDevice>;

public:
    static core::smart_refctd_ptr<COpenGLPhysicalDevice> create(const egl::CEGL* _egl)
    {
        const EGLint
            red = 8,
            green = 8,
            blue = 8,
            alpha = 0;
        const EGLint bufsz = red + green + blue;
        const EGLint depth = 24;
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
        eglChooseConfig(_egl->display, egl_attributes, &config, 1, &ccnt);
        if (ccnt < 1)
            return nullptr;

        EGLint ctx_attributes[] = {
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 6,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

            EGL_NONE
        };

        EGLContext ctx = EGL_NO_CONTEXT;
        do
        {
            ctx = eglCreateContext(_egl->display, config, EGL_NO_CONTEXT, ctx_attributes);
            --ctx_attributes[3];
        } while (ctx == EGL_NO_CONTEXT && ctx_attributes[3] >= 3); // fail if cant create >=4.3 context
        ++ctx_attributes[3];

        if (ctx == EGL_NO_CONTEXT)
            return nullptr;

        return core::make_smart_refctd_ptr<COpenGLPhysicalDevice>(_egl, config, ctx, ctx_attributes[1], ctx_attributes[3]);
    }

private:
    explicit COpenGLPhysicalDevice(const egl::CEGL* _egl, EGLConfig config, EGLContext ctx, EGLint major, EGLint minor) : 
        base_t(_egl), 
        m_config(config),
        gl_major(major), gl_minor(minor)
    {
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
		m_features.isIntelGPU = (vendor.find("Intel") != vendor.npos || vendor.find("INTEL") != vendor.npos);

		float ogl_ver;
		sscanf(reinterpret_cast<const char*>(GetString(GL_VERSION)), "%f", &ogl_ver);
		m_features.Version = static_cast<uint16_t>(core::round(ogl_ver * 100.0f));

		const GLubyte* shaderVersion = GetString(GL_SHADING_LANGUAGE_VERSION);
		float sl_ver;
		sscanf(reinterpret_cast<const char*>(shaderVersion), "%f", &sl_ver);
		m_features.ShaderLanguageVersion = static_cast<uint16_t>(core::round(sl_ver * 100.0f));

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

				for (uint32_t j = 0; j < m_features.NBL_OpenGL_Feature_Count; ++j)
				{
					if (!strcmp(OpenGLFeatureStrings[j], extensionName))
					{
						m_features.FeatureAvailable[j] = true;
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
				for (uint32_t j = 0; j < m_features.NBL_OpenGL_Feature_Count; ++j)
				{
					if (extname == OpenGLFeatureStrings[j])
					{
						m_features.FeatureAvailable[j] = true;
						break;
					}
				}
			}
		}

		GLint num = 0;

		GetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &m_features.reqUBOAlignment);
		assert(core::is_alignment(reqUBOAlignment));
		GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &m_features.reqSSBOAlignment);
		assert(core::is_alignment(reqSSBOAlignment));
		GetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &m_features.reqTBOAlignment);
		assert(core::is_alignment(reqTBOAlignment));

		GetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, reinterpret_cast<GLint64*>(&m_features.maxUBOSize));
		GetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, reinterpret_cast<GLint64*>(&m_features.maxSSBOSize));
		GetInteger64v(GL_MAX_TEXTURE_BUFFER_SIZE, reinterpret_cast<GLint64*>(&m_features.maxTBOSizeInTexels));
		m_features.maxBufferSize = std::max(m_features.maxUBOSize, m_features.maxSSBOSize);

		GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, reinterpret_cast<GLint*>(&m_features.maxUBOBindings));
		GetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, reinterpret_cast<GLint*>(&m_features.maxSSBOBindings));
		GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_features.maxTextureBindings));
		GetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_features.maxTextureBindingsCompute));
		GetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, reinterpret_cast<GLint*>(&m_features.maxImageBindings));

		GetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, &m_features.minMemoryMapAlignment);

		GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, m_features.MaxComputeWGSize);
		GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, m_features.MaxComputeWGSize + 1);
		GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, m_features.MaxComputeWGSize + 2);


		GetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &num);
		m_features.MaxArrayTextureLayers = num;

		if (m_features.isFeatureAvailable(m_features.NBL_EXT_texture_filter_anisotropic))
		{
			GetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &num);
			m_features.MaxAnisotropy = static_cast<uint8_t>(num);
		}


		if (m_features.isFeatureAvailable(m_features.NBL_ARB_geometry_shader4))
		{
			GetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &num);
			m_features.MaxGeometryVerticesOut = static_cast<uint32_t>(num);
		}

		if (m_features.isFeatureAvailable(m_features.NBL_EXT_texture_lod_bias))
			GetFloatv(GL_MAX_TEXTURE_LOD_BIAS_EXT, &m_features.MaxTextureLODBias);


		GetIntegerv(GL_MAX_CLIP_DISTANCES, &num);
		m_features.MaxUserClipPlanes = static_cast<uint8_t>(num);
		GetIntegerv(GL_MAX_DRAW_BUFFERS, &num);
		m_features.MaxMultipleRenderTargets = static_cast<uint8_t>(num);

		GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, m_features.DimAliasedLine);
		GetFloatv(GL_ALIASED_POINT_SIZE_RANGE, m_features.DimAliasedPoint);
		GetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE, m_features.DimSmoothedLine);
		GetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, m_features.DimSmoothedPoint);

		// we dont need this any more
		_egl->call.peglMakeCurrent(_egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		_egl->call.peglDestroyContext(_egl->display, ctx);
		_egl->call.peglDestroySurface(_egl->display, pbuf);
    }

    COpenGLFeatureMap m_features;
    EGLConfig m_config;
    EGLint gl_major, gl_minor;
};

}
}

#endif