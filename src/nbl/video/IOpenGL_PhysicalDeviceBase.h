#ifndef __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__

#include <regex>
#include "nbl/video/CEGL.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/COpenGLFeatureMap.h"
#include "nbl/video/COpenGLDebug.h"
#define GL_GLEXT_PROTOTYPES
#define GL_APICALL extern
#define GL_APIENTRY // im not sure about calling convention...
#undef GL_KHR_debug
#include "GLES3/gl2ext.h"

namespace nbl { 
namespace video
{

template <typename LogicalDeviceType>
class IOpenGL_PhysicalDeviceBase : public IPhysicalDevice
{
    static inline constexpr EGLint EGL_API_TYPE = LogicalDeviceType::FunctionTableType::EGL_API_TYPE;
    static inline constexpr bool IsGLES = (EGL_API_TYPE == EGL_OPENGL_ES_API);

	static inline constexpr uint32_t MaxQueues = 8u;

protected:
	struct SInitResult
	{
		EGLConfig config;
		EGLContext ctx;
		EGLint major;
		EGLint minor;
	};
#if 0
	static void print_cfg(const egl::CEGL* _egl, EGLConfig cfg)
	{
		auto getAttrib = [&] (EGLint a) -> EGLint
		{
			EGLint val;
			_egl->call.peglGetConfigAttrib(_egl->display, cfg, a, &val);
			return val;
		};
		EGLint alpha = getAttrib(EGL_ALPHA_SIZE);
		EGLint blue = getAttrib(EGL_BLUE_SIZE);
		EGLint conformant = getAttrib(EGL_CONFORMANT);
		EGLint green = getAttrib(EGL_GREEN_SIZE);
		EGLint red = getAttrib(EGL_RED_SIZE);
		EGLint renderable = getAttrib(EGL_RENDERABLE_TYPE);
		EGLint surface = getAttrib(EGL_SURFACE_TYPE);
		EGLint colorbuf = getAttrib(EGL_COLOR_BUFFER_TYPE);

		if (colorbuf != EGL_RGB_BUFFER || !(conformant&EGL_OPENGL_ES3_BIT) || !(renderable&EGL_OPENGL_ES3_BIT))
			return;

		printf("alpha=%d\nred=%d\ngreen=%d\nblue=%d\n", alpha, red, green, blue);
		if (surface&EGL_PBUFFER_BIT)
			printf("pbuffer, ");
		if (surface&EGL_WINDOW_BIT)
			printf("window");
		printf("\n");
	}
#endif
	static SInitResult createContext(const egl::CEGL* _egl, EGLenum api_type, const std::pair<EGLint, EGLint>& bestApiVer, EGLint minMinorVer)
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

		_egl->call.peglBindAPI(api_type);

		const EGLint egl_attributes[] = {
			EGL_RED_SIZE, red,
			EGL_GREEN_SIZE, green,
			EGL_BLUE_SIZE, blue,
			//EGL_BUFFER_SIZE, bufsz,
			EGL_DEPTH_SIZE, depth,
			EGL_STENCIL_SIZE, stencil,
			EGL_ALPHA_SIZE, alpha,
			EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
			EGL_CONFORMANT, IsGLES ? EGL_OPENGL_ES3_BIT:EGL_OPENGL_BIT,
			EGL_RENDERABLE_TYPE, IsGLES ? EGL_OPENGL_ES3_BIT:EGL_OPENGL_BIT,
			//Params.Stereobuffer
			//Params.Vsync
			EGL_SURFACE_TYPE, (EGL_WINDOW_BIT | EGL_PBUFFER_BIT),

			EGL_NONE
		};

		SInitResult res;
		res.config = NULL;
		res.ctx = EGL_NO_CONTEXT;
		res.major = 0;
		res.minor = 0;

#if 0
		EGLConfig cfgs[1024];
		EGLint cfgs_count;
		_egl->call.peglGetConfigs(_egl->display, cfgs, 1024, &cfgs_count);
		for (int i = 0; i < cfgs_count; ++i)
		{
			printf("PRINTING CONFIG %d\n", i);
			print_cfg(_egl, cfgs[i]);
		}
#endif

		EGLint ccnt = 1;
		int chCfgRes = _egl->call.peglChooseConfig(_egl->display, egl_attributes, &res.config, 1, &ccnt);
		if (ccnt < 1)
		{
			//LOGI("Couldnt find EGL fb config!");
			return res;
		}

		EGLint ctx_attributes[] = {
			EGL_CONTEXT_MAJOR_VERSION, bestApiVer.first,
			EGL_CONTEXT_MINOR_VERSION, bestApiVer.second,
#ifdef _NBL_DEBUG
			//EGL_CONTEXT_OPENGL_DEBUG, EGL_TRUE,
#endif
			//EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,

			EGL_NONE
		};
		EGLint& gl_major = ctx_attributes[1];
		EGLint& gl_minor = ctx_attributes[3];

		EGLContext ctx = EGL_NO_CONTEXT;
		do
		{
			res.ctx = _egl->call.peglCreateContext(_egl->display, res.config, EGL_NO_CONTEXT, ctx_attributes);
			--gl_minor;
			//LOGI("eglCreateContext() tryout result = %d", res.ctx == EGL_NO_CONTEXT ? 0 : 1);
		} while (res.ctx == EGL_NO_CONTEXT && gl_minor >= minMinorVer); // fail if cant create >=4.3 context
		++gl_minor;

		//LOGI("glCreateContext() bestApiVer was { %u, %u }", bestApiVer.first, bestApiVer.second);
		//LOGI("glCreateContext() last tried api ver was { %d, %d }", gl_major, gl_minor);
		if (res.ctx == EGL_NO_CONTEXT)
		{
			//LOGI("Couldnt create context!");
			return res;
		}

		res.major = gl_major;
		res.minor = gl_minor;

		return res;
	}

public:
    IOpenGL_PhysicalDeviceBase(core::smart_refctd_ptr<io::IFileSystem>&& fs, core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc, 
		const egl::CEGL* _egl, EGLConfig _config, EGLContext ctx, EGLint _major, EGLint _minor, SDebugCallback* dbgCb) :
		IPhysicalDevice(std::move(fs), std::move(glslc)),
        m_egl(_egl), m_config(_config), m_gl_major(_major), m_gl_minor(_minor), m_dbgCb(dbgCb)
    {
        // OpenGL backend emulates presence of just one queue family with all capabilities (graphics, compute, transfer, ... what about sparse binding?)
        SQueueFamilyProperties qprops;
        qprops.queueFlags = EQF_GRAPHICS_BIT | EQF_COMPUTE_BIT | EQF_TRANSFER_BIT;
        qprops.queueCount = MaxQueues;
        qprops.timestampValidBits = 64u; // ??? TODO
        qprops.minImageTransferGranularity = { 1u,1u,1u }; // ??? TODO

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(1u, qprops);

		_egl->call.peglBindAPI(EGL_API_TYPE);

		const EGLint pbuffer_attributes[] = {
			EGL_WIDTH,  1,
			EGL_HEIGHT, 1,

			EGL_NONE
		};
		EGLSurface pbuf = _egl->call.peglCreatePbufferSurface(_egl->display, m_config, pbuffer_attributes);

		_egl->call.peglMakeCurrent(_egl->display, pbuf, pbuf, ctx);

		auto GetString = reinterpret_cast<decltype(glGetString)*>(_egl->call.peglGetProcAddress("glGetString"));
		auto GetStringi = reinterpret_cast<PFNGLGETSTRINGIPROC>(_egl->call.peglGetProcAddress("glGetStringi"));
		auto GetIntegerv = reinterpret_cast<decltype(glGetIntegerv)*>(_egl->call.peglGetProcAddress("glGetIntegerv"));
		auto GetInteger64v = reinterpret_cast<PFNGLGETINTEGER64VPROC>(_egl->call.peglGetProcAddress("glGetInteger64v"));
		auto GetIntegeri_v = reinterpret_cast<PFNGLGETINTEGERI_VPROC>(_egl->call.peglGetProcAddress("glGetIntegeri_v"));
		auto GetFloatv = reinterpret_cast<decltype(glGetFloatv)*>(_egl->call.peglGetProcAddress("glGetFloatv"));
		auto GetBooleanv = reinterpret_cast<decltype(glGetBooleanv)*>(_egl->call.peglGetProcAddress("glGetBooleanv"));

#ifdef _NBL_DEBUG
		if (dbgCb)
		{
			if constexpr (IsGLES)
			{
				PFNGLDEBUGMESSAGECALLBACKPROC DebugMessageCallback = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKPROC>(_egl->call.peglGetProcAddress("glDebugMessageCallback"));
				PFNGLDEBUGMESSAGECALLBACKKHRPROC DebugMessageCallbackKHR = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKKHRPROC>(_egl->call.peglGetProcAddress("glDebugMessageCallbackKHR"));
				if (DebugMessageCallback)
					DebugMessageCallback(&opengl_debug_callback, dbgCb);
				else if (DebugMessageCallbackKHR)
					DebugMessageCallbackKHR(&opengl_debug_callback, dbgCb);
			}
			else
			{
				PFNGLDEBUGMESSAGECALLBACKPROC DebugMessageCallback = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKPROC>(_egl->call.peglGetProcAddress("glDebugMessageCallback"));
				PFNGLDEBUGMESSAGECALLBACKARBPROC DebugMessageCallbackARB = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKARBPROC>(_egl->call.peglGetProcAddress("glDebugMessageCallbackARB"));
				if (DebugMessageCallback)
					DebugMessageCallback(&opengl_debug_callback, dbgCb);
				else if (DebugMessageCallbackARB)
					DebugMessageCallbackARB(&opengl_debug_callback, dbgCb);
			}
		}
#endif

		// initialize features
		std::string vendor = reinterpret_cast<const char*>(GetString(GL_VENDOR));
		m_glfeatures.isIntelGPU = (vendor.find("Intel") != vendor.npos || vendor.find("INTEL") != vendor.npos);

		const std::regex version_re("([1-9]\\.[0-9])");
		std::cmatch re_match;

		float ogl_ver = 0.f;
		const char* ogl_ver_str = reinterpret_cast<const char*>(GetString(GL_VERSION));
		if (std::regex_search(ogl_ver_str, re_match, version_re) && re_match.size() >= 2)
		{
			sscanf(re_match[1].str().c_str(), "%f", &ogl_ver);
			m_glfeatures.Version = static_cast<uint16_t>(core::round(ogl_ver * 100.0f));
		}
		else
		{
			sscanf(ogl_ver_str, "%f", &ogl_ver);
		}
		assert(ogl_ver != 0.f);

		float sl_ver;
		const char* shaderVersion = reinterpret_cast<const char*>(GetString(GL_SHADING_LANGUAGE_VERSION));
		if (std::regex_search(shaderVersion, re_match, version_re) && re_match.size() >= 2)
		{
			sscanf(re_match[1].str().c_str(), "%f", &sl_ver);
			m_glfeatures.ShaderLanguageVersion = static_cast<uint16_t>(core::round(sl_ver * 100.0f));
		}
		else
		{
			sscanf(shaderVersion, "%f", &sl_ver);
		}
		assert(sl_ver != 0.f);

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
					if (!strcmp(m_glfeatures.OpenGLFeatureStrings[j], extensionName))
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
					if (extname == m_glfeatures.OpenGLFeatureStrings[j])
					{
						m_glfeatures.FeatureAvailable[j] = true;
						break;
					}
				}
			}
		}

		GLint num = 0;

		GetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqUBOAlignment);
		assert(core::is_alignment(m_glfeatures.reqUBOAlignment));
		GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqSSBOAlignment);
		assert(core::is_alignment(m_glfeatures.reqSSBOAlignment));
		GetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqTBOAlignment);
		assert(core::is_alignment(m_glfeatures.reqTBOAlignment));

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

		if constexpr (!IsGLES)
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
		else m_glfeatures.MaxAnisotropy = 0u;


		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_ARB_geometry_shader4))
		{
			GetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &num);
			m_glfeatures.MaxGeometryVerticesOut = static_cast<uint32_t>(num);
		}

		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_texture_lod_bias))
			GetFloatv(GL_MAX_TEXTURE_LOD_BIAS_EXT, &m_glfeatures.MaxTextureLODBias);

		if constexpr (!IsGLES)
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

		if constexpr (!IsGLES)
		{
			GetFloatv(GL_SMOOTH_LINE_WIDTH_RANGE, m_glfeatures.DimSmoothedLine);
			GetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, m_glfeatures.DimSmoothedPoint);
		}
		else
		{
			m_glfeatures.DimSmoothedLine[0] = 0;
			m_glfeatures.DimSmoothedLine[1] = 0;
			m_glfeatures.DimSmoothedPoint[0] = 0;
			m_glfeatures.DimSmoothedPoint[1] = 0;
		}

		bool runningInRenderDoc = false;
#ifdef _NBL_PLATFORM_WINDOWS_
		if (GetModuleHandleA("renderdoc.dll"))
#elif defined(_NBL_PLATFORM_ANDROID_)
		if (dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD))
#elif defined(_NBL_PLATFORM_LINUX_)
		if (dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
#else
		if (false)
#endif
			runningInRenderDoc = true;
		m_glfeatures.runningInRenderDoc = runningInRenderDoc;

		// physical device features
		{
			m_features.logicOp = !IsGLES;
			m_features.multiViewport = IsGLES ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_OES_viewport_array) : true;
			m_features.multiDrawIndirect = IsGLES ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_multi_draw_indirect) : true;
			m_features.imageCubeArray = true; //we require OES_texture_cube_map_array on GLES
			m_features.robustBufferAccess = false;
			m_features.vertexAttributeDouble = !IsGLES;

			if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_shader_subgroup))
			{
				GLboolean subgroupQuadAllStages = GL_FALSE;
				GetBooleanv(GL_SUBGROUP_QUAD_ALL_STAGES_KHR, &subgroupQuadAllStages);
				m_features.shaderSubgroupQuadAllStages = static_cast<bool>(subgroupQuadAllStages);

				GLint subgroup = 0;
				GetIntegerv(GL_SUBGROUP_SUPPORTED_FEATURES_KHR, &subgroup);

				m_features.shaderSubgroupBasic = (subgroup & GL_SUBGROUP_FEATURE_BASIC_BIT_KHR);
				m_features.shaderSubgroupVote = (subgroup & GL_SUBGROUP_FEATURE_VOTE_BIT_KHR);
				m_features.shaderSubgroupArithmetic = (subgroup & GL_SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR);
				m_features.shaderSubgroupBallot = (subgroup & GL_SUBGROUP_FEATURE_BALLOT_BIT_KHR);
				m_features.shaderSubgroupShuffle = (subgroup & GL_SUBGROUP_FEATURE_SHUFFLE_BIT_KHR);
				m_features.shaderSubgroupShuffleRelative = (subgroup & GL_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR);
				m_features.shaderSubgroupClustered = (subgroup & GL_SUBGROUP_FEATURE_CLUSTERED_BIT_KHR);
				m_features.shaderSubgroupQuad = (subgroup & GL_SUBGROUP_FEATURE_QUAD_BIT_KHR);
			}
		}

		// physical device limits
		{
			m_limits.UBOAlignment = m_glfeatures.reqUBOAlignment;
			m_limits.SSBOAlignment = m_glfeatures.reqSSBOAlignment;
			m_limits.bufferViewAlignment = m_glfeatures.reqTBOAlignment;

			m_limits.maxUBOSize = m_glfeatures.maxUBOSize;
			m_limits.maxSSBOSize = m_glfeatures.maxSSBOSize;
			m_limits.maxBufferViewSizeTexels = m_glfeatures.maxTBOSizeInTexels;
			m_limits.maxBufferSize = std::max(m_limits.maxUBOSize, m_limits.maxSSBOSize);

			GLint max_ssbos[5];
			GetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, max_ssbos + 0);
			GetIntegerv(GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS, max_ssbos + 1);
			GetIntegerv(GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, max_ssbos + 2);
			GetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, max_ssbos + 3);
			GetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, max_ssbos + 4);
			uint32_t maxSSBOsPerStage = static_cast<uint32_t>(*std::min_element(max_ssbos, max_ssbos + 5));

			m_limits.maxPerStageSSBOs = maxSSBOsPerStage;

			m_limits.maxSSBOs = m_glfeatures.maxSSBOBindings;
			m_limits.maxUBOs = m_glfeatures.maxUBOBindings;
			m_limits.maxTextures = m_glfeatures.maxTextureBindings;
			m_limits.maxStorageImages = m_glfeatures.maxImageBindings;

			GetFloatv(GL_POINT_SIZE_RANGE, m_limits.pointSizeRange);
			GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, m_limits.lineWidthRange);

			GLint maxViewportExtent[2]{0,0};
			GetIntegerv(GL_MAX_VIEWPORT_DIMS, maxViewportExtent);

			GLint maxViewports = 16;
			GetIntegerv(GL_MAX_VIEWPORTS, &maxViewports);

			m_limits.maxViewports = maxViewports;

			m_limits.maxViewportDims[0] = maxViewportExtent[0];
			m_limits.maxViewportDims[1] = maxViewportExtent[1];

			m_limits.maxWorkgroupSize[0] = m_glfeatures.MaxComputeWGSize[0];
			m_limits.maxWorkgroupSize[1] = m_glfeatures.MaxComputeWGSize[1];
			m_limits.maxWorkgroupSize[2] = m_glfeatures.MaxComputeWGSize[2];

			m_limits.subgroupSize = 0u;
			m_limits.subgroupOpsShaderStages = 0u;

			if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_shader_subgroup))
			{
				GLint subgroupSize = 0;
				GetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);

				GLint subgroupOpsStages = 0;
				GetIntegerv(GL_SUBGROUP_SUPPORTED_STAGES_KHR, &subgroupOpsStages);
				if (subgroupOpsStages & GL_VERTEX_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::ISpecializedShader::ESS_VERTEX;
				if (subgroupOpsStages & GL_TESS_CONTROL_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::ISpecializedShader::ESS_TESSELATION_CONTROL;
				if (subgroupOpsStages & GL_TESS_EVALUATION_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::ISpecializedShader::ESS_TESSELATION_EVALUATION;
				if (subgroupOpsStages & GL_GEOMETRY_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::ISpecializedShader::ESS_GEOMETRY;
				if (subgroupOpsStages & GL_FRAGMENT_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::ISpecializedShader::ESS_FRAGMENT;
				if (subgroupOpsStages & GL_COMPUTE_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::ISpecializedShader::ESS_COMPUTE;
			}
		}


		// we dont need this any more
		_egl->call.peglMakeCurrent(_egl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		_egl->call.peglDestroyContext(_egl->display, ctx);
		_egl->call.peglDestroySurface(_egl->display, pbuf);
    }

protected:
    virtual ~IOpenGL_PhysicalDeviceBase() = default;

    const egl::CEGL* m_egl;
    EGLConfig m_config;
    EGLint m_gl_major, m_gl_minor;

	COpenGLFeatureMap m_glfeatures;

	SDebugCallback* m_dbgCb;
};

}
}

#endif
