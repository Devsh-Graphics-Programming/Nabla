#ifndef __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/renderdoc.h"
#include <regex>

#include "nbl/video/COpenGLFeatureMap.h"

#include "nbl/video/CEGL.h"


#include "nbl/video/debug/COpenGLDebugCallback.h"
#ifndef EGL_CONTEXT_OPENGL_NO_ERROR_KHR
#	define EGL_CONTEXT_OPENGL_NO_ERROR_KHR 0x31B3
#endif

namespace nbl::video
{

template <typename LogicalDeviceType>
class IOpenGL_PhysicalDeviceBase : public IPhysicalDevice
{
    using function_table_t = typename LogicalDeviceType::FunctionTableType;
    static inline constexpr EGLint EGL_API_TYPE = function_table_t::EGL_API_TYPE;
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
#if 1
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
	static SInitResult createContext(const egl::CEGL* _egl, EGLenum api_type, const std::pair<EGLint,EGLint>& bestApiVer, EGLint minMinorVer)
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
#ifdef _NBL_PLATFORM_ANDROID_
			EGL_CONTEXT_OPENGL_NO_ERROR_KHR, EGL_TRUE,
#endif
#if defined(_NBL_DEBUG) && !defined(_NBL_PLATFORM_ANDROID_)
			EGL_CONTEXT_OPENGL_DEBUG, EGL_TRUE,
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
    IOpenGL_PhysicalDeviceBase(IAPIConnection* api, renderdoc_api_t* rdoc, core::smart_refctd_ptr<system::ISystem>&& s, egl::CEGL&& _egl, COpenGLDebugCallback&& _dbgCb, EGLConfig _config, EGLContext ctx, EGLint _major, EGLint _minor)
		: IPhysicalDevice(std::move(s),core::make_smart_refctd_ptr<asset::IGLSLCompiler>(s.get())), m_api(api), m_rdoc_api(rdoc), m_egl(std::move(_egl)), m_dbgCb(std::move(_dbgCb)), m_config(_config), m_gl_major(_major), m_gl_minor(_minor)
    {
        // OpenGL backend emulates presence of just one queue family with all capabilities (graphics, compute, transfer, ... what about sparse binding?)
        SQueueFamilyProperties qprops;
        qprops.queueFlags = core::bitflag(EQF_GRAPHICS_BIT)|EQF_COMPUTE_BIT|EQF_TRANSFER_BIT;
        qprops.queueCount = MaxQueues;
        qprops.timestampValidBits = 30u; // ??? TODO: glGetQueryiv(GL_TIMESTAMP,GL_QUERY_COUNTER_BITS,&qprops.timestampValidBits)
        qprops.minImageTransferGranularity = { 1u,1u,1u };

        m_qfamProperties = core::make_refctd_dynamic_array<qfam_props_array_t>(1u, qprops);

		m_egl.call.peglBindAPI(EGL_API_TYPE);

		const EGLint pbuffer_attributes[] = {
			EGL_WIDTH,  1,
			EGL_HEIGHT, 1,

			EGL_NONE
		};
		EGLSurface pbuf = m_egl.call.peglCreatePbufferSurface(m_egl.display, m_config, pbuffer_attributes);

		m_egl.call.peglMakeCurrent(m_egl.display, pbuf, pbuf, ctx);

		auto GetString = reinterpret_cast<decltype(glGetString)*>(m_egl.call.peglGetProcAddress("glGetString"));
		auto GetStringi = reinterpret_cast<PFNGLGETSTRINGIPROC>(m_egl.call.peglGetProcAddress("glGetStringi"));
		auto GetIntegerv = reinterpret_cast<decltype(glGetIntegerv)*>(m_egl.call.peglGetProcAddress("glGetIntegerv"));
		auto GetInteger64v = reinterpret_cast<PFNGLGETINTEGER64VPROC>(m_egl.call.peglGetProcAddress("glGetInteger64v"));
		auto GetIntegeri_v = reinterpret_cast<PFNGLGETINTEGERI_VPROC>(m_egl.call.peglGetProcAddress("glGetIntegeri_v"));
		auto GetFloatv = reinterpret_cast<decltype(glGetFloatv)*>(m_egl.call.peglGetProcAddress("glGetFloatv"));
		auto GetBooleanv = reinterpret_cast<decltype(glGetBooleanv)*>(m_egl.call.peglGetProcAddress("glGetBooleanv"));

		if (m_dbgCb.getLogger() && m_dbgCb.m_callback)
		{
			if constexpr (IsGLES)
			{
				PFNGLDEBUGMESSAGECALLBACKPROC DebugMessageCallback = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKPROC>(m_egl.call.peglGetProcAddress("glDebugMessageCallback"));
				PFNGLDEBUGMESSAGECALLBACKKHRPROC DebugMessageCallbackKHR = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKKHRPROC>(m_egl.call.peglGetProcAddress("glDebugMessageCallbackKHR"));
				if (DebugMessageCallback)
					DebugMessageCallback(m_dbgCb.m_callback,&m_dbgCb);
				else if (DebugMessageCallbackKHR)
					DebugMessageCallbackKHR(m_dbgCb.m_callback,&m_dbgCb);
			}
			else
			{
				PFNGLDEBUGMESSAGECALLBACKPROC DebugMessageCallback = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKPROC>(m_egl.call.peglGetProcAddress("glDebugMessageCallback"));
				PFNGLDEBUGMESSAGECALLBACKARBPROC DebugMessageCallbackARB = reinterpret_cast<PFNGLDEBUGMESSAGECALLBACKARBPROC>(m_egl.call.peglGetProcAddress("glDebugMessageCallbackARB"));
				if (DebugMessageCallback)
					DebugMessageCallback(m_dbgCb.m_callback,&m_dbgCb);
				else if (DebugMessageCallbackARB)
					DebugMessageCallbackARB(m_dbgCb.m_callback,&m_dbgCb);
			}
		}

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
			m_glfeatures.minMemoryMapAlignment = 0; // TODO: probably wise to set it to 4

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
		}
		else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_clip_cull_distance)) // ES
		{
				GetIntegerv(GL_MAX_CLIP_DISTANCES_EXT, &num);
		}
		m_glfeatures.MaxUserClipPlanes = static_cast<uint8_t>(num);

		GetIntegerv(GL_MAX_DRAW_BUFFERS, &num);
		m_glfeatures.MaxMultipleRenderTargets = static_cast<uint8_t>(num);

		// TODO: move this to IPhysicalDevice::SFeatures
		const bool runningInRenderDoc = (m_rdoc_api != nullptr);
		m_glfeatures.runningInRenderDoc = runningInRenderDoc;

		// physical device features
		{
			m_features.logicOp = !IsGLES;
			m_features.multiViewport = IsGLES ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_OES_viewport_array) : true;
			m_features.imageCubeArray = true; //we require OES_texture_cube_map_array on GLES
			m_features.robustBufferAccess = false; // TODO: there's an extension for that in GL
			m_features.vertexAttributeDouble = !IsGLES;
			m_features.multiDrawIndirect = IsGLES ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_multi_draw_indirect) : true;
			m_features.drawIndirectCount = IsGLES ? false : (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_indirect_parameters) || m_glfeatures.Version >= 460u);

			// TODO: @achal handle ARB, EXT, NVidia and AMD extensions which can be used to spoof
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
			// GL doesnt have any limit on this (???)
			m_limits.maxDrawIndirectCount = std::numeric_limits<decltype(m_limits.maxDrawIndirectCount)>::max();

			m_limits.UBOAlignment = m_glfeatures.reqUBOAlignment;
			m_limits.SSBOAlignment = m_glfeatures.reqSSBOAlignment;
			m_limits.bufferViewAlignment = m_glfeatures.reqTBOAlignment;

			m_limits.maxUBOSize = m_glfeatures.maxUBOSize;
			m_limits.maxSSBOSize = m_glfeatures.maxSSBOSize;
			m_limits.maxBufferViewSizeTexels = m_glfeatures.maxTBOSizeInTexels;
			m_limits.maxBufferSize = std::max(m_limits.maxUBOSize, m_limits.maxSSBOSize);

			m_limits.maxImageArrayLayers = m_glfeatures.MaxArrayTextureLayers;

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

			// TODO: get this from OpenCL interop, or just a GPU Device & Vendor ID table
			GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,reinterpret_cast<int32_t*>(&m_limits.maxOptimallyResidentWorkgroupInvocations));
			m_limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(m_limits.maxOptimallyResidentWorkgroupInvocations),512u);
			constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
			m_limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*m_limits.maxOptimallyResidentWorkgroupInvocations;

			// TODO: better subgroup exposal
			m_limits.subgroupSize = 0u;
			m_limits.subgroupOpsShaderStages = static_cast<asset::IShader::E_SHADER_STAGE>(0u);
			
			m_limits.nonCoherentAtomSize = 256ull;

			if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_shader_subgroup))
			{
				GLint subgroupSize = 0;
				GetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);

				GLint subgroupOpsStages = 0;
				GetIntegerv(GL_SUBGROUP_SUPPORTED_STAGES_KHR, &subgroupOpsStages);
				if (subgroupOpsStages & GL_VERTEX_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::IShader::ESS_VERTEX;
				if (subgroupOpsStages & GL_TESS_CONTROL_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::IShader::ESS_TESSELATION_CONTROL;
				if (subgroupOpsStages & GL_TESS_EVALUATION_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::IShader::ESS_TESSELATION_EVALUATION;
				if (subgroupOpsStages & GL_GEOMETRY_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::IShader::ESS_GEOMETRY;
				if (subgroupOpsStages & GL_FRAGMENT_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::IShader::ESS_FRAGMENT;
				if (subgroupOpsStages & GL_COMPUTE_SHADER_BIT)
					m_limits.subgroupOpsShaderStages |= asset::IShader::ESS_COMPUTE;
			}
		}
		
		std::ostringstream pool;
        addCommonGLSLDefines(pool,runningInRenderDoc);
		{
			std::string define;
			for (size_t j=0ull; j<std::extent<decltype(COpenGLFeatureMap::m_GLSLExtensions)>::value; ++j)
			{
				auto nativeGLExtension = COpenGLFeatureMap::m_GLSLExtensions[j];
				if (m_glfeatures.isFeatureAvailable(nativeGLExtension))
				{
					define = "NBL_IMPL_";
					define += COpenGLFeatureMap::OpenGLFeatureStrings[nativeGLExtension];
					addGLSLDefineToPool(pool,define.c_str());
				}
			}
		}
        finalizeGLSLDefinePool(std::move(pool));

		// we dont need this any more
		m_egl.call.peglMakeCurrent(m_egl.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		m_egl.call.peglDestroyContext(m_egl.display, ctx);
		m_egl.call.peglDestroySurface(m_egl.display, pbuf);
    }

	IDebugCallback* getDebugCallback()
	{
		return static_cast<IDebugCallback*>(&m_dbgCb);
	}

	bool isSwapchainSupported() const override { return true; }

	inline bool isAllowedTextureFormat(asset::E_FORMAT _fmt) const
	{
		using namespace asset;
		// opengl spec section 8.5.1
		switch (_fmt)
		{
			// formats checked as "Req. tex"
		case EF_R8_UNORM:
		case EF_R8_SNORM:
		case EF_R16_UNORM:
		case EF_R16_SNORM:
		case EF_R8G8_UNORM:
		case EF_R8G8_SNORM:
		case EF_R16G16_UNORM:
		case EF_R16G16_SNORM:
		case EF_R8G8B8_UNORM:
		case EF_R8G8B8_SNORM:
		case EF_A1R5G5B5_UNORM_PACK16:
		case EF_R8G8B8A8_SRGB:
		case EF_A8B8G8R8_UNORM_PACK32:
		case EF_A8B8G8R8_SNORM_PACK32:
		case EF_A8B8G8R8_SRGB_PACK32:
		case EF_R16_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_R16G16B16_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
		case EF_R32_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R32G32B32_SFLOAT:
		case EF_R32G32B32A32_SFLOAT:
		case EF_B10G11R11_UFLOAT_PACK32:
		case EF_E5B9G9R9_UFLOAT_PACK32:
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_A2B10G10R10_UINT_PACK32:
		case EF_R16G16B16A16_UNORM:
		case EF_R8_UINT:
		case EF_R8_SINT:
		case EF_R8G8_UINT:
		case EF_R8G8_SINT:
		case EF_R8G8B8_UINT:
		case EF_R8G8B8_SINT:
		case EF_R8G8B8A8_UNORM:
		case EF_R8G8B8A8_SNORM:
		case EF_R8G8B8A8_UINT:
		case EF_R8G8B8A8_SINT:
		case EF_B8G8R8A8_UINT:
		case EF_R16_UINT:
		case EF_R16_SINT:
		case EF_R16G16_UINT:
		case EF_R16G16_SINT:
		case EF_R16G16B16_UINT:
		case EF_R16G16B16_SINT:
		case EF_R16G16B16A16_UINT:
		case EF_R16G16B16A16_SINT:
		case EF_R32_UINT:
		case EF_R32_SINT:
		case EF_R32G32_UINT:
		case EF_R32G32_SINT:
		case EF_R32G32B32_UINT:
		case EF_R32G32B32_SINT:
		case EF_R32G32B32A32_UINT:
		case EF_R32G32B32A32_SINT:

			// depth/stencil/depth+stencil formats checked as "Req. format"
		case EF_D16_UNORM:
		case EF_X8_D24_UNORM_PACK32:
		case EF_D32_SFLOAT:
		case EF_D24_UNORM_S8_UINT:
		case EF_S8_UINT:

			// specific compressed formats
		case EF_BC6H_UFLOAT_BLOCK:
		case EF_BC6H_SFLOAT_BLOCK:
		case EF_BC7_UNORM_BLOCK:
		case EF_BC7_SRGB_BLOCK:
		case EF_ETC2_R8G8B8_UNORM_BLOCK:
		case EF_ETC2_R8G8B8_SRGB_BLOCK:
		case EF_ETC2_R8G8B8A1_UNORM_BLOCK:
		case EF_ETC2_R8G8B8A1_SRGB_BLOCK:
		case EF_ETC2_R8G8B8A8_UNORM_BLOCK:
		case EF_ETC2_R8G8B8A8_SRGB_BLOCK:
		case EF_EAC_R11_UNORM_BLOCK:
		case EF_EAC_R11_SNORM_BLOCK:
		case EF_EAC_R11G11_UNORM_BLOCK:
		case EF_EAC_R11G11_SNORM_BLOCK:
			return true;

			// astc
		case EF_ASTC_4x4_UNORM_BLOCK:
		case EF_ASTC_5x4_UNORM_BLOCK:
		case EF_ASTC_5x5_UNORM_BLOCK:
		case EF_ASTC_6x5_UNORM_BLOCK:
		case EF_ASTC_6x6_UNORM_BLOCK:
		case EF_ASTC_8x5_UNORM_BLOCK:
		case EF_ASTC_8x6_UNORM_BLOCK:
		case EF_ASTC_8x8_UNORM_BLOCK:
		case EF_ASTC_10x5_UNORM_BLOCK:
		case EF_ASTC_10x6_UNORM_BLOCK:
		case EF_ASTC_10x8_UNORM_BLOCK:
		case EF_ASTC_10x10_UNORM_BLOCK:
		case EF_ASTC_12x10_UNORM_BLOCK:
		case EF_ASTC_12x12_UNORM_BLOCK:
		case EF_ASTC_4x4_SRGB_BLOCK:
		case EF_ASTC_5x4_SRGB_BLOCK:
		case EF_ASTC_5x5_SRGB_BLOCK:
		case EF_ASTC_6x5_SRGB_BLOCK:
		case EF_ASTC_6x6_SRGB_BLOCK:
		case EF_ASTC_8x5_SRGB_BLOCK:
		case EF_ASTC_8x6_SRGB_BLOCK:
		case EF_ASTC_8x8_SRGB_BLOCK:
		case EF_ASTC_10x5_SRGB_BLOCK:
		case EF_ASTC_10x6_SRGB_BLOCK:
		case EF_ASTC_10x8_SRGB_BLOCK:
		case EF_ASTC_10x10_SRGB_BLOCK:
		case EF_ASTC_12x10_SRGB_BLOCK:
		case EF_ASTC_12x12_SRGB_BLOCK:
			return m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_texture_compression_astc_ldr);

		default: return false;
		}
	}

	inline bool isAllowedImageStoreFormat(asset::E_FORMAT _fmt) const
	{
		using namespace asset;
		switch (_fmt)
		{
		case EF_R32G32B32A32_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_B10G11R11_UFLOAT_PACK32:
		case EF_R32_SFLOAT:
		case EF_R16_SFLOAT:
		case EF_R16G16B16A16_UNORM:
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_R8G8B8A8_UNORM:
		case EF_R16G16_UNORM:
		case EF_R8G8_UNORM:
		case EF_R16_UNORM:
		case EF_R8_UNORM:
		case EF_R16G16B16A16_SNORM:
		case EF_R8G8B8A8_SNORM:
		case EF_R16G16_SNORM:
		case EF_R8G8_SNORM:
		case EF_R16_SNORM:
		case EF_R32G32B32A32_UINT:
		case EF_R16G16B16A16_UINT:
		case EF_A2B10G10R10_UINT_PACK32:
		case EF_R8G8B8A8_UINT:
		case EF_R32G32_UINT:
		case EF_R16G16_UINT:
		case EF_R8G8_UINT:
		case EF_R32_UINT:
		case EF_R16_UINT:
		case EF_R8_UINT:
		case EF_R32G32B32A32_SINT:
		case EF_R16G16B16A16_SINT:
		case EF_R8G8B8A8_SINT:
		case EF_R32G32_SINT:
		case EF_R16G16_SINT:
		case EF_R8G8_SINT:
		case EF_R32_SINT:
		case EF_R16_SINT:
		case EF_R8_SINT:
			return true;
		default: return false;
		}
	}

	inline bool isAllowedImageStoreAtomicFormat(asset::E_FORMAT format) const
	{
		switch (format)
		{
		case asset::EF_R32_SINT:
		case asset::EF_R32_UINT:
			return true;
		default:
			return false;
		}
	}

	inline bool isAllowedBufferViewFormat(asset::E_FORMAT _fmt) const
	{
		using namespace asset;
		switch (_fmt)
		{
		case EF_R8_UNORM: [[fallthrough]];
		case EF_R16_UNORM: [[fallthrough]];
		case EF_R16_SFLOAT: [[fallthrough]];
		case EF_R32_SFLOAT: [[fallthrough]];
		case EF_R8_SINT: [[fallthrough]];
		case EF_R16_SINT: [[fallthrough]];
		case EF_R32_SINT: [[fallthrough]];
		case EF_R8_UINT: [[fallthrough]];
		case EF_R16_UINT: [[fallthrough]];
		case EF_R32_UINT: [[fallthrough]];
		case EF_R8G8_UNORM: [[fallthrough]];
		case EF_R16G16_UNORM: [[fallthrough]];
		case EF_R16G16_SFLOAT: [[fallthrough]];
		case EF_R32G32_SFLOAT: [[fallthrough]];
		case EF_R8G8_SINT: [[fallthrough]];
		case EF_R16G16_SINT: [[fallthrough]];
		case EF_R32G32_SINT: [[fallthrough]];
		case EF_R8G8_UINT: [[fallthrough]];
		case EF_R16G16_UINT: [[fallthrough]];
		case EF_R32G32_UINT: [[fallthrough]];
		case EF_R32G32B32_SFLOAT: [[fallthrough]];
		case EF_R32G32B32_SINT: [[fallthrough]];
		case EF_R32G32B32_UINT: [[fallthrough]];
		case EF_R8G8B8A8_UNORM: [[fallthrough]];
		case EF_R16G16B16A16_UNORM: [[fallthrough]];
		case EF_R16G16B16A16_SFLOAT: [[fallthrough]];
		case EF_R32G32B32A32_SFLOAT: [[fallthrough]];
		case EF_R8G8B8A8_SINT: [[fallthrough]];
		case EF_R16G16B16A16_SINT: [[fallthrough]];
		case EF_R32G32B32A32_SINT: [[fallthrough]];
		case EF_R8G8B8A8_UINT: [[fallthrough]];
		case EF_R16G16B16A16_UINT: [[fallthrough]];
		case EF_R32G32B32A32_UINT:
			return true;
		default:
			return false;
		}
	}

	inline bool isAllowedVertexAttribFormat(asset::E_FORMAT _fmt) const
	{
		using namespace asset;
		switch (_fmt)
		{
			// signed/unsigned byte
		case EF_R8_UNORM:
		case EF_R8_SNORM:
		case EF_R8_UINT:
		case EF_R8_SINT:
		case EF_R8G8_UNORM:
		case EF_R8G8_SNORM:
		case EF_R8G8_UINT:
		case EF_R8G8_SINT:
		case EF_R8G8B8_UNORM:
		case EF_R8G8B8_SNORM:
		case EF_R8G8B8_UINT:
		case EF_R8G8B8_SINT:
		case EF_R8G8B8A8_UNORM:
		case EF_R8G8B8A8_SNORM:
		case EF_R8G8B8A8_UINT:
		case EF_R8G8B8A8_SINT:
		case EF_R8_USCALED:
		case EF_R8_SSCALED:
		case EF_R8G8_USCALED:
		case EF_R8G8_SSCALED:
		case EF_R8G8B8_USCALED:
		case EF_R8G8B8_SSCALED:
		case EF_R8G8B8A8_USCALED:
		case EF_R8G8B8A8_SSCALED:
			// unsigned byte BGRA (normalized only)
		case EF_B8G8R8A8_UNORM:
			// unsigned/signed short
		case EF_R16_UNORM:
		case EF_R16_SNORM:
		case EF_R16_UINT:
		case EF_R16_SINT:
		case EF_R16G16_UNORM:
		case EF_R16G16_SNORM:
		case EF_R16G16_UINT:
		case EF_R16G16_SINT:
		case EF_R16G16B16_UNORM:
		case EF_R16G16B16_SNORM:
		case EF_R16G16B16_UINT:
		case EF_R16G16B16_SINT:
		case EF_R16G16B16A16_UNORM:
		case EF_R16G16B16A16_SNORM:
		case EF_R16G16B16A16_UINT:
		case EF_R16G16B16A16_SINT:
		case EF_R16_USCALED:
		case EF_R16_SSCALED:
		case EF_R16G16_USCALED:
		case EF_R16G16_SSCALED:
		case EF_R16G16B16_USCALED:
		case EF_R16G16B16_SSCALED:
		case EF_R16G16B16A16_USCALED:
		case EF_R16G16B16A16_SSCALED:
			// unsigned/signed int
		case EF_R32_UINT:
		case EF_R32_SINT:
		case EF_R32G32_UINT:
		case EF_R32G32_SINT:
		case EF_R32G32B32_UINT:
		case EF_R32G32B32_SINT:
		case EF_R32G32B32A32_UINT:
		case EF_R32G32B32A32_SINT:
			// unsigned/signed rgb10a2 BGRA (normalized only)
		case EF_A2R10G10B10_UNORM_PACK32:
		case EF_A2R10G10B10_SNORM_PACK32:
			// unsigned/signed rgb10a2
		case EF_A2B10G10R10_UNORM_PACK32:
		case EF_A2B10G10R10_SNORM_PACK32:
		case EF_A2B10G10R10_UINT_PACK32:
		case EF_A2B10G10R10_SINT_PACK32:
		case EF_A2B10G10R10_SSCALED_PACK32:
		case EF_A2B10G10R10_USCALED_PACK32:
			// GL_UNSIGNED_INT_10F_11F_11F_REV
		case EF_B10G11R11_UFLOAT_PACK32:
			// half float
		case EF_R16_SFLOAT:
		case EF_R16G16_SFLOAT:
		case EF_R16G16B16_SFLOAT:
		case EF_R16G16B16A16_SFLOAT:
			// float
		case EF_R32_SFLOAT:
		case EF_R32G32_SFLOAT:
		case EF_R32G32B32_SFLOAT:
		case EF_R32G32B32A32_SFLOAT:
			// double
		case EF_R64_SFLOAT:
		case EF_R64G64_SFLOAT:
		case EF_R64G64B64_SFLOAT:
		case EF_R64G64B64A64_SFLOAT:
			return true;
		default: return false;
		}
	}

	SFormatProperties getFormatProperties(asset::E_FORMAT format) const override
	{
		SFormatProperties result = {};
		// EFF_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000200,
		// EFF_BLIT_SRC_BIT = 0x00000400,
		// EFF_BLIT_DST_BIT = 0x00000800,
		// EFF_SAMPLED_IMAGE_FILTER_LINEAR_BIT = 0x00001000,
		// EFF_TRANSFER_SRC_BIT = 0x00004000,
		// EFF_TRANSFER_DST_BIT = 0x00008000,
		// EFF_MIDPOINT_CHROMA_SAMPLES_BIT = 0x00020000,
		// EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT = 0x00040000,
		// EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT = 0x00080000,
		// EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT = 0x00100000,
		// EFF_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT = 0x00200000,
		// EFF_DISJOINT_BIT = 0x00400000,
		// EFF_COSITED_CHROMA_SAMPLES_BIT = 0x00800000,
		// EFF_SAMPLED_IMAGE_FILTER_MINMAX_BIT = 0x00010000,
		// EFF_SAMPLED_IMAGE_FILTER_CUBIC_BIT_IMG = 0x00002000,
		// EFF_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT = 0x20000000,
		// EFF_FRAGMENT_DENSITY_MAP_BIT = 0x01000000,
		// EFF_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT = 0x40000000,
		
		// result.optimalTilingFeatures = ;
		// result.bufferFeatures = ;

		if (isAllowedTextureFormat(format))
			result.optimalTilingFeatures |= asset::EFF_SAMPLED_IMAGE_BIT;
		if (isAllowedImageStoreFormat(format))
		{
			result.optimalTilingFeatures |= asset::EFF_STORAGE_IMAGE_BIT;
			if (isAllowedImageStoreAtomicFormat(format))
				result.optimalTilingFeatures |= asset::EFF_STORAGE_IMAGE_ATOMIC_BIT;
		}
		if (isAllowedBufferViewFormat(format))
		{
			result.optimalTilingFeatures |= asset::EFF_UNIFORM_TEXEL_BUFFER_BIT;
			if (isAllowedImageStoreAtomicFormat(format))
			{
				result.optimalTilingFeatures |= asset::EFF_STORAGE_TEXEL_BUFFER_BIT;
				result.optimalTilingFeatures |= asset::EFF_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
			}
		}
		if (isAllowedVertexAttribFormat(format))
			result.optimalTilingFeatures |= asset::EFF_VERTEX_BUFFER_BIT;

		if (!asset::isBlockCompressionFormat(format))
			result.optimalTilingFeatures |= asset::EFF_COLOR_ATTACHMENT_BIT;

		if (asset::isFloatingPointFormat(format))
			result.optimalTilingFeatures |= asset::EFF_COLOR_ATTACHMENT_BLEND_BIT;



		_NBL_TODO();



		result.linearTilingFeatures = result.optimalTilingFeatures;

		return result;
	}

protected:
	virtual ~IOpenGL_PhysicalDeviceBase()
	{
		m_egl.deinitialize();
	}

	IAPIConnection* m_api; // dumb pointer to avoid circ ref
	renderdoc_api_t* m_rdoc_api;
    egl::CEGL m_egl;
	COpenGLDebugCallback m_dbgCb;

    EGLConfig m_config;
    EGLint m_gl_major, m_gl_minor;

	COpenGLFeatureMap m_glfeatures;
};

}

#endif
