#ifndef __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/renderdoc.h"
#include <regex>

#include "nbl/video/COpenGLFeatureMap.h"
#include "nbl/video/SOpenGLContextLocalCache.h"

#include "nbl/video/CEGL.h"


#include "nbl/video/debug/COpenGLDebugCallback.h"
#ifndef EGL_CONTEXT_OPENGL_NO_ERROR_KHR
#	define EGL_CONTEXT_OPENGL_NO_ERROR_KHR 0x31B3
#endif

namespace nbl::video
{

class IOpenGLPhysicalDeviceBase : public IPhysicalDevice
{
	public:
		static inline constexpr uint32_t MaxQueues = 8u;

		IOpenGLPhysicalDeviceBase(
			core::smart_refctd_ptr<system::ISystem>&& s,
			core::smart_refctd_ptr<asset::IGLSLCompiler>&& glslc,
			egl::CEGL&& _egl
		) : IPhysicalDevice(std::move(s),std::move(glslc)), m_egl(std::move(_egl))
		{}
	
		const egl::CEGL& getInternalObject() const {return m_egl;}
		
	protected:
		virtual ~IOpenGLPhysicalDeviceBase()
		{
			m_egl.deinitialize();
		}

		egl::CEGL m_egl;
};

template <typename LogicalDeviceType>
class IOpenGL_PhysicalDeviceBase : public IOpenGLPhysicalDeviceBase
{
	using function_table_t = typename LogicalDeviceType::FunctionTableType;
	static inline constexpr EGLint EGL_API_TYPE = function_table_t::EGL_API_TYPE;
	static inline constexpr bool IsGLES = (EGL_API_TYPE == EGL_OPENGL_ES_API);

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
		: IOpenGLPhysicalDeviceBase(std::move(s),core::make_smart_refctd_ptr<asset::IGLSLCompiler>(s.get()),std::move(_egl)), m_api(api), m_rdoc_api(rdoc), m_dbgCb(std::move(_dbgCb)), m_config(_config), m_gl_major(_major), m_gl_minor(_minor)
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

		auto GetError = reinterpret_cast<decltype(glGetError)*>(m_egl.call.peglGetProcAddress("glGetError"));
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

		// Some usefull stringy searchy functions
		auto hasInString = [](const std::string& str, const std::string& toFind, bool matchWholeWord = false, bool matchCase = false) -> bool
		{
			std::string base = str;
			std::string to_find = toFind;
			if(!matchCase)
			{
				std::transform(base.begin(), base.end(), base.begin(), [](unsigned char c){ return std::tolower(c); });
				std::transform(to_find.begin(), to_find.end(), to_find.begin(), [](unsigned char c){ return std::tolower(c); });
			}

			if(matchWholeWord) // Because we don't want to detect "armadillo" as an "arm" driver, I couldn't come up with a better example
			{
				std::regex r("\\b" + toFind + "\\b"); // the pattern \b matches a word boundary
				std::smatch m;
				return std::regex_search(base, m, r);
			}
			else
			{
				return (base.find(toFind) != base.npos);
			}
		};

		// Some tests to ensure "matchWholeWord" works as expected
		assert(hasInString("RADV/ACO FIJI", "radv", true));
		assert(hasInString("Intel(R)", "intel", true));
		assert(hasInString("ATI Technologies Inc.", "ati technologies", true));

		// initialize features
		const char* vendor = reinterpret_cast<const char*>(GetString(GL_VENDOR));
		const char* renderer = reinterpret_cast<const char*>(GetString(GL_RENDERER));
		const char* ogl_ver_str = reinterpret_cast<const char*>(GetString(GL_VERSION));

		// Detecting DriverID from vendor, renderer and gl_version:
		// the logic comes from exhaustive search and matching between OpenGL and Vulkan GPUInfo website
		// https://vulkan.gpuinfo.org/displaycoreproperty.php?core=1.2&name=driverID&platform=all
		if(hasInString(vendor, "mesa", true) || hasInString(renderer, "mesa", true) || hasInString(ogl_ver_str, "mesa", true))
		{
			// Mesa Driver
			if(hasInString(vendor, "intel", true) || hasInString(renderer, "intel", true))
				m_properties.driverID = E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA;
			else if(hasInString(renderer, "radv", true))
				m_properties.driverID = E_DRIVER_ID::EDI_MESA_RADV;
			else if(hasInString(renderer, "llvmpipe", true))
				m_properties.driverID = E_DRIVER_ID::EDI_MESA_LLVMPIPE;
			else if(hasInString(renderer, "amd", true) || hasInString(renderer, "radeon", true) || hasInString(vendor, "ati technologies", true))
				m_properties.driverID = E_DRIVER_ID::EDI_AMD_OPEN_SOURCE;
		}
		else if(hasInString(vendor, "intel", true) || hasInString(renderer, "intel", true))
			m_properties.driverID = E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS;
		else if(hasInString(vendor, "ati technologies", true) || hasInString(vendor, "amd", true) || hasInString(renderer, "amd", true))
			m_properties.driverID = E_DRIVER_ID::EDI_AMD_PROPRIETARY;
		else if(hasInString(vendor, "nvidia", true)) // easiest to detect :D
			m_properties.driverID = E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY;
		else 
			m_properties.driverID = E_DRIVER_ID::EDI_UNKNOWN;

		m_glfeatures.isIntelGPU = (m_properties.driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || m_properties.driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);

		// Heuristic to detect Physical Device Type until we have something better:
		if(m_properties.driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || m_properties.driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS)
			m_properties.deviceType = E_TYPE::ET_INTEGRATED_GPU;
		else if(m_properties.driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || m_properties.driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY)
			m_properties.deviceType = E_TYPE::ET_DISCRETE_GPU;
		else if(m_properties.driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY)
			m_properties.deviceType = E_TYPE::ET_DISCRETE_GPU;
		else if(hasInString(renderer, "virgl", true))
			m_properties.deviceType = E_TYPE::ET_VIRTUAL_GPU;
		else
			m_properties.deviceType = E_TYPE::ET_UNKNOWN;
		
		// Query VRAM Size 
		GetError();
		size_t VRAMSize = 0u;
		GLint tmp[4] = {0,0,0,0};
		switch (m_properties.deviceType)
		{
		case E_DRIVER_ID::EDI_AMD_PROPRIETARY:
			GetIntegerv(0x87FC,tmp); //TEXTURE_FREE_MEMORY_ATI, only textures
			VRAMSize = size_t(tmp[0])*1024ull;
			assert(VRAMSize > 0u);
			break;
		case EDI_INTEL_OPEN_SOURCE_MESA: [[fallthrough]];
		case EDI_MESA_RADV: [[fallthrough]];
		case EDI_MESA_LLVMPIPE: [[fallthrough]];
		case EDI_AMD_OPEN_SOURCE: [[fallthrough]];
			// TODO https://www.khronos.org/registry/OpenGL/extensions/MESA/GLX_MESA_query_renderer.txt
		default: // other vendors sometimes implement the NVX extension
			GetIntegerv(0x9047,tmp); // dedicated as per https://www.khronos.org/registry/OpenGL/extensions/NVX/NVX_gpu_memory_info.txt
			VRAMSize = size_t(tmp[0])*1024ull;
			assert(VRAMSize > 0u);
			break;
		}
		if (GetError()!=GL_NO_ERROR)
			VRAMSize = 2047u;

		// Spoof Memory Types and Heaps
		m_memoryProperties = {};
		m_memoryProperties.memoryHeapCount = 3u;
		m_memoryProperties.memoryHeaps[0u].flags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS>(IDeviceMemoryAllocation::EMHF_DEVICE_LOCAL_BIT);
		m_memoryProperties.memoryHeaps[0u].size = VRAMSize; // VRAM SIZE
		m_memoryProperties.memoryHeaps[1u].flags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS>(0u);
		m_memoryProperties.memoryHeaps[1u].size = size_t(0.7f * float(m_system->getSystemInfo().totalMemory)); // 70% System memory
		m_memoryProperties.memoryHeaps[2u].flags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_HEAP_FLAGS>(IDeviceMemoryAllocation::EMHF_DEVICE_LOCAL_BIT);
		m_memoryProperties.memoryHeaps[2u].size = 256u * 1024u * 1024u; // 256MB

		m_memoryProperties.memoryTypeCount = 14u;
		m_memoryProperties.memoryTypes[0u].heapIndex = 0u;
		m_memoryProperties.memoryTypes[0u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT);

		m_memoryProperties.memoryTypes[1u].heapIndex = 2u;
		m_memoryProperties.memoryTypes[1u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT;
		m_memoryProperties.memoryTypes[2u].heapIndex = 2u;
		m_memoryProperties.memoryTypes[2u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT;
		m_memoryProperties.memoryTypes[3u].heapIndex = 2u;
		m_memoryProperties.memoryTypes[3u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT;
		
		m_memoryProperties.memoryTypes[4u].heapIndex = 2u;
		m_memoryProperties.memoryTypes[4u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT;
		m_memoryProperties.memoryTypes[5u].heapIndex = 2u;
		m_memoryProperties.memoryTypes[5u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT;
		m_memoryProperties.memoryTypes[6u].heapIndex = 2u;
		m_memoryProperties.memoryTypes[6u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_DEVICE_LOCAL_BIT) | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT;
		
		m_memoryProperties.memoryTypes[7u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[7u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(0u);
		
		m_memoryProperties.memoryTypes[8u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[8u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT ;
		m_memoryProperties.memoryTypes[9u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[9u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT;
		m_memoryProperties.memoryTypes[10u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[10u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT) | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT;
		
		m_memoryProperties.memoryTypes[11u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[11u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT ;
		m_memoryProperties.memoryTypes[12u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[12u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT) | IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT;
		m_memoryProperties.memoryTypes[13u].heapIndex = 1u;
		m_memoryProperties.memoryTypes[13u].propertyFlags = core::bitflag<IDeviceMemoryAllocation::E_MEMORY_PROPERTY_FLAGS>(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT) | IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT;

		const std::regex version_re("([1-9]\\.[0-9])");
		std::cmatch re_match;

		float ogl_ver = 0.f;
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
		// TODO: GLES has a problem with reporting this
		GetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, &m_glfeatures.reqTBOAlignment);
		if (!core::is_alignment(m_glfeatures.reqTBOAlignment))
			m_glfeatures.reqTBOAlignment = 16u;
		//assert(core::is_alignment(m_glfeatures.reqTBOAlignment)); 
		//m_glfeatures.reqSSBOAlignment = 64u; // uncomment to emulate chromebook

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
		
		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_ARB_query_buffer_object))
		{
			m_features.allowCommandBufferQueryCopies = true;
		}
		m_features.inheritedQueries = true; // We emulate secondary command buffers so enable by default

		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_texture_filter_anisotropic))
		{
			GetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &num);
			m_glfeatures.MaxAnisotropy = static_cast<uint8_t>(num);
			m_features.samplerAnisotropy = true;
			m_properties.limits.maxSamplerAnisotropyLog2 = std::log2((float)m_glfeatures.MaxAnisotropy);
		}
		else m_glfeatures.MaxAnisotropy = 0u;

		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_ARB_geometry_shader4))
		{
			GetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &num);
			m_glfeatures.MaxGeometryVerticesOut = static_cast<uint32_t>(num);
			m_features.geometryShader = true;
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

			if(m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_fragment_shader_interlock))
			{
				// Can't check individualy (???)
				m_features.fragmentShaderPixelInterlock = true;
				m_features.fragmentShaderSampleInterlock = true;
				m_features.fragmentShaderShadingRateInterlock = true;
			}
		}

		// physical device limits
		{
			// TODO: get deviceUUID
			// TODO: get deviceType
			int majorVer = 0;
			int minorVer = 0;
			GetIntegerv(GL_MAJOR_VERSION, &majorVer);
			GetIntegerv(GL_MINOR_VERSION, &minorVer);
			m_properties.apiVersion.major = majorVer;
			m_properties.apiVersion.minor = minorVer;
			m_properties.apiVersion.patch = 0u;
			
			// GL doesnt have any limit on this (???)
			m_properties.limits.maxDrawIndirectCount = std::numeric_limits<decltype(m_properties.limits.maxDrawIndirectCount)>::max();

			m_properties.limits.UBOAlignment = m_glfeatures.reqUBOAlignment;
			m_properties.limits.SSBOAlignment = m_glfeatures.reqSSBOAlignment;
			m_properties.limits.bufferViewAlignment = m_glfeatures.reqTBOAlignment;

			m_properties.limits.maxUBOSize = m_glfeatures.maxUBOSize;
			m_properties.limits.maxSSBOSize = m_glfeatures.maxSSBOSize;
			m_properties.limits.maxBufferViewSizeTexels = m_glfeatures.maxTBOSizeInTexels;
			m_properties.limits.maxBufferSize = std::max(m_properties.limits.maxUBOSize, m_properties.limits.maxSSBOSize);

			m_properties.limits.maxImageArrayLayers = m_glfeatures.MaxArrayTextureLayers;
			m_properties.limits.timestampPeriodInNanoSeconds = 1.0f;

			GLint max_ssbos[5];
			GetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, max_ssbos + 0);
			GetIntegerv(GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS, max_ssbos + 1);
			GetIntegerv(GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, max_ssbos + 2);
			GetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, max_ssbos + 3);
			GetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, max_ssbos + 4);
			uint32_t maxSSBOsPerStage = static_cast<uint32_t>(*std::min_element(max_ssbos, max_ssbos + 5));

			m_properties.limits.maxPerStageDescriptorSSBOs = maxSSBOsPerStage;

			m_properties.limits.maxDescriptorSetSSBOs = m_glfeatures.maxSSBOBindings;
			m_properties.limits.maxDescriptorSetUBOs = m_glfeatures.maxUBOBindings;
			m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs = SOpenGLState::MaxDynamicOffsetSSBOs;
			m_properties.limits.maxDescriptorSetDynamicOffsetUBOs = SOpenGLState::MaxDynamicOffsetUBOs;
			m_properties.limits.maxDescriptorSetImages = m_glfeatures.maxTextureBindings;
			m_properties.limits.maxDescriptorSetStorageImages = m_glfeatures.maxImageBindings;
			GetInteger64v(GL_MAX_TEXTURE_SIZE, reinterpret_cast<GLint64*>(&m_properties.limits.maxTextureSize));

			GetFloatv(GL_POINT_SIZE_RANGE, m_properties.limits.pointSizeRange);
			GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, m_properties.limits.lineWidthRange);

			GLint maxViewportExtent[2]{0,0};
			GetIntegerv(GL_MAX_VIEWPORT_DIMS, maxViewportExtent);

			GLint maxViewports = 16;
			GetIntegerv(GL_MAX_VIEWPORTS, &maxViewports);

			m_properties.limits.maxViewports = maxViewports;

			m_properties.limits.maxViewportDims[0] = maxViewportExtent[0];
			m_properties.limits.maxViewportDims[1] = maxViewportExtent[1];

			GLint maxComputeSharedMemorySize = 0;
			GetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &maxComputeSharedMemorySize);
			m_properties.limits.maxComputeSharedMemorySize = maxComputeSharedMemorySize;

			m_properties.limits.maxWorkgroupSize[0] = m_glfeatures.MaxComputeWGSize[0];
			m_properties.limits.maxWorkgroupSize[1] = m_glfeatures.MaxComputeWGSize[1];
			m_properties.limits.maxWorkgroupSize[2] = m_glfeatures.MaxComputeWGSize[2];

			// TODO: get this from OpenCL interop, or just a GPU Device & Vendor ID table
			GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,reinterpret_cast<int32_t*>(&m_properties.limits.maxOptimallyResidentWorkgroupInvocations));
			m_properties.limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(m_properties.limits.maxOptimallyResidentWorkgroupInvocations),512u);
			constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
			m_properties.limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*m_properties.limits.maxOptimallyResidentWorkgroupInvocations;

			// TODO: better subgroup exposal
			m_properties.limits.subgroupSize = 0u;
			m_properties.limits.subgroupOpsShaderStages = static_cast<asset::IShader::E_SHADER_STAGE>(0u);
			
			m_properties.limits.nonCoherentAtomSize = 256ull;

			m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_6;

			if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_shader_subgroup))
			{
				GLint subgroupSize = 0;
				GetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);

				GLint subgroupOpsStages = 0;
				GetIntegerv(GL_SUBGROUP_SUPPORTED_STAGES_KHR, &subgroupOpsStages);
				if (subgroupOpsStages & GL_VERTEX_SHADER_BIT)
					m_properties.limits.subgroupOpsShaderStages |= asset::IShader::ESS_VERTEX;
				if (subgroupOpsStages & GL_TESS_CONTROL_SHADER_BIT)
					m_properties.limits.subgroupOpsShaderStages |= asset::IShader::ESS_TESSELATION_CONTROL;
				if (subgroupOpsStages & GL_TESS_EVALUATION_SHADER_BIT)
					m_properties.limits.subgroupOpsShaderStages |= asset::IShader::ESS_TESSELATION_EVALUATION;
				if (subgroupOpsStages & GL_GEOMETRY_SHADER_BIT)
					m_properties.limits.subgroupOpsShaderStages |= asset::IShader::ESS_GEOMETRY;
				if (subgroupOpsStages & GL_FRAGMENT_SHADER_BIT)
					m_properties.limits.subgroupOpsShaderStages |= asset::IShader::ESS_FRAGMENT;
				if (subgroupOpsStages & GL_COMPUTE_SHADER_BIT)
					m_properties.limits.subgroupOpsShaderStages |= asset::IShader::ESS_COMPUTE;
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

		// we dont need this context any more
		m_egl.call.peglMakeCurrent(m_egl.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		m_egl.call.peglDestroyContext(m_egl.display, ctx);
		m_egl.call.peglDestroySurface(m_egl.display, pbuf);
	}

	IDebugCallback* getDebugCallback()
	{
		return static_cast<IDebugCallback*>(&m_dbgCb);
	}

	bool isSwapchainSupported() const override { return true; }

protected:
	IAPIConnection* m_api; // dumb pointer to avoid circ ref
	renderdoc_api_t* m_rdoc_api;
	COpenGLDebugCallback m_dbgCb;

	EGLConfig m_config;
	EGLint m_gl_major, m_gl_minor;

	COpenGLFeatureMap m_glfeatures;
};

}

#endif
