#ifndef __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__
#define __NBL_I_OPENGL_PHYSICAL_DEVICE_BASE_H_INCLUDED__

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/utilities/renderdoc.h"
#include <regex>

#include "nbl/video/COpenGLFeatureMap.h"
#include "nbl/video/SOpenGLContextLocalCache.h"

#include "nbl/video/CEGL.h"
#include "nbl/core/xxHash256.h"

#include "nbl/video/debug/COpenGLDebugCallback.h"
#ifndef EGL_CONTEXT_OPENGL_NO_ERROR_KHR
#	define EGL_CONTEXT_OPENGL_NO_ERROR_KHR 0x31B3
#endif

#include "nbl/asset/ICPUMeshBuffer.h" // for MAX_PUSH_CONSTANT_BYTESIZE

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
		{
#if defined(_NBL_PLATFORM_WINDOWS_)
			m_properties.driverID = E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS;
#else
			m_properties.driverID = E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA;
			_NBL_DEBUG_BREAK_IF(true); // Should've captured it in the previous if, need to update logic
#endif
		}
		else if(hasInString(vendor, "ati technologies", true) || hasInString(vendor, "amd", true) || hasInString(renderer, "amd", true))
			m_properties.driverID = E_DRIVER_ID::EDI_AMD_PROPRIETARY;
		else if(hasInString(vendor, "nvidia", true)) // easiest to detect :D
			m_properties.driverID = E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY;
		else 
			m_properties.driverID = E_DRIVER_ID::EDI_UNKNOWN;

		memset(m_properties.deviceUUID, 0, VK_UUID_SIZE);
		strcpy(m_properties.driverInfo, renderer);
		// driverName
		switch (m_properties.driverID)
		{
		case E_DRIVER_ID::EDI_AMD_PROPRIETARY: strcpy(m_properties.driverName, "AMD proprietary driver"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_AMD_OPEN_SOURCE: strcpy(m_properties.driverName, "AMD open-source driver"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_MESA_RADV: strcpy(m_properties.driverName, "radv"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY: strcpy(m_properties.driverName, "NVIDIA"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS: strcpy(m_properties.driverName, "Intel Corporation"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA: strcpy(m_properties.driverName, "Intel open-source Mesa driver"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_IMAGINATION_PROPRIETARY: strcpy(m_properties.driverName, "Imagination Proprietary driver"); break;
		case E_DRIVER_ID::EDI_QUALCOMM_PROPRIETARY: strcpy(m_properties.driverName, "Qualcomm Proprietary driver"); break;
		case E_DRIVER_ID::EDI_ARM_PROPRIETARY: strcpy(m_properties.driverName, "ARM Proprietary driver"); break;
		case E_DRIVER_ID::EDI_GOOGLE_SWIFTSHADER: strcpy(m_properties.driverName, "SwiftShader driver"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_GGP_PROPRIETARY: strcpy(m_properties.driverName, "GGP Proprietary driver"); break;
		case E_DRIVER_ID::EDI_BROADCOM_PROPRIETARY: strcpy(m_properties.driverName, "BROADCOM Proprietary driver"); break;
		case E_DRIVER_ID::EDI_MESA_LLVMPIPE: strcpy(m_properties.driverName, "llvmpipe"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_MOLTENVK: strcpy(m_properties.driverName, "MoltenVk Driver"); break;
		case E_DRIVER_ID::EDI_COREAVI_PROPRIETARY: strcpy(m_properties.driverName, "COREAVI Proprietary driver"); break;
		case E_DRIVER_ID::EDI_JUICE_PROPRIETARY: strcpy(m_properties.driverName, "JUICE Proprietary driver"); break;
		case E_DRIVER_ID::EDI_VERISILICON_PROPRIETARY: strcpy(m_properties.driverName, "VERISILICON Proprietary driver"); break;
		case E_DRIVER_ID::EDI_MESA_TURNIP: strcpy(m_properties.driverName, "turnip Mesa driver"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_MESA_V3DV: strcpy(m_properties.driverName, "V3DV Mesa driver"); break;
		case E_DRIVER_ID::EDI_MESA_PANVK: strcpy(m_properties.driverName, "PANVK Mesa driver"); break;
		case E_DRIVER_ID::EDI_SAMSUNG_PROPRIETARY: strcpy(m_properties.driverName, "Samsung Driver"); break;
		case E_DRIVER_ID::EDI_MESA_VENUS: strcpy(m_properties.driverName, "venus"); break; // from vulkan.gpuinfo.org
		case E_DRIVER_ID::EDI_UNKNOWN:
		default: strcpy(m_properties.driverName, "UNKNOWN"); break;
		}

		bool isIntelGPU = (m_properties.driverID == E_DRIVER_ID::EDI_INTEL_OPEN_SOURCE_MESA || m_properties.driverID == E_DRIVER_ID::EDI_INTEL_PROPRIETARY_WINDOWS);
		bool isAMDGPU = (m_properties.driverID == E_DRIVER_ID::EDI_AMD_OPEN_SOURCE || m_properties.driverID == E_DRIVER_ID::EDI_AMD_PROPRIETARY);
		bool isNVIDIAGPU = (m_properties.driverID == E_DRIVER_ID::EDI_NVIDIA_PROPRIETARY);

		// conformanceVersion
		if(isIntelGPU)
			m_properties.conformanceVersion = {4u, 4u, 0u, 0u};
		else if(isAMDGPU)
			m_properties.conformanceVersion = {3u, 3u, 0u, 0u};
		else if(isNVIDIAGPU)
			m_properties.conformanceVersion = {4u, 4u, 0u, 0u};
		else
			m_properties.conformanceVersion = {3u, 1u, 0u, 0u};

		m_glfeatures.isIntelGPU = isIntelGPU;

		m_properties.driverVersion = 0u;
		m_properties.vendorID = ~0u;
		m_properties.deviceID = 0u;
		strcpy(m_properties.deviceName, renderer);
		uint64_t deviceNameHash[4] = {};
		static_assert(VK_MAX_PHYSICAL_DEVICE_NAME_SIZE >= sizeof(uint64_t)*4);
		core::XXHash_256(m_properties.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE, deviceNameHash);
		memcpy(m_properties.pipelineCacheUUID, &deviceNameHash, sizeof(uint64_t)*4);
		
		memset(m_properties.driverUUID, 0, VK_UUID_SIZE);
		memset(m_properties.deviceLUID, 0, VK_LUID_SIZE);
		m_properties.deviceNodeMask = 0x00000001;
		m_properties.deviceLUIDValid = false;

		// Heuristic to detect Physical Device Type until we have something better:
		if(isIntelGPU)
			m_properties.deviceType = E_TYPE::ET_INTEGRATED_GPU;
		else if(isAMDGPU || isNVIDIAGPU)
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

		GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, reinterpret_cast<GLint*>(&m_glfeatures.maxUBOBindings));
		GetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, reinterpret_cast<GLint*>(&m_glfeatures.maxSSBOBindings));
		GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_glfeatures.maxTextureBindings));
		GetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_glfeatures.maxTextureBindingsCompute));
		GetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, reinterpret_cast<GLint*>(&m_glfeatures.maxImageBindings));

		GLuint maxElementIndex = 0u;
		GetIntegerv(GL_MAX_ELEMENT_INDEX, reinterpret_cast<GLint*>(&maxElementIndex));
		
		m_features.robustBufferAccess = false; // TODO: there's an extension for that in GL
		m_features.fullDrawIndexUint32 = (maxElementIndex == 0xffff'ffff);
		m_features.imageCubeArray = true; //we require OES_texture_cube_map_array on GLES
		m_features.independentBlend = IsGLES 
			? (m_glfeatures.Version >= 320u || m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_OES_draw_buffers_indexed) || m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_draw_buffers_indexed))
			: true;

		if (!IsGLES || m_glfeatures.Version >= 320u)
		{
			#define GLENUM_WITH_SUFFIX(X) X
			#include "nbl/video/GL/limit_queries/tessellation_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_OES_tessellation_shader))
		{
			#define GLENUM_WITH_SUFFIX(X) X##_OES
			#include "nbl/video/GL/limit_queries/tessellation_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_tessellation_shader))
		{
			#define GLENUM_WITH_SUFFIX(X) X##_EXT
			#include "nbl/video/GL/limit_queries/tessellation_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		// else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_INTEL_tessellation_shader))
		{
			//!! no _INTEL suffix
			#define GLENUM_WITH_SUFFIX(X) X##_INTEL
			// #include "src/nbl/video/GL/limit_queries/tessellation_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		
		if (!IsGLES || m_glfeatures.Version >= 320u)
		{
			#define GLENUM_WITH_SUFFIX(X) X
			#include "nbl/video/GL/limit_queries/geometry_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_OES_geometry_shader))
		{
			#define GLENUM_WITH_SUFFIX(X) X##_OES
			#include "nbl/video/GL/limit_queries/geometry_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_geometry_shader))
		{
			#define GLENUM_WITH_SUFFIX(X) X##_EXT
			#include "nbl/video/GL/limit_queries/geometry_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}
		// else if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_INTEL_geometry_shader))
		{
			//!! no _INTEL suffix
			#define GLENUM_WITH_SUFFIX(X) X##_INTEL
			// #include "nbl/video/GL/limit_queries/geometry_shader.h"
			#undef GLENUM_WITH_SUFFIX
		}

		m_features.logicOp = !IsGLES;
		m_features.multiDrawIndirect = IsGLES ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_multi_draw_indirect) : true;

		m_features.drawIndirectFirstInstance = (IsGLES)
			? (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_multi_draw_indirect) || m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_base_instance)) 
			: true;

		m_features.depthClamp = m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_depth_clamp);
		m_features.depthBiasClamp = (IsGLES) ? m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_depth_clamp) : true;

		m_features.fillModeNonSolid = (IsGLES) ? m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_NV_polygon_mode) : true;
		m_features.depthBounds = m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_depth_bounds_test);
		m_features.wideLines = true;
		m_features.largePoints = true;
		m_features.alphaToOne = (IsGLES) ? m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_multisample_compatibility) : true; // GLES?
		
		m_features.multiViewport = IsGLES 
			? (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_OES_viewport_array) || m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_NV_viewport_array))
			: true;

		if (m_glfeatures.Version >= 460u || m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_texture_filter_anisotropic))
		{
			GLint maxAnisotropy = 0;
			GetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAnisotropy);
			if(maxAnisotropy)
			{
				m_features.samplerAnisotropy = true;
				m_properties.limits.maxSamplerAnisotropyLog2 = core::findMSB(static_cast<uint32_t>(maxAnisotropy));
			}
		}
		
		m_features.shaderStorageImageMultisample = true; // true in our minimum supported GL and GLES

		if constexpr (IsGLES)
		{
			if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_clip_cull_distance))
			{
				m_features.shaderClipDistance = true;
				m_features.shaderCullDistance = true;
				GetIntegerv(GL_MAX_CLIP_DISTANCES_EXT, reinterpret_cast<GLint*>(&m_properties.limits.maxClipDistances));
				GetIntegerv(GL_MAX_CULL_DISTANCES_EXT, reinterpret_cast<GLint*>(&m_properties.limits.maxCullDistances));
				GetIntegerv(GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES_EXT, reinterpret_cast<GLint*>(&m_properties.limits.maxCombinedClipAndCullDistances));
			}
		}
		else
		{
			m_features.shaderClipDistance = true;
			GetIntegerv(GL_MAX_CLIP_DISTANCES, reinterpret_cast<GLint*>(&m_properties.limits.maxClipDistances));
			if (m_glfeatures.Version >= 460u || m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_ARB_cull_distance))
			{
				m_features.shaderCullDistance = true;
				GetIntegerv(GL_MAX_CULL_DISTANCES, reinterpret_cast<GLint*>(&m_properties.limits.maxCullDistances));
				GetIntegerv(GL_MAX_COMBINED_CLIP_AND_CULL_DISTANCES, reinterpret_cast<GLint*>(&m_properties.limits.maxCombinedClipAndCullDistances));
			}
		}
		
		m_features.vertexAttributeDouble = !IsGLES;
		m_features.inheritedQueries = true; // We emulate secondary command buffers so enable by default
		m_features.shaderDrawParameters = IsGLES ? false : (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_shader_draw_parameters) || m_glfeatures.Version >= 460u);
		m_features.samplerMirrorClampToEdge = (IsGLES) ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_texture_mirror_clamp_to_edge) : true;
		m_features.drawIndirectCount = IsGLES ? false : (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_indirect_parameters) || m_glfeatures.Version >= 460u);
			
		m_features.samplerFilterMinmax = false; // no such sampler in GL
		m_features.bufferDeviceAddress = false; // no such capability in GL
		m_features.subgroupSizeControl = false;
		m_features.computeFullSubgroups = false;

		if(m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_fragment_shader_interlock))
		{
			// Can't check individualy (???)
			m_features.fragmentShaderPixelInterlock = true;
			m_features.fragmentShaderSampleInterlock = true;
			m_features.fragmentShaderShadingRateInterlock = true;
		}

		if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_texture_lod_bias))
			GetFloatv(GL_MAX_TEXTURE_LOD_BIAS_EXT, &m_glfeatures.MaxTextureLODBias);

		GLint num = 0;
		GetIntegerv(GL_MAX_DRAW_BUFFERS, &num);
		m_glfeatures.MaxMultipleRenderTargets = static_cast<uint8_t>(num);

		// TODO: move this to IPhysicalDevice::SFeatures
		const bool runningInRenderDoc = (m_rdoc_api != nullptr);
		m_glfeatures.runningInRenderDoc = runningInRenderDoc;

		// physical device limits
		{
			int majorVer = 0;
			int minorVer = 0;
			GetIntegerv(GL_MAJOR_VERSION, &majorVer);
			GetIntegerv(GL_MINOR_VERSION, &minorVer);
			m_properties.apiVersion.major = majorVer;
			m_properties.apiVersion.minor = minorVer;
			m_properties.apiVersion.patch = 0u;
			
			/* Vulkan 1.0 Core  */
			GLint64 maxTextureSize = 0u; // 1D + 2D
			GLint64 max3DTextureSize = 0u; // 1D + 2D
			GLint64 maxCubeMapTextureSize = 0u;
			GetInteger64v(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
			GetInteger64v(GL_MAX_3D_TEXTURE_SIZE, &max3DTextureSize);
			GetInteger64v(GL_MAX_CUBE_MAP_TEXTURE_SIZE, &maxCubeMapTextureSize);
			m_properties.limits.maxImageDimension1D = maxTextureSize;
			m_properties.limits.maxImageDimension2D	= maxTextureSize;
			m_properties.limits.maxImageDimension3D	= max3DTextureSize;
			m_properties.limits.maxImageDimensionCube =	maxCubeMapTextureSize;
			GetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, reinterpret_cast<GLint*>(&m_properties.limits.maxImageArrayLayers));
			GetInteger64v(GL_MAX_TEXTURE_BUFFER_SIZE, reinterpret_cast<GLint64*>(&m_properties.limits.maxBufferViewTexels));
			GetInteger64v(GL_MAX_UNIFORM_BLOCK_SIZE, reinterpret_cast<GLint64*>(&m_properties.limits.maxUBOSize));
			GetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, reinterpret_cast<GLint64*>(&m_properties.limits.maxSSBOSize));

			m_properties.limits.maxPushConstantsSize = asset::ICPUMeshBuffer::MAX_PUSH_CONSTANT_BYTESIZE;
			m_properties.limits.maxMemoryAllocationCount = 1000'000'000;
			m_properties.limits.maxSamplerAllocationCount = 1000'000;
			
			m_properties.limits.bufferImageGranularity = std::numeric_limits<size_t>::max(); // buffer and image in the same memory can't be done in gl
			
			GLuint maxCombinedShaderOutputResources;
			GLuint maxFragmentShaderUniformBlocks;
			GLuint maxTextureImageUnits;
			GetIntegerv(GL_MAX_COMBINED_SHADER_OUTPUT_RESOURCES, reinterpret_cast<GLint*>(&maxCombinedShaderOutputResources));
			GetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, reinterpret_cast<GLint*>(&maxFragmentShaderUniformBlocks));
			GetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&maxTextureImageUnits));

			uint32_t maxFragmentShaderResources = maxCombinedShaderOutputResources + maxFragmentShaderUniformBlocks + maxTextureImageUnits;
			uint32_t maxComputeShaderResources = 0u;
			uint32_t maxVertexShaderResources = 0u;
			uint32_t maxTessControlShaderResources = 0u;
			uint32_t maxTessEvalShaderResources = 0u;
			uint32_t maxGeometryShaderResources = 0u;

			GLint maxSSBO[6];
			GetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, maxSSBO + 0);
			GetIntegerv(GL_MAX_VERTEX_SHADER_STORAGE_BLOCKS, maxSSBO + 1);
			GetIntegerv(GL_MAX_TESS_CONTROL_SHADER_STORAGE_BLOCKS, maxSSBO + 2);
			GetIntegerv(GL_MAX_TESS_EVALUATION_SHADER_STORAGE_BLOCKS, maxSSBO + 3);
			GetIntegerv(GL_MAX_GEOMETRY_SHADER_STORAGE_BLOCKS, maxSSBO + 4);
			GetIntegerv(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, maxSSBO + 5);
			maxComputeShaderResources += maxSSBO[0u];
			maxVertexShaderResources += maxSSBO[1u];
			maxTessControlShaderResources += maxSSBO[2u];
			maxTessEvalShaderResources += maxSSBO[3u];
			maxGeometryShaderResources += maxSSBO[4u];

			uint32_t maxPerStageSSBOs = static_cast<uint32_t>(*std::min_element(maxSSBO, maxSSBO + 6));

			GLint maxSampler[5];
			GetIntegerv(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS, maxSampler + 0);
			GetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, maxSampler + 1);
			GetIntegerv(GL_MAX_TESS_CONTROL_TEXTURE_IMAGE_UNITS, maxSampler + 2);
			GetIntegerv(GL_MAX_TESS_EVALUATION_TEXTURE_IMAGE_UNITS, maxSampler + 3);
			GetIntegerv(GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS, maxSampler + 4);
			maxComputeShaderResources += maxSampler[0u];
			maxVertexShaderResources += maxSampler[1u];
			maxTessControlShaderResources += maxSampler[2u];
			maxTessEvalShaderResources += maxSampler[3u];
			maxGeometryShaderResources += maxSampler[4u];
			uint32_t maxPerStageSamplers = static_cast<uint32_t>(*std::min_element(maxSampler, maxSampler + 5));
			
			GLint maxUBOs[6];
			GetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, maxUBOs + 0);
			GetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS, maxUBOs + 1);
			GetIntegerv(GL_MAX_TESS_CONTROL_UNIFORM_BLOCKS, maxUBOs + 2);
			GetIntegerv(GL_MAX_TESS_EVALUATION_UNIFORM_BLOCKS, maxUBOs + 3);
			GetIntegerv(GL_MAX_GEOMETRY_UNIFORM_BLOCKS, maxUBOs + 4);
			GetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, maxUBOs + 5);
			maxComputeShaderResources += maxUBOs[0u];
			maxVertexShaderResources += maxUBOs[1u];
			maxTessControlShaderResources += maxUBOs[2u];
			maxTessEvalShaderResources += maxUBOs[3u];
			maxGeometryShaderResources += maxUBOs[4u];
			uint32_t maxPerStageUBOs = static_cast<uint32_t>(*std::min_element(maxUBOs, maxUBOs + 6));
			
			GLint maxStorageImages[6];
			GetIntegerv(GL_MAX_COMPUTE_IMAGE_UNIFORMS, maxStorageImages + 0);
			GetIntegerv(GL_MAX_VERTEX_IMAGE_UNIFORMS, maxStorageImages + 1);
			GetIntegerv(GL_MAX_TESS_CONTROL_IMAGE_UNIFORMS, maxStorageImages + 2);
			GetIntegerv(GL_MAX_TESS_EVALUATION_IMAGE_UNIFORMS, maxStorageImages + 3);
			GetIntegerv(GL_MAX_GEOMETRY_IMAGE_UNIFORMS, maxStorageImages + 4);
			GetIntegerv(GL_MAX_FRAGMENT_IMAGE_UNIFORMS, maxStorageImages + 5);
			maxComputeShaderResources += maxStorageImages[0u];
			maxVertexShaderResources += maxStorageImages[1u];
			maxTessControlShaderResources += maxStorageImages[2u];
			maxTessEvalShaderResources += maxStorageImages[3u];
			maxGeometryShaderResources += maxStorageImages[4u];
			uint32_t maxPerStageStorageImages = static_cast<uint32_t>(*std::min_element(maxStorageImages, maxStorageImages + 6));
			
			// Max PerStage Descriptors
			m_properties.limits.maxPerStageDescriptorSamplers = maxPerStageSamplers;
			m_properties.limits.maxPerStageDescriptorUBOs = maxPerStageUBOs;
			m_properties.limits.maxPerStageDescriptorSSBOs = maxPerStageSSBOs;
			m_properties.limits.maxPerStageDescriptorImages = m_properties.limits.maxPerStageDescriptorSamplers; // OpenGL glBindTextures is used to bind a BufferView (UTB), so they use the same slots as regular textures
			m_properties.limits.maxPerStageDescriptorStorageImages = maxPerStageStorageImages;
			m_properties.limits.maxPerStageDescriptorInputAttachments = 0u;
			
			//m_properties.limits.maxPerStageResources
			{
				m_properties.limits.maxPerStageResources = maxFragmentShaderResources;
				m_properties.limits.maxPerStageResources = core::max(maxComputeShaderResources, m_properties.limits.maxPerStageResources);
				m_properties.limits.maxPerStageResources = core::max(maxVertexShaderResources, m_properties.limits.maxPerStageResources);
				m_properties.limits.maxPerStageResources = core::max(maxTessControlShaderResources, m_properties.limits.maxPerStageResources);
				m_properties.limits.maxPerStageResources = core::max(maxTessEvalShaderResources, m_properties.limits.maxPerStageResources);
				m_properties.limits.maxPerStageResources = core::max(maxGeometryShaderResources, m_properties.limits.maxPerStageResources);
			}

			// Max Descriptors
			GetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, reinterpret_cast<GLint*>(&m_properties.limits.maxDescriptorSetSamplers));
			GetIntegerv(GL_MAX_COMBINED_UNIFORM_BLOCKS, reinterpret_cast<GLint*>(&m_properties.limits.maxDescriptorSetUBOs));
			m_properties.limits.maxDescriptorSetDynamicOffsetUBOs = m_properties.limits.maxDescriptorSetUBOs;
			GetIntegerv(GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS, reinterpret_cast<GLint*>(&m_properties.limits.maxDescriptorSetSSBOs));
			m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs = m_properties.limits.maxDescriptorSetSSBOs;
			m_properties.limits.maxDescriptorSetImages = m_properties.limits.maxDescriptorSetSamplers; // OpenGL glBindTextures is used to bind a BufferView (UTB), so they use the same slots as regular textures
			GetIntegerv(GL_MAX_COMBINED_IMAGE_UNIFORMS, reinterpret_cast<GLint*>(&m_properties.limits.maxDescriptorSetStorageImages));
			m_properties.limits.maxDescriptorSetInputAttachments = 0u;

			// m_properties.limits.maxDescriptorSetUBOs = m_glfeatures.maxUBOBindings;
			// m_properties.limits.maxDescriptorSetDynamicOffsetUBOs = SOpenGLState::MaxDynamicOffsetUBOs;
			// m_properties.limits.maxDescriptorSetSSBOs = m_glfeatures.maxSSBOBindings;
			// m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs = SOpenGLState::MaxDynamicOffsetSSBOs;
			// m_properties.limits.maxDescriptorSetImages = m_glfeatures.maxTextureBindings;
			// m_properties.limits.maxDescriptorSetStorageImages = m_glfeatures.maxImageBindings;
			
			GetIntegerv(GL_MAX_VERTEX_OUTPUT_COMPONENTS, reinterpret_cast<GLint*>(&m_properties.limits.maxVertexOutputComponents));

			GetIntegerv(GL_MAX_FRAGMENT_INPUT_COMPONENTS, reinterpret_cast<GLint*>(&m_properties.limits.maxFragmentInputComponents));
			GetIntegerv(GL_MAX_DRAW_BUFFERS, reinterpret_cast<GLint*>(&m_properties.limits.maxFragmentOutputAttachments));
			GetIntegerv(GL_MAX_DUAL_SOURCE_DRAW_BUFFERS, reinterpret_cast<GLint*>(&m_properties.limits.maxFragmentDualSrcAttachments));
			GetIntegerv(GL_MAX_COMBINED_IMAGE_UNITS_AND_FRAGMENT_OUTPUTS, reinterpret_cast<GLint*>(&m_properties.limits.maxFragmentCombinedOutputResources));

			GetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, reinterpret_cast<GLint*>(&m_properties.limits.maxComputeSharedMemorySize));
			
			GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, reinterpret_cast<GLint*>(&m_properties.limits.maxComputeWorkGroupInvocations));
			GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, reinterpret_cast<GLint*>(m_properties.limits.maxComputeWorkGroupCount));
			GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, reinterpret_cast<GLint*>(m_properties.limits.maxComputeWorkGroupCount + 1));
			GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, reinterpret_cast<GLint*>(m_properties.limits.maxComputeWorkGroupCount + 2));

			GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, reinterpret_cast<GLint*>(m_properties.limits.maxWorkgroupSize));
			GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, reinterpret_cast<GLint*>(m_properties.limits.maxWorkgroupSize + 1));
			GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, reinterpret_cast<GLint*>(m_properties.limits.maxWorkgroupSize + 2));
			
			
			GetIntegerv(GL_SUBPIXEL_BITS, reinterpret_cast<GLint*>(&m_properties.limits.subPixelPrecisionBits));

			// GL doesnt have any limit on this (???)
			m_properties.limits.maxDrawIndirectCount = std::numeric_limits<decltype(m_properties.limits.maxDrawIndirectCount)>::max();
			
			GetFloatv(GL_MAX_TEXTURE_LOD_BIAS, &m_properties.limits.maxSamplerLodBias);

			GLint maxViewportExtent[2]{0,0};
			GetIntegerv(GL_MAX_VIEWPORT_DIMS, maxViewportExtent);
			GLint maxViewports = 16;
			GetIntegerv(GL_MAX_VIEWPORTS, &maxViewports);
			m_properties.limits.maxViewports = maxViewports;
			m_properties.limits.maxViewportDims[0] = maxViewportExtent[0];
			m_properties.limits.maxViewportDims[1] = maxViewportExtent[1];
			
			int32_t maxDim = static_cast<int32_t>(core::max(m_properties.limits.maxViewportDims[0], m_properties.limits.maxViewportDims[1]));
			m_properties.limits.viewportBoundsRange[0] = -2 * maxDim;
			m_properties.limits.viewportBoundsRange[1] = 2 * maxDim - 1;

			GetIntegerv(GL_VIEWPORT_SUBPIXEL_BITS, reinterpret_cast<GLint*>(&m_properties.limits.viewportSubPixelBits));

			if(IsGLES)
				m_properties.limits.minMemoryMapAlignment = 16ull;
			else
				GetIntegerv(GL_MIN_MAP_BUFFER_ALIGNMENT, reinterpret_cast<GLint*>(&m_properties.limits.minMemoryMapAlignment));

			
			GetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, reinterpret_cast<GLint*>(&m_properties.limits.minUBOAlignment));
			assert(core::is_alignment(m_properties.limits.minUBOAlignment));
			GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, reinterpret_cast<GLint*>(&m_properties.limits.minSSBOAlignment));
			assert(core::is_alignment(m_properties.limits.minSSBOAlignment));
			// TODO: GLES has a problem with reporting this
			GetIntegerv(GL_TEXTURE_BUFFER_OFFSET_ALIGNMENT, reinterpret_cast<GLint*>(&m_properties.limits.bufferViewAlignment));

			if (!core::is_alignment(m_properties.limits.bufferViewAlignment))
				m_properties.limits.bufferViewAlignment = 16u;
			// assert(core::is_alignment(m_properties.limits.bufferViewAlignment)); 

			GetIntegerv(GL_MIN_PROGRAM_TEXEL_OFFSET, reinterpret_cast<GLint*>(&m_properties.limits.minTexelOffset));
			GetIntegerv(GL_MAX_PROGRAM_TEXEL_OFFSET, reinterpret_cast<GLint*>(&m_properties.limits.maxTexelOffset));
			GetIntegerv(GL_MIN_PROGRAM_TEXTURE_GATHER_OFFSET, reinterpret_cast<GLint*>(&m_properties.limits.minTexelGatherOffset));
			GetIntegerv(GL_MAX_PROGRAM_TEXTURE_GATHER_OFFSET, reinterpret_cast<GLint*>(&m_properties.limits.maxTexelGatherOffset));
	
			if(!IsGLES || m_glfeatures.Version >= 320u)
			{
				GetFloatv(GL_MIN_FRAGMENT_INTERPOLATION_OFFSET, &m_properties.limits.minInterpolationOffset);
				GetFloatv(GL_MAX_FRAGMENT_INTERPOLATION_OFFSET, &m_properties.limits.maxInterpolationOffset);
			}

			GetIntegerv(GL_MAX_FRAMEBUFFER_WIDTH, reinterpret_cast<GLint*>(&m_properties.limits.maxFramebufferWidth));
			GetIntegerv(GL_MAX_FRAMEBUFFER_HEIGHT, reinterpret_cast<GLint*>(&m_properties.limits.maxFramebufferHeight));

			if(!IsGLES || m_glfeatures.Version>=320u)
				GetIntegerv(GL_MAX_FRAMEBUFFER_LAYERS, reinterpret_cast<GLint*>(&m_properties.limits.maxFramebufferLayers));
			else if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_OES_geometry_shader))
				GetIntegerv(GL_MAX_FRAMEBUFFER_LAYERS_OES, reinterpret_cast<GLint*>(&m_properties.limits.maxFramebufferLayers));
			else if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_geometry_shader))
				GetIntegerv(GL_MAX_FRAMEBUFFER_LAYERS_EXT, reinterpret_cast<GLint*>(&m_properties.limits.maxFramebufferLayers));

			auto getSampleCountFlagsFromSampleCount = [](GLint samples) -> core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>
			{
				return core::bitflag<asset::IImage::E_SAMPLE_COUNT_FLAGS>((0x1u<<(1+core::findMSB(static_cast<uint32_t>(samples))))-1u);
			};

			GLint maxFramebufferSamples = 0;
			GetIntegerv(GL_MAX_FRAMEBUFFER_SAMPLES, &maxFramebufferSamples);
			auto framebufferSampleCountFlags = getSampleCountFlagsFromSampleCount(maxFramebufferSamples);
			m_properties.limits.framebufferColorSampleCounts = framebufferSampleCountFlags;
			m_properties.limits.framebufferDepthSampleCounts = framebufferSampleCountFlags;
			m_properties.limits.framebufferStencilSampleCounts = framebufferSampleCountFlags;
			m_properties.limits.framebufferNoAttachmentsSampleCounts = framebufferSampleCountFlags;

			GetIntegerv(GL_MAX_COLOR_ATTACHMENTS, reinterpret_cast<GLint*>(&m_properties.limits.maxColorAttachments));
			
			GLint maxColorTextureSamples = 0;
			GetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &maxColorTextureSamples);
			auto colorSampleCountFlags = getSampleCountFlagsFromSampleCount(maxColorTextureSamples);
			GLint maxDepthTextureSamples = 0;
			GetIntegerv(GL_MAX_DEPTH_TEXTURE_SAMPLES, &maxDepthTextureSamples);
			auto depthSampleCountFlags = getSampleCountFlagsFromSampleCount(maxDepthTextureSamples);
			m_properties.limits.sampledImageColorSampleCounts = colorSampleCountFlags;
			m_properties.limits.sampledImageIntegerSampleCounts = colorSampleCountFlags;
			m_properties.limits.sampledImageDepthSampleCounts = depthSampleCountFlags;
			m_properties.limits.sampledImageStencilSampleCounts = depthSampleCountFlags;
			m_properties.limits.storageImageSampleCounts = colorSampleCountFlags;
			
			GetIntegerv(GL_MAX_SAMPLE_MASK_WORDS, reinterpret_cast<GLint*>(&m_properties.limits.maxSampleMaskWords));
			
			m_properties.limits.timestampComputeAndGraphics = (IsGLES) ? m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_EXT_disjoint_timer_query) : true;
			m_properties.limits.timestampPeriodInNanoSeconds = 1.0f;

			m_properties.limits.discreteQueuePriorities = 1u;

			GetFloatv(GL_POINT_SIZE_RANGE, m_properties.limits.pointSizeRange);
			GetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, m_properties.limits.lineWidthRange);
			
			GetFloatv(GL_POINT_SIZE_GRANULARITY, &m_properties.limits.pointSizeGranularity);
			GetFloatv(GL_LINE_WIDTH_GRANULARITY, &m_properties.limits.lineWidthGranularity);
			
			m_properties.limits.strictLines = false;
			m_properties.limits.standardSampleLocations = false; // TODO: Investigate

			m_properties.limits.optimalBufferCopyOffsetAlignment = 8ull;
			m_properties.limits.optimalBufferCopyRowPitchAlignment = 8ull;
			m_properties.limits.nonCoherentAtomSize = 256ull;

			const uint64_t maxTBOSizeInBytes = (IsGLES) 
				? (m_properties.limits.maxBufferViewTexels * getTexelOrBlockBytesize(asset::EF_R32G32B32A32_UINT))     // m_properties.limits.maxBufferViewTexels * GLES Fattest Format 
				: (m_properties.limits.maxBufferViewTexels * getTexelOrBlockBytesize(asset::EF_R64G64B64A64_SFLOAT)); // m_properties.limits.maxBufferViewTexels * GL Fattest Format 

			const uint64_t maxBufferSize = std::max(std::max((uint64_t)m_properties.limits.maxUBOSize, (uint64_t)m_properties.limits.maxSSBOSize), maxTBOSizeInBytes);

			/* Vulkan 1.1 Core  */
			
			m_properties.limits.maxPerSetDescriptors = m_glfeatures.maxUBOBindings + m_glfeatures.maxSSBOBindings + m_glfeatures.maxTextureBindings + m_glfeatures.maxImageBindings;
			m_properties.limits.maxMemoryAllocationSize = maxBufferSize; // TODO(Erfan): 

			/* Vulkan 1.2 Core  */

			/*		VK_KHR_shader_float_controls */
			m_properties.limits.shaderSignedZeroInfNanPreserveFloat16   = false;
			m_properties.limits.shaderSignedZeroInfNanPreserveFloat32   = false;
			m_properties.limits.shaderSignedZeroInfNanPreserveFloat64   = false;
			m_properties.limits.shaderDenormPreserveFloat16             = false;
			m_properties.limits.shaderDenormPreserveFloat32             = false;
			m_properties.limits.shaderDenormPreserveFloat64             = false;
			m_properties.limits.shaderDenormFlushToZeroFloat16          = false;
			m_properties.limits.shaderDenormFlushToZeroFloat32          = false;
			m_properties.limits.shaderDenormFlushToZeroFloat64          = false;
			m_properties.limits.shaderRoundingModeRTEFloat16            = false;
			m_properties.limits.shaderRoundingModeRTEFloat32            = false;
			m_properties.limits.shaderRoundingModeRTEFloat64            = false;
			m_properties.limits.shaderRoundingModeRTZFloat16            = false;
			m_properties.limits.shaderRoundingModeRTZFloat32            = false;
			m_properties.limits.shaderRoundingModeRTZFloat64            = false;

			/*		VK_EXT_descriptor_indexing */
			m_properties.limits.maxUpdateAfterBindDescriptorsInAllPools					= ~0u;
			bool nonUniformIndexing = m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_NV_gpu_shader5) || m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_EXT_nonuniform_qualifier);
			m_properties.limits.shaderUniformBufferArrayNonUniformIndexingNative		= nonUniformIndexing;
			m_properties.limits.shaderSampledImageArrayNonUniformIndexingNative			= nonUniformIndexing;
			m_properties.limits.shaderStorageBufferArrayNonUniformIndexingNative		= nonUniformIndexing;
			m_properties.limits.shaderStorageImageArrayNonUniformIndexingNative			= nonUniformIndexing;
			m_properties.limits.shaderInputAttachmentArrayNonUniformIndexingNative		= false; //	No Input Attachments in	GL
			m_properties.limits.robustBufferAccessUpdateAfterBind						= false; //	TODO
			m_properties.limits.quadDivergentImplicitLod								= nonUniformIndexing;
			m_properties.limits.maxPerStageDescriptorUpdateAfterBindSamplers			= m_properties.limits.maxPerStageDescriptorSamplers;
			m_properties.limits.maxPerStageDescriptorUpdateAfterBindUBOs				= m_properties.limits.maxPerStageDescriptorUBOs;
			m_properties.limits.maxPerStageDescriptorUpdateAfterBindSSBOs				= m_properties.limits.maxPerStageDescriptorSSBOs;
			m_properties.limits.maxPerStageDescriptorUpdateAfterBindImages				= m_properties.limits.maxPerStageDescriptorImages;
			m_properties.limits.maxPerStageDescriptorUpdateAfterBindStorageImages		= m_properties.limits.maxPerStageDescriptorStorageImages;
			m_properties.limits.maxPerStageDescriptorUpdateAfterBindInputAttachments	= 0u;  // No Input Attachments in GL
			m_properties.limits.maxPerStageUpdateAfterBindResources						= m_properties.limits.maxPerStageResources;
			m_properties.limits.maxDescriptorSetUpdateAfterBindSamplers					= m_properties.limits.maxDescriptorSetSamplers;
			m_properties.limits.maxDescriptorSetUpdateAfterBindUBOs						= m_properties.limits.maxDescriptorSetUBOs;
			m_properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetUBOs		= m_properties.limits.maxDescriptorSetDynamicOffsetUBOs;
			m_properties.limits.maxDescriptorSetUpdateAfterBindSSBOs					= m_properties.limits.maxDescriptorSetSSBOs;
			m_properties.limits.maxDescriptorSetUpdateAfterBindDynamicOffsetSSBOs		= m_properties.limits.maxDescriptorSetDynamicOffsetSSBOs;
			m_properties.limits.maxDescriptorSetUpdateAfterBindImages					= m_properties.limits.maxDescriptorSetImages;
			m_properties.limits.maxDescriptorSetUpdateAfterBindStorageImages			= m_properties.limits.maxDescriptorSetStorageImages;
			m_properties.limits.maxDescriptorSetUpdateAfterBindInputAttachments			= 0u; // No	Input Attachments in GL
			
			m_properties.limits.filterMinmaxSingleComponentFormats = false;
			m_properties.limits.filterMinmaxImageComponentMapping = false;

			m_properties.limits.framebufferIntegerColorSampleCounts = framebufferSampleCountFlags;

			/* Vulkan 1.3 Core  */
			m_properties.limits.maxBufferSize = maxBufferSize;
			
			// maxSubgroupSize can be overrided later by KHR_shader_subgroup::GL_SUBGROUP_SIZE_KHR
			getMinMaxSubgroupSizeFromDriverID(m_properties.driverID, m_properties.limits.minSubgroupSize, m_properties.limits.maxSubgroupSize);

			m_properties.limits.maxComputeWorkgroupSubgroups = m_properties.limits.maxComputeWorkGroupInvocations/m_properties.limits.minSubgroupSize;
			m_properties.limits.requiredSubgroupSizeStages = core::bitflag<asset::IShader::E_SHADER_STAGE>(0u);
			
			
			// https://github.com/KhronosGroup/OpenGL-API/issues/51
			// https://www.khronos.org/opengl/wiki/Vertex_Post-Processing#Clipping
			if(isNVIDIAGPU || IsGLES)
				m_properties.limits.pointClippingBehavior = SLimits::EPCB_USER_CLIP_PLANES_ONLY;
			else
				m_properties.limits.pointClippingBehavior = SLimits::EPCB_ALL_CLIP_PLANES;

			/* SubgroupProperties */
			m_properties.limits.subgroupSize = 0u;
			m_properties.limits.subgroupOpsShaderStages = static_cast<asset::IShader::E_SHADER_STAGE>(0u);
			
			if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_shader_subgroup))
			{
				GLint subgroupSize = 0;
				GetIntegerv(GL_SUBGROUP_SIZE_KHR, &subgroupSize);
				m_properties.limits.maxSubgroupSize = subgroupSize;

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
			
			// TODO: @achal handle ARB, EXT, NVidia and AMD extensions which can be used to spoof
			if (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_KHR_shader_subgroup))
			{
				GLboolean subgroupQuadAllStages = GL_FALSE;
				GetBooleanv(GL_SUBGROUP_QUAD_ALL_STAGES_KHR, &subgroupQuadAllStages);
				m_properties.limits.shaderSubgroupQuadAllStages = static_cast<bool>(subgroupQuadAllStages);

				GLint subgroup = 0;
				GetIntegerv(GL_SUBGROUP_SUPPORTED_FEATURES_KHR, &subgroup);

				m_properties.limits.shaderSubgroupBasic = (subgroup & GL_SUBGROUP_FEATURE_BASIC_BIT_KHR);
				m_properties.limits.shaderSubgroupVote = (subgroup & GL_SUBGROUP_FEATURE_VOTE_BIT_KHR);
				m_properties.limits.shaderSubgroupArithmetic = (subgroup & GL_SUBGROUP_FEATURE_ARITHMETIC_BIT_KHR);
				m_properties.limits.shaderSubgroupBallot = (subgroup & GL_SUBGROUP_FEATURE_BALLOT_BIT_KHR);
				m_properties.limits.shaderSubgroupShuffle = (subgroup & GL_SUBGROUP_FEATURE_SHUFFLE_BIT_KHR);
				m_properties.limits.shaderSubgroupShuffleRelative = (subgroup & GL_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT_KHR);
				m_properties.limits.shaderSubgroupClustered = (subgroup & GL_SUBGROUP_FEATURE_CLUSTERED_BIT_KHR);
				m_properties.limits.shaderSubgroupQuad = (subgroup & GL_SUBGROUP_FEATURE_QUAD_BIT_KHR);
			}


			// https://github.com/KhronosGroup/SPIRV-Cross/issues/1350
			// https://github.com/KhronosGroup/SPIRV-Cross/issues/1351
			// https://github.com/KhronosGroup/SPIRV-Cross/issues/1352
			if(m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_NV_shader_thread_group) || m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_shader_ballot))
				m_properties.limits.shaderSubgroupBasic = true;
			if(m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_shader_group_vote) || m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_NV_gpu_shader5) || m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_AMD_gcn_shader))
				m_properties.limits.shaderSubgroupVote = true;
			if(m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_NV_shader_thread_group) || (m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_gpu_shader_int64) && m_glfeatures.isFeatureAvailable(COpenGLFeatureMap::NBL_ARB_shader_ballot)))
				m_properties.limits.shaderSubgroupBallot = true;

			/* !NOT SUPPORTED: AccelerationStructurePropertiesKHR  */
			/* !NOT SUPPORTED: RayTracingPipelinePropertiesKHR */
			
			/* Nabla */
			
			if (m_glfeatures.isFeatureAvailable(m_glfeatures.NBL_ARB_query_buffer_object))
				m_properties.limits.allowCommandBufferQueryCopies = true;

			// TODO: get this from OpenCL interop, or just a GPU Device & Vendor ID table
			GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,reinterpret_cast<int32_t*>(&m_properties.limits.maxOptimallyResidentWorkgroupInvocations));
			m_properties.limits.maxOptimallyResidentWorkgroupInvocations = core::min(core::roundDownToPoT(m_properties.limits.maxOptimallyResidentWorkgroupInvocations),512u);
			constexpr auto beefyGPUWorkgroupMaxOccupancy = 256u; // TODO: find a way to query and report this somehow, persistent threads are very useful!
			m_properties.limits.maxResidentInvocations = beefyGPUWorkgroupMaxOccupancy*m_properties.limits.maxOptimallyResidentWorkgroupInvocations;

			m_properties.limits.spirvVersion = asset::IGLSLCompiler::ESV_1_6;
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
