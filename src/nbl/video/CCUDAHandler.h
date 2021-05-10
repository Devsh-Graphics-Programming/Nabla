// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_CUDA_HANDLER_H__
#define __NBL_VIDEO_C_CUDA_HANDLER_H__

#include "nbl/macros.h"
#include "IReadFile.h"
#include "nbl/core/compile_config.h"
#include "nbl/system/system.h"


#ifdef _NBL_COMPILE_WITH_CUDA_

#include "cuda.h"
#include "nvrtc.h"
#if CUDA_VERSION < 9000
	#error "Need CUDA 9.0 SDK or higher."
#endif

#ifdef _NBL_COMPILE_WITH_OPENGL_
	#include "COpenGLDriver.h"
	// make CUDA play nice
	#define WGL_NV_gpu_affinity 0
	#include "cudaGL.h"
	#undef WGL_NV_gpu_affinity
#endif // _NBL_COMPILE_WITH_OPENGL_

// useful includes in the future
//#include "cudaEGL.h"
//#include "cudaVDPAU.h"

#include "os.h"

namespace nbl
{
namespace cuda
{

#define _NBL_DEFAULT_NVRTC_OPTIONS "--std=c++14",virtualCUDAArchitecture,"-dc","-use_fast_math"


class CCUDAHandler
{
    public:
		using LibLoader = system::DefaultFuncPtrLoader;

		NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CUDA, LibLoader
			,cuCtxCreate_v2
			,cuDevicePrimaryCtxRetain
			,cuDevicePrimaryCtxRelease
			,cuDevicePrimaryCtxSetFlags
			,cuDevicePrimaryCtxGetState
			,cuCtxDestroy_v2
			,cuCtxEnablePeerAccess
			,cuCtxGetApiVersion
			,cuCtxGetCurrent
			,cuCtxGetDevice
			,cuCtxGetSharedMemConfig
			,cuCtxPopCurrent_v2
			,cuCtxPushCurrent_v2
			,cuCtxSetCacheConfig
			,cuCtxSetCurrent
			,cuCtxSetSharedMemConfig
			,cuCtxSynchronize
			,cuDeviceComputeCapability
			,cuDeviceCanAccessPeer
			,cuDeviceGetCount
			,cuDeviceGet
			,cuDeviceGetAttribute
			,cuDeviceGetLuid
			,cuDeviceGetUuid
			,cuDeviceTotalMem_v2
			,cuDeviceGetName
			,cuDriverGetVersion
			,cuEventCreate
			,cuEventDestroy_v2
			,cuEventElapsedTime
			,cuEventQuery
			,cuEventRecord
			,cuEventSynchronize
			,cuFuncGetAttribute
			,cuFuncSetCacheConfig
			,cuGetErrorName
			,cuGetErrorString
			,cuGraphicsGLRegisterBuffer
			,cuGraphicsGLRegisterImage
			,cuGraphicsMapResources
			,cuGraphicsResourceGetMappedPointer_v2
			,cuGraphicsResourceGetMappedMipmappedArray
			,cuGraphicsSubResourceGetMappedArray
			,cuGraphicsUnmapResources
			,cuGraphicsUnregisterResource
			,cuInit
			,cuLaunchKernel
			,cuMemAlloc_v2
			,cuMemcpyDtoD_v2
			,cuMemcpyDtoH_v2
			,cuMemcpyHtoD_v2
			,cuMemcpyDtoDAsync_v2
			,cuMemcpyDtoHAsync_v2
			,cuMemcpyHtoDAsync_v2
			,cuMemGetAddressRange_v2
			,cuMemFree_v2
			,cuMemFreeHost
			,cuMemGetInfo_v2
			,cuMemHostAlloc
			,cuMemHostRegister_v2
			,cuMemHostUnregister
			,cuMemsetD32_v2
			,cuMemsetD32Async
			,cuMemsetD8_v2
			,cuMemsetD8Async
			,cuModuleGetFunction
			,cuModuleGetGlobal_v2
			,cuModuleLoadDataEx
			,cuModuleLoadFatBinary
			,cuModuleUnload
			,cuOccupancyMaxActiveBlocksPerMultiprocessor
			,cuPointerGetAttribute
			,cuStreamAddCallback
			,cuStreamCreate
			,cuStreamDestroy_v2
			,cuStreamQuery
			,cuStreamSynchronize
			,cuStreamWaitEvent
			,cuSurfObjectCreate
			,cuSurfObjectDestroy
			,cuTexObjectCreate
			,cuTexObjectDestroy
			,cuGLGetDevices_v2
		);
		static CUDA cuda;

		struct Device
		{
			Device() {}
			Device(int ordinal);
			~Device()
			{
			}

			CUdevice handle = -1;
			char name[122] = {};
			char luid = -1;
			unsigned int deviceNodeMask = 0;
			CUuuid uuid = {};
			size_t vram_size = 0ull;
			int attributes[CU_DEVICE_ATTRIBUTE_MAX] = {};
		};

		NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(NVRTC, LibLoader,
			nvrtcGetErrorString,
			nvrtcVersion,
			nvrtcAddNameExpression,
			nvrtcCompileProgram,
			nvrtcCreateProgram,
			nvrtcDestroyProgram,
			nvrtcGetLoweredName,
			nvrtcGetPTX,
			nvrtcGetPTXSize,
			nvrtcGetProgramLog,
			nvrtcGetProgramLogSize
		);
		static NVRTC nvrtc;

	protected:
        CCUDAHandler() = default;

		_NBL_STATIC_INLINE int CudaVersion = 0;
		_NBL_STATIC_INLINE int DeviceCount = 0;
		static core::vector<Device> devices;
		
		static core::vector<core::smart_refctd_ptr<const io::IReadFile> > headers;
		static core::vector<const char*> headerContents;
		static core::vector<const char*> headerNames;

		_NBL_STATIC_INLINE_CONSTEXPR const char* virtualCUDAArchitectures[] = {	"-arch=compute_30",
																				"-arch=compute_32",
																				"-arch=compute_35",
																				"-arch=compute_37",
																				"-arch=compute_50",
																				"-arch=compute_52",
																				"-arch=compute_53",
																				"-arch=compute_60",
																				"-arch=compute_61",
																				"-arch=compute_62",
																				"-arch=compute_70",
																				"-arch=compute_72",
																				"-arch=compute_75",
																				"-arch=compute_80"};
		_NBL_STATIC_INLINE const char* virtualCUDAArchitecture = nullptr;

		#ifdef _MSC_VER
			_NBL_STATIC_INLINE_CONSTEXPR const char* CUDA_EXTRA_DEFINES = "#ifndef _WIN64\n#define _WIN64\n#endif\n";
		#else
			_NBL_STATIC_INLINE_CONSTEXPR const char* CUDA_EXTRA_DEFINES = "#ifndef __LP64__\n#define __LP64__\n#endif\n";
		#endif

	public:
		static CUresult init();
		static void deinit();

		static const char* getCommonVirtualCUDAArchitecture() {return virtualCUDAArchitecture;}

		static bool defaultHandleResult(CUresult result);

		static CUresult getDefaultGLDevices(uint32_t* foundCount, CUdevice* pCudaDevices, uint32_t cudaDeviceCount)
		{
			return cuda.pcuGLGetDevices_v2(foundCount,pCudaDevices,cudaDeviceCount,CU_GL_DEVICE_LIST_ALL);
		}

		template<typename T>
		static T* cast_CUDA_ptr(CUdeviceptr ptr) {return reinterpret_cast<T*>(ptr);}

		template<typename ObjType>
		struct GraphicsAPIObjLink
		{
				GraphicsAPIObjLink() : obj(nullptr), cudaHandle(nullptr), acquired(false)
				{
					asImage = {nullptr};
				}
				GraphicsAPIObjLink(core::smart_refctd_ptr<ObjType>&& _obj) : GraphicsAPIObjLink()
				{
					obj = std::move(_obj);
				}
				GraphicsAPIObjLink(GraphicsAPIObjLink&& other) : GraphicsAPIObjLink()
				{
					operator=(std::move(other));
				}

				GraphicsAPIObjLink(const GraphicsAPIObjLink& other) = delete;
				GraphicsAPIObjLink& operator=(const GraphicsAPIObjLink& other) = delete;
				GraphicsAPIObjLink& operator=(GraphicsAPIObjLink&& other)
				{
					std::swap(obj,other.obj);
					std::swap(cudaHandle,other.cudaHandle);
					std::swap(acquired,other.acquired);
					std::swap(asImage,other.asImage);
					return *this;
				}

				~GraphicsAPIObjLink()
				{
					assert(!acquired); // you've fucked up, there's no way for us to fix it, you need to release the objects on a proper stream
					if (obj)
						CCUDAHandler::cuda.pcuGraphicsUnregisterResource(cudaHandle);
				}

				//
				auto* getObject() const {return obj.get();}

			private:
				core::smart_refctd_ptr<ObjType> obj;
				CUgraphicsResource cudaHandle;
				bool acquired;

				friend class CCUDAHandler;
			public:
				union
				{
					struct
					{
						CUdeviceptr pointer;
					} asBuffer;
					struct
					{
						CUmipmappedArray mipmappedArray;
						CUarray array;
					} asImage;
				};
		};

		//
		static CUresult registerBuffer(GraphicsAPIObjLink<video::IGPUBuffer>* link, uint32_t flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
		static CUresult registerImage(GraphicsAPIObjLink<video::IGPUImage>* link, uint32_t flags = CU_GRAPHICS_REGISTER_FLAGS_NONE);
		

		template<typename ObjType>
		static CUresult acquireResourcesFromGraphics(void* tmpStorage, GraphicsAPIObjLink<ObjType>* linksBegin, GraphicsAPIObjLink<ObjType>* linksEnd, CUstream stream)
		{
			auto count = std::distance(linksBegin,linksEnd);

			auto resources = reinterpret_cast<CUgraphicsResource*>(tmpStorage);
			auto rit = resources;
			for (auto iit=linksBegin; iit!=linksEnd; iit++,rit++)
			{
				if (iit->acquired)
					return CUDA_ERROR_UNKNOWN;
				*rit = iit->cudaHandle;
			}

			auto retval = cuda.pcuGraphicsMapResources(count,resources,stream);
			for (auto iit=linksBegin; iit!=linksEnd; iit++)
				iit->acquired = true;
			return retval;
		}
		template<typename ObjType>
		static CUresult releaseResourcesToGraphics(void* tmpStorage, GraphicsAPIObjLink<ObjType>* linksBegin, GraphicsAPIObjLink<ObjType>* linksEnd, CUstream stream)
		{
			auto count = std::distance(linksBegin,linksEnd);

			auto resources = reinterpret_cast<CUgraphicsResource*>(tmpStorage);
			auto rit = resources;
			for (auto iit=linksBegin; iit!=linksEnd; iit++,rit++)
			{
				if (!iit->acquired)
					return CUDA_ERROR_UNKNOWN;
				*rit = iit->cudaHandle;
			}

			auto retval = cuda.pcuGraphicsUnmapResources(count,resources,stream);
			for (auto iit=linksBegin; iit!=linksEnd; iit++)
				iit->acquired = false;
			return retval;
		}

		static CUresult acquireAndGetPointers(GraphicsAPIObjLink<video::IGPUBuffer>* linksBegin, GraphicsAPIObjLink<video::IGPUBuffer>* linksEnd, CUstream stream, size_t* outbufferSizes = nullptr);
		static CUresult acquireAndGetMipmappedArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, CUstream stream);
		static CUresult acquireAndGetArray(GraphicsAPIObjLink<video::IGPUImage>* linksBegin, GraphicsAPIObjLink<video::IGPUImage>* linksEnd, uint32_t* arrayIndices, uint32_t* mipLevels, CUstream stream);


		static bool defaultHandleResult(nvrtcResult result)
		{
			switch (result)
			{
				case NVRTC_SUCCESS:
					return true;
					break;
				default:
					if (nvrtc.pnvrtcGetErrorString)
						printf("%s\n",nvrtc.pnvrtcGetErrorString(result));
					else
						printf(R"===(CudaHandler: `pnvrtcGetErrorString` is nullptr, the nvrtc library probably not found on the system.\n)===");
					break;
			}
			_NBL_DEBUG_BREAK_IF(true);
			return false;
		}

		//
		static core::SRange<const io::IReadFile* const> getCUDASTDHeaders()
		{
			auto begin = headers.empty() ? nullptr:reinterpret_cast<const io::IReadFile* const*>(&headers[0].get());
			return {begin,begin+headers.size()};
		}
		static const auto& getCUDASTDHeaderContents() { return headerContents; }
		static const auto& getCUDASTDHeaderNames() { return headerNames; }

		//
		static nvrtcResult createProgram(	nvrtcProgram* prog, const char* source, const char* name,
											const char* const* headersBegin=nullptr, const char* const* headersEnd=nullptr,
											const char* const* includeNamesBegin=nullptr, const char* const* includeNamesEnd=nullptr)
		{
			auto headerCount = std::distance(headersBegin, headersEnd);
			if (headerCount)
			{
				if (std::distance(includeNamesBegin,includeNamesEnd)!=headerCount)
					return NVRTC_ERROR_INVALID_INPUT;
			}
			else
			{
				headersBegin = nullptr;
				includeNamesBegin = nullptr;
			}
			auto extraLen = strlen(CUDA_EXTRA_DEFINES);
			auto origLen = strlen(source);
			auto totalLen = extraLen+origLen;
			auto tmp = _NBL_NEW_ARRAY(char,totalLen+1u);
			memcpy(tmp, CUDA_EXTRA_DEFINES, extraLen);
			memcpy(tmp+extraLen, source, origLen);
			tmp[totalLen] = 0;
			auto result = nvrtc.pnvrtcCreateProgram(prog, tmp, name, headerCount, headersBegin, includeNamesBegin);
			_NBL_DELETE_ARRAY(tmp,totalLen);
			return result;
		}

		template<typename HeaderFileIt>
		static nvrtcResult createProgram(	nvrtcProgram* prog, nbl::io::IReadFile* main,
											const HeaderFileIt includesBegin, const HeaderFileIt includesEnd)
		{
			int numHeaders = std::distance(includesBegin,includesEnd);
			core::vector<const char*> headers(numHeaders);
			core::vector<const char*> includeNames(numHeaders);
			size_t sourceIt = strlen(CUDA_EXTRA_DEFINES);
			size_t sourceSize = sourceIt+main->getSize();
			sourceSize++;
			for (auto it=includesBegin; it!=includesEnd; it++)
			{
				sourceSize += it->getSize()+1u;
				includeNames.emplace_back(it->getFileName().c_str());
			}
			core::vector<char> sources(sourceSize);
			memcpy(sources.data(),CUDA_EXTRA_DEFINES,sourceIt);
			auto filesize = main->getSize();
			main->read(sources.data()+sourceIt,filesize);
			sourceIt += filesize;
			sources[sourceIt++] = 0;
			for (auto it=includesBegin; it!=includesEnd; it++)
			{
				auto oldpos = it->getPos();
				it->seek(0ull);

				auto ptr = sources.data()+sourceIt;
				headers.push_back(ptr);
				filesize = it->getSize();
				it->read(ptr,filesize);
				sourceIt += filesize;
				sources[sourceIt++] = 0;

				it->seek(oldpos);
			}
			return nvrtc.pnvrtcCreateProgram(prog, sources.data(), main->getFileName().c_str(), numHeaders, headers.data(), includeNames.data());
		}
		
		template<typename OptionsT = const std::initializer_list<const char*>&>
		static nvrtcResult compileProgram(nvrtcProgram prog, OptionsT options={_NBL_DEFAULT_NVRTC_OPTIONS})
		{
			return nvrtc.pnvrtcCompileProgram(prog, options.size(), options.begin());
		}

		static nvrtcResult compileProgram(nvrtcProgram prog, const std::vector<const char*>& options)
		{
			return nvrtc.pnvrtcCompileProgram(prog, options.size(), options.data());
		}

		//
		static nvrtcResult getProgramLog(nvrtcProgram prog, std::string& log);

		//
		static nvrtcResult getPTX(nvrtcProgram prog, std::string& ptx);

		//
		template<typename OptionsT = const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX(std::string& ptx, const char* source, const char* filename,
			const char* const* headersBegin = nullptr, const char* const* headersEnd = nullptr,
			const char* const* includeNamesBegin = nullptr, const char* const* includeNamesEnd = nullptr,
			OptionsT options = { _NBL_DEFAULT_NVRTC_OPTIONS },
			std::string* log = nullptr)
		{
			nvrtcProgram program = nullptr;
			nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
			auto cleanup = core::makeRAIIExiter([&program, &result]() -> void {
				if (result != NVRTC_SUCCESS && program)
					nvrtc.pnvrtcDestroyProgram(&program);
				});

			result = createProgram(&program, source, filename, headersBegin, headersEnd, includeNamesBegin, includeNamesEnd);

			if (result != NVRTC_SUCCESS)
				return result;

			return result = compileDirectlyToPTX_helper<OptionsT>(ptx, program, std::forward<OptionsT>(options), log);
		}

		template<typename OptionsT = const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX(std::string& ptx, nbl::io::IReadFile* main,
			const char* const* headersBegin = nullptr, const char* const* headersEnd = nullptr,
			const char* const* includeNamesBegin = nullptr, const char* const* includeNamesEnd = nullptr,
			OptionsT options = { _NBL_DEFAULT_NVRTC_OPTIONS },
			std::string* log = nullptr)
		{
			char* data = new char[main->getSize()+1ull];
			main->read(data, main->getSize());
			data[main->getSize()] = 0;
			auto result = compileDirectlyToPTX<OptionsT>(ptx, data, main->getFileName().c_str(), headersBegin, headersEnd, std::forward<OptionsT>(options), log);
			delete[] data;

			return result;
		}

		template<typename CompileArgsT, typename OptionsT=const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX(std::string& ptx, nbl::io::IReadFile* main,
												CompileArgsT includesBegin, CompileArgsT includesEnd,
												OptionsT options={_NBL_DEFAULT_NVRTC_OPTIONS},
												std::string* log=nullptr)
		{
			nvrtcProgram program = nullptr;
			nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
			auto cleanup = core::makeRAIIExiter([&program,&result]() -> void {
				if (result!=NVRTC_SUCCESS && program)
					nvrtc.pnvrtcDestroyProgram(&program);
			});
			result = createProgram(&program, main, includesBegin, includesEnd);
			if (result!=NVRTC_SUCCESS)
				return result;

			return result = compileDirectlyToPTX_helper<OptionsT>(ptx,program,std::forward<OptionsT>(options),log);
		}


	protected:
		template<typename OptionsT = const std::initializer_list<const char*>&>
		static nvrtcResult compileDirectlyToPTX_helper(std::string& ptx, nvrtcProgram program, OptionsT options, std::string* log=nullptr)
		{
			nvrtcResult result = compileProgram(program,options);
			if (log)
				getProgramLog(program, *log);
			if (result!=NVRTC_SUCCESS)
				return result;

			return getPTX(program, ptx);
		}
};

}
}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
