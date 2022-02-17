// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_HANDLER_H_
#define _NBL_VIDEO_C_CUDA_HANDLER_H_

#include "nbl/system/declarations.h"

#include "nbl/video/CCUDADevice.h"


#ifdef _NBL_COMPILE_WITH_CUDA_
namespace nbl::video
{

class CCUDAHandler : public core::IReferenceCounted
{
    public:
		static bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger=nullptr);
		inline bool defaultHandleResult(CUresult result)
		{
			core::smart_refctd_ptr<system::ILogger> logger = m_logger.get();
			return defaultHandleResult(result,logger.get());
		}

		//
		bool defaultHandleResult(nvrtcResult result);

		//
		template<typename T>
		static T* cast_CUDA_ptr(CUdeviceptr ptr) { return reinterpret_cast<T*>(ptr); }

		//
		core::smart_refctd_ptr<CCUDAHandler> create(system::ISystem* system, core::smart_refctd_ptr<system::ILogger>&& _logger);

		//
		using LibLoader = system::DefaultFuncPtrLoader;
		NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CUDA,LibLoader
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
		);
		const CUDA& getCUDAFunctionTable() const {return m_cuda;}

		NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(NVRTC,LibLoader,
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
		const NVRTC& getNVRTCFunctionTable() const {return m_nvrtc;}

#if 0
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
#endif	
		//
		inline nvrtcResult compileProgram(nvrtcProgram prog, const size_t optionCount, const char* const* options)
		{
			return m_nvrtc.pnvrtcCompileProgram(prog, optionCount, options);
		}
		template<typename OptionsT = const std::initializer_list<const char*>&>
		inline nvrtcResult compileProgram(nvrtcProgram prog, OptionsT options)
		{
			return compileProgram(prog, options.size(), options.begin());
		}
		inline nvrtcResult compileProgram(nvrtcProgram prog, const std::vector<const char*>& options)
		{
			return compileProgram(prog, options.size(), options.data());
		}

		//
		nvrtcResult getProgramLog(nvrtcProgram prog, std::string& log);

		//
		std::pair<core::smart_refctd_ptr<asset::ICPUBuffer>,nvrtcResult> getPTX(nvrtcProgram prog);

#if 0
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
#endif

		core::smart_refctd_ptr<CCUDADevice> createDevice(core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, IPhysicalDevice* physicalDevice);

	protected:
		CCUDAHandler(
			CUDA&& _cuda,
			NVRTC&& _nvrtc,
			core::vector<core::smart_refctd_ptr<system::IFile>>&& _headers,
			core::smart_refctd_ptr<system::ILogger>&& _logger,
			int _version
		);
		~CCUDAHandler() = default;

#if 0
		static core::vector<const char*> headerContents;
		static core::vector<const char*> headerNames;

#ifdef _MSC_VER
		_NBL_STATIC_INLINE_CONSTEXPR const char* CUDA_EXTRA_DEFINES = "#ifndef _WIN64\n#define _WIN64\n#endif\n";
#else
		_NBL_STATIC_INLINE_CONSTEXPR const char* CUDA_EXTRA_DEFINES = "#ifndef __LP64__\n#define __LP64__\n#endif\n";
#endif

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
#endif
		// function tables
		CUDA m_cuda;
		NVRTC m_nvrtc;

		//
		core::vector<core::smart_refctd_ptr<system::IFile>> m_headers;
		system::logger_opt_smart_ptr m_logger;
		int m_version;
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
