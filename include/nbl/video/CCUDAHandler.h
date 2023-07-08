// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_HANDLER_H_
#define _NBL_VIDEO_C_CUDA_HANDLER_H_

#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"

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
		static core::smart_refctd_ptr<CCUDAHandler> create(system::ISystem* system, core::smart_refctd_ptr<system::ILogger>&& _logger);

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

		//
		inline core::SRange<system::IFile* const> getSTDHeaders()
		{
			auto begin = m_headers.empty() ? nullptr:(&m_headers[0].get());
			return {begin,begin+m_headers.size()};
		}
		inline const auto& getSTDHeaderContents() { return m_headerContents; }
		inline const auto& getSTDHeaderNames() { return m_headerNames; }

		//
		nvrtcResult createProgram(nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
		inline nvrtcResult createProgram(nvrtcProgram* prog, const char* source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
		{
			return createProgram(prog,std::string(source),name,headerCount,headerContents,includeNames);
		}
		inline nvrtcResult createProgram(nvrtcProgram* prog, system::IFile* file, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr)
		{
			const auto filesize = file->getSize();
			std::string source(filesize+1u,'0');

			system::IFile::success_t bytesRead;
			file->read(bytesRead,source.data(),0u,file->getSize());
			source.resize(bytesRead.getBytesProcessed());

			return createProgram(prog,std::move(source),file->getFileName().string().c_str(),headerCount,headerContents,includeNames);
		}

		//
		inline nvrtcResult compileProgram(nvrtcProgram prog, core::SRange<const char* const> options)
		{
			return m_nvrtc.pnvrtcCompileProgram(prog,options.size(),options.begin());
		}

		//
		nvrtcResult getProgramLog(nvrtcProgram prog, std::string& log);

		//
		struct ptx_and_nvrtcResult_t
		{
			core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
			nvrtcResult result;
		};
		ptx_and_nvrtcResult_t getPTX(nvrtcProgram prog);

		//
		inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
			std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
			const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
			std::string* log=nullptr
		)
		{
			nvrtcProgram program = nullptr;
			nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
			auto cleanup = core::makeRAIIExiter([&]() -> void
			{
				if (result!=NVRTC_SUCCESS && program)
					m_nvrtc.pnvrtcDestroyProgram(&program); // TODO: do we need to destroy the program if we successfully get PTX?
			});

			result = createProgram(&program,std::move(source),filename,headerCount,headerContents,includeNames);
			return compileDirectlyToPTX_impl(result,program,nvrtcOptions,log);
		}
		inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
			const char* source, const char* filename, core::SRange<const char* const> nvrtcOptions,
			const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
			std::string* log=nullptr
		)
		{
			return compileDirectlyToPTX(std::string(source),filename,nvrtcOptions,headerCount,headerContents,includeNames,log);
		}
		inline ptx_and_nvrtcResult_t compileDirectlyToPTX(
			system::IFile* file, core::SRange<const char* const> nvrtcOptions,
			const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr,
			std::string* log=nullptr
		)
		{
			nvrtcProgram program = nullptr;
			nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
			auto cleanup = core::makeRAIIExiter([&]() -> void
			{
				if (result!=NVRTC_SUCCESS && program)
					m_nvrtc.pnvrtcDestroyProgram(&program); // TODO: do we need to destroy the program if we successfully get PTX?
			});

			result = createProgram(&program,file,headerCount,headerContents,includeNames);
			return compileDirectlyToPTX_impl(result,program,nvrtcOptions,log);
		}

		core::smart_refctd_ptr<CCUDADevice> createDevice(core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, IPhysicalDevice* physicalDevice);
protected:
		CCUDAHandler(CUDA&& _cuda, NVRTC&& _nvrtc, core::vector<core::smart_refctd_ptr<system::IFile>>&& _headers, core::smart_refctd_ptr<system::ILogger>&& _logger, int _version)
			: m_cuda(std::move(_cuda)), m_nvrtc(std::move(_nvrtc)), m_headers(std::move(_headers)), m_logger(std::move(_logger)), m_version(_version)
		{
			for (auto& header : m_headers)
			{
				m_headerContents.push_back(reinterpret_cast<const char*>(header->getMappedPointer()));
				m_headerNamesStorage.push_back(header->getFileName().string());
				m_headerNames.push_back(m_headerNamesStorage.back().c_str());
			}
		}
		~CCUDAHandler() = default;


		//
		inline ptx_and_nvrtcResult_t compileDirectlyToPTX_impl(nvrtcResult result, nvrtcProgram program, core::SRange<const char* const> nvrtcOptions, std::string* log)
		{
			if (result!=NVRTC_SUCCESS)
				return {nullptr,result};

			result = compileProgram(program,nvrtcOptions);
			if (log)
				getProgramLog(program,*log);
			if (result!=NVRTC_SUCCESS)
				return {nullptr,result};
			
			return getPTX(program);
		}

		// function tables
		CUDA m_cuda;
		NVRTC m_nvrtc;

		//
		core::vector<core::smart_refctd_ptr<system::IFile>> m_headers;
		core::vector<const char*> m_headerContents;
		core::vector<std::string> m_headerNamesStorage;
		core::vector<const char*> m_headerNames;
		system::logger_opt_smart_ptr m_logger;
		int m_version;
};

}

#endif // _NBL_COMPILE_WITH_CUDA_

#endif
