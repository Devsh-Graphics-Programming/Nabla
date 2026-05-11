// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_HANDLER_H_
#define _NBL_VIDEO_C_CUDA_HANDLER_H_

#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/declarations.h"
#include "nbl/system/path.h"
#include "nbl/video/CUDAInteropHandles.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace nbl::video
{
class CCUDADevice;
class CVulkanConnection;
class IPhysicalDevice;

namespace cuda_native
{
// SDK-free forward declarations for the dynamic CUDA/NVRTC tables exposed by the opt-in native header.
class CUDA;
class NVRTC;
}

namespace cuda_interop
{
inline constexpr const char* RuntimePathsFileName = "nbl_cuda_interop_runtime.json";
inline constexpr uint32_t RuntimeVersionComponentCount = 2u;
using SRuntimeVersion = std::array<int,RuntimeVersionComponentCount>;

struct SRuntimeIncludeDir
{
	system::path path;
	std::string source;
	uint32_t cudaVersion = 0u;
	bool completeRuntimeHeaderSet = false;
};

struct SRuntimeCompileEnvironment
{
	core::vector<system::path> includeDirs;
	core::vector<SRuntimeIncludeDir> includeDirInfos;
};

NBL_API2 SRuntimeCompileEnvironment findRuntimeCompileEnvironment();
NBL_API2 SRuntimeCompileEnvironment findRuntimeCompileEnvironment(const core::vector<system::path>& explicitIncludeDirs);
NBL_API2 SRuntimeCompileEnvironment findRuntimeCompileEnvironment(const core::vector<system::path>& explicitIncludeDirs, const core::vector<system::path>& runtimePathFiles);
inline core::vector<std::string> makeNVRTCIncludeOptions(const SRuntimeCompileEnvironment& environment)
{
	core::vector<std::string> options;
	for (const auto& includeDir : environment.includeDirs)
		options.push_back("-I" + includeDir.generic_string());
	return options;
}
}

class NBL_API2 CCUDAHandler : public core::IReferenceCounted
{
	public:
		static core::smart_refctd_ptr<CCUDAHandler> create(system::ISystem* system, core::smart_refctd_ptr<system::ILogger>&& _logger);
		static uint32_t getBuildCUDASDKVersion();
		uint32_t getLoadedCUDADriverVersion() const;
		cuda_interop::SRuntimeVersion getLoadedNVRTCVersion() const;
		const cuda_native::CUDA& getCUDAFunctionTable() const;
		const cuda_native::NVRTC& getNVRTCFunctionTable() const;
		core::SRange<const char* const> getDefaultRuntimeIncludeOptions() const;
		inline system::logger_opt_ptr getLogger() const { return m_logger.getOptRawPtr(); }

		struct SPTXResult
		{
			core::smart_refctd_ptr<asset::ICPUBuffer> ptx;
			cuda_interop::SNVRTCResult result;
		};

		static bool defaultHandleResult(cuda_interop::SCUresult result, const system::logger_opt_ptr& logger);
		bool defaultHandleResult(cuda_interop::SCUresult result) const;
		bool defaultHandleResult(cuda_interop::SNVRTCResult result) const;

		cuda_interop::SNVRTCResult createProgram(cuda_interop::SOutput<cuda_interop::SNVRTCProgram> prog, std::string&& source, const char* name, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr);
		cuda_interop::SNVRTCResult compileProgram(cuda_interop::SNVRTCProgram prog, core::SRange<const char* const> options) const;
		cuda_interop::SNVRTCResult getProgramLog(cuda_interop::SNVRTCProgram prog, std::string& log) const;
		SPTXResult getPTX(cuda_interop::SNVRTCProgram prog) const;
		SPTXResult compileDirectlyToPTX(
			std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
			std::string& log, const int headerCount=0, const char* const* headerContents=nullptr, const char* const* includeNames=nullptr
		);

		inline core::SRange<system::IFile* const> getSTDHeaders()
		{
			auto begin = m_headers.empty() ? nullptr:(&m_headers[0].get());
			return {begin,begin+m_headers.size()};
		}
		inline const auto& getSTDHeaderContents() { return m_headerContents; }
		inline const auto& getSTDHeaderNames() { return m_headerNames; }

		struct SCUDADeviceInfo
		{
			std::array<uint8_t,16> uuid = {};
		};

		inline core::vector<SCUDADeviceInfo> const& getAvailableDevices() const
		{
			return m_availableDevices;
		}

		core::smart_refctd_ptr<CCUDADevice> createDevice(core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, IPhysicalDevice* physicalDevice);

	protected:
		~CCUDAHandler() override;

	private:
		struct SNativeState;
		CCUDAHandler(std::unique_ptr<SNativeState>&& nativeState, core::vector<core::smart_refctd_ptr<system::IFile>>&& _headers, core::smart_refctd_ptr<system::ILogger>&& _logger);

		std::unique_ptr<SNativeState> m_native;
		core::vector<SCUDADeviceInfo> m_availableDevices;
		core::vector<core::smart_refctd_ptr<system::IFile>> m_headers;
		core::vector<const char*> m_headerContents;
		core::vector<std::string> m_headerNamesStorage;
		core::vector<const char*> m_headerNames;
		system::logger_opt_smart_ptr m_logger;
};

}

#endif
