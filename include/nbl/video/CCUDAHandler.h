// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_CUDA_HANDLER_H_
#define _NBL_VIDEO_C_CUDA_HANDLER_H_

#include "nbl/core/declarations.h"
#include "nbl/core/definitions.h"

#include "nbl/system/declarations.h"
#include "nbl/system/path.h"

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
struct SAccess;
}

namespace cuda_interop
{
inline constexpr const char* RuntimePathsFileName = "nbl_cuda_interop_runtime.json";

struct SRuntimeCompileEnvironment
{
	core::vector<system::path> includeDirs;
};

NBL_API2 SRuntimeCompileEnvironment findRuntimeCompileEnvironment(core::vector<system::path> explicitIncludeDirs = {});
NBL_API2 SRuntimeCompileEnvironment findRuntimeCompileEnvironment(core::vector<system::path> explicitIncludeDirs, core::vector<system::path> runtimePathFiles);
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
		friend struct cuda_native::SAccess;

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
