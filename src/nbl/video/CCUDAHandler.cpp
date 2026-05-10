// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CUDAInterop.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>

namespace nbl::video::cuda_interop
{
namespace
{

#if defined(_NBL_PLATFORM_WINDOWS_)
inline constexpr char EnvironmentPathListSeparator = ';';
#else
inline constexpr char EnvironmentPathListSeparator = ':';
#endif

std::string readEnvironmentVariable(const char* name)
{
	if (const char* value = std::getenv(name))
		return value;
	return {};
}

bool isDirectory(const system::path& path)
{
	std::error_code error;
	return std::filesystem::exists(path,error) && std::filesystem::is_directory(path,error);
}

bool isRegularFile(const system::path& path)
{
	std::error_code error;
	return std::filesystem::exists(path,error) && std::filesystem::is_regular_file(path,error);
}

system::path normalizedAbsolute(system::path path)
{
	std::error_code error;
	auto absolute = std::filesystem::absolute(path,error);
	if (error)
		absolute = std::move(path);
	return absolute.lexically_normal();
}

bool looksLikeCUDAIncludeDir(const system::path& path)
{
	if (!isDirectory(path))
		return false;

	return isRegularFile(path/"cuda_fp16.h") ||
		isRegularFile(path/"cuda_runtime_api.h") ||
		isRegularFile(path/"vector_types.h") ||
		isRegularFile(path/"cuda.h") ||
		isRegularFile(path/"nv"/"target");
}

uint32_t readCUDAVersion(const system::path& includeDir)
{
	std::ifstream input(includeDir/"cuda.h");
	if (!input)
		return 0u;

	std::string line;
	while (std::getline(input,line))
	{
		std::istringstream stream(line);
		std::string directive;
		stream >> directive;
		if (directive!="#define")
			continue;

		std::string name;
		stream >> name;
		if (name!="CUDA_VERSION")
			continue;

		uint32_t version = 0u;
		if (stream >> version)
			return version;
	}
	return 0u;
}

bool looksLikeCompleteRuntimeHeaderSet(const system::path& includeDir)
{
	return isRegularFile(includeDir/"cuda.h") &&
		isRegularFile(includeDir/"cuda_runtime_api.h") &&
		isRegularFile(includeDir/"vector_types.h");
}

void appendIncludeDir(SRuntimeCompileEnvironment& environment, system::path path, std::string source)
{
	if (path.empty() || !looksLikeCUDAIncludeDir(path))
		return;

	path = normalizedAbsolute(std::move(path));
	const auto pathString = path.generic_string();
	const auto alreadyAdded = std::find_if(environment.includeDirs.begin(),environment.includeDirs.end(),[&](const system::path& existing) {
		return existing.generic_string()==pathString;
	});
	if (alreadyAdded==environment.includeDirs.end())
	{
		SRuntimeIncludeDir info;
		info.path = path;
		info.source = std::move(source);
		info.cudaVersion = readCUDAVersion(path);
		info.completeRuntimeHeaderSet = looksLikeCompleteRuntimeHeaderSet(path);

		environment.includeDirs.push_back(std::move(path));
		environment.includeDirInfos.push_back(std::move(info));
	}
}

void appendCUDAIncludeDirsBelow(SRuntimeCompileEnvironment& environment, const system::path& root, uint32_t maxDepth, std::string source)
{
	if (!isDirectory(root))
		return;

	if (looksLikeCUDAIncludeDir(root))
	{
		appendIncludeDir(environment,root,std::move(source));
		return;
	}
	if (maxDepth==0u)
		return;

	core::vector<system::path> candidates;
	std::error_code error;
	for (const auto& entry : std::filesystem::directory_iterator(root,error))
	{
		if (error)
			break;

		std::error_code entryError;
		if (!entry.is_directory(entryError))
			continue;
		candidates.push_back(entry.path());
	}

	std::sort(candidates.begin(),candidates.end(),[](const system::path& lhs, const system::path& rhs) {
		return lhs.generic_string()>rhs.generic_string();
	});
	for (const auto& candidate : candidates)
		appendCUDAIncludeDirsBelow(environment,candidate,maxDepth-1u,source);
}

void appendCUDAIncludeRoot(SRuntimeCompileEnvironment& environment, const system::path& root, std::string source)
{
	if (root.empty())
		return;

	appendIncludeDir(environment,root,source);
	appendIncludeDir(environment,root/"include",std::move(source));
}

void appendRuntimePathsConfig(SRuntimeCompileEnvironment& environment, const system::path& configFile, const char* source)
{
	if (!isRegularFile(configFile))
		return;

	std::ifstream input(configFile);
	if (!input)
		return;

	const auto json = nlohmann::json::parse(input,nullptr,false);
	if (json.is_discarded())
		return;

	const auto paths = json.find("cudaRuntimeIncludeDirs");
	if (paths==json.end() || !paths->is_array())
		return;

	for (const auto& path : *paths)
		if (path.is_string())
			appendIncludeDir(environment,system::path(path.get<std::string>()),std::string(source)+": "+configFile.generic_string());
}

template<typename Append>
void appendPathListEnv(const char* name, Append append)
{
	const auto value = readEnvironmentVariable(name);
	if (value.empty())
		return;

	size_t begin = 0;
	while (begin<value.size())
	{
		const auto end = value.find(EnvironmentPathListSeparator,begin);
		const auto segment = value.substr(begin,end==std::string::npos ? std::string::npos:end-begin);
		if (!segment.empty())
			append(system::path(segment));
		if (end==std::string::npos)
			break;
		begin = end+1;
	}
}

void appendRuntimePathsConfigs(SRuntimeCompileEnvironment& environment, const core::vector<system::path>& explicitRuntimePathFiles)
{
	for (const auto& runtimePathFile : explicitRuntimePathFiles)
		appendRuntimePathsConfig(environment,runtimePathFile,"explicit runtime JSON");

	appendPathListEnv("NBL_CUDA_INTEROP_RUNTIME_JSON",[&](const system::path& path) {
		appendRuntimePathsConfig(environment,path,"NBL_CUDA_INTEROP_RUNTIME_JSON");
	});
	appendPathListEnv("Nabla_CUDA_INTEROP_RUNTIME_JSON",[&](const system::path& path) {
		appendRuntimePathsConfig(environment,path,"Nabla_CUDA_INTEROP_RUNTIME_JSON");
	});

	const auto exeDir = system::executableDirectory();
	if (!exeDir.empty())
		appendRuntimePathsConfig(environment,exeDir/RuntimePathsFileName,"executable-local runtime JSON");
}

void appendAppLocalIncludeDirs(SRuntimeCompileEnvironment& environment)
{
	const auto exeDir = system::executableDirectory();
	if (exeDir.empty())
		return;

	appendIncludeDir(environment,exeDir/"cuda"/"include","app-local cuda/include");
	appendCUDAIncludeDirsBelow(environment,exeDir/"nvidia",4u,"app-local nvidia package");
	appendIncludeDir(environment,exeDir/"Libraries"/"cuda"/"include","app-local Libraries/cuda/include");
	appendIncludeDir(environment,exeDir.parent_path()/"cuda"/"include","parent app-local cuda/include");
	appendCUDAIncludeDirsBelow(environment,exeDir.parent_path()/"nvidia",4u,"parent app-local nvidia package");
}

void appendPythonPackageIncludeDirs(SRuntimeCompileEnvironment& environment, const system::path& root, const char* source)
{
	if (root.empty())
		return;

	appendCUDAIncludeDirsBelow(environment,root/"Lib"/"site-packages"/"nvidia",4u,std::string(source)+" Python nvidia package");
	appendCUDAIncludeDirsBelow(environment,root/"lib"/"site-packages"/"nvidia",4u,std::string(source)+" Python nvidia package");
	appendIncludeDir(environment,root/"Library"/"include",std::string(source)+" Library/include");
	appendIncludeDir(environment,root/"include",std::string(source)+" include");
}

void appendEnvironmentIncludeDirs(SRuntimeCompileEnvironment& environment)
{
	appendPathListEnv("NBL_CUDA_RUNTIME_INCLUDE_DIRS",[&](const system::path& path) {
		appendIncludeDir(environment,path,"NBL_CUDA_RUNTIME_INCLUDE_DIRS");
	});
	appendPathListEnv("Nabla_CUDA_RUNTIME_INCLUDE_DIRS",[&](const system::path& path) {
		appendIncludeDir(environment,path,"Nabla_CUDA_RUNTIME_INCLUDE_DIRS");
	});

	appendCUDAIncludeRoot(environment,readEnvironmentVariable("CUDA_PATH"),"CUDA_PATH");
	appendCUDAIncludeRoot(environment,readEnvironmentVariable("CUDA_HOME"),"CUDA_HOME");
	appendCUDAIncludeRoot(environment,readEnvironmentVariable("CUDA_ROOT"),"CUDA_ROOT");
	appendCUDAIncludeRoot(environment,readEnvironmentVariable("CUDAToolkit_ROOT"),"CUDAToolkit_ROOT");

	appendPythonPackageIncludeDirs(environment,readEnvironmentVariable("VIRTUAL_ENV"),"VIRTUAL_ENV");
	appendPythonPackageIncludeDirs(environment,readEnvironmentVariable("CONDA_PREFIX"),"CONDA_PREFIX");
}

void appendCUDAInstallRoots(SRuntimeCompileEnvironment& environment, const system::path& root, const char* source)
{
	if (!isDirectory(root))
		return;

	core::vector<system::path> candidates;
	std::error_code error;
	for (const auto& entry : std::filesystem::directory_iterator(root,error))
	{
		if (error)
			break;
		if (!entry.is_directory(error))
			continue;
		candidates.push_back(entry.path()/"include");
	}

	std::sort(candidates.begin(),candidates.end(),[](const system::path& lhs, const system::path& rhs) {
		return lhs.generic_string()>rhs.generic_string();
	});
	for (const auto& candidate : candidates)
		appendIncludeDir(environment,candidate,source);
}

void appendSystemIncludeDirs(SRuntimeCompileEnvironment& environment)
{
	#if defined(_NBL_PLATFORM_WINDOWS_)
	appendCUDAInstallRoots(environment,"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA","system CUDA Toolkit install root");
	#else
	appendIncludeDir(environment,"/usr/local/cuda/include","system /usr/local/cuda");
	appendCUDAInstallRoots(environment,"/usr/local","system /usr/local CUDA install root");
	appendIncludeDir(environment,"/usr/include","system /usr/include");
	#endif
}

}

SRuntimeCompileEnvironment findRuntimeCompileEnvironment(core::vector<system::path> explicitIncludeDirs, core::vector<system::path> runtimePathFiles)
{
	SRuntimeCompileEnvironment environment;

	/*
		Runtime header discovery builds the ordered include list passed to NVRTC. It is not a lock to the CUDA SDK
		used to build Nabla. A packaged Nabla must stay relocatable, so host-specific include paths are accepted
		only when the application provides them at runtime: direct arguments, JSON next to the executable, an
		override JSON, app-local header bundles, environment variables, or finally common toolkit install roots.

		The first root containing a requested header wins exactly like normal C/C++ include search. Keep every
		accepted root with its source and parsed CUDA_VERSION so startup logs can explain what NVRTC will see.
		This is also why mismatched or partial roots produce diagnostics instead of changing discovery order or
		hard-failing before the user kernel is compiled.
	*/
	for (auto& includeDir : explicitIncludeDirs)
		appendIncludeDir(environment,std::move(includeDir),"explicit include dir");

	appendRuntimePathsConfigs(environment,runtimePathFiles);
	appendAppLocalIncludeDirs(environment);
	appendEnvironmentIncludeDirs(environment);
	appendSystemIncludeDirs(environment);

	return environment;
}

SRuntimeCompileEnvironment findRuntimeCompileEnvironment(core::vector<system::path> explicitIncludeDirs)
{
	return findRuntimeCompileEnvironment(std::move(explicitIncludeDirs),{});
}

}

#ifdef _NBL_COMPILE_WITH_CUDA_
#include "CUDAInteropNativeState.hpp"
#include "nbl/system/CFileView.h"
#include "jitify/jitify.hpp"


namespace nbl::video
{

namespace
{

int cudaVersionMajor(int version)
{
	return version/1000;
}

int cudaVersionMinor(int version)
{
	return (version%1000)/10;
}

int cudaVersionCode(int major, int minor)
{
	return major*1000+minor*10;
}

system::path loadedRuntimeModulePath(const char* moduleName)
{
	#if defined(_NBL_PLATFORM_WINDOWS_)
	const auto moduleDir = system::loadedModuleDirectory(moduleName);
	if (moduleDir.empty())
		return {};
	return moduleDir/(std::string(moduleName)+".dll");
	#else
	return {};
	#endif
}

std::string cudaVersionString(int version)
{
	std::ostringstream stream;
	stream << cudaVersionMajor(version) << "." << cudaVersionMinor(version);
	return stream.str();
}

std::string cudaVersionString(const std::array<int,2>& version)
{
	std::ostringstream stream;
	stream << version[0] << "." << version[1];
	return stream.str();
}

std::string runtimeIncludeDirDescription(const cuda_interop::SRuntimeIncludeDir& includeDir)
{
	std::ostringstream stream;
	stream << includeDir.path.generic_string() << " (" << includeDir.source;
	if (includeDir.cudaVersion!=0u)
		stream << ", CUDA_VERSION " << includeDir.cudaVersion << " / " << cudaVersionString(includeDir.cudaVersion);
	else
		stream << ", CUDA_VERSION unknown";
	if (!includeDir.completeRuntimeHeaderSet)
		stream << ", partial header root";
	stream << ")";
	return stream.str();
}

std::string cudaRuntimeReport(
	const int buildVersion, const int cudaDriverVersion, const system::path& cudaDriverPath,
	const std::array<int,2>& nvrtcVersion, const std::string& nvrtcLibraryName, const system::path& nvrtcPath,
	const cuda_interop::SRuntimeCompileEnvironment& runtimeEnvironment)
{
	std::ostringstream stream;
	stream << "CCUDAHandler: CUDA interop runtime report:\n";
	stream << "  - Nabla build CUDA SDK: " << cudaVersionString(buildVersion) << "\n";
	stream << "  - CUDA Driver API: " << cudaVersionString(cudaDriverVersion);
	if (!cudaDriverPath.empty())
		stream << " (" << cudaDriverPath.generic_string() << ")";
	stream << "\n";
	stream << "  - NVRTC runtime: " << cudaVersionString(nvrtcVersion) << " (" << nvrtcLibraryName;
	if (!nvrtcPath.empty())
		stream << ", " << nvrtcPath.generic_string();
	stream << ")\n";

	if (runtimeEnvironment.includeDirs.empty())
	{
		stream << "  - NVRTC runtime header search path: none discovered";
	}
	else
	{
		stream << "  - Primary NVRTC runtime header path: " << runtimeIncludeDirDescription(runtimeEnvironment.includeDirInfos.front()) << "\n";
		stream << "  - NVRTC runtime header search order (first path containing the requested header wins):\n";
		for (const auto& includeDir : runtimeEnvironment.includeDirInfos)
			stream << "    - " << runtimeIncludeDirDescription(includeDir) << "\n";
	}
	return stream.str();
}

}
	
CCUDAHandler::CCUDAHandler(
	std::unique_ptr<SNativeState>&& nativeState,
	core::vector<core::smart_refctd_ptr<system::IFile>>&& _headers, 
	core::smart_refctd_ptr<system::ILogger>&& _logger)
	: m_native(std::move(nativeState))
	, m_headers(std::move(_headers))
	, m_logger(std::move(_logger))
{
	assert(m_native);

	for (auto& header : m_headers)
	{
		m_headerContents.push_back(reinterpret_cast<const char*>(header->getMappedPointer()));
		m_headerNamesStorage.push_back(header->getFileName().string());
		m_headerNames.push_back(m_headerNamesStorage.back().c_str());
	}
	for (const auto& option : m_native->runtimeIncludeOptions)
		m_native->runtimeIncludeOptionPtrs.push_back(option.c_str());

	int deviceCount = 0;
	if (m_native->cuda.pcuDeviceGetCount(&deviceCount) != CUDA_SUCCESS || deviceCount <= 0)
		return;

	for (int device_i = 0; device_i < deviceCount; device_i++)
	{
		CUdevice handle = -1;
		if (m_native->cuda.pcuDeviceGet(&handle, device_i) != CUDA_SUCCESS || handle < 0)
			continue;

		CUuuid uuid = {};
		if (m_native->cuda.pcuDeviceGetUuid_v2(&uuid, handle) != CUDA_SUCCESS)
			continue;

		auto& nativeDevice = m_native->deviceStates.emplace_back();
		nativeDevice.handle = handle;
		nativeDevice.uuid = uuid;
		auto& cleanDevice = m_availableDevices.emplace_back();
		memcpy(cleanDevice.uuid.data(),&uuid,cleanDevice.uuid.size());

		for (size_t i = 0; i < nativeDevice.attributes.size(); i++)
			m_native->cuda.pcuDeviceGetAttribute(&nativeDevice.attributes[i], static_cast<CUdevice_attribute>(i), handle);

	}
}

CCUDAHandler::~CCUDAHandler() = default;

uint32_t CCUDAHandler::getBuildCUDASDKVersion()
{
	return CUDA_VERSION;
}

uint32_t CCUDAHandler::getLoadedCUDADriverVersion() const
{
	return m_native->cudaDriverVersion;
}

std::array<int,2> CCUDAHandler::getLoadedNVRTCVersion() const
{
	return m_native->nvrtcVersion;
}

const cuda_native::CUDA& CCUDAHandler::getCUDAFunctionTable() const
{
	return m_native->cuda;
}

const cuda_native::NVRTC& CCUDAHandler::getNVRTCFunctionTable() const
{
	return m_native->nvrtc;
}

core::SRange<const char* const> CCUDAHandler::getDefaultRuntimeIncludeOptions() const
{
	if (m_native->runtimeIncludeOptionPtrs.empty())
		return {nullptr,nullptr};
	const auto* begin = m_native->runtimeIncludeOptionPtrs.data();
	return {begin,begin+m_native->runtimeIncludeOptionPtrs.size()};
}

namespace cuda_native
{

bool defaultHandleResult(CUresult result, const system::logger_opt_ptr& logger)
{
	switch (result)
	{
		case CUDA_SUCCESS:
			return true;
			break;
		case CUDA_ERROR_INVALID_VALUE:
			logger.log(R"===(CCUDAHandler:
				This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_OUT_OF_MEMORY:
			logger.log(R"===(CCUDAHandler:
				The API call failed because it was unable to allocate enough memory to perform the requested operation.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_INITIALIZED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUDA driver has not been initialized with cuInit() or that initialization has failed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_DEINITIALIZED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUDA driver is in the process of shutting down.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PROFILER_DISABLED:
			logger.log(R"===(CCUDAHandler:
				This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NO_DEVICE:
			logger.log(R"===(CCUDAHandler:
				This indicates that no CUDA-capable devices were detected by the installed CUDA driver. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_DEVICE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_IMAGE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_CONTEXT:
			logger.log(R"===(CCUDAHandler:
				This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See cuCtxGetApiVersion() for more details.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_MAP_FAILED:
			logger.log(R"===(CCUDAHandler:
				This indicates that a map or register operation has failed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNMAP_FAILED:
			logger.log(R"===(CCUDAHandler:
				This indicates that an unmap or unregister operation has failed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the specified array is currently mapped and thus cannot be destroyed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ALREADY_MAPPED:
			logger.log(R"===(CCUDAHandler:
				This indicates that the resource is already mapped.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			logger.log(R"===(CCUDAHandler:
				This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ALREADY_ACQUIRED:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource has already been acquired. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_MAPPED:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource is not mapped.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			logger.log(R"===(CCUDAHandler:
				This indicates that a mapped resource is not available for access as an array. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			logger.log(R"===(CCUDAHandler:
				This indicates that a mapped resource is not available for access as a pointer. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			logger.log(R"===(CCUDAHandler:
				This indicates that an uncorrectable ECC error was detected during execution. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUlimit passed to the API call is not supported by the active device. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
			logger.log(R"===(CCUDAHandler:
				This indicates that peer access is not supported across the given devices. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_PTX:
			logger.log(R"===(CCUDAHandler:
				This indicates that a PTX JIT compilation failed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
			logger.log(R"===(CCUDAHandler:
				This indicates that the PTX version is unsupported by the CUDA driver. Check that the CUDA driver runtime can consume PTX produced by the loaded NVRTC runtime.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
			logger.log(R"===(CCUDAHandler:
				This indicates an error with OpenGL or DirectX context. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NVLINK_UNCORRECTABLE:
			logger.log(R"===(CCUDAHandler:
				This indicates that an uncorrectable NVLink error was detected during the execution. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that the PTX JIT compiler library was not found. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_SOURCE:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device kernel source is invalid. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_FILE_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that the file specified was not found. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that a link to a shared object failed to resolve.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			logger.log(R"===(CCUDAHandler:
				This indicates that initialization of a shared object failed.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_OPERATING_SYSTEM:
			logger.log(R"===(CCUDAHandler:
				This indicates that an OS call failed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_HANDLE:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like CUstream and CUevent. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ILLEGAL_STATE:
			logger.log(R"===(CCUDAHandler:
				This indicates that a resource required by the API call is not in a valid state to perform the requested operation. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_FOUND:
			logger.log(R"===(CCUDAHandler:
				This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_READY:
			logger.log(R"===(CCUDAHandler:
				This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than CUDA_SUCCESS (which indicates completion). Calls that may return this value include cuEventQuery() and cuStreamQuery().
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ILLEGAL_ADDRESS:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			logger.log(R"===(CCUDAHandler:
				This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			logger.log(R"===(CCUDAHandler:
				This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			logger.log(R"===(CCUDAHandler:
				This error indicates a kernel launch that uses an incompatible texturing mode. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that a call to cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via cuCtxEnablePeerAccess(). 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the primary context for the specified device has already been initialized. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the context current to the calling thread has been destroyed using cuCtxDestroy, or is a primary context which has not yet been initialized. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ASSERT:
			logger.log(R"===(CCUDAHandler:
				A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_TOO_MANY_PEERS:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to cuCtxEnablePeerAccess(). 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the memory range passed to cuMemHostRegister() has already been registered. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the pointer passed to cuMemHostUnregister() does not correspond to any currently registered memory region. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_HARDWARE_STACK_ERROR:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_ILLEGAL_INSTRUCTION:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_MISALIGNED_ADDRESS:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_INVALID_ADDRESS_SPACE:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;				
		case CUDA_ERROR_INVALID_PC:
			logger.log(R"===(CCUDAHandler:
				While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.
			)===",system::ILogger::ELL_ERROR);
			break;				
		case CUDA_ERROR_LAUNCH_FAILED:
			logger.log(R"===(CCUDAHandler:
				An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the number of blocks launched per grid for a kernel that was launched via either cuLaunchCooperativeKernel or cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by cuOccupancyMaxActiveBlocksPerMultiprocessor or cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
			)===",system::ILogger::ELL_ERROR);
			break;			
		case CUDA_ERROR_NOT_PERMITTED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the attempted operation is not permitted.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_NOT_SUPPORTED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the attempted operation is not supported on the current system or device.
			)===",system::ILogger::ELL_ERROR);
			break;				
		case CUDA_ERROR_SYSTEM_NOT_READY:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the system is not yet ready to start any CUDA work. To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
			logger.log(R"===(CCUDAHandler:
				This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the operation is not permitted when the stream is capturing. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the current capture sequence on the stream has been invalidated due to a previous error. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_MERGE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the operation would have resulted in a merge of two independent capture sequences. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the capture was not initiated in this stream. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the capture sequence contains a fork that was not joined to the primary stream.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
			logger.log(R"===(CCUDAHandler:
				This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
			logger.log(R"===(CCUDAHandler:
				This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_CAPTURED_EVENT:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
			logger.log(R"===(CCUDAHandler:
				A stream capture sequence not initiated with the CU_STREAM_CAPTURE_MODE_RELAXED argument to cuStreamBeginCapture was passed to cuStreamEndCapture in a different thread. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_TIMEOUT:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the timeout specified for the wait operation has lapsed. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
			logger.log(R"===(CCUDAHandler:
				This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update. 
			)===",system::ILogger::ELL_ERROR);
			break;
		case CUDA_ERROR_UNKNOWN:
		default:
			logger.log("CCUDAHandler: Unknown CUDA error code %d.",system::ILogger::ELL_ERROR,static_cast<int>(result));
			break;
	}
	return false;
}

bool defaultHandleResult(const CCUDAHandler& handler, CUresult result)
{
	if (result==CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
	{
		const auto cudaVersion = handler.getLoadedCUDADriverVersion();
		const auto nvrtcVersion = handler.getLoadedNVRTCVersion();
		handler.getLogger().log(
			"CCUDAHandler: CUDA driver API %d.%d rejected PTX produced through NVRTC %d.%d. Install a newer NVIDIA driver or use an NVRTC/runtime-header set compatible with the installed driver.",
			system::ILogger::ELL_ERROR,
			cudaVersionMajor(cudaVersion),cudaVersionMinor(cudaVersion),
			nvrtcVersion[0],nvrtcVersion[1]
		);
	}
	return defaultHandleResult(result,handler.getLogger());
}

bool defaultHandleResult(const CCUDAHandler& handler, nvrtcResult result)
{
	const auto& nvrtc = handler.getNVRTCFunctionTable();
	const auto logger = handler.getLogger();
	switch (result)
	{
		case NVRTC_SUCCESS:
			return true;
			break;
		default:
			if (nvrtc.pnvrtcGetErrorString)
				logger.log("%s\n",system::ILogger::ELL_ERROR,nvrtc.pnvrtcGetErrorString(result));
			else
				logger.log(R"===(CudaHandler: `pnvrtcGetErrorString` is nullptr, the nvrtc library probably not found on the system.\n)===",system::ILogger::ELL_ERROR);
			break;
	}
	return false;
}

}

core::smart_refctd_ptr<CCUDAHandler> CCUDAHandler::create(system::ISystem* system, core::smart_refctd_ptr<system::ILogger>&& _logger)
{
	const system::logger_opt_ptr logger(_logger.get());

	cuda_native::CUDA cuda = cuda_native::CUDA(
		#if defined(_NBL_WINDOWS_API_)
			"nvcuda"
		#elif defined(_NBL_POSIX_API_)
			"cuda"
		#else
			#error "Unsuported Platform"
		#endif
	);

	// need a complex safe calling chain because DLL/SO might not have loaded
	#define SAFE_CUDA_CALL(FUNC,...) \
	{\
		if (!cuda.p ## FUNC)\
		{\
			logger.log("CCUDAHandler: CUDA Driver API function %s was not found. Need CUDA driver runtime %d.%d or newer.",system::ILogger::ELL_ERROR,#FUNC,cudaVersionMajor(cuda_native::MinimumCUDADriverVersion),cudaVersionMinor(cuda_native::MinimumCUDADriverVersion));\
			return nullptr;\
		}\
		auto result = cuda.p ## FUNC(__VA_ARGS__);\
		if (result!=CUDA_SUCCESS)\
		{\
			logger.log("CCUDAHandler: %s failed with CUDA error code %d.",system::ILogger::ELL_ERROR,#FUNC,static_cast<int>(result));\
			return nullptr;\
		}\
	}
	
	SAFE_CUDA_CALL(cuInit,0)
				
	int cudaVersion = 0;
	SAFE_CUDA_CALL(cuDriverGetVersion,&cudaVersion)
	if (cudaVersion<cuda_native::MinimumCUDADriverVersion)
	{
		logger.log(
			"CCUDAHandler: CUDA driver runtime %d.%d is below required %d.%d.",
			system::ILogger::ELL_ERROR,
			cudaVersionMajor(cudaVersion),cudaVersionMinor(cudaVersion),
			cudaVersionMajor(cuda_native::MinimumCUDADriverVersion),cudaVersionMinor(cuda_native::MinimumCUDADriverVersion)
		);
		return nullptr;
	}

	// stop the pollution
	#undef SAFE_CUDA_CALL

	auto readNVRTCVersion = [&](const cuda_native::NVRTC& candidate, std::array<int,2>& version, const char* name) -> bool
	{
		if (!candidate.pnvrtcVersion)
			return false;

		const auto result = candidate.pnvrtcVersion(version.data(),version.data()+1);
		if (result==NVRTC_SUCCESS)
			return true;

		logger.log("CCUDAHandler: nvrtcVersion failed for %s with NVRTC error code %d.",system::ILogger::ELL_WARNING,name,static_cast<int>(result));
		version = {-1,-1};
		return false;
	};

	cuda_native::NVRTC nvrtc = {};
	std::array<int,2> nvrtcVersion = {-1,-1};
	std::string nvrtcLibraryName;

	#if defined(_NBL_WINDOWS_API_)
	cuda_native::NVRTC fallbackNVRTC = {};
	std::array<int,2> fallbackNVRTCVersion = {-1,-1};
	std::string fallbackNVRTCLibraryName;

	/*
		The CUDA driver consumes the final PTX, not the toolkit that provided headers or nvrtc*.dll.
		A real machine can have an older NVIDIA driver and a newer CUDA Toolkit side by side, for example
		driver API 13.1 from nvcuda.dll with CUDA 13.2 Toolkit/NVRTC in CUDA_PATH. In that setup NVRTC can
		emit PTX the installed driver rejects with CUDA_ERROR_UNSUPPORTED_PTX_VERSION. Prefer an NVRTC runtime
		that is not newer than the loaded driver and log the full version matrix when no compatible one exists.
	*/
	const char* nvrtc64_versions[] = {
		"nvrtc64_132",
		"nvrtc64_131",
		"nvrtc64_130",
		nullptr
	};

	const char* nvrtc64_suffices[] = {"","_","_0","_1","_2",nullptr};
	for (auto verpath=nvrtc64_versions; *verpath && !nvrtc.pnvrtcVersion; verpath++)
	{
		for (auto suffix=nvrtc64_suffices; *suffix; suffix++)
		{
			std::string candidateName(*verpath);
			candidateName += *suffix;

			cuda_native::NVRTC candidate(candidateName.c_str());
			std::array<int,2> candidateVersion = {-1,-1};
			if (!readNVRTCVersion(candidate,candidateVersion,candidateName.c_str()))
				continue;

			if (cudaVersionCode(candidateVersion[0],candidateVersion[1])<=cudaVersion)
			{
				nvrtc = std::move(candidate);
				nvrtcVersion = candidateVersion;
				nvrtcLibraryName = std::move(candidateName);
				break;
			}

			if (!fallbackNVRTC.pnvrtcVersion)
			{
				fallbackNVRTC = std::move(candidate);
				fallbackNVRTCVersion = candidateVersion;
				fallbackNVRTCLibraryName = std::move(candidateName);
			}
		}
	}

	if (!nvrtc.pnvrtcVersion && fallbackNVRTC.pnvrtcVersion)
	{
		nvrtc = std::move(fallbackNVRTC);
		nvrtcVersion = fallbackNVRTCVersion;
		nvrtcLibraryName = std::move(fallbackNVRTCLibraryName);
	}
	#elif defined(_NBL_POSIX_API_)
	nvrtcLibraryName = "nvrtc";
	nvrtc = cuda_native::NVRTC(nvrtcLibraryName.c_str());
	readNVRTCVersion(nvrtc,nvrtcVersion,nvrtcLibraryName.c_str());
	#else
	#error "Unsuported Platform"
	#endif

	// check nvrtc existence and compatibility
	if (!nvrtc.pnvrtcVersion)
	{
		logger.log("CCUDAHandler: NVRTC runtime was not found. Need NVRTC %d.x or newer.",system::ILogger::ELL_ERROR,cuda_native::MinimumNVRTCMajorVersion);
		return nullptr;
	}
	if (nvrtcVersion[0]<cuda_native::MinimumNVRTCMajorVersion)
	{
		logger.log(
			"CCUDAHandler: NVRTC runtime %d.%d is below required %d.x.",
			system::ILogger::ELL_ERROR,
			nvrtcVersion[0],nvrtcVersion[1],cuda_native::MinimumNVRTCMajorVersion
		);
		return nullptr;
	}

	const auto buildVersion = CCUDAHandler::getBuildCUDASDKVersion();
	auto runtimeEnvironment = cuda_interop::findRuntimeCompileEnvironment();
	const auto cudaDriverPath = loadedRuntimeModulePath("nvcuda");
	const auto nvrtcPath = loadedRuntimeModulePath(nvrtcLibraryName.c_str());
	const auto report = cudaRuntimeReport(buildVersion,cudaVersion,cudaDriverPath,nvrtcVersion,nvrtcLibraryName,nvrtcPath,runtimeEnvironment);
	logger.log("%s",system::ILogger::ELL_INFO,report.c_str());

	if (cudaVersionCode(nvrtcVersion[0],nvrtcVersion[1])>cudaVersion)
	{
		logger.log(
			"CCUDAHandler: NVRTC runtime %d.%d is newer than CUDA driver API %d.%d. PTX generated by this NVRTC may be unsupported by the installed driver.",
			system::ILogger::ELL_WARNING,
			nvrtcVersion[0],nvrtcVersion[1],
			cudaVersionMajor(cudaVersion),cudaVersionMinor(cudaVersion)
		);
	}
	if (runtimeEnvironment.includeDirs.empty())
	{
		logger.log("CCUDAHandler: no CUDA runtime headers were discovered for NVRTC include paths.",system::ILogger::ELL_WARNING);
	}
	else
	{
		const auto& primaryIncludeDir = runtimeEnvironment.includeDirInfos.front();
		if (!primaryIncludeDir.completeRuntimeHeaderSet)
		{
			logger.log(
				"CCUDAHandler: primary NVRTC runtime header path %s does not contain cuda.h, cuda_runtime_api.h, and vector_types.h together. NVRTC may use later include paths for missing headers.",
				system::ILogger::ELL_WARNING,
				primaryIncludeDir.path.generic_string().c_str()
			);
		}

		const auto nvrtcVersionCode = cudaVersionCode(nvrtcVersion[0],nvrtcVersion[1]);
		if (primaryIncludeDir.cudaVersion!=0u && primaryIncludeDir.cudaVersion!=static_cast<uint32_t>(nvrtcVersionCode))
		{
			logger.log(
				"CCUDAHandler: primary NVRTC runtime headers report CUDA_VERSION %u (%s), while loaded NVRTC is %s. This is allowed by discovery policy, but kernels using version-specific CUDA headers may fail to compile.",
				system::ILogger::ELL_WARNING,
				primaryIncludeDir.cudaVersion,
				cudaVersionString(primaryIncludeDir.cudaVersion).c_str(),
				cudaVersionString(nvrtcVersion).c_str()
			);
		}
	}

	// add headers
	core::vector<core::smart_refctd_ptr<system::IFile>> headers;
	for (const auto& it : jitify::detail::get_jitsafe_headers_map())
	{
		const void* contents = it.second.data();
		headers.push_back(core::make_smart_refctd_ptr<system::CFileView<system::CNullAllocator>>(
			it.first.c_str(),
			core::bitflag(system::IFile::ECF_READ)|system::IFile::ECF_MAPPABLE,
			std::chrono::clock_cast<system::IFile::time_point_t::clock>(std::chrono::system_clock::now()),
			const_cast<void*>(contents),it.second.size()+1u
		));
	}

	return core::smart_refctd_ptr<CCUDAHandler>(
		new CCUDAHandler(std::make_unique<SNativeState>(std::move(cuda),std::move(nvrtc),cudaVersion,nvrtcVersion,std::move(runtimeEnvironment)),std::move(headers),std::move(_logger)),
		core::dont_grab
	);
}

namespace cuda_native
{

nvrtcResult createProgram(CCUDAHandler& handler, nvrtcProgram* prog, std::string&& source, const char* name, const int headerCount, const char* const* headerContents, const char* const* includeNames)
{
#if defined(_NBL_WINDOWS_API_)
	source.insert(0ull,"#ifndef _WIN64\n#define _WIN64\n#endif\n");
#elif defined(_NBL_POSIX_API_)
	source.insert(0ull,"#ifndef __LP64__\n#define __LP64__\n#endif\n");
#else
#error "Unsuported Platform"
#endif
	return handler.getNVRTCFunctionTable().pnvrtcCreateProgram(prog,source.c_str(),name,headerCount,headerContents,includeNames);
}

nvrtcResult compileProgram(const CCUDAHandler& handler, nvrtcProgram prog, core::SRange<const char* const> options)
{
	return handler.getNVRTCFunctionTable().pnvrtcCompileProgram(prog,options.size(),options.begin());
}

nvrtcResult getProgramLog(const CCUDAHandler& handler, nvrtcProgram prog, std::string& log)
{
	size_t _size = 0ull;
	const auto& nvrtc = handler.getNVRTCFunctionTable();
	nvrtcResult sizeRes = nvrtc.pnvrtcGetProgramLogSize(prog, &_size);
	if (sizeRes != NVRTC_SUCCESS)
		return sizeRes;
	if (_size == 0ull)
		return NVRTC_ERROR_INVALID_INPUT;

	log.resize(_size);
	return nvrtc.pnvrtcGetProgramLog(prog,log.data());
}

SPTXResult getPTX(const CCUDAHandler& handler, nvrtcProgram prog)
{
	size_t _size = 0ull;
	const auto& nvrtc = handler.getNVRTCFunctionTable();
	nvrtcResult sizeRes = nvrtc.pnvrtcGetPTXSize(prog,&_size);
	if (sizeRes!=NVRTC_SUCCESS)
		return {nullptr,sizeRes};
	if (_size==0ull)
		return {nullptr,NVRTC_ERROR_INVALID_INPUT};

	asset::ICPUBuffer::SCreationParams ptxParams = {};
	ptxParams.size = _size;
	auto ptx = asset::ICPUBuffer::create(std::move(ptxParams));
	auto ptxPtr = static_cast<char*>(ptx->getPointer());
	return {std::move(ptx),nvrtc.pnvrtcGetPTX(prog,ptxPtr)};
}

static SPTXResult compileDirectlyToPTX_impl(CCUDAHandler& handler, nvrtcResult result, nvrtcProgram program, core::SRange<const char* const> nvrtcOptions, std::string& log)
{
	log.clear();
	if (result!=NVRTC_SUCCESS)
		return {nullptr,result};

	const auto runtimeIncludeOptions = handler.getDefaultRuntimeIncludeOptions();
	core::vector<const char*> options;
	options.reserve(nvrtcOptions.size()+runtimeIncludeOptions.size());
	for (const auto option : nvrtcOptions)
		options.push_back(option);
	for (const auto option : runtimeIncludeOptions)
		options.push_back(option);

	const auto* optionsBegin = options.empty() ? nullptr:options.data();
	const auto* optionsEnd = options.empty() ? nullptr:optionsBegin+options.size();
	result = compileProgram(handler,program,{optionsBegin,optionsEnd});
	getProgramLog(handler,program,log);
	if (result!=NVRTC_SUCCESS)
		return {nullptr,result};

	return getPTX(handler,program);
}

SPTXResult compileDirectlyToPTX(
	CCUDAHandler& handler, std::string&& source, const char* filename, core::SRange<const char* const> nvrtcOptions,
	std::string& log, const int headerCount, const char* const* headerContents, const char* const* includeNames)
{
	nvrtcProgram program = nullptr;
	nvrtcResult result = NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
	auto cleanup = core::makeRAIIExiter([&]() -> void
	{
		if (program)
			handler.getNVRTCFunctionTable().pnvrtcDestroyProgram(&program);
	});

	result = createProgram(handler,&program,std::move(source),filename,headerCount,headerContents,includeNames);
	return compileDirectlyToPTX_impl(handler,result,program,nvrtcOptions,log);
}

}

core::smart_refctd_ptr<CCUDADevice> CCUDAHandler::createDevice(core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection, IPhysicalDevice* physicalDevice)
{
	if (!vulkanConnection)
		return nullptr;
	const auto devices = vulkanConnection->getPhysicalDevices();
	if (std::find(devices.begin(),devices.end(),physicalDevice)==devices.end())
		return nullptr;

	for (const auto& device : m_native->deviceStates)
	{
		if (!memcmp(&device.uuid,&physicalDevice->getProperties().deviceUUID,VK_UUID_SIZE))
		{
			CCUDADevice::E_VIRTUAL_ARCHITECTURE arch = CCUDADevice::EVA_COUNT;
			const int& archMajor = device.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR];
			const int& archMinor = device.attributes[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR];
			switch (archMajor)
			{
				case 3:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_30;
							break;
						case 2:
							arch = CCUDADevice::EVA_32;
							break;
						case 5:
							arch = CCUDADevice::EVA_35;
							break;
						case 7:
							arch = CCUDADevice::EVA_37;
							break;
						default:
							break;
					}
					break;
				case 5:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_50;
							break;
						case 2:
							arch = CCUDADevice::EVA_52;
							break;
						case 3:
							arch = CCUDADevice::EVA_53;
							break;
						default:
							break;
					}
					break;
				case 6:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_60;
							break;
						case 1:
							arch = CCUDADevice::EVA_61;
							break;
						case 2:
							arch = CCUDADevice::EVA_62;
							break;
						default:
							break;
					}
					break;
				case 7:
					switch (archMinor)
					{
						case 0:
							arch = CCUDADevice::EVA_70;
							break;
						case 2:
							arch = CCUDADevice::EVA_72;
							break;
						case 5:
							arch = CCUDADevice::EVA_75;
							break;
						default:
							break;
					}
					break;
				default:
					if (archMajor>=8)
						arch = CCUDADevice::EVA_80;
					break;
			}
			if (arch==CCUDADevice::EVA_COUNT)
				continue;

			auto cudaDevice = core::smart_refctd_ptr<CCUDADevice>(
				new CCUDADevice(std::move(vulkanConnection),physicalDevice,arch,std::make_unique<CCUDADevice::SNativeState>(device.handle),core::smart_refctd_ptr<CCUDAHandler>(this)),
				core::dont_grab
			);
			if (!cudaDevice->isValid())
				return nullptr;
			return std::move(cudaDevice);
		}
	}
	return nullptr;
}

}

#else

namespace nbl::video
{

// CUDA OFF stub keeps the clean public API linkable and reports feature absence with nullptr instead of unresolved symbols.
struct CCUDAHandler::SNativeState {};

CCUDAHandler::CCUDAHandler(
	std::unique_ptr<SNativeState>&& nativeState,
	core::vector<core::smart_refctd_ptr<system::IFile>>&& _headers,
	core::smart_refctd_ptr<system::ILogger>&& _logger)
	: m_native(std::move(nativeState))
	, m_headers(std::move(_headers))
	, m_logger(std::move(_logger))
{
	assert(m_native);
}

CCUDAHandler::~CCUDAHandler() = default;

uint32_t CCUDAHandler::getBuildCUDASDKVersion()
{
	return 0u;
}

uint32_t CCUDAHandler::getLoadedCUDADriverVersion() const
{
	return 0u;
}

std::array<int,2> CCUDAHandler::getLoadedNVRTCVersion() const
{
	return {-1,-1};
}

const cuda_native::CUDA& CCUDAHandler::getCUDAFunctionTable() const
{
	std::abort();
}

const cuda_native::NVRTC& CCUDAHandler::getNVRTCFunctionTable() const
{
	std::abort();
}

core::SRange<const char* const> CCUDAHandler::getDefaultRuntimeIncludeOptions() const
{
	return {nullptr,nullptr};
}

core::smart_refctd_ptr<CCUDAHandler> CCUDAHandler::create(system::ISystem*, core::smart_refctd_ptr<system::ILogger>&&)
{
	return nullptr;
}

core::smart_refctd_ptr<CCUDADevice> CCUDAHandler::createDevice(core::smart_refctd_ptr<CVulkanConnection>&&, IPhysicalDevice*)
{
	return nullptr;
}

}

#endif // _NBL_COMPILE_WITH_CUDA_
