#include "nbl/ext/CUDAInterop/CUDAInterop.h"

namespace nbl::video
{

struct CCUDAHandler::SNativeState {};
struct CCUDADevice::SNativeState {};
struct CCUDAExportableMemory::SNativeState {};
struct CCUDAImportedMemory::SNativeState {};
struct CCUDAImportedSemaphore::SNativeState {};

CCUDAHandler::CCUDAHandler(
	std::unique_ptr<SNativeState>&& nativeState,
	core::vector<core::smart_refctd_ptr<system::IFile>>&& _headers,
	core::smart_refctd_ptr<system::ILogger>&& _logger,
	int _version)
	: m_native(std::move(nativeState))
	, m_headers(std::move(_headers))
	, m_logger(std::move(_logger))
	, m_version(_version)
{}

CCUDAHandler::~CCUDAHandler() = default;

core::smart_refctd_ptr<CCUDAHandler> CCUDAHandler::create(system::ISystem*, core::smart_refctd_ptr<system::ILogger>&&)
{
	return nullptr;
}

core::smart_refctd_ptr<CCUDADevice> CCUDAHandler::createDevice(core::smart_refctd_ptr<CVulkanConnection>&&, IPhysicalDevice*)
{
	return nullptr;
}

CCUDADevice::CCUDADevice(
	core::smart_refctd_ptr<CVulkanConnection>&& vulkanConnection,
	IPhysicalDevice* const vulkanDevice,
	const E_VIRTUAL_ARCHITECTURE virtualArchitecture,
	std::unique_ptr<SNativeState>&& nativeState,
	core::smart_refctd_ptr<CCUDAHandler>&& handler)
	: m_logger(nullptr)
	, m_vulkanConnection(std::move(vulkanConnection))
	, m_physicalDevice(vulkanDevice)
	, m_virtualArchitecture(virtualArchitecture)
	, m_handler(std::move(handler))
	, m_native(std::move(nativeState))
{}

CCUDADevice::~CCUDADevice() = default;

size_t CCUDADevice::roundToGranularity(ECUDAMemoryLocation, size_t size) const
{
	return size;
}

core::smart_refctd_ptr<CCUDAExportableMemory> CCUDADevice::createExportableMemory(CCUDAExportableMemory::SCreationParams&&)
{
	return nullptr;
}

core::smart_refctd_ptr<CCUDAImportedMemory> CCUDADevice::importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>&&)
{
	return nullptr;
}

core::smart_refctd_ptr<CCUDAImportedSemaphore> CCUDADevice::importExternalSemaphore(core::smart_refctd_ptr<ISemaphore>&&)
{
	return nullptr;
}

CCUDAExportableMemory::CCUDAExportableMemory(core::smart_refctd_ptr<CCUDADevice> device, SCachedCreationParams&& params, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_params(std::move(params))
	, m_native(std::move(nativeState))
{}

CCUDAExportableMemory::~CCUDAExportableMemory() = default;

core::smart_refctd_ptr<IDeviceMemoryAllocation> CCUDAExportableMemory::exportAsMemory(ILogicalDevice*, IDeviceMemoryBacked*) const
{
	return nullptr;
}

CCUDAImportedMemory::CCUDAImportedMemory(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<IDeviceMemoryAllocation> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{}

CCUDAImportedMemory::~CCUDAImportedMemory() = default;

CCUDAImportedSemaphore::CCUDAImportedSemaphore(core::smart_refctd_ptr<CCUDADevice> device, core::smart_refctd_ptr<ISemaphore> src, std::unique_ptr<SNativeState>&& nativeState)
	: m_device(std::move(device))
	, m_src(std::move(src))
	, m_native(std::move(nativeState))
{}

CCUDAImportedSemaphore::~CCUDAImportedSemaphore() = default;

}
