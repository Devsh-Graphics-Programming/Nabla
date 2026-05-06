#include "nbl/ext/CUDAInterop/CUDAInterop.h"
#include "nbl/system/IApplicationFramework.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#ifndef _NBL_COMPILE_WITH_CUDA_
#error "CUDA interop consumers must opt in through Nabla::ext::CUDAInterop."
#endif

namespace
{
using namespace nbl;
using namespace nbl::video;

[[maybe_unused]] bool compileVulkanCudaInteropRecipe(
	CCUDADevice& cudaDevice,
	ILogicalDevice* vulkanDevice,
	core::smart_refctd_ptr<IDeviceMemoryAllocation> vulkanMemory,
	core::smart_refctd_ptr<ISemaphore> vulkanSemaphore)
{
	auto cudaMemory = cudaDevice.createExportableMemory({
		.size = 4096,
		.alignment = 4096,
		.location = CU_MEM_LOCATION_TYPE_DEVICE,
	});
	if (!cudaMemory)
		return false;

	auto exportedToVulkan = cudaMemory->exportAsMemory(vulkanDevice);
	auto importedFromVulkan = cudaDevice.importExternalMemory(std::move(vulkanMemory));
	auto importedSemaphore = cudaDevice.importExternalSemaphore(std::move(vulkanSemaphore));

	CUdeviceptr mappedVulkanMemory = 0;
	if (importedFromVulkan)
		importedFromVulkan->getMappedBuffer(&mappedVulkanMemory);

	const CUexternalSemaphore cudaSemaphore = importedSemaphore ? importedSemaphore->getInternalObject():nullptr;
	return exportedToVulkan.get() && mappedVulkanMemory && cudaSemaphore;
}

bool cudaDriverRoundtrip(CCUDAHandler& handler, CUdevice device)
{
	auto& cuda = handler.getCUDAFunctionTable();

	CUcontext context = nullptr;
	if (cuda.pcuDevicePrimaryCtxRetain(&context, device)!=CUDA_SUCCESS)
		return false;

	CUcontext poppedContext = nullptr;
	auto releaseContext = [&]()
	{
		if (context)
		{
			cuda.pcuCtxPopCurrent_v2(&poppedContext);
			cuda.pcuDevicePrimaryCtxRelease_v2(device);
		}
	};

	if (cuda.pcuCtxPushCurrent_v2(context)!=CUDA_SUCCESS)
	{
		releaseContext();
		return false;
	}

	constexpr std::array<uint32_t, 4> input = {0x12345678u, 0x90abcdefu, 0xfedcba09u, 0x87654321u};
	std::array<uint32_t, input.size()> output = {};

	CUdeviceptr deviceMemory = 0;
	bool ok = cuda.pcuMemAlloc_v2(&deviceMemory, sizeof(input))==CUDA_SUCCESS;
	if (ok)
		ok = cuda.pcuMemcpyHtoD_v2(deviceMemory,input.data(),sizeof(input))==CUDA_SUCCESS;
	if (ok)
		ok = cuda.pcuMemcpyDtoH_v2(output.data(),deviceMemory,sizeof(output))==CUDA_SUCCESS;
	if (deviceMemory)
		ok = cuda.pcuMemFree_v2(deviceMemory)==CUDA_SUCCESS && ok;

	releaseContext();
	return ok && std::ranges::equal(input, output);
}
}

class CUDAInteropOptInSmoke final : public nbl::system::IApplicationFramework
{
	using base_t = nbl::system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(nbl::core::smart_refctd_ptr<nbl::system::ISystem>&&) override
	{
		if (!isAPILoaded())
			return false;

		static_assert(std::is_same_v<decltype(std::declval<const nbl::video::CCUDADevice&>().getInternalObject()), CUdevice>);

		auto handler = nbl::video::CCUDAHandler::create(nullptr, nullptr);
		if (!handler)
			return true;

		const auto& devices = handler->getAvailableDevices();
		if (devices.empty())
			return true;

		return cudaDriverRoundtrip(*handler, devices.front().handle);
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(CUDAInteropOptInSmoke)
