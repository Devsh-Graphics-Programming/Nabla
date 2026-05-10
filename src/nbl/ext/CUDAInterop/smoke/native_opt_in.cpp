#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"
#include "nbl/system/IApplicationFramework.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <utility>

#ifndef CUDA_VERSION
#error "Nabla::ext::CUDAInterop must expose CUDA SDK headers."
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
		.locationType = CU_MEM_LOCATION_TYPE_DEVICE,
	});
	if (!cudaMemory)
		return false;

	auto exportedToVulkan = cudaMemory->exportAsMemory(vulkanDevice);
	auto importedFromVulkan = cudaDevice.importExternalMemory(std::move(vulkanMemory));
	auto importedSemaphore = cudaDevice.importExternalSemaphore(std::move(vulkanSemaphore));

	cuda_native::SCUdeviceptr mappedVulkanMemory;
	if (importedFromVulkan)
		importedFromVulkan->getMappedBuffer(mappedVulkanMemory);

	const cuda_native::SCUdeviceptr cudaDevicePtr = cudaMemory->getDeviceptr();
	cuda_native::SCUexternalSemaphore cudaSemaphore;
	if (importedSemaphore)
		cudaSemaphore = cuda_native::SCUexternalSemaphore(importedSemaphore->getInternalObject());
	return exportedToVulkan.get() && mappedVulkanMemory && cudaDevicePtr && cudaSemaphore;
}

bool cudaDriverRoundtrip(CCUDAHandler& handler, CUdevice device)
{
	auto& cuda = handler.getCUDAFunctionTable();

	CUcontext context = nullptr;
	if (cuda.pcuDevicePrimaryCtxRetain(&context, device)!=CUDA_SUCCESS)
		return false;

	CUcontext poppedContext = nullptr;
	bool contextPushed = false;
	auto releaseContext = [&]()
	{
		if (context)
		{
			if (contextPushed)
				cuda.pcuCtxPopCurrent_v2(&poppedContext);
			cuda.pcuDevicePrimaryCtxRelease_v2(device);
		}
	};

	if (cuda.pcuCtxPushCurrent_v2(context)!=CUDA_SUCCESS)
	{
		releaseContext();
		return false;
	}
	contextPushed = true;

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

bool cudaFp16HeaderCompileProbe(CCUDAHandler& handler)
{
	constexpr const char* Source = R"cuda(
		#include <cuda_fp16.h>
		extern "C" __global__ void fp16_probe(unsigned short* out)
		{
			out[0] = sizeof(__half);
		}
	)cuda";

	std::string log;
	auto compile = cuda_native::compileDirectlyToPTX(
		handler,
		std::string(Source),
		"cuda_fp16_discovery_probe.cu",
		{nullptr,nullptr},
		log,
		0,
		nullptr,
		nullptr
	);
	return compile.result==NVRTC_SUCCESS && compile.ptx && compile.ptx->getSize()>0u;
}
}

class CUDAInteropNativeOptInSmoke final : public nbl::system::IApplicationFramework
{
	using base_t = nbl::system::IApplicationFramework;

public:
	using base_t::base_t;

	bool onAppInitialized(nbl::core::smart_refctd_ptr<nbl::system::ISystem>&&) override
	{
		if (!isAPILoaded())
			return false;

		static_assert(std::is_same_v<nbl::video::cuda_native::SCUdevice::cuda_t, CUdevice>);
		[[maybe_unused]] const bool exactBuildSDK = nbl::video::cuda_native::isBuildCUDASDKVersionExactMatch();

		#ifdef NBL_CUDA_INTEROP_SMOKE_RUNTIME_JSON
		const nbl::core::vector<nbl::system::path> explicitIncludeDirs;
		const nbl::core::vector<nbl::system::path> runtimePathFiles = {NBL_CUDA_INTEROP_SMOKE_RUNTIME_JSON};
		const auto runtimeEnvironment = nbl::video::cuda_interop::findRuntimeCompileEnvironment(explicitIncludeDirs, runtimePathFiles);
		if (!std::filesystem::exists(NBL_CUDA_INTEROP_SMOKE_RUNTIME_JSON))
			return false;
		#else
		const auto runtimeEnvironment = nbl::video::cuda_interop::findRuntimeCompileEnvironment();
		#endif
		const auto includeOptions = nbl::video::cuda_interop::makeNVRTCIncludeOptions(runtimeEnvironment);
		const auto hasRuntimeHeaders = std::find_if(runtimeEnvironment.includeDirs.begin(),runtimeEnvironment.includeDirs.end(),[](const auto& includeDir) {
			return std::filesystem::exists(includeDir/"cuda_fp16.h") || std::filesystem::exists(includeDir/"cuda_runtime_api.h");
		})!=runtimeEnvironment.includeDirs.end();
		if (includeOptions.empty() || !hasRuntimeHeaders)
			return false;

		auto handler = nbl::video::CCUDAHandler::create(nullptr, nullptr);
		if (!handler)
			return true;

		auto pcuDriverGetVersion = NBL_SYSTEM_LOAD_DYNLIB_FUNCPTR(handler->getCUDAFunctionTable(), cuDriverGetVersion);
		int loadedDriverVersion = 0;
		if (!pcuDriverGetVersion || pcuDriverGetVersion(&loadedDriverVersion)!=CUDA_SUCCESS || loadedDriverVersion==0)
			return false;

		if (!cudaFp16HeaderCompileProbe(*handler))
			return false;

		int deviceCount = 0;
		if (handler->getCUDAFunctionTable().pcuDeviceGetCount(&deviceCount)!=CUDA_SUCCESS || deviceCount==0)
			return true;

		CUdevice device = {};
		if (handler->getCUDAFunctionTable().pcuDeviceGet(&device,0)!=CUDA_SUCCESS)
			return false;

		return cudaDriverRoundtrip(*handler, device);
	}

	void workLoopBody() override {}
	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(CUDAInteropNativeOptInSmoke)
