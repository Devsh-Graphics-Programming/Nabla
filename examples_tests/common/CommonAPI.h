#define _NBL_STATIC_LIB_
#include <nabla.h>

#if defined(_NBL_PLATFORM_WINDOWS_)
#include <nbl/ui/CWindowWin32.h>
using CWindowT = nbl::ui::CWindowWin32;
#elif defined(_NBL_PLATFORM_LINUX_)
#include <nbl/ui/CWindowLinux.h>
using CWindowT = nbl::ui::CWindowLinux;
#endif


class CommonAPI
{
	CommonAPI() = delete;
public:
	template<uint32_t sc_image_count>
	struct InitOutput
	{
		nbl::core::smart_refctd_ptr<CWindowT> window;
		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IGPUQueue* queue;
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, sc_image_count> fbo;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool;

	};
	template<uint32_t window_width, uint32_t window_height, uint32_t sc_image_count>
	static InitOutput<sc_image_count> Init(nbl::video::E_API_TYPE api_type, const std::string_view app_name)
	{
		using namespace nbl;
		InitOutput<sc_image_count> result = {};
		result.window = CWindowT::create(window_width, window_height, ui::IWindow::ECF_NONE);

		video::SDebugCallback dbgcb;
		dbgcb.callback = &defaultDebugCallback;
		dbgcb.userData = nullptr;
		result.apiConnection = video::IAPIConnection::create(api_type, 0, app_name.data(), dbgcb);
		result.surface = result.apiConnection->createSurface(result.window.get());

		auto gpus = result.apiConnection->getPhysicalDevices();
		assert(!gpus.empty());
		auto gpu = gpus.begin()[0];
		int familyIndex = getQueueFamilyIndex(gpu, video::IPhysicalDevice::EQF_GRAPHICS_BIT |
			video::IPhysicalDevice::EQF_COMPUTE_BIT |
			video::IPhysicalDevice::EQF_TRANSFER_BIT);
		assert(result.surface->isSupported(gpu.get(), familyIndex));

		video::ILogicalDevice::SCreationParams dev_params;
		dev_params.queueParamsCount = 1u;
		video::ILogicalDevice::SQueueCreationParams q_params;
		q_params.familyIndex = familyIndex;
		q_params.count = 1u;
		q_params.flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
		float priority = 1.f;
		q_params.priorities = &priority;
		dev_params.queueCreateInfos = &q_params;
		result.logicalDevice = gpu->createLogicalDevice(dev_params);

		result.queue = result.logicalDevice->getQueue(familyIndex, 0);

		result.swapchain = createSwapchain(window_width, window_height, sc_image_count, result.logicalDevice, result.surface, video::ISurface::EPM_FIFO_RELAXED);
		assert(result.swapchain);

		result.renderpass = createRenderpass(result.logicalDevice);

		result.fbo = createFBOWithSwapchainImages<sc_image_count, window_width, window_height>(result.logicalDevice, result.swapchain, result.renderpass);

		result.commandPool = result.logicalDevice->createCommandPool(familyIndex, static_cast<video::IGPUCommandPool::E_CREATE_FLAGS>(0));
		assert(result.commandPool);


		return result;

	}
	static nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> createSwapchain(uint32_t width,
		uint32_t height,
		uint32_t imageCount,
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		const nbl::core::smart_refctd_ptr<nbl::video::ISurface>& surface,
		nbl::video::ISurface::E_PRESENT_MODE presentMode)
	{
		using namespace nbl;
		video::ISwapchain::SCreationParams sc_params;
		sc_params.width = width;
		sc_params.height = height;
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = imageCount;
		sc_params.presentMode = presentMode;
		sc_params.surface = surface;
		sc_params.surfaceFormat.format = asset::EF_R8G8B8A8_SRGB;
		sc_params.surfaceFormat.colorSpace.eotf = asset::EOTF_sRGB;
		sc_params.surfaceFormat.colorSpace.primary = asset::ECP_SRGB;

		return device->createSwapchain(std::move(sc_params));
	}
	static nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device)
	{
		using namespace nbl;

		video::IGPURenderpass::SCreationParams::SAttachmentDescription a;
		a.initialLayout = asset::EIL_UNDEFINED;
		a.finalLayout = asset::EIL_UNDEFINED;
		a.format = asset::EF_R8G8B8A8_SRGB;
		a.samples = asset::IImage::ESCF_1_BIT;
		a.loadOp = video::IGPURenderpass::ELO_CLEAR;
		a.storeOp = video::IGPURenderpass::ESO_STORE;

		video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::EIL_UNDEFINED;
		video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		sp.depthStencilAttachment = nullptr;
		sp.flags = video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = 1u;
		rp_params.attachments = &a;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		return device->createGPURenderpass(rp_params);
	}

	template<size_t imageCount, size_t width, size_t height>
	static auto createFBOWithSwapchainImages(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass)->std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, imageCount>
	{
		using namespace nbl;
		std::array<nbl::core::smart_refctd_ptr<video::IGPUFramebuffer>, imageCount> fbo;
		auto sc_images = swapchain->getImages();
		assert(sc_images.size() == imageCount);
		for (uint32_t i = 0u; i < imageCount; ++i)
		{
			auto img = sc_images.begin()[i];
			core::smart_refctd_ptr<video::IGPUImageView> view;
			{
				video::IGPUImageView::SCreationParams view_params;
				view_params.format = img->getCreationParameters().format;
				view_params.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
				view_params.subresourceRange.baseMipLevel = 0u;
				view_params.subresourceRange.levelCount = 1u;
				view_params.subresourceRange.baseArrayLayer = 0u;
				view_params.subresourceRange.layerCount = 1u;
				view_params.image = std::move(img);

				view = device->createGPUImageView(std::move(view_params));
				assert(view);
			}

			video::IGPUFramebuffer::SCreationParams fb_params;
			fb_params.width = width;
			fb_params.height = height;
			fb_params.layers = 1u;
			fb_params.renderpass = renderpass;
			fb_params.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
			fb_params.attachmentCount = 1u;
			fb_params.attachments = &view;

			fbo[i] = device->createGPUFramebuffer(std::move(fb_params));
			assert(fbo[i]);
		}
		return fbo;
	}

	static void Submit(nbl::video::ILogicalDevice* device,
		nbl::video::ISwapchain* sc,
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer>* cmdbuf,
		nbl::video::IGPUQueue* queue,
		nbl::video::IGPUSemaphore* imgAcqSemaphore,
		nbl::video::IGPUSemaphore* renderFinishedSemaphore,
		size_t imageCount,
		uint32_t imgNum)
	{
		using namespace nbl;
		video::IGPUQueue::SSubmitInfo submit;
		{
			auto* cb = cmdbuf[imgNum].get();
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cb;
			video::IGPUSemaphore* signalsem = renderFinishedSemaphore;
			submit.signalSemaphoreCount = 1u;
			submit.pSignalSemaphores = &signalsem;
			video::IGPUSemaphore* waitsem = imgAcqSemaphore;
			asset::E_PIPELINE_STAGE_FLAGS dstWait = asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
			submit.waitSemaphoreCount = 1u;
			submit.pWaitSemaphores = &waitsem;
			submit.pWaitDstStageMask = &dstWait;

			queue->submit(1u, &submit, nullptr);
		}
	}

	static void Present(nbl::video::ILogicalDevice* device,
		nbl::video::ISwapchain* sc,
		nbl::video::IGPUQueue* queue,
		nbl::video::IGPUSemaphore* renderFinishedSemaphore,
		uint32_t imageNum)
	{
		using namespace nbl;
		video::IGPUQueue::SPresentInfo present;
		{
			present.swapchainCount = 1u;
			present.imgIndices = &imageNum;
			video::ISwapchain* swapchain = sc;
			present.swapchains = &swapchain;
			video::IGPUSemaphore* waitsem = renderFinishedSemaphore;
			present.waitSemaphoreCount = 1u;
			present.waitSemaphores = &waitsem;

			queue->present(present);
		}
	}
	static std::pair<nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>, nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>> createEmpty2DTexture(
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		uint32_t width,
		uint32_t height,
		nbl::asset::E_FORMAT format = nbl::asset::E_FORMAT::EF_R8G8B8A8_UNORM)
	{
		nbl::video::IGPUImage::SCreationParams gpu_image_params;
		gpu_image_params.mipLevels = 1;
		gpu_image_params.extent = { width, height, 1 };
		gpu_image_params.format = format;
		gpu_image_params.arrayLayers = 1u;
		gpu_image_params.type = nbl::asset::IImage::ET_2D;
		gpu_image_params.samples = nbl::asset::IImage::ESCF_1_BIT;
		gpu_image_params.flags = static_cast<nbl::asset::IImage::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image = device->createGPUImageOnDedMem(std::move(gpu_image_params), device->getDeviceLocalGPUMemoryReqs());

		nbl::video::IGPUImageView::SCreationParams creation_params;
		creation_params.format = image->getCreationParameters().format;
		creation_params.image = image;
		creation_params.viewType = nbl::video::IGPUImageView::ET_2D;
		creation_params.subresourceRange = { static_cast<nbl::asset::IImage::E_ASPECT_FLAGS>(0u), 0, 1, 0, 1 };
		creation_params.flags = static_cast<nbl::video::IGPUImageView::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image_view = device->createGPUImageView(std::move(creation_params));
		return std::pair(image, image_view);
	}

	static int getQueueFamilyIndex(const nbl::core::smart_refctd_ptr<nbl::video::IPhysicalDevice>& gpu, uint32_t requiredQueueFlags)
	{
		auto props = gpu->getQueueFamilyProperties();
		int currentIndex = 0;
		for (const auto& property : props)
		{
			if ((property.queueFlags & requiredQueueFlags) == requiredQueueFlags)
			{
				return currentIndex;
			}
			++currentIndex;
		}
		return -1;
	}
	static void defaultDebugCallback(nbl::video::E_DEBUG_MESSAGE_SEVERITY severity, nbl::video::E_DEBUG_MESSAGE_TYPE type, const char* msg, void* userData)
	{
		using namespace nbl;
		const char* sev = nullptr;
		switch (severity)
		{
		case video::EDMS_VERBOSE:
			sev = "verbose"; break;
		case video::EDMS_INFO:
			sev = "info"; break;
		case video::EDMS_WARNING:
			sev = "warning"; break;
		case video::EDMS_ERROR:
			sev = "error"; break;
		}
		std::cout << "OpenGL " << sev << ": " << msg << std::endl;
	}

};