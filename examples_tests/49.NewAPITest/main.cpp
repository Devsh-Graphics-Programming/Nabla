// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <nabla.h>

#include <iostream>
#include <cstdio>

// TODO: remove as will be replaced with ISystem
#include "CFileSystem.h" // tmp, this should be accessible via IFileSystem and not required to be created explicitely by user

#if defined(_NBL_PLATFORM_WINDOWS_)
#include <nbl/ui/CWindowWin32.h>
using CWindowT = nbl::ui::CWindowWin32;
#elif defined(_NBL_PLATFORM_LINUX_)
#ifdef _NBL_TEST_WAYLAND
#include <nbl/ui/CWindowWayland.h>
using CWindowT = nbl::ui::CWindowWayland;
#else
#include <nbl/ui/CWindowX11.h>
using CWindowT = nbl::ui::CWindowX11;
#endif
#endif

using namespace nbl;

// TODO: replace with a engine-wide system::ILogger
static void debugCallback(video::E_DEBUG_MESSAGE_SEVERITY severity, video::E_DEBUG_MESSAGE_TYPE type, const char* msg, void* userData)
{
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

int main()
{
	constexpr uint32_t WIN_W = 800u;
	constexpr uint32_t WIN_H = 600u;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	const char* vs_source = R"(#version 430

layout (location = 0) in vec2 Pos;
layout (location = 1) in vec3 Color;

layout (location = 0) out vec3 OutColor;

void main()
{
	OutColor = Color;
	gl_Position = vec4(Pos, 0.0, 1.0);
}
)";
	const char* fs_source = R"(#version 430

layout (location = 0) in vec3 InColor;
layout (location = 0) out vec4 OutColor;

void main()
{
	OutColor = vec4(InColor, 1.0);
}
)";

	// change window creation order because they seem to not be movable!?
	auto win2 = CWindowT::create(WIN_W, WIN_H, ui::IWindow::ECF_NONE);
	auto win = CWindowT::create(WIN_W, WIN_H, ui::IWindow::ECF_NONE);
	// TODO: can I move the window?

	core::smart_refctd_ptr<video::IAPIConnection> api;
	 // TODO: Change API Connection creation (the copy this code to CommonAPI.h)
#if 0
	{
		std::cout <<
			R"(
Choose Graphics API:
0) Vulkan
1) OpenGL ES
2) OpenGL core
)" << std::endl;
		uint8_t apiType;
		std::cin >> apiType;
		switch (apiType)
		{
			case 1:
				api = video::COpenGLConnection::create(0,"New API Test",std::move(logger_smart_ptr));
				break;
			case 2:
				api = video::COpenGLESConnection::create(0,"New API Test",std::move(logger_smart_ptr));
				break;
			default:
				api = video::CVulkanConnection::create(0, "New API Test", std::move(logger_smart_ptr));
				break;
		}
	}
#else
	{
		video::SDebugCallback dbgcb;
		dbgcb.callback = &debugCallback;
		dbgcb.userData = nullptr;
		api = video::IAPIConnection::create(video::EAT_OPENGL, 0, "New API Test", dbgcb);
	}
#endif

	auto surface = api->createSurface(win.get());
	auto surface2 = api->createSurface(win2.get());

	auto gpus = api->getPhysicalDevices();
	assert(!gpus.empty());
	auto gpu = gpus.begin()[0];

	assert(surface->isSupported(gpu.get(), 0u));

	video::ILogicalDevice::SCreationParams dev_params;
	dev_params.queueParamsCount = 1u;
	video::ILogicalDevice::SQueueCreationParams q_params;
	q_params.familyIndex = 0u;
	q_params.count = 4u;
	q_params.flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
	float priority[4] = {1.f,1.f,1.f,1.f};
	q_params.priorities = priority;
	dev_params.queueCreateInfos = &q_params;
	auto device = gpu->createLogicalDevice(dev_params);

	auto* queue = device->getQueue(0u, 0u);
	auto* queue2 = device->getQueue(0u, 1u);
	/*
	uint8_t stackmem[1u << 14];
	float farray[3]{ 1.f, 3.f, 4.f };
	memcpy(stackmem, farray, 12);
	auto memreqs = device->getDeviceLocalGPUMemoryReqs();
	memreqs.vulkanReqs.size = sizeof(stackmem);
	auto somebuffer = device->createGPUBufferOnDedMem(memreqs, true);
	asset::SBufferRange<video::IGPUBuffer> bufrng;
	bufrng.offset = 0;
	bufrng.size = somebuffer->getSize();
	bufrng.buffer = somebuffer;
	device->updateBufferRangeViaStagingBuffer(queue, bufrng, stackmem);
	*/

	// those shouldnt be like that (filesystem is already created in IAPIConnection but also temporarily i think -- we're going to have system::ISystem instead anyway)
	auto fs = core::make_smart_refctd_ptr<io::CFileSystem>("");
	auto am = core::make_smart_refctd_ptr<asset::IAssetManager>(std::move(fs));

	asset::IAssetLoader::SAssetLoadParams lp;
	auto bundle = am->getAsset("../../media/dwarf.jpg", lp);
	assert(!bundle.getContents().empty());

	video::IGPUObjectFromAssetConverter cpu2gpu;
	video::IGPUObjectFromAssetConverter::SParams c2gparams;
	c2gparams.device = device.get();
	c2gparams.perQueue[video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = queue;
	c2gparams.perQueue[video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = queue2;
	c2gparams.finalQueueFamIx = queue->getFamilyIndex();
	c2gparams.sharingMode = asset::ESM_CONCURRENT;
	c2gparams.limits = gpu->getLimits();
	c2gparams.assetManager = am.get();
	
	//auto cpubuf = new asset::ICPUBuffer(1024);
	auto gpuimgs = cpu2gpu.getGPUObjectsFromAssets<asset::ICPUImage>(bundle.getContents(), c2gparams);
	auto gpuimg = (*gpuimgs)[0];

	core::smart_refctd_ptr<video::ISwapchain> sc;
	{
		video::ISwapchain::SCreationParams sc_params;
		sc_params.width = WIN_W;
		sc_params.height = WIN_H;
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = SC_IMG_COUNT;
		sc_params.presentMode = video::ISurface::EPM_FIFO_RELAXED;
		sc_params.surface = surface;
		sc_params.surfaceFormat.format = asset::EF_R8G8B8A8_SRGB;
		sc_params.surfaceFormat.colorSpace.eotf = asset::EOTF_sRGB;
		sc_params.surfaceFormat.colorSpace.primary = asset::ECP_SRGB;

		sc = device->createSwapchain(std::move(sc_params));
		assert(sc);
	}
	core::smart_refctd_ptr<video::ISwapchain> sc2;
	{
		video::ISwapchain::SCreationParams sc_params;
		sc_params.width = WIN_W;
		sc_params.height = WIN_H;
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = SC_IMG_COUNT;
		sc_params.presentMode = video::ISurface::EPM_FIFO_RELAXED;
		sc_params.surface = surface2;
		sc_params.surfaceFormat.format = asset::EF_R8G8B8A8_SRGB;
		sc_params.surfaceFormat.colorSpace.eotf = asset::EOTF_sRGB;
		sc_params.surfaceFormat.colorSpace.primary = asset::ECP_SRGB;

		sc2 = device->createSwapchain(std::move(sc_params));
		assert(sc2);
	}

	core::smart_refctd_ptr<video::IGPURenderpass> renderpass;
	{
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

		renderpass = device->createGPURenderpass(rp_params);
	}

	auto sc_images = sc->getImages();
	assert(sc_images.size() == SC_IMG_COUNT);

	core::smart_refctd_ptr<video::IGPUFramebuffer> fbo[SC_IMG_COUNT];
	for (uint32_t i = 0u; i < sc_images.size(); ++i)
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
		fb_params.width = WIN_W;
		fb_params.height = WIN_H;
		fb_params.layers = 1u;
		fb_params.renderpass = renderpass;
		fb_params.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fb_params.attachmentCount = 1u;
		fb_params.attachments = &view;

		fbo[i] = device->createGPUFramebuffer(std::move(fb_params));
		assert(fbo[i]);
	}

	auto cmdpool = device->createCommandPool(0u, static_cast<video::IGPUCommandPool::E_CREATE_FLAGS>(0));
	assert(cmdpool);

#include "nbl/nblpack.h"
	struct SVertex
	{
		float pos[2];
		float color[3];
	} PACK_STRUCT;
#include "nbl/nblunpack.h"

	auto layout = device->createGPUPipelineLayout();
	assert(layout);

	core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpindependent_pipeline;
	{
		auto vs_unspec = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(vs_source));
		auto fs_unspec = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(fs_source));

		asset::ISpecializedShader::SInfo vsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX, "vs");
		auto vs = device->createGPUSpecializedShader(vs_unspec.get(), vsinfo);
		asset::ISpecializedShader::SInfo fsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT, "fs");
		auto fs = device->createGPUSpecializedShader(fs_unspec.get(), fsinfo);

		video::IGPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

		asset::SVertexInputParams vtxinput;
		vtxinput.attributes[0].binding = 0;
		vtxinput.attributes[0].format = asset::EF_R32G32_SFLOAT;
		vtxinput.attributes[0].relativeOffset = offsetof(SVertex, pos);

		vtxinput.attributes[1].binding = 0;
		vtxinput.attributes[1].format = asset::EF_R32G32B32_SFLOAT;
		vtxinput.attributes[1].relativeOffset = offsetof(SVertex, color);

		vtxinput.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
		vtxinput.bindings[0].stride = sizeof(SVertex);

		vtxinput.enabledAttribFlags = 0b0011;
		vtxinput.enabledBindingFlags = 0b0001;
		
		asset::SRasterizationParams raster;
		raster.depthTestEnable = 0;
		raster.depthWriteEnable = 0;
		raster.faceCullingMode = asset::EFCM_NONE;
		
		asset::SPrimitiveAssemblyParams primitive;
		primitive.primitiveType = asset::EPT_TRIANGLE_LIST;

		asset::SBlendParams blend;

		rpindependent_pipeline = device->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(layout), shaders, shaders+2, vtxinput, blend, primitive, raster);
		assert(rpindependent_pipeline);
	}

	core::smart_refctd_ptr<video::IGPUGraphicsPipeline> pipeline;
	{
		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = renderpass;
		gp_params.renderpassIndependent = rpindependent_pipeline;
		gp_params.subpassIx = 0u;

		pipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));
	}

	core::smart_refctd_ptr<video::IGPUBuffer> buffer;
	{
		const SVertex vertices[3]{
			{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
			{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
			{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
		};

		video::IDriverMemoryBacked::SDriverMemoryRequirements mreq;
		
		auto mreqs = device->getDeviceLocalGPUMemoryReqs();
		mreqs.vulkanReqs.size = sizeof(vertices);
		buffer = device->createGPUBufferOnDedMem(mreqs, true);
		assert(buffer);

		core::smart_refctd_ptr<video::IGPUCommandBuffer> cb;
		device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &cb);
		assert(cb);

		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		asset::SViewport vp;
		vp.minDepth = 1.f;
		vp.maxDepth = 0.f;
		vp.x = 0u;
		vp.y = 0u;
		vp.width = WIN_W;
		vp.height = WIN_H;
		cb->setViewport(0u, 1u, &vp);

		cb->updateBuffer(buffer.get(), 0u, sizeof(vertices), vertices);

		video::IGPUCommandBuffer::SBufferMemoryBarrier bufMemBarrier;
		bufMemBarrier.srcQueueFamilyIndex = 0u;
		bufMemBarrier.dstQueueFamilyIndex = 0u;
		bufMemBarrier.offset = 0u;
		bufMemBarrier.size = buffer->getSize();
		bufMemBarrier.buffer = buffer;
		bufMemBarrier.barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		bufMemBarrier.barrier.dstAccessMask = asset::EAF_VERTEX_ATTRIBUTE_READ_BIT;
		cb->pipelineBarrier(asset::EPSF_TRANSFER_BIT, asset::EPSF_VERTEX_INPUT_BIT, 0, 0u, nullptr, 1u, &bufMemBarrier, 0u, nullptr);

		cb->end();
		
		video::IGPUQueue::SSubmitInfo info;
		auto* cb_ = cb.get();
		info.commandBufferCount = 1u;
		info.commandBuffers = &cb_;
		info.pSignalSemaphores = nullptr;
		info.signalSemaphoreCount = 0u;
		info.pWaitSemaphores = nullptr;
		info.waitSemaphoreCount = 0u;
		info.pWaitDstStageMask = nullptr;
		queue->submit(1u, &info, nullptr);
	}

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
	device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = cmdbuf[i];
		auto& fb = fbo[i];

		cb->begin(0);
		
		const video::IGPUBuffer* buf = buffer.get();
		size_t offset = 0u;
		cb->bindVertexBuffers(0u,1u,&buf,&offset);
		cb->bindGraphicsPipeline(pipeline.get());
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		asset::VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = { WIN_W, WIN_H };
		clear.color.float32[0] = 1.f;
		clear.color.float32[1] = 0.f;
		clear.color.float32[2] = 0.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = renderpass;
		info.framebuffer = fb;
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		cb->draw(3u, 1u, 0u, 0u);
		cb->endRenderPass();

		cb->end();
	}

	constexpr uint32_t FRAME_COUNT = 5000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		auto img_acq_sem = device->createSemaphore();
		auto render_finished_sem = device->createSemaphore();

		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		video::IGPUQueue::SSubmitInfo submit;
		{
			auto* cb = cmdbuf[imgnum].get();
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cb;
			video::IGPUSemaphore* signalsem = render_finished_sem.get();
			submit.signalSemaphoreCount = 1u;
			submit.pSignalSemaphores = &signalsem;
			video::IGPUSemaphore* waitsem = img_acq_sem.get();
			asset::E_PIPELINE_STAGE_FLAGS dstWait = asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
			submit.waitSemaphoreCount = 1u;
			submit.pWaitSemaphores = &waitsem;
			submit.pWaitDstStageMask = &dstWait;

			queue->submit(1u, &submit, nullptr);
		}

		video::IGPUQueue::SPresentInfo present;
		{
			present.swapchainCount = 1u;
			present.imgIndices = &imgnum;
			video::ISwapchain* swapchain = sc.get();
			present.swapchains = &swapchain;
			video::IGPUSemaphore* waitsem = render_finished_sem.get();
			present.waitSemaphoreCount = 1u;
			present.waitSemaphores = &waitsem;

			queue->present(present);
		}
	}

	device->waitIdle();

	return 0;
}
