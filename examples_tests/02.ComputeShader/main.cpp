#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;
using namespace core;


const char* src = R"(#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform pushConstants {
    layout (offset = 0) uvec2 imgSize;
} u_pushConstants;

layout (set = 0, binding = 0, rgba8) uniform readonly image2D inImage;
layout (set = 0, binding = 1, rgba8) uniform image2D outImage;

void main()
{
	if (all(lessThan(gl_GlobalInvocationID.xy, u_pushConstants.imgSize)))
	{
		vec3 rgb = imageLoad(inImage, ivec2(gl_GlobalInvocationID.xy)).rgb;
		
		imageStore(outImage, ivec2(gl_GlobalInvocationID.xy), vec4(1, 1, 0, 1));
	}
})";


int getQueueFamilyIndex(const core::smart_refctd_ptr<nbl::video::IPhysicalDevice>& gpu, uint32_t requiredQueueFlags)
{
	auto props = gpu->getQueueFamilyProperties();
	int currentIndex = 0;
	for (const auto& property : props)
	{
		if (property.queueFlags & requiredQueueFlags)
		{
			return currentIndex;
		}
		++currentIndex;
	}
	return -1;
}

int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	auto win = CWindowT::create(WIN_W, WIN_H, system::IWindow::ECF_NONE);

	video::SDebugCallback dbgcb;
	dbgcb.callback = &CommonAPI::defaultDebugCallback;
	dbgcb.userData = nullptr;
	auto gl = video::IAPIConnection::create(video::EAT_OPENGL, 0, "New API Test", &dbgcb);
	auto surface = gl->createSurface(win.get());

	auto gpus = gl->getPhysicalDevices();
	assert(!gpus.empty());
	auto gpu = gpus.begin()[0];
	int familyIndex = getQueueFamilyIndex(gpu, video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT | 
		video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_COMPUTE_BIT | 
		video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_TRANSFER_BIT);
	assert(surface->isSupported(gpu.get(), 0u));

	video::ILogicalDevice::SCreationParams dev_params;
	dev_params.queueParamsCount = 1u;
	video::ILogicalDevice::SQueueCreationParams q_params;
	q_params.familyIndex = 0u;
	q_params.count = 4u;
	q_params.flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
	float priority[4] = { 1.f,1.f,1.f,1.f };
	q_params.priorities = priority;
	dev_params.queueCreateInfos = &q_params;
	auto device = gpu->createLogicalDevice(dev_params);

	auto queue = device->getQueue(familyIndex, 0);

	core::smart_refctd_ptr<video::ISwapchain> sc = CommonAPI::createSwapchain(WIN_W, WIN_H, SC_IMG_COUNT, device, surface, video::ISurface::EPM_FIFO_RELAXED);
	assert(sc);

	core::smart_refctd_ptr<video::IGPURenderpass> renderpass = CommonAPI::createRenderpass(device);

	auto fbo = CommonAPI::createFBOWithSwapchainImages<SC_IMG_COUNT, WIN_W, WIN_H>(device, sc, renderpass);

	auto cmdpool = device->createCommandPool(familyIndex, static_cast<video::IGPUCommandPool::E_CREATE_FLAGS>(0));
	assert(cmdpool);

	core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool;
	{
		video::IDescriptorPool::E_CREATE_FLAGS flags = video::IDescriptorPool::ECF_FREE_DESCRIPTOR_SET_BIT;
		video::IDescriptorPool::SDescriptorPoolSize poolSize{ nbl::asset::E_DESCRIPTOR_TYPE::EDT_STORAGE_IMAGE, 2 };

		descriptorPool = device->createDescriptorPool(flags, 1, 1, &poolSize);
	}

	//TODO: Load inImgPair from "../../media/color_space_test/R8G8B8A8_2.png" instead of creating empty GPU IMAGE
	auto inImgPair = CommonAPI::createEmpty2DTexture(device, WIN_W, WIN_H);
	auto outImgPair = CommonAPI::createEmpty2DTexture(device, WIN_W, WIN_H);

	core::smart_refctd_ptr<video::IGPUImage> inImg = inImgPair.first;
	core::smart_refctd_ptr<video::IGPUImage> outImg = outImgPair.first;
	core::smart_refctd_ptr<video::IGPUImageView> inImgView = inImgPair.second;
	core::smart_refctd_ptr<video::IGPUImageView> outImgView = outImgPair.second;

	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> ds0layout;
	{
		video::IGPUDescriptorSetLayout::SBinding bnd[2];
		bnd[0].binding = 0u;
		bnd[0].type = asset::EDT_STORAGE_IMAGE;
		bnd[0].count = 1u;
		bnd[0].stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
		bnd[0].samplers = nullptr;
		bnd[1] = bnd[0];
		bnd[1].binding = 1u;
		ds0layout = device->createGPUDescriptorSetLayout(bnd, bnd + 2);
	}

	core::smart_refctd_ptr<video::IGPUDescriptorSet> ds0_gpu;
	ds0_gpu = device->createGPUDescriptorSet(descriptorPool.get(), ds0layout);

	core::smart_refctd_ptr<video::IGPUComputePipeline> compPipeline;
	core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
	{
		{
			asset::SPushConstantRange range;
			range.offset = 0u;
			range.size = sizeof(uint32_t) * 2u;
			range.stageFlags = asset::ISpecializedShader::ESS_COMPUTE;
			layout = device->createGPUPipelineLayout(&range, &range + 1, std::move(ds0layout));
		}
		core::smart_refctd_ptr<video::IGPUSpecializedShader> shader;
		{
			//TODO: Load from compute.comp instead of getting source from src
			auto cs_unspec = device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(src));
			asset::ISpecializedShader::SInfo csinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_COMPUTE, "cs");
			auto cs = device->createGPUSpecializedShader(cs_unspec.get(), csinfo);

			compPipeline = device->createGPUComputePipeline(nullptr, core::smart_refctd_ptr(layout), std::move(cs));
		}
	}


	{
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
		cb->end();
	}

	core::smart_refctd_ptr<video::IGPUCommandBuffer> cmdbuf[SC_IMG_COUNT];
	device->createCommandBuffers(cmdpool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMG_COUNT, cmdbuf);
	auto sc_images = sc->getImages();
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = cmdbuf[i];
		auto& fb = fbo[i];

		asset::IImage::SImageCopy region;
		region.dstOffset = { 0, 0, 0 };
		region.extent = { WIN_W, WIN_H, 0 };
		cb->begin(0);
		cb->bindDescriptorSets(nbl::asset::E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout.get(), 0, 1, (nbl::video::IGPUDescriptorSet**)&ds0_gpu.get());
		cb->pushConstants(layout.get(), 0, 0, sizeof(uint32_t) * 2u, &core::vector2du32_SIMD(WIN_W, WIN_H));
		cb->bindComputePipeline(compPipeline.get());
		cb->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);
		cb->copyImage(inImg.get(), nbl::asset::E_IMAGE_LAYOUT::EIL_UNDEFINED, sc_images.begin()[i].get(), nbl::asset::E_IMAGE_LAYOUT::EIL_UNDEFINED, 1, &region);
		
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		asset::VkRect2D area;
		region.srcOffset = { 0, 0, 0 };
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
		cb->endRenderPass();

		cb->end();
	}


	constexpr uint32_t FRAME_COUNT = 500000u;
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		CommonAPI::Present<SC_IMG_COUNT>(device, sc, cmdbuf, queue);
	}

	device->waitIdle();

	return 0;
}
