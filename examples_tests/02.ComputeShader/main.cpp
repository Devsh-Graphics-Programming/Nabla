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



int main()
{
	constexpr uint32_t WIN_W = 1280;
	constexpr uint32_t WIN_H = 720;
	constexpr uint32_t SC_IMG_COUNT = 3u;

	auto initOutp = CommonAPI::Init<WIN_W, WIN_H, SC_IMG_COUNT>(video::EAT_OPENGL, "Compute Shader");
	auto win = initOutp.window;
	auto gl = initOutp.apiConnection;
	auto surface = initOutp.surface;
	auto device = initOutp.logicalDevice;
	auto queue = initOutp.queue;
	auto sc = initOutp.swapchain;
	auto renderpass = initOutp.renderpass;
	auto fbo = initOutp.fbo;
	auto cmdpool = initOutp.commandPool;

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
	{
		video::IGPUDescriptorSet::SWriteDescriptorSet write[2];
		video::IGPUDescriptorSet::SDescriptorInfo info[2];
		write[0].arrayElement = 0u;
		write[0].binding = 0u;
		write[0].count = 1u;
		write[0].descriptorType = asset::EDT_STORAGE_IMAGE;
		write[0].dstSet = ds0_gpu.get();
		info[0].desc = inImgView;
		info[0].image.imageLayout = asset::EIL_GENERAL;
		write[0].info = info;
		write[1] = write[0];
		write[1].binding = 1u;
		info[1].desc = outImgView;
		info[1].image.imageLayout = asset::EIL_GENERAL;
		write[1].info = info+1;
		device->updateDescriptorSets(2u, write, 0u, nullptr);
	}

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
			//TODO: Load from "../compute.comp" instead of getting source from src
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
	auto sc_images = sc->getImages();
	for (uint32_t i = 0u; i < SC_IMG_COUNT; ++i)
	{
		auto& cb = cmdbuf[i];
		auto& fb = fbo[i];

		asset::IImage::SImageCopy region;
		region.dstOffset = { 0, 0, 0 };
		region.srcOffset = { 0, 0, 0 };
		region.extent = { WIN_W, WIN_H, 1 };
		region.dstSubresource.baseArrayLayer = 0;
		region.dstSubresource.mipLevel = 0;
		region.dstSubresource.layerCount = 1;
		region.srcSubresource.baseArrayLayer = 0;
		region.srcSubresource.mipLevel = 0;
		region.srcSubresource.layerCount = 1;
		cb->begin(0);
		cb->bindDescriptorSets(nbl::asset::E_PIPELINE_BIND_POINT::EPBP_COMPUTE, layout.get(), 0, 1, (nbl::video::IGPUDescriptorSet**)&ds0_gpu.get());
		cb->pushConstants(layout.get(), asset::ISpecializedShader::ESS_COMPUTE, 0, sizeof(uint32_t) * 2u, &core::vector2du32_SIMD(WIN_W, WIN_H));
		cb->bindComputePipeline(compPipeline.get());
		cb->dispatch((WIN_W + 15u) / 16u, (WIN_H + 15u) / 16u, 1u);
		video::IGPUCommandBuffer::SImageMemoryBarrier b;
		b.dstQueueFamilyIndex = 0;
		b.srcQueueFamilyIndex = 0;
		b.image = outImg;
		b.newLayout = asset::EIL_UNDEFINED;
		b.oldLayout = asset::EIL_UNDEFINED;
		b.subresourceRange.baseArrayLayer = 0;
		b.subresourceRange.baseMipLevel = 0;
		b.subresourceRange.layerCount = 1;
		b.subresourceRange.levelCount = 1;
		b.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		b.barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
		cb->pipelineBarrier(asset::EPSF_COMPUTE_SHADER_BIT, asset::EPSF_TRANSFER_BIT, 0, 0u, nullptr, 0u, nullptr, 1, &b);
		cb->copyImage(outImg.get(), nbl::asset::E_IMAGE_LAYOUT::EIL_UNDEFINED, sc_images.begin()[i].get(), nbl::asset::E_IMAGE_LAYOUT::EIL_UNDEFINED, 1, &region);
		
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
		//cb->beginRenderPass(&info, asset::ESC_INLINE);
		//cb->endRenderPass();

		cb->end();
	}


	constexpr uint32_t FRAME_COUNT = 50000u;
	constexpr uint64_t MAX_TIMEOUT = 99999999999999ull; //ns
	for (uint32_t i = 0u; i < FRAME_COUNT; ++i)
	{
		auto img_acq_sem = device->createSemaphore();
		auto render1_finished_sem = device->createSemaphore();

		uint32_t imgnum = 0u;
		sc->acquireNextImage(MAX_TIMEOUT, img_acq_sem.get(), nullptr, &imgnum);

		CommonAPI::Submit(device.get(), sc.get(), cmdbuf, queue, img_acq_sem.get(), render1_finished_sem.get(), SC_IMG_COUNT, imgnum);

		CommonAPI::Present(device.get(), sc.get(), queue, render1_finished_sem.get(), imgnum);
	}

	device->waitIdle();

	return 0;
}
