// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "nbl/ext/DebugDraw/Draw3DLineShaders.h"
#include "../../examples_tests/common/CommonAPI.h" // Temporary
using namespace nbl;
using namespace video;
using namespace scene;
using namespace asset;
using namespace ext;
using namespace DebugDraw;


CDraw3DLine::CDraw3DLine(video::ILogicalDevice* device,
	video::IGPUQueue* queue,
	video::ISwapchain* swapchain,
	video::IGPURenderpass* renderpass,
	video::IGPUCommandPool* cmdPool,
	core::smart_refctd_ptr<video::IGPUFramebuffer>* fbos,
	uint32_t W,
	uint32_t H,
	uint32_t SC_IMAGE_COUNT) :
	m_device(device), m_queue(queue),
	m_swapchain(swapchain), m_renderpass(renderpass),
	m_scImageCount(SC_IMAGE_COUNT), m_framebuffers(fbos),
	m_imgSize(W, H), m_commandBuffers(SC_IMAGE_COUNT)
{
	m_device->createCommandBuffers(cmdPool, video::IGPUCommandBuffer::EL_PRIMARY, SC_IMAGE_COUNT, m_commandBuffers.data());

	core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
	{
		asset::SPushConstantRange range;
		range.offset = 0u;
		range.size = sizeof(core::matrix4SIMD);
		range.stageFlags = asset::ISpecializedShader::ESS_VERTEX;
		layout = device->createGPUPipelineLayout(&range, &range + 1);
	}
	assert(layout);
	core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> rpindependent_pipeline;
	{

		auto vs_unspec = m_device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(Draw3DLineVertexShader));
		auto fs_unspec = m_device->createGPUShader(core::make_smart_refctd_ptr<asset::ICPUShader>(Draw3DLineFragmentShader));

		asset::ISpecializedShader::SInfo vsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX, "vs");
		auto vs = m_device->createGPUSpecializedShader(vs_unspec.get(), vsinfo);
		asset::ISpecializedShader::SInfo fsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT, "fs");
		auto fs = m_device->createGPUSpecializedShader(fs_unspec.get(), fsinfo);

		video::IGPUSpecializedShader* shaders[2]{ vs.get(), fs.get() };

		asset::SVertexInputParams vtxinput;
		vtxinput.attributes[0].binding = 0;
		vtxinput.attributes[0].format = asset::EF_R32G32_SFLOAT;
		vtxinput.attributes[0].relativeOffset = offsetof(S3DLineVertex, Position[0]);

		vtxinput.attributes[1].binding = 0;
		vtxinput.attributes[1].format = asset::EF_R32G32B32_SFLOAT;
		vtxinput.attributes[1].relativeOffset = offsetof(S3DLineVertex, Color[0]);

		vtxinput.bindings[0].inputRate = asset::EVIR_PER_VERTEX;
		vtxinput.bindings[0].stride = sizeof(S3DLineVertex);

		vtxinput.enabledAttribFlags = 0b0011;
		vtxinput.enabledBindingFlags = 0b0001;

		asset::SRasterizationParams raster;
		raster.depthTestEnable = 0;
		raster.depthWriteEnable = 0;
		raster.faceCullingMode = asset::EFCM_NONE;

		asset::SPrimitiveAssemblyParams primitive;
		primitive.primitiveType = asset::EPT_LINE_LIST;

		asset::SBlendParams blend;

		rpindependent_pipeline = m_device->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(layout), shaders, shaders + 2, vtxinput, blend, primitive, raster);
		assert(rpindependent_pipeline);
	}

	{
		video::IGPUGraphicsPipeline::SCreationParams gp_params;
		gp_params.rasterizationSamplesHint = asset::IImage::ESCF_1_BIT;
		gp_params.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(renderpass);
		gp_params.renderpassIndependent = rpindependent_pipeline;
		gp_params.subpassIx = 0u;

		m_pipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));
	}
	asset::SBufferBinding<video::IGPUBuffer> bindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = { 0u,core::smart_refctd_ptr<video::IGPUBuffer>(m_device->getDefaultUpStreamingBuffer()->getBuffer()) };
	m_meshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(rpindependent_pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{});
	m_meshBuffer->setIndexType(asset::EIT_UNKNOWN);
	m_meshBuffer->setIndexCount(2);
}

void CDraw3DLine::recordToCommandBuffer()
{
	for (uint32_t i = 0u; i < m_scImageCount; ++i)
	{
		auto cb = m_commandBuffers[i];
		auto fb = m_framebuffers[i];
		cb->reset(0);
		cb->begin(0);

		auto* buf = m_meshBuffer->getVertexBufferBindings()->buffer.get();
		size_t offset = 0u;
		//TODO: change the interfaces to take const pointer
		cb->bindVertexBuffers(0u, 1u, const_cast<video::IGPUBuffer**>(&buf), &offset);
		cb->pushConstants(const_cast<nbl::video::IGPUPipelineLayout*>(m_meshBuffer->getPipeline()->getLayout()), asset::ISpecializedShader::ESS_VERTEX, 0, sizeof(m_viewProj), &m_viewProj);
		cb->bindGraphicsPipeline(m_pipeline.get());
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = VkExtent2D{ uint32_t(m_imgSize.X), uint32_t(m_imgSize.Y) };
		info.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(m_renderpass);
		info.framebuffer = core::smart_refctd_ptr<video::IGPUFramebuffer>(fb);
		info.clearValueCount = 0u;
		info.clearValues = nullptr;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		cb->draw(m_lines.size() * 2, 1u, 0u, 0u);
		cb->endRenderPass();

		cb->end();
	}
}

void CDraw3DLine::draw(video::IGPUSemaphore* waitSem, video::IGPUSemaphore* signalSem, uint32_t imgNum)
{
	auto upStreamBuff = m_device->getDefaultUpStreamingBuffer();
	const void* lineData[1] = { m_lines.data() };

	const uint32_t sizes[1] = { sizeof(S3DLineVertex) * m_lines.size() * 2 };
	uint32_t offset[1] = { video::StreamingTransientDataBufferMT<>::invalid_address };
	upStreamBuff->multi_place(1u, (const void* const*)lineData, (uint32_t*)&offset, (uint32_t*)&sizes, (uint32_t*)&alignments);
	if (upStreamBuff->needsManualFlushOrInvalidate())
	{
		auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
		const video::IDriverMemoryAllocation::MappedMemoryRange mappedRange(upStreamMem, offset[0], sizes[0]);
		core::SRange<const video::IDriverMemoryAllocation::MappedMemoryRange> range(&mappedRange, &mappedRange + 1);
		m_device->flushMappedMemoryRanges(range);
	}

	m_meshBuffer->setBaseVertex(offset[0] / sizeof(S3DLineVertex));
	m_meshBuffer->setIndexCount(m_lines.size() * 2);

	CommonAPI::Submit(m_device, m_swapchain, m_commandBuffers.data(), m_queue, waitSem, signalSem, m_scImageCount, imgNum);

	upStreamBuff->multi_free(1u, (uint32_t*)&offset, (uint32_t*)&sizes, m_device->createFence(video::IGPUFence::ECF_SIGNALED_BIT));
}