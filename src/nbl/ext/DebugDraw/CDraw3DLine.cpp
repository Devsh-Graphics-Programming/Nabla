// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/ext/DebugDraw/CDraw3DLine.h"
#include "nbl/ext/DebugDraw/Draw3DLineShaders.h"
//#include "../../examples_tests/common/CommonAPI.h" // Temporary
using namespace nbl;
using namespace video;
using namespace scene;
using namespace asset;
using namespace ext;
using namespace DebugDraw;



void CDraw3DLine::recordToCommandBuffer()
{
	for (uint32_t i = 0u; i < m_scImageCount; ++i)
	{
		auto cb = m_commandBuffers[i];
		auto fb = m_framebuffers[i];

		cb->begin(0);

		auto* buf = m_meshBuffer->getVertexBufferBindings()->buffer.get();
		size_t offset = 0u;
		cb->bindVertexBuffers(0u, 1u, const_cast<video::IGPUBuffer**>(&buf), &offset);
		cb->bindGraphicsPipeline(m_pipeline.get());
		video::IGPUCommandBuffer::SRenderpassBeginInfo info;
		asset::SClearValue clear;
		asset::VkRect2D area;
		area.offset = { 0, 0 };
		area.extent = VkExtent2D{ uint32_t(m_imgSize.X), uint32_t(m_imgSize.Y) };
		clear.color.float32[0] = 1.f;
		clear.color.float32[1] = 0.f;
		clear.color.float32[2] = 0.f;
		clear.color.float32[3] = 1.f;
		info.renderpass = core::smart_refctd_ptr<video::IGPURenderpass>(m_renderpass);
		info.framebuffer = core::smart_refctd_ptr<video::IGPUFramebuffer>(fb);
		info.clearValueCount = 1u;
		info.clearValues = &clear;
		info.renderArea = area;
		cb->beginRenderPass(&info, asset::ESC_INLINE);
		cb->draw(2u, 1u, 0u, 0u);
		cb->endRenderPass();

		cb->end();
	}


}

void CDraw3DLine::draw(const core::matrix4SIMD& viewProjMat,
	float fromX, float fromY, float fromZ,
	float toX, float toY, float toZ,
	float r, float g, float b, float a)
{
	S3DLineVertex vertices[2] = {
		{{ fromX, fromY, fromZ }, { r, g, b, a }},
		{{ toX, toY, toZ }, { r, g, b, a }}
	};

	auto upStreamBuff = m_device->getDefaultUpStreamingBuffer();
	void* lineData[1] = { vertices };

	static const uint32_t sizes[1] = { sizeof(S3DLineVertex) * 2 };
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


	recordToCommandBuffer();
	
	upStreamBuff->multi_free(1u, (uint32_t*)&offset, (uint32_t*)&sizes, m_device->createFence(video::IGPUFence::ECF_SIGNALED_BIT));

	//TODO: real image count
	//CommonAPI::Present<3>(m_device, m_swapchain, m_commandBuffers.data(), m_queue);
}

//void CDraw3DLine::draw(const core::matrix4SIMD& viewProjMat, const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
//{
//	auto upStreamBuff = m_driver->getDefaultUpStreamingBuffer();
//	const void* lineData[1] = { linesData.data() };
//
//	const uint32_t sizes[1] = { sizeof(S3DLineVertex) * linesData.size() * 2 };
//	uint32_t offset[1] = { video::StreamingTransientDataBufferMT<>::invalid_address };
//	upStreamBuff->multi_place(1u, (const void* const*)lineData, (uint32_t*)&offset, (uint32_t*)&sizes, (uint32_t*)&alignments);
//	if (upStreamBuff->needsManualFlushOrInvalidate())
//	{
//		auto upStreamMem = upStreamBuff->getBuffer()->getBoundMemory();
//		m_driver->flushMappedMemoryRanges({ { upStreamMem,offset[0],sizes[0] } });
//	}
//
//	m_meshBuffer->setBaseVertex(offset[0] / sizeof(S3DLineVertex));
//	m_meshBuffer->setIndexCount(linesData.size() * 2);
//
//	m_driver->bindGraphicsPipeline(m_meshBuffer->getPipeline());
//	m_driver->pushConstants(m_meshBuffer->getPipeline()->getLayout(), ISpecializedShader::ESS_VERTEX, 0u, sizeof(core::matrix4SIMD), viewProjMat.pointer());
//
//	m_driver->drawMeshBuffer(m_meshBuffer.get());
//
//	upStreamBuff->multi_free(1u, (uint32_t*)&offset, (uint32_t*)&sizes, std::move(m_driver->placeFence()));
//}