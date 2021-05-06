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


CDraw3DLine::CDraw3DLine(const core::smart_refctd_ptr<video::ILogicalDevice>& device) :
	m_device(core::smart_refctd_ptr<video::ILogicalDevice>(device))
{
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

	asset::SBufferBinding<video::IGPUBuffer> bindings[video::IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = { 0u,core::smart_refctd_ptr<video::IGPUBuffer>(m_device->getDefaultUpStreamingBuffer()->getBuffer()) };
	m_meshBuffer = core::make_smart_refctd_ptr<video::IGPUMeshBuffer>(std::move(rpindependent_pipeline), nullptr, bindings, asset::SBufferBinding<video::IGPUBuffer>{});
	m_meshBuffer->setIndexType(asset::EIT_UNKNOWN);
	m_meshBuffer->setIndexCount(2);
}

void CDraw3DLine::recordToCommandBuffer(video::IGPUCommandBuffer* cmdBuffer, video::IGPUGraphicsPipeline* graphics_pipeline)
{

	auto cb = cmdBuffer;
	assert(cb->getState() == IGPUCommandBuffer::ES_RECORDING);
	auto* buf = m_meshBuffer->getVertexBufferBindings()->buffer.get();
	size_t offset = 0u;
	cb->bindVertexBuffers(0u, 1u, &buf, &offset);
	cb->pushConstants(const_cast<nbl::video::IGPUPipelineLayout*>(m_meshBuffer->getPipeline()->getLayout()), asset::ISpecializedShader::ESS_VERTEX, 0, sizeof(m_viewProj), &m_viewProj);
	cb->bindGraphicsPipeline(graphics_pipeline);
	cb->draw(m_lines.size() * 2, 1u, 0u, 0u);

}

void CDraw3DLine::updateMeshBuffer()
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

	upStreamBuff->multi_free(1u, (uint32_t*)&offset, (uint32_t*)&sizes, m_device->createFence(video::IGPUFence::ECF_SIGNALED_BIT));
}