// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_C_DRAW_3D_LINE_INCLUDED_
#define _NBL_EXT_C_DRAW_3D_LINE_INCLUDED_

#include "nabla.h"

namespace nbl
{
namespace ext
{
namespace DebugDraw
{


#include "nbl/nblpack.h"
struct S3DLineVertex
{
    float Position[3];
    float Color[4];
} PACK_STRUCT;
#include "nbl/nblunpack.h"

class CDraw3DLine : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		template<uint32_t SC_IMAGE_COUNT, uint32_t W, uint32_t H>	
		static core::smart_refctd_ptr<CDraw3DLine> create(video::ILogicalDevice* _device,
			video::IGPUQueue* queue,
			video::ISwapchain* swapchain,
			video::IGPURenderpass* renderpass,
			video::IGPUCommandPool* commandPool)
		{
			return core::smart_refctd_ptr<CDraw3DLine>(new CDraw3DLine<SC_IMAGE_COUNT, W, H>(device, queue, swapchain, renderpass, commandPool, scImageCount), core::dont_grab);
		}

        void draw(const core::matrix4SIMD& viewProjMat,
            float fromX, float fromY, float fromZ,
            float toX, float toY, float toZ,
            float r, float g, float b, float a
        );

        void draw(const core::matrix4SIMD& viewProjMat, const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData);

		inline void enqueueBox(core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData, const core::aabbox3df& box, float r, float g, float b, float a, const core::matrix3x4SIMD& tform=core::matrix3x4SIMD())
		{
			auto addLine = [&](auto s, auto e) -> void
			{
				linesData.emplace_back(S3DLineVertex{{s.X,s.Y,s.Z},{r,g,b,a}},S3DLineVertex{{e.X,e.Y,e.Z},{r,g,b,a}});
			};

			core::vectorSIMDf verts[8];
			box.getEdges(verts);
			for (auto i=0; i<8; i++)
				tform.pseudoMulWith4x1(verts[i]);

			addLine(verts[0], verts[1]);
			addLine(verts[0], verts[2]);
			addLine(verts[1], verts[3]);
			addLine(verts[2], verts[3]);

			addLine(verts[0], verts[4]);
			addLine(verts[1], verts[5]);
			addLine(verts[2], verts[6]);
			addLine(verts[3], verts[7]);

			addLine(verts[4], verts[5]);
			addLine(verts[4], verts[6]);
			addLine(verts[5], verts[7]);
			addLine(verts[6], verts[7]);
		}

    private:
		template<uint32_t SC_IMAGE_COUNT, uint32_t W, uint32_t H>
		CDraw3DLine(video::ILogicalDevice* _device,
			video::IGPUQueue* queue,
			video::ISwapchain* swapchain,
			video::IGPURenderpass* renderpass,
			video::IGPUCommandPool* commandPool,
			core::smart_refctd_ptr<video::IGPUFramebuffer> fbos[SC_IMAGE_COUNT]);
		virtual ~CDraw3DLine() {}
		void recordToCommandBuffer();

        video::ILogicalDevice* m_device;
		video::IGPUQueue* m_queue;
		video::ISwapchain* m_swapchain;
		video::IGPURenderpass* m_renderpass;
		video::IGPUCommandPool* m_commandPool;
        video::IGPUMeshBuffer* m_meshBuffer;
		uint32_t m_scImageCount;
		core::vector2di m_imgSize;
		core::vector<core::smart_refctd_ptr<video::IGPUCommandBuffer>> m_commandBuffers;
        core::vector<video::IGPUFramebuffer*> m_framebuffers;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_pipeline;
        const uint32_t alignments[1] = { sizeof(S3DLineVertex) };
};

template<uint32_t SC_IMAGE_COUNT, uint32_t W, uint32_t H>
CDraw3DLine::CDraw3DLine(video::ILogicalDevice* _device,
	video::IGPUQueue* queue,
	video::ISwapchain* swapchain,
	video::IGPURenderpass* renderpass,
	video::IGPUCommandPool* commandPool,
	core::smart_refctd_ptr<video::IGPUFramebuffer> fbos[SC_IMAGE_COUNT]) :
	m_device(device), m_queue(queue),
	m_swapchain(swapchain), m_renderpass(renderpass),
	m_commandPool(commandPool), m_scImageCount(scImageCount),
	m_commandBuffers(scImageCount), m_framebuffers(scImageCount), m_imgSize(W, H)
{
	m_device->createCommandBuffers(m_commandPool.get(), video::IGPUCommandBuffer::EL_PRIMARY, SC_IMAGE_COUNT, m_commandBuffers.data());
	for (int i = 0; i < SC_IMAGE_COUNT; ++i) m_framebuffers[i] = fbos[i].get();
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
		gp_params.renderpass = renderpass;
		gp_params.renderpassIndependent = rpindependent_pipeline;
		gp_params.subpassIx = 0u;

		m_pipeline = device->createGPUGraphicsPipeline(nullptr, std::move(gp_params));
	}
	SBufferBinding<IGPUBuffer> bindings[IGPUMeshBuffer::MAX_ATTR_BUF_BINDING_COUNT];
	bindings[0u] = { 0u,core::smart_refctd_ptr<video::IGPUBuffer>(m_device->getDefaultUpStreamingBuffer()->getBuffer()) };
	m_meshBuffer = core::make_smart_refctd_ptr<IGPUMeshBuffer>(std::move(m_pipeline), nullptr, bindings, SBufferBinding<IGPUBuffer>{});
	m_meshBuffer->setIndexType(EIT_UNKNOWN);
	m_meshBuffer->setIndexCount(2);
}
} // namespace DebugDraw
} // namespace ext
} // namespace nbl

#endif
