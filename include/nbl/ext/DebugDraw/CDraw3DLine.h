// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXT_C_DRAW_3D_LINE_INCLUDED_
#define _NBL_EXT_C_DRAW_3D_LINE_INCLUDED_

#include "nabla.h"
#include "nbl/ext/DebugDraw/Draw3DLineShaders.h"

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
		static core::smart_refctd_ptr<CDraw3DLine> create(const core::smart_refctd_ptr<video::ILogicalDevice>& device,
			const core::smart_refctd_ptr<video::ISwapchain>& swapchain,
			const core::smart_refctd_ptr<video::IGPURenderpass>& renderpass,
			video::IGPUQueue* queue,
			video::IGPUCommandPool* commandPool,
			core::smart_refctd_ptr<video::IGPUFramebuffer>* fbos)
		{
			return core::smart_refctd_ptr<CDraw3DLine>(new CDraw3DLine(device, swapchain, renderpass, queue, commandPool, fbos, W, H, SC_IMAGE_COUNT), core::dont_grab);
		}

        void draw(video::IGPUSemaphore* imgAcqSem, video::IGPUSemaphore* renderFinishedSem, uint32_t imgNum);

		void setData(const core::matrix4SIMD& viewProjMat, const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
		{
			m_viewProj = viewProjMat;
			m_lines = linesData;
			recordToCommandBuffer();
		}

		void setLine(const core::matrix4SIMD& viewProjMat,
			float fromX, float fromY, float fromZ,
			float toX, float toY, float toZ,
			float r, float g, float b, float a
		)
		{
			m_lines = core::vector<std::pair<S3DLineVertex, S3DLineVertex>>{ std::pair(S3DLineVertex{{ fromX, fromY, fromZ }, { r, g, b, a }}, S3DLineVertex{{ toX, toY, toZ }, { r, g, b, a }}) };
			recordToCommandBuffer();
		}

		void setLinesData(const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
		{
			m_lines = linesData;
			recordToCommandBuffer();
		}

		void setViewProjMatrix(const core::matrix4SIMD& viewProjMat)
		{
			m_viewProj = viewProjMat;
			recordToCommandBuffer();
		}

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
		CDraw3DLine(const core::smart_refctd_ptr<video::ILogicalDevice>& device,
			const core::smart_refctd_ptr<video::ISwapchain>& swapchain,
			const core::smart_refctd_ptr<video::IGPURenderpass>& renderpass,
			video::IGPUQueue* queue,
			video::IGPUCommandPool* commandPool,
			core::smart_refctd_ptr<video::IGPUFramebuffer>* fbos,
			uint32_t W,
			uint32_t H,
			uint32_t SC_IMAGE_COUNT);
		virtual ~CDraw3DLine() {}
		void recordToCommandBuffer();
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::ISwapchain> m_swapchain;
		core::smart_refctd_ptr<video::IGPURenderpass> m_renderpass;
		video::IGPUQueue* m_queue;
        nbl::core::smart_refctd_ptr<video::IGPUMeshBuffer> m_meshBuffer;
		uint32_t m_scImageCount;
		core::vector2di m_imgSize;
		core::vector<core::smart_refctd_ptr<video::IGPUCommandBuffer>> m_commandBuffers;
		std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>> m_framebuffers;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_pipeline;

		core::matrix4SIMD m_viewProj;
		core::vector<std::pair<S3DLineVertex, S3DLineVertex>> m_lines;
        const uint32_t alignments[1] = { sizeof(S3DLineVertex) };
};

} // namespace DebugDraw
} // namespace ext
} // namespace nbl

#endif
