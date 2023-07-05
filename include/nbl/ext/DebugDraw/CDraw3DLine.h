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

class CDraw3DLine : public core::IReferenceCounted
{
    public:
		static core::smart_refctd_ptr<CDraw3DLine> create(const core::smart_refctd_ptr<video::ILogicalDevice>& device)
		{
			return core::smart_refctd_ptr<CDraw3DLine>(new CDraw3DLine(device), core::dont_grab);
		}


		void setData(const core::matrix4SIMD& viewProjMat, const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
		{
			m_viewProj = viewProjMat;
			m_lines = linesData;

		}

		void clearData()
		{
			m_lines.clear();
		}

		void setLine(const core::matrix4SIMD& viewProjMat,
			float fromX, float fromY, float fromZ,
			float toX, float toY, float toZ,
			float r, float g, float b, float a
		)
		{
			m_lines = core::vector<std::pair<S3DLineVertex, S3DLineVertex>>{ std::pair(S3DLineVertex{{ fromX, fromY, fromZ }, { r, g, b, a }}, S3DLineVertex{{ toX, toY, toZ }, { r, g, b, a }}) };
		}

		void addLine(const core::matrix4SIMD& viewProjMat,
			float fromX, float fromY, float fromZ,
			float toX, float toY, float toZ,
			float r, float g, float b, float a
		)
		{
			m_lines.emplace_back(S3DLineVertex{{ fromX, fromY, fromZ }, { r, g, b, a }}, S3DLineVertex{{ toX, toY, toZ }, { r, g, b, a }});
		}

		void setLinesData(const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
		{
			m_lines = linesData;
		}

		void addLines(const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData)
		{
			m_lines.insert(m_lines.end(), linesData.begin(), linesData.end());
		}

		void setViewProjMatrix(const core::matrix4SIMD& viewProjMat)
		{
			m_viewProj = viewProjMat;
		}

		video::IGPURenderpassIndependentPipeline* getRenderpassIndependentPipeline()
		{
			return m_rpindependent_pipeline.get();
		}
		/*
			The function which records the debug draw call into the command buffer @cmdBuffer.
			The function assumes that the cmdBuffer is in the recording state, so you should call cmdBuffer->begin() 
			before calling CDraw3DLine::recordToCommandBuffer.
			The @graphics_pipeline parameter is the graphics pipeline, built up on the renderpass independent pipeline, which you can 
			retrieve with CDraw3DLine::getRenderpassIndependentPipeline()
		*/
		void recordToCommandBuffer(video::IGPUCommandBuffer* cmdBuffer, video::IGPUGraphicsPipeline* graphics_pipeline);

		inline void addBox(const core::aabbox3df& box, float r, float g, float b, float a, const core::matrix3x4SIMD& tform=core::matrix3x4SIMD())
		{
			auto addLine = [&](auto s, auto e) -> void
			{
				m_lines.emplace_back(S3DLineVertex{{s.X,s.Y,s.Z},{r,g,b,a}},S3DLineVertex{{e.X,e.Y,e.Z},{r,g,b,a}});
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
		// If @fence is not nullptr, you'll get a new fence assigned to @fence that you can wait for,
		// If @fence is nullptr, the function will automatically manage fence waiting
        void updateVertexBuffer(video::IUtilities* utilities, video::IQueue* queue, core::smart_refctd_ptr<video::IGPUFence>* fence = nullptr);
    private:
		CDraw3DLine(const core::smart_refctd_ptr<video::ILogicalDevice>& device);
		virtual ~CDraw3DLine() {}
	private:
		core::smart_refctd_ptr<video::ILogicalDevice> m_device;
		core::smart_refctd_ptr<video::IGPUBuffer> m_linesBuffer =  nullptr;
		core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> m_rpindependent_pipeline;
		core::matrix4SIMD m_viewProj;
		core::vector<std::pair<S3DLineVertex, S3DLineVertex>> m_lines;
        const uint32_t alignments[1] = { sizeof(S3DLineVertex) };
};

} // namespace DebugDraw
} // namespace ext
} // namespace nbl

#endif
