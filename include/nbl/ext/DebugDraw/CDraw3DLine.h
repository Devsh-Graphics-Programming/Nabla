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
        template<typename T>
        using shared_ptr = core::smart_refctd_ptr<T>;

        using CDraw3DLinePtr        = shared_ptr<CDraw3DLine>;
        using LogicalDevicePtr      = shared_ptr<video::ILogicalDevice>;
        using S3DLineVertexPairList = core::vector<std::pair<S3DLineVertex, S3DLineVertex>>;

        public:
          inline static CDraw3DLinePtr create(
            const LogicalDevicePtr& device,
            bool isDepthEnabled = false
          ) noexcept
          { return CDraw3DLinePtr(new CDraw3DLine(device, isDepthEnabled), core::dont_grab); }

        public:
          inline video::IGPURenderpassIndependentPipeline* getRenderpassIndependentPipeline() noexcept
          { return m_rpindependent_pipeline.get(); }

          inline void setData(
            const core::matrix4SIMD& viewProj,
            const S3DLineVertexPairList& linesData
          ) noexcept
          {
            m_viewProj = viewProj;
            m_lines = linesData;
          }

          inline void setViewProj(const core::matrix4SIMD& viewProj) noexcept
          { m_viewProj = viewProj; }

          inline void setLinesData(const S3DLineVertexPairList& linesData) noexcept
          { m_lines = linesData; }

          inline void setLine(
            const core::matrix4SIMD& viewProj,
            const core::vectorSIMDf& from, const core::vectorSIMDf& to,
            const core::vectorSIMDf& fromColor  = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f),
            const core::vectorSIMDf& toColor    = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f)
          ) noexcept
          {
            m_viewProj = viewProj;

            m_lines = S3DLineVertexPairList{
              std::pair(
                S3DLineVertex{{ from.x, from.y, from.z }, { fromColor.r, fromColor.g, fromColor.b, fromColor.a }},
                S3DLineVertex{{ to.x, to.y, to.z }, { toColor.r, toColor.g, toColor.b, toColor.a }}
              )
            };
          }

          inline void clearData() noexcept { m_lines.clear(); }

          inline void addLine(
            const core::matrix4SIMD& viewProj,
            const core::vectorSIMDf& from, const core::vectorSIMDf& to,
            const core::vectorSIMDf& fromColor  = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f),
            const core::vectorSIMDf& toColor    = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f)
          ) noexcept
          {
            m_viewProj = viewProj;

            m_lines.emplace_back(
              S3DLineVertex{{ from.x, from.y, from.z }, { fromColor.r, fromColor.g, fromColor.b, fromColor.a }},
              S3DLineVertex{{ to.x, to.y, to.z }, { toColor.r, toColor.g, toColor.b, toColor.a }}
            );
          }

          inline void addLine(
            const core::vectorSIMDf& from, const core::vectorSIMDf& to,
            const core::vectorSIMDf& fromColor  = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f),
            const core::vectorSIMDf& toColor    = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f)
          ) noexcept
          {
            m_lines.emplace_back(
              S3DLineVertex{{ from.x, from.y, from.z }, { fromColor.r, fromColor.g, fromColor.b, fromColor.a }},
              S3DLineVertex{{ to.x, to.y, to.z }, { toColor.r, toColor.g, toColor.b, toColor.a }}
            );
          }

          inline void addLines(const S3DLineVertexPairList& linesData) noexcept
          { m_lines.insert(m_lines.end(), linesData.begin(), linesData.end()); }

          // to use for both AABB and OBB (cannot/should not alter bbox here!)
          template<class TBox = core::aabbox3dsf>
          inline void addBox(
            const TBox& box,
            const core::matrix3x4SIMD& tform    = core::matrix3x4SIMD(),
            const core::vectorSIMDf& fromColor  = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f),
            const core::vectorSIMDf& toColor    = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f)
          ) noexcept
          {
            static_assert(
              std::is_base_of<core::aabbox3dsf, TBox>::value ||
              std::is_base_of<core::aabbox3df,  TBox>::value ||
              std::is_base_of<core::aabbox3di,  TBox>::value,
              "TBox should derive from aabbox3d<T, UVector>"
            );

            core::vectorSIMDf verts[10];

            box.getEdges(verts);

            for(auto i=0; i<10; i++)
            { tform.pseudoMulWith4x1(verts[i]); }

            addLine(verts[0], verts[1], fromColor, toColor);
            addLine(verts[0], verts[2], fromColor, toColor);
            addLine(verts[1], verts[3], fromColor, toColor);
            addLine(verts[2], verts[3], fromColor, toColor);

            addLine(verts[0], verts[4], fromColor, toColor);
            addLine(verts[1], verts[5], fromColor, toColor);
            addLine(verts[2], verts[6], fromColor, toColor);
            addLine(verts[3], verts[7], fromColor, toColor);

            addLine(verts[4], verts[5], fromColor, toColor);
            addLine(verts[4], verts[6], fromColor, toColor);
            addLine(verts[5], verts[7], fromColor, toColor);
            addLine(verts[6], verts[7], fromColor, toColor);

            // TODO: remove once done testing
            addLine(verts[8], verts[9], core::vectorSIMDf(1,0,0,1), core::vectorSIMDf(0,0,1,1));
          }

          inline void addKDOP(
            const core::OBB& dito,
            const core::matrix3x4SIMD& tform    = core::matrix3x4SIMD(),
            const core::vectorSIMDf& fromColor  = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f),
            const core::vectorSIMDf& toColor    = core::vectorSIMDf(0.f, 1.f, 0.f, 1.f)
          ) noexcept
          {}

        public:
          // If @fence is not nullptr, you'll get a new fence assigned to @fence that you can wait for,
          // If @fence is nullptr, the function will automatically manage fence waiting
          void updateVertexBuffer(
            video::IUtilities* utilities,
            video::IGPUQueue* queue,
            shared_ptr<video::IGPUFence>* fence = nullptr
          ) noexcept;

          /*
            The function which records the debug draw call into the command buffer @cmdBuffer.
            The function assumes that the cmdBuffer is in the recording state, so you should call cmdBuffer->begin()
            before calling CDraw3DLine::recordToCommandBuffer.
            The @graphics_pipeline parameter is the graphics pipeline, built up on the renderpass independent pipeline, which you can
            retrieve with CDraw3DLine::getRenderpassIndependentPipeline()
          */
          void recordToCommandBuffer(
            video::IGPUCommandBuffer* cmdBuffer,
            video::IGPUGraphicsPipeline* graphics_pipeline,
            uint32_t pushConstOffset = 0
          ) noexcept;

        private:
          explicit CDraw3DLine(const LogicalDevicePtr& device, bool isDepthEnabled = false);
          ~CDraw3DLine() override = default;

        private:
          LogicalDevicePtr m_device;
          shared_ptr<video::IGPUBuffer> m_linesBuffer = nullptr;
          shared_ptr<video::IGPURenderpassIndependentPipeline> m_rpindependent_pipeline;
          core::matrix4SIMD m_viewProj;
          S3DLineVertexPairList m_lines;

          const uint32_t alignments[1] = { sizeof(S3DLineVertex) };
      };
    } // namespace DebugDraw
  } // namespace ext
} // namespace nbl

#endif
