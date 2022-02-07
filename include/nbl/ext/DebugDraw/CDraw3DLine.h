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
    static core::smart_refctd_ptr<CDraw3DLine> create(video::IVideoDriver* _driver);

    void draw(const core::matrix4SIMD& viewProjMat,
        float fromX, float fromY, float fromZ,
        float toX, float toY, float toZ,
        float r, float g, float b, float a);

    void draw(const core::matrix4SIMD& viewProjMat, const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData);

    inline void enqueueBox(core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData, const core::aabbox3df& box, float r, float g, float b, float a, const core::matrix3x4SIMD& tform = core::matrix3x4SIMD())
    {
        auto addLine = [&](auto s, auto e) -> void {
            linesData.emplace_back(S3DLineVertex{{s.X, s.Y, s.Z}, {r, g, b, a}}, S3DLineVertex{{e.X, e.Y, e.Z}, {r, g, b, a}});
        };

        core::vectorSIMDf verts[8];
        box.getEdges(verts);
        for(auto i = 0; i < 8; i++)
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
    CDraw3DLine(video::IVideoDriver* _driver);
    virtual ~CDraw3DLine() {}

    core::smart_refctd_ptr<video::IVideoDriver> m_driver;
    core::smart_refctd_ptr<video::IGPUMeshBuffer> m_meshBuffer;
    const uint32_t alignments[1] = {sizeof(S3DLineVertex)};
};

}  // namespace DebugDraw
}  // namespace ext
}  // namespace nbl

#endif
