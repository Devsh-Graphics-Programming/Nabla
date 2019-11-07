#ifndef _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_
#define _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace DebugDraw
{

class Draw3DLineCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    Draw3DLineCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};

#include "irr/irrpack.h"
struct S3DLineVertex
{
    float Position[3];
    float Color[4];
} PACK_STRUCT;
#include "irr/irrunpack.h"

class CDraw3DLine : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
        static core::smart_refctd_ptr<CDraw3DLine> create(video::IVideoDriver* _driver);

        void draw(
            float fromX, float fromY, float fromZ,
            float toX, float toY, float toZ,
            float r, float g, float b, float a
        );

        void draw(const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData);

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
        CDraw3DLine(video::IVideoDriver* _driver);
		virtual ~CDraw3DLine() {}

        core::smart_refctd_ptr<video::IVideoDriver> m_driver;
        video::SGPUMaterial m_material;
        core::smart_refctd_ptr<video::IGPUMeshBuffer> m_meshBuffer;
        const uint32_t alignments[1] = { sizeof(S3DLineVertex) };
};

} // namespace DebugDraw
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_
