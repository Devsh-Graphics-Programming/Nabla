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
        static CDraw3DLine* create(video::IVideoDriver* _driver);

        void draw(
            float fromX, float fromY, float fromZ,
            float toX, float toY, float toZ,
            float r, float g, float b, float a
        );

        void draw(const core::vector<std::pair<S3DLineVertex, S3DLineVertex>>& linesData);

    private:
        CDraw3DLine(video::IVideoDriver* _driver);
        ~CDraw3DLine();

        video::IVideoDriver* m_driver;
        video::SGPUMaterial m_material;
        video::IGPUMeshDataFormatDesc* m_desc;
        video::IGPUMeshBuffer* m_meshBuffer;
        const uint32_t alignments[1] = { sizeof(S3DLineVertex) };
};

} // namespace DebugDraw
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_
