#ifndef _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_
#define _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_

#include "irrlicht.h"

namespace irr
{
namespace ext
{
namespace draw
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
        // services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};

#include "irr/irrpack.h"
struct S3DLineVertex
{
    float Position[3];
    std::uint32_t Color[4];
} PACK_STRUCT;
#include "irr/irrunpack.h"

class CDraw3DLine : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
        static CDraw3DLine* create(video::IVideoDriver* _driver);

        void draw(
            float fromX, float fromY, float fromZ,
            float toX, float toY, float toZ,
            std::uint32_t r, std::uint32_t g, std::uint32_t b, std::uint32_t a
        );

    private:
        CDraw3DLine(video::IVideoDriver* _driver);
        ~CDraw3DLine();

    video::IVideoDriver* m_driver;
    video::SMaterial m_material;
    scene::IGPUMeshDataFormatDesc* m_desc;

    scene::IGPUMeshBuffer* m_meshBuffer;
    void* m_lineData[2];

    static const std::uint16_t m_indices[2];
    static uint32_t offsets[2];
    static const uint32_t alignments[2];
    static const uint32_t sizes[2];
};

} // namespace draw
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_
