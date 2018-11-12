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
struct S3DLine
{
    float Start[3];
    float End[3];
    std::uint32_t Color[4];
} PACK_STRUCT;
#include "irr/irrunpack.h"

class CDraw3DLine : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
        static CDraw3DLine* create(video::IVideoDriver* _driver);

        void draw(const S3DLine& line);

    private:
        CDraw3DLine(video::IVideoDriver* _driver);
        ~CDraw3DLine();

    video::IVideoDriver* m_driver;
    video::SMaterial m_material;
    scene::IGPUMeshDataFormatDesc* m_desc;

    scene::IGPUMeshBuffer* m_meshBuffer;
    void* m_lineData[3];
    static constexpr uint32_t offsets[3] =
        {
        video::StreamingTransientDataBufferMT<>::invalid_address,
        video::StreamingTransientDataBufferMT<>::invalid_address
        };

    static constexpr uint32_t alignments[3] =
        {
        sizeof(decltype(S3DLine::Start[0])),
        sizeof(decltype(S3DLine::End[0])),
        sizeof(decltype(S3DLine::Color[0]))
        };

    static constexpr uint32_t sizes[3] =
        { sizeof(S3DLine::Start), sizeof(S3DLine::End), sizeof(S3DLine::Color) };
};

} // namespace draw
} // namespace ext
} // namespace irr

#endif // _IRR_EXT_C_DRAW_3D_LINE_INCLUDED_
