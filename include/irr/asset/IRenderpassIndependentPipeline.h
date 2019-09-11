#ifndef __IRR_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/asset/format/EFormat.h"
#include "irr/core/math/irrMath.h"
#include "irr/asset/ShaderCommons.h"
#include <algorithm>

namespace irr {
namespace asset
{

enum E_PRIMITIVE_TYPE
{
    //! All vertices are non-connected points.
    EPT_POINTS = 0,

    //! All vertices form a single connected line.
    EPT_LINE_STRIP,

    //! Just as LINE_STRIP, but the last and the first vertex is also connected.
    EPT_LINE_LOOP,

    //! Every two vertices are connected creating n/2 lines.
    EPT_LINES,

    //! After the first two vertices each vertex defines a new triangle.
    //! Always the two last and the new one form a new triangle.
    EPT_TRIANGLE_STRIP,

    //! After the first two vertices each vertex defines a new triangle.
    //! All around the common first vertex.
    EPT_TRIANGLE_FAN,

    //! Explicitly set all vertices for each triangle.
    EPT_TRIANGLES

    // missing adjacency types and patches
};

struct SVertexInputParams
{
    bool enabled;
    bool perInstance;
    E_FORMAT format;
    uint32_t stride;
    uint32_t offset;
};

struct SPrimitiveAssemblyParams
{
    E_PRIMITIVE_TYPE primitiveType;
    bool primitiveRestartEnable;
    uint32_t tessPatchVertCount;
};

enum E_STENCIL_OP
{
    ESO_KEEP = 0,
    ESO_ZERO = 1,
    ESO_REPLACE = 2,
    ESO_INCREMENT_AND_CLAMP = 3,
    ESO_DECREMENT_AND_CLAMP = 4,
    ESO_INVERT = 5,
    ESO_INCREMENT_AND_WRAP = 6,
    ESO_DECREMENT_AND_WRAP = 7
};

enum E_COMPARE_OP
{
    ECO_NEVER = 0,
    ECO_LESS = 1,
    ECO_EQUAL = 2,
    ECO_LESS_OR_EQUAL = 3,
    ECO_GREATER = 4,
    ECO_NOT_EQUAL = 5,
    ECO_GREATER_OR_EQUAL = 6,
    ECO_ALWAYS = 7
};

struct SStencilOpParams
{
    E_STENCIL_OP failOp;
    E_STENCIL_OP passOp;
    E_STENCIL_OP depthFailOp;
    E_COMPARE_OP compareOp;
};

enum E_POLYGON_MODE
{
    EPM_FILL = 0,
    EPM_LINE = 1,
    EPM_POINT = 2
};

enum E_FACE_CULL_MODE
{
    EFCM_NONE = 0,
    EFCM_FRONT_BIT = 1,
    EFCM_BACK_BIT = 2,
    EFCM_FRONT_AND_BACK = 3
};

enum E_SAMPLE_COUNT
{
    ESC_1_BIT = 0x00000001,
    ESC_2_BIT = 0x00000002,
    ESC_4_BIT = 0x00000004,
    ESC_8_BIT = 0x00000008,
    ESC_16_BIT = 0x00000010,
    ESC_32_BIT = 0x00000020,
    ESC_64_BIT = 0x00000040
};

struct SRaserizationParams
{
    SStencilOpParams frontStencilOps;
    SStencilOpParams backStencilOps;
    uint32_t viewportCount;
    E_POLYGON_MODE polygonMode;
    E_FACE_CULL_MODE faceCullingMode;
    E_SAMPLE_COUNT raserizationSamplesHint;
    uint32_t sampleMask[2];
    E_COMPARE_OP depthCompareOp;
    float minSampleShading;
    bool depthClampEnable;
    bool rasterizerDiscard;
    bool fronFaceIsCCW;
    bool depthBiasEnable;
    bool sampleShadingEnable;
    bool alphaToCoverageEnable;
    bool alphaToOneEnable;
    bool depthTestEnable;
    bool depthWriteEnable;
    bool depthBoundsTestEnable;
    bool stencilTestEnable;
};

enum E_LOGIC_OP
{
    ELO_CLEAR = 0,
    ELO_AND = 1,
    ELO_AND_REVERSE = 2,
    ELO_COPY = 3,
    ELO_AND_INVERTED = 4,
    ELO_NO_OP = 5,
    ELO_XOR = 6,
    ELO_OR = 7,
    ELO_NOR = 8,
    ELO_EQUIVALENT = 9,
    ELO_INVERT = 10,
    ELO_OR_REVERSE = 11,
    ELO_COPY_INVERTED = 12,
    ELO_OR_INVERTED = 13,
    ELO_NAND = 14,
    ELO_SET = 15
};

enum E_BLEND_FACTOR
{
    EBF_ZERO = 0,
    EBF_ONE = 1,
    EBF_SRC_COLOR = 2,
    EBF_ONE_MINUS_SRC_COLOR = 3,
    EBF_DST_COLOR = 4,
    EBF_ONE_MINUS_DST_COLOR = 5,
    EBF_SRC_ALPHA = 6,
    EBF_ONE_MINUS_SRC_ALPHA = 7,
    EBF_DST_ALPHA = 8,
    EBF_ONE_MINUS_DST_ALPHA = 9,
    EBF_CONSTANT_COLOR = 10,
    EBF_ONE_MINUS_CONSTANT_COLOR = 11,
    EBF_CONSTANT_ALPHA = 12,
    EBF_ONE_MINUS_CONSTANT_ALPHA = 13,
    EBF_SRC_ALPHA_SATURATE = 14,
    EBF_SRC1_COLOR = 15,
    EBF_ONE_MINUS_SRC1_COLOR = 16,
    EBF_SRC1_ALPHA = 17,
    EBF_ONE_MINUS_SRC1_ALPHA = 18
};

enum E_BLEND_OP
{
    EBO_ADD = 0,
    EBO_SUBTRACT = 1,
    EBO_REVERSE_SUBTRACT = 2,
    EBO_MIN = 3
};

struct SColorAttachmentBlendParams
{
    uint32_t attachmentEnabled : 1;
    uint32_t blendEnable : 1;
    uint32_t srcColorFactor : 5;
    uint32_t dstColorFactor : 5;
    uint32_t colorBlendOp : 2;
    uint32_t srcAlphaFactor : 5;
    uint32_t dstAlphaFactor : 5;
    uint32_t alphaBlendOp : 2;
    uint32_t colorWriteMask : 4;
};

struct SBlendParams
{
    bool logicOpEnable;
    E_LOGIC_OP logicOp;
    SColorAttachmentBlendParams blendParams[8];
};

template<typename SpecShaderType, typename LayoutType>
class IRenderpassIndependentPipeline
{
protected:
    IRenderpassIndependentPipeline(
        core::smart_refctd_ptr<LayoutType> _layout,
        core::smart_refctd_ptr<SpecShaderType> _vs,
        core::smart_refctd_ptr<SpecShaderType> _tcs,
        core::smart_refctd_ptr<SpecShaderType> _tes,
        core::smart_refctd_ptr<SpecShaderType> _gs,
        core::smart_refctd_ptr<SpecShaderType> _fs,
        const SVertexInputParams* _vertexInputParams,
        SBlendParams _blendParams,
        SPrimitiveAssemblyParams _primAsmParams,
        SRaserizationParams _rasterParams
    ) : m_layout(_layout),
        m_shaders{_vs, _tcs, _tes, _gs, _fs},
        m_blendParams(_blendParams),
        m_primAsmParams(_primAsmParams),
        m_rasterParams(_rasterParams)
    {
        std::copy(_vertexInputParams, _vertexInputParams+16, m_vertexInputParams);
    }
    virtual ~IRenderpassIndependentPipeline() = default;

public:
    inline const LayoutType* getLayout() const { return m_layout.get(); }
    inline const SpecShaderType* getShaderAtStage(E_SHADER_STAGE _stage) const { return m_shaders[core::findLSB<uint32_t>(_stage)].get(); }

    inline const SBlendParams& getBlendParams() const { return m_blendParams; }
    inline const SPrimitiveAssemblyParams& getPrimitiveAssemblyParams() const { return m_primAsmParams; }
    inline const SRaserizationParams& getRasterizationParams() const { return m_rasterParams; }
    inline const SVertexInputParams* getVertexInputParams() const { return m_vertexInputParams; }
    inline const SVertexInputParams& getVertexInputParams(uint32_t _ix) const { return m_vertexInputParams[_ix]; }

protected:
    core::smart_refctd_ptr<LayoutType> m_layout;
    core::smart_refctd_ptr<SpecShaderType> m_shaders[5];

    SBlendParams m_blendParams;
    SPrimitiveAssemblyParams m_primAsmParams;
    SRaserizationParams m_rasterParams;
    SVertexInputParams m_vertexInputParams[16];
};

}
}

#endif