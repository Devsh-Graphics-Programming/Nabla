#ifndef __IRR_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __IRR_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include "irr/asset/format/EFormat.h"
#include "irr/core/math/irrMath.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/IPipeline.h"
#include "irr/macros.h"
#include <algorithm>

namespace irr {
namespace asset
{

enum E_PRIMITIVE_TOPOLOGY
{
    EPT_POINT_LIST = 0,
    EPT_LINE_LIST = 1,
    EPT_LINE_STRIP = 2,
    EPT_TRIANGLE_LIST = 3,
    EPT_TRIANGLE_STRIP = 4,
    EPT_TRIANGLE_FAN = 5,
    EPT_LINE_LIST_WITH_ADJACENCY = 6,
    EPT_LINE_STRIP_WITH_ADJACENCY = 7,
    EPT_TRIANGLE_LIST_WITH_ADJACENCY = 8,
    EPT_TRIANGLE_STRIP_WITH_ADJACENCY = 9,
    EPT_PATCH_LIST = 10
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
    E_PRIMITIVE_TOPOLOGY primitiveType;
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

struct SRasterizationParams
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

//! Available vertex attribute ids
/** As of 2018 most OpenGL and Vulkan implementations support 16 attributes (some CAD GPUs more) */
enum E_VERTEX_ATTRIBUTE_ID
{
    EVAI_ATTR0 = 0,
    EVAI_ATTR1,
    EVAI_ATTR2,
    EVAI_ATTR3,
    EVAI_ATTR4,
    EVAI_ATTR5,
    EVAI_ATTR6,
    EVAI_ATTR7,
    EVAI_ATTR8,
    EVAI_ATTR9,
    EVAI_ATTR10,
    EVAI_ATTR11,
    EVAI_ATTR12,
    EVAI_ATTR13,
    EVAI_ATTR14,
    EVAI_ATTR15,
    EVAI_COUNT
};

template<typename SpecShaderType, typename LayoutType>
class IRenderpassIndependentPipeline : public IPipeline<LayoutType>
{
    IRR_INLINE_STATIC_CONSTEXPR size_t SHADER_STAGE_COUNT = 5ull;
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
        SRasterizationParams _rasterParams
    ) : IPipeline<LayoutType>(std::move(_layout)),
        m_shaders{_vs, _tcs, _tes, _gs, _fs},
        m_blendParams(_blendParams),
        m_primAsmParams(_primAsmParams),
        m_rasterParams(_rasterParams)
    {
        std::copy(_vertexInputParams, _vertexInputParams+EVAI_COUNT, m_vertexInputParams);
    }
    virtual ~IRenderpassIndependentPipeline() = default;

public:
    inline const LayoutType* getLayout() const { return m_layout.get(); }
    inline const SpecShaderType* getShaderAtStage(E_SHADER_STAGE _stage) const { return m_shaders[core::findLSB<uint32_t>(_stage)].get(); }

    inline const SBlendParams& getBlendParams() const { return m_blendParams; }
    inline const SPrimitiveAssemblyParams& getPrimitiveAssemblyParams() const { return m_primAsmParams; }
    inline const SRasterizationParams& getRasterizationParams() const { return m_rasterParams; }
    inline const SVertexInputParams* getVertexInputParams() const { return m_vertexInputParams; }
    inline const SVertexInputParams& getVertexInputParams(uint32_t _ix) const { return m_vertexInputParams[_ix]; }

protected:
    core::smart_refctd_ptr<SpecShaderType> m_shaders[SHADER_STAGE_COUNT];

    SBlendParams m_blendParams;
    SPrimitiveAssemblyParams m_primAsmParams;
    SRasterizationParams m_rasterParams;
    SVertexInputParams m_vertexInputParams[EVAI_COUNT];
};

}
}

#endif