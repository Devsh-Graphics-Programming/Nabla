// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__
#define __NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED__

#include <algorithm>


#include "nbl/macros.h"

#include "nbl/core/declarations.h"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/ISpecializedShader.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IImage.h"


namespace nbl
{
namespace asset
{

enum E_PRIMITIVE_TOPOLOGY : uint8_t
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

enum E_VERTEX_INPUT_RATE : uint8_t
{
    EVIR_PER_VERTEX = 0,
    EVIR_PER_INSTANCE = 1
};

#include "nbl/nblpack.h"
struct SVertexInputAttribParams
{
	SVertexInputAttribParams() : binding(0u), format(EF_UNKNOWN), relativeOffset(0u) {}
	SVertexInputAttribParams(uint32_t _binding, uint32_t _format, uint32_t _relativeOffset) :
						binding(_binding), format(_format), relativeOffset(_relativeOffset) {}

    inline bool operator==(const SVertexInputAttribParams& rhs) const
    {
        return binding==rhs.binding&&format==rhs.format&&relativeOffset==rhs.relativeOffset;
    }

    uint32_t binding : 4;
    uint32_t format : 8;//asset::E_FORMAT
    uint32_t relativeOffset : 13;//assuming max=2048

    constexpr static size_t serializedSize() { return sizeof(uint32_t); }

    void serialize(void* mem) const
    {
        auto* dst = reinterpret_cast<uint32_t*>(mem);
        *dst = core::bitfieldInsert(*dst, binding, 0, 4);
        *dst = core::bitfieldInsert(*dst, format, 4, 8);
        *dst = core::bitfieldInsert(*dst, relativeOffset, 12, 13);
    }
} PACK_STRUCT;
static_assert(sizeof(SVertexInputAttribParams)==(4u), "Unexpected size!");
struct SVertexInputBindingParams
{

    inline bool operator==(const SVertexInputBindingParams& rhs) const
    {
        return stride==rhs.stride&&inputRate==rhs.inputRate;
    }

    uint32_t stride = 0u; // could have packed the stride and input rate together since there are limits on those
    E_VERTEX_INPUT_RATE inputRate = EVIR_PER_VERTEX;

    constexpr static size_t serializedSize() { return sizeof(SVertexInputBindingParams); }

    void serialize(void* mem) const
    {
        memcpy(mem, this, serializedSize());
    }
} PACK_STRUCT;
static_assert(sizeof(SVertexInputBindingParams)==5u, "Unexpected size!");
struct SVertexInputParams
{
    _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_VERTEX_ATTRIB_COUNT = 16u;
    _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_ATTR_BUF_BINDING_COUNT = 16u;

    inline bool operator==(const SVertexInputParams& rhs) const
    {
        if (enabledAttribFlags!=rhs.enabledAttribFlags||enabledBindingFlags!=rhs.enabledBindingFlags)
            return false;

        return std::equal(attributes,attributes+MAX_VERTEX_ATTRIB_COUNT,rhs.attributes)&&std::equal(bindings,bindings+MAX_ATTR_BUF_BINDING_COUNT,rhs.bindings);
    }


    uint16_t enabledAttribFlags = 0u;
    uint16_t enabledBindingFlags = 0u;
    //! index in array denotes location (attribute ID)
	SVertexInputAttribParams attributes[MAX_VERTEX_ATTRIB_COUNT] = {};
    //! index in array denotes binding number
	SVertexInputBindingParams bindings[MAX_ATTR_BUF_BINDING_COUNT] = {};

    static_assert(sizeof(enabledAttribFlags)*8 >= MAX_VERTEX_ATTRIB_COUNT, "Insufficient flag bits for number of supported attributes");
    static_assert(sizeof(enabledBindingFlags)*8 >= MAX_ATTR_BUF_BINDING_COUNT, "Insufficient flag bits for number of supported bindings");

    constexpr static size_t serializedSize() { return 2u*sizeof(uint16_t) + MAX_VERTEX_ATTRIB_COUNT*SVertexInputAttribParams::serializedSize() + MAX_ATTR_BUF_BINDING_COUNT*SVertexInputBindingParams::serializedSize(); }

    void serialize(void* mem) const
    {
        auto* flags = reinterpret_cast<uint16_t*>(mem);
        flags[0] = enabledAttribFlags;
        flags[1] = enabledBindingFlags;
        auto* dst = reinterpret_cast<uint8_t*>(flags + 2);
        for (uint32_t i = 0u; i < MAX_VERTEX_ATTRIB_COUNT; ++i)
            attributes[i].serialize(dst + i*attributes[i].serializedSize());
        dst += MAX_VERTEX_ATTRIB_COUNT * SVertexInputAttribParams::serializedSize();
        for (uint32_t i = 0u; i < MAX_VERTEX_ATTRIB_COUNT; ++i)
            bindings[i].serialize(dst + i*bindings[i].serializedSize());
    }
} PACK_STRUCT;
static_assert(sizeof(SVertexInputParams) == (2u * 2u + SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT * sizeof(SVertexInputAttribParams) + SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT * sizeof(SVertexInputBindingParams)), "Unexpected size!");

struct SPrimitiveAssemblyParams
{
    E_PRIMITIVE_TOPOLOGY primitiveType = EPT_TRIANGLE_LIST;
    uint8_t primitiveRestartEnable = false;
    uint16_t tessPatchVertCount = 3u;

    constexpr static size_t serializedSize() { return sizeof(SPrimitiveAssemblyParams); }

    void serialize(void* mem) const
    {
        memcpy(mem, this, sizeof(*this));
    }
} PACK_STRUCT;
static_assert(sizeof(SPrimitiveAssemblyParams)==4u, "Unexpected size!");

enum E_STENCIL_OP : uint8_t
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

enum E_COMPARE_OP : uint8_t
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
    E_STENCIL_OP failOp = ESO_KEEP;
    E_STENCIL_OP passOp = ESO_KEEP;
    E_STENCIL_OP depthFailOp = ESO_KEEP;
    E_COMPARE_OP compareOp = ECO_ALWAYS;
    uint32_t writeMask = ~0u;
    uint32_t reference = 0u;
} PACK_STRUCT;
static_assert(sizeof(SStencilOpParams)==(4u*1u + 2u*4u), "Unexpected size!");

enum E_POLYGON_MODE : uint8_t
{
    EPM_FILL = 0,
    EPM_LINE = 1,
    EPM_POINT = 2
};

enum E_FACE_CULL_MODE : uint8_t
{
    EFCM_NONE = 0,
    EFCM_FRONT_BIT = 1,
    EFCM_BACK_BIT = 2,
    EFCM_FRONT_AND_BACK = 3
};

struct SRasterizationParams
{
	SRasterizationParams()
	{
		depthClampEnable = false;
		rasterizerDiscard = false;
		frontFaceIsCCW = true;
		depthBiasEnable = false;
		sampleShadingEnable = false;
		alphaToCoverageEnable = false;
		alphaToOneEnable = false;
		depthTestEnable = true;
		depthWriteEnable = true;
		depthBoundsTestEnable = false;
		stencilTestEnable = false;
	}

    uint8_t viewportCount = 1u;
    E_POLYGON_MODE polygonMode = EPM_FILL;
    E_FACE_CULL_MODE faceCullingMode = EFCM_BACK_BIT;
	E_COMPARE_OP depthCompareOp = ECO_GREATER;
    IImage::E_SAMPLE_COUNT_FLAGS rasterizationSamplesHint = IImage::ESCF_1_BIT; //TODO depr
	uint32_t sampleMask[2] = {~0u,~0u};
    float minSampleShading = 0.f;
    float depthBiasSlopeFactor = 0.f;
    float depthBiasConstantFactor = 0.f;
    SStencilOpParams frontStencilOps;
    SStencilOpParams backStencilOps;
    struct {
        uint16_t depthClampEnable : 1;
        uint16_t rasterizerDiscard : 1;
        uint16_t frontFaceIsCCW : 1;
        uint16_t depthBiasEnable : 1;
        uint16_t sampleShadingEnable : 1;
        uint16_t alphaToCoverageEnable : 1;
        uint16_t alphaToOneEnable : 1;
        uint16_t depthTestEnable : 1;
        uint16_t depthWriteEnable : 1;
        uint16_t depthBoundsTestEnable : 1;
        uint16_t stencilTestEnable : 1;
    } PACK_STRUCT;

    constexpr static size_t serializedSize() { return offsetof(SRasterizationParams, backStencilOps) + sizeof(SStencilOpParams) + sizeof(uint16_t); }

    void serialize(void* _mem) const
    {
        memcpy(_mem, this, sizeof(*this));
        auto* bf_dst = reinterpret_cast<uint8_t*>(_mem);
        bf_dst += offsetof(SRasterizationParams, backStencilOps) + sizeof(SStencilOpParams);
        uint16_t bf = 0;
        bf = core::bitfieldInsert(bf, depthClampEnable, 0, 1);
        bf = core::bitfieldInsert(bf, rasterizerDiscard, 1, 1);
        bf = core::bitfieldInsert(bf, frontFaceIsCCW, 2, 1);
        bf = core::bitfieldInsert(bf, depthBiasEnable, 3, 1);
        bf = core::bitfieldInsert(bf, sampleShadingEnable, 4, 1);
        bf = core::bitfieldInsert(bf, alphaToCoverageEnable, 5, 1);
        bf = core::bitfieldInsert(bf, alphaToOneEnable, 6, 1);
        bf = core::bitfieldInsert(bf, depthTestEnable, 7, 1);
        bf = core::bitfieldInsert(bf, depthWriteEnable, 8, 1);
        bf = core::bitfieldInsert(bf, depthBoundsTestEnable, 9, 1);
        bf = core::bitfieldInsert(bf, stencilTestEnable, 10, 1);

        reinterpret_cast<uint16_t*>(bf_dst)[0] = bf;
    }
} PACK_STRUCT;
static_assert(sizeof(SRasterizationParams)==4u*sizeof(uint8_t) + 3u*sizeof(uint32_t) + 3u*sizeof(float) + 2u*sizeof(SStencilOpParams) + sizeof(uint16_t), "Unexpected size!");

enum E_LOGIC_OP : uint8_t
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

enum E_BLEND_FACTOR : uint8_t
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

enum E_BLEND_OP : uint8_t
{
    EBO_ADD = 0,
    EBO_SUBTRACT = 1,
    EBO_REVERSE_SUBTRACT = 2,
    EBO_MIN = 3,
    EBO_MAX = 4,
    EBO_ZERO_EXT,
    EBO_SRC_EXT,
    EBO_DST_EXT,
    EBO_SRC_OVER_EXT,
    EBO_DST_OVER_EXT,
    EBO_SRC_IN_EXT,
    EBO_DST_IN_EXT,
    EBO_SRC_OUT_EXT,
    EBO_DST_OUT_EXT,
    EBO_SRC_ATOP_EXT,
    EBO_DST_ATOP_EXT,
    EBO_XOR_EXT,
    EBO_MULTIPLY_EXT,
    EBO_SCREEN_EXT,
    EBO_OVERLAY_EXT,
    EBO_DARKEN_EXT,
    EBO_LIGHTEN_EXT,
    EBO_COLORDODGE_EXT,
    EBO_COLORBURN_EXT,
    EBO_HARDLIGHT_EXT,
    EBO_SOFTLIGHT_EXT,
    EBO_DIFFERENCE_EXT,
    EBO_EXCLUSION_EXT,
    EBO_INVERT_EXT,
    EBO_INVERT_RGB_EXT,
    EBO_LINEARDODGE_EXT,
    EBO_LINEARBURN_EXT,
    EBO_VIVIDLIGHT_EXT,
    EBO_LINEARLIGHT_EXT,
    EBO_PINLIGHT_EXT,
    EBO_HARDMIX_EXT,
    EBO_HSL_HUE_EXT,
    EBO_HSL_SATURATION_EXT,
    EBO_HSL_COLOR_EXT,
    EBO_HSL_LUMINOSITY_EXT,
    EBO_PLUS_EXT,
    EBO_PLUS_CLAMPED_EXT,
    EBO_PLUS_CLAMPED_ALPHA_EXT,
    EBO_PLUS_DARKER_EXT,
    EBO_MINUS_EXT,
    EBO_MINUS_CLAMPED_EXT,
    EBO_CONTRAST_EXT,
    EBO_INVERT_OVG_EXT,
    EBO_RED_EXT,
    EBO_GREEN_EXT,
    EBO_BLUE_EXT
};

struct SColorAttachmentBlendParams
{
	SColorAttachmentBlendParams() : 
		attachmentEnabled(true),
		blendEnable(false),
		srcColorFactor(EBF_ONE),
		dstColorFactor(EBF_ZERO),
		colorBlendOp(EBO_ADD),
		srcAlphaFactor(EBF_ONE),
		dstAlphaFactor(EBF_ZERO),
		alphaBlendOp(EBO_ADD),
		colorWriteMask(0xfu)
	{}

    uint8_t attachmentEnabled : 1;
    uint8_t blendEnable : 1;
    uint8_t srcColorFactor : 5;
    
    uint8_t dstColorFactor : 5;
    
    uint8_t colorBlendOp : 6;
    
    uint8_t srcAlphaFactor : 5;
    
    uint8_t dstAlphaFactor : 5;
    
    uint8_t alphaBlendOp : 2;
    //RGBA, LSB is R, MSB is A
    uint8_t colorWriteMask : 4;

    constexpr static size_t serializedSize() { return 5ull; }

    void serialize(void* _mem) const
    {
        auto* bf_dst = reinterpret_cast<uint8_t*>(_mem);
        uint64_t bf = 0;
        bf = core::bitfieldInsert<uint64_t>(bf, attachmentEnabled, 0, 1);
        bf = core::bitfieldInsert<uint64_t>(bf, blendEnable, 1, 1);
        bf = core::bitfieldInsert<uint64_t>(bf, srcColorFactor, 2, 5);
        bf = core::bitfieldInsert<uint64_t>(bf, dstColorFactor, 7, 5);
        bf = core::bitfieldInsert<uint64_t>(bf, colorBlendOp, 12, 6);
        bf = core::bitfieldInsert<uint64_t>(bf, srcAlphaFactor, 18, 5);
        bf = core::bitfieldInsert<uint64_t>(bf, dstAlphaFactor, 23, 5);
        bf = core::bitfieldInsert<uint64_t>(bf, alphaBlendOp, 28, 2);
        bf = core::bitfieldInsert<uint64_t>(bf, colorWriteMask, 30, 4);

        memcpy(bf_dst, &bf, serializedSize());
    }
} PACK_STRUCT;
//static_assert(sizeof(SColorAttachmentBlendParams)==5u, "Unexpected size of SColorAttachmentBlendParams (should be 5)");

struct SBlendParams
{
	SBlendParams() : logicOpEnable(false), logicOp(ELO_NO_OP), blendParams{} {}

    _NBL_STATIC_INLINE_CONSTEXPR size_t MAX_COLOR_ATTACHMENT_COUNT = 8u;
    uint8_t logicOpEnable : 1;
    uint8_t logicOp : 4;
    SColorAttachmentBlendParams blendParams[MAX_COLOR_ATTACHMENT_COUNT];

    constexpr static size_t serializedSize() { return 1u + MAX_COLOR_ATTACHMENT_COUNT * SColorAttachmentBlendParams::serializedSize(); }

    void serialize(void* _mem) const
    {
        auto* bf_dst = reinterpret_cast<uint8_t*>(_mem);
        *bf_dst = core::bitfieldInsert(*bf_dst, logicOpEnable, 0, 1);
        *bf_dst = core::bitfieldInsert(*bf_dst, logicOp, 1, 4);
        ++bf_dst;
        for (uint32_t i = 0u; i < MAX_COLOR_ATTACHMENT_COUNT; ++i)
            blendParams[i].serialize(bf_dst + i*SColorAttachmentBlendParams::serializedSize());
    }
} PACK_STRUCT;
static_assert(sizeof(SBlendParams)==(1u + sizeof(SColorAttachmentBlendParams)*SBlendParams::MAX_COLOR_ATTACHMENT_COUNT), "Unexpected size!");

#include "nbl/nblunpack.h"

//TODO put into legacy namespace later
/*
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
*/

//! Base class for Renderpass Independent Pipeline - Graphics Pipeline
/*
    IRenderpassIndependentPipeline has been introduced by us because we
    disliked how Vulkan forced the user to know about the types and formats
    of Framebuffer Attachments (Render Targets in DirectX parlance) that will
    be used when creating the pipeline. This would have made it impossible to
    load models in a "screen agnostic way".

    Graphics pipelines consist of multiple shader stages,
    multiple fixed-function pipeline stages, 
    and a pipeline layout.

    @see IPipeline
*/

template<typename SpecShaderType, typename LayoutType>
class IRenderpassIndependentPipeline : public IPipeline<LayoutType>
{
	public:
		_NBL_STATIC_INLINE_CONSTEXPR size_t SHADER_STAGE_COUNT = 5u;

		enum E_SHADER_STAGE_IX : uint32_t
		{
			ESSI_VERTEX_SHADER_IX = 0,
			ESSI_TESS_CTRL_SHADER_IX = 1,
			ESSI_TESS_EVAL_SHADER_IX = 2,
			ESSI_GEOMETRY_SHADER_IX = 3,
			ESSI_FRAGMENT_SHADER_IX = 4
		};

		IRenderpassIndependentPipeline(
			core::smart_refctd_ptr<LayoutType>&& _layout,
			SpecShaderType*const * _shadersBegin, SpecShaderType*const * _shadersEnd, 
			const SVertexInputParams& _vertexInputParams,
			const SBlendParams& _blendParams,
			const SPrimitiveAssemblyParams& _primAsmParams,
			const SRasterizationParams& _rasterParams
		) : IPipeline<LayoutType>(std::move(_layout)),
			m_blendParams(_blendParams),
			m_primAsmParams(_primAsmParams),
			m_rasterParams(_rasterParams),
			m_vertexInputParams(_vertexInputParams)
		{
			auto shaders = core::SRange<SpecShaderType* const>(_shadersBegin, _shadersEnd);
			for (auto shdr : shaders)
			{
				const int32_t ix = core::findLSB<uint32_t>(shdr->getStage());
				assert(ix < static_cast<int32_t>(SHADER_STAGE_COUNT));
				assert(!m_shaders[ix]);//must be maximum of 1 for each stage
				m_shaders[ix] = core::smart_refctd_ptr<SpecShaderType>(shdr);
			}
		}
		virtual ~IRenderpassIndependentPipeline() = default;

	public:
		inline const LayoutType* getLayout() const { return IPipeline<LayoutType>::m_layout.get(); }

		inline const SpecShaderType* getShaderAtStage(IShader::E_SHADER_STAGE _stage) const { return m_shaders[core::findLSB<uint32_t>(_stage)].get(); }
		inline const SpecShaderType* getShaderAtIndex(uint32_t _ix) const { return m_shaders[_ix].get(); }

		inline const SBlendParams& getBlendParams() const { return m_blendParams; }
		inline const SPrimitiveAssemblyParams& getPrimitiveAssemblyParams() const { return m_primAsmParams; }
		inline const SRasterizationParams& getRasterizationParams() const { return m_rasterParams; }
		inline const SVertexInputParams& getVertexInputParams() const { return m_vertexInputParams; }

	protected:
		core::smart_refctd_ptr<SpecShaderType> m_shaders[SHADER_STAGE_COUNT];

		SBlendParams m_blendParams;
		SPrimitiveAssemblyParams m_primAsmParams;
		SRasterizationParams m_rasterParams;
		SVertexInputParams m_vertexInputParams;
};

}
}

#endif