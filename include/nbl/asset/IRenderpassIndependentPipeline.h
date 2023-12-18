// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_


#include "nbl/macros.h"

#include "nbl/core/declarations.h"

#include "nbl/builtin/cache/ICacheKeyCreator.h"

#include "nbl/asset/format/EFormat.h"
#include "nbl/asset/IImage.h"
#include "nbl/asset/IShader.h"
#include "nbl/asset/RasterizationStates.h"

#include <algorithm>


namespace nbl::asset
{

struct SVertexInputAttribParams
{
    inline auto operator<=>(const SVertexInputAttribParams& rhs) const = default;

    uint32_t binding : 4 = 0;
    uint32_t format : 8  = EF_UNKNOWN; // asset::E_FORMAT
    uint32_t relativeOffset : 13 = 0; // assuming max=2048
};
static_assert(sizeof(SVertexInputAttribParams)==4u, "Unexpected size!");

struct SVertexInputBindingParams
{
    enum E_VERTEX_INPUT_RATE : uint32_t
    {
        EVIR_PER_VERTEX = 0,
        EVIR_PER_INSTANCE = 1
    };

    inline auto operator<=>(const SVertexInputBindingParams& rhs) const = default;

    uint32_t stride : 31 = 0u;
    E_VERTEX_INPUT_RATE inputRate : 1 = EVIR_PER_VERTEX;
};
static_assert(sizeof(SVertexInputBindingParams)==4u, "Unexpected size!");

struct SVertexInputParams
{
    constexpr static inline size_t MAX_VERTEX_ATTRIB_COUNT = 16u;
    constexpr static inline size_t MAX_ATTR_BUF_BINDING_COUNT = 16u;

    inline auto operator<=>(const SVertexInputParams& rhs) const = default;


    uint16_t enabledAttribFlags = 0u;
    uint16_t enabledBindingFlags = 0u;
    //! index in array denotes location (attribute ID)
	SVertexInputAttribParams attributes[MAX_VERTEX_ATTRIB_COUNT] = {};
    //! index in array denotes binding number
	SVertexInputBindingParams bindings[MAX_ATTR_BUF_BINDING_COUNT] = {};
};
static_assert(sizeof(SVertexInputParams)==(sizeof(uint16_t)*2u+SVertexInputParams::MAX_VERTEX_ATTRIB_COUNT*sizeof(SVertexInputAttribParams)+SVertexInputParams::MAX_ATTR_BUF_BINDING_COUNT*sizeof(SVertexInputBindingParams)),"Unexpected size!");


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
*/

template<typename ShaderType>
class IRenderpassIndependentPipeline
{
	public:
        struct SCachedCreationParams final
        {
            SVertexInputParams vertexInput = {};
            SPrimitiveAssemblyParams primitiveAssembly = {};
            SRasterizationParams rasterization = {};
            SBlendParams blend = {};
        };
        struct SCreationParams
        {
            protected:
                using SpecInfo = ShaderType::SSpecInfo;
                template<typename ExtraLambda>
                inline bool impl_valid(ExtraLambda&& extra) const
                {
                    const ShaderType* pVertexShader = nullptr;
                    std::bitset<GRAPHICS_SHADER_STAGE_COUNT> stagePresence = {};
                    for (const auto info : shaders)
                    if (info.shader)
                    {
                        if (!extra(info))
                            return false;
                        const auto stage = info.shader->getStage();
                        if (stage>=GRAPHICS_SHADER_STAGE_COUNT)
                            return false;
                        const auto stageIx = core::findLSB(stage);
                        if (stagePresence.test(stageIx))
                            return false;
                        stagePresence.set(stageIx);
                    }
                    if (!pVertexShader)
                        return false;
                    return true;
                }

            public:
                inline bool valid() const
                {
                    return impl_valid([](const SpecInfo& info)->bool
                    {
                        if (!info.valid())
                            return false;
                        return false;
                    });
                }

                std::span<const SpecInfo> shaders = {};
                SCachedCreationParams cached = {};
        };

        inline const SCachedCreationParams& getCachedCreationParams() const {return m_cachedParams;}

	protected:
        constexpr static inline size_t GRAPHICS_SHADER_STAGE_COUNT = 5u;

        IRenderpassIndependentPipeline(const SCachedCreationParams& _cachedParams) : m_cachedParams(_cachedParams) {}
        virtual ~IRenderpassIndependentPipeline() = default;


        SCachedCreationParams m_cachedParams;
};

}
#endif