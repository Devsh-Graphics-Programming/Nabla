// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RENDERPASS_INDEPENDENT_PIPELINE_H_INCLUDED_


#include "nbl/macros.h"

#include "nbl/core/declarations.h"

#include "nbl/asset/IGraphicsPipeline.h"

#include <algorithm>


namespace nbl::asset
{

//! Deprecated class but needs to stay around till Material Compiler 2
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
            using SpecInfo = ShaderType::SSpecInfo;
            std::span<const SpecInfo> shaders = {};
            SCachedCreationParams cached = {};
        };

        inline const SCachedCreationParams& getCachedCreationParams() const {return m_cachedParams;}

        constexpr static inline size_t GRAPHICS_SHADER_STAGE_COUNT = 5u;

	protected:
        IRenderpassIndependentPipeline(const SCachedCreationParams& _cachedParams) : m_cachedParams(_cachedParams) {}
        virtual ~IRenderpassIndependentPipeline() = default;

        SCachedCreationParams m_cachedParams;
};

}
#endif