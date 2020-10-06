// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED__
#define __NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED__

#include <utility>
#include "irr/asset/IPipeline.h"
#include "irr/asset/ISpecializedShader.h"

namespace irr
{
namespace asset
{

template<typename SpecShaderType, typename LayoutType>
class IComputePipeline : public IPipeline<LayoutType>
{
    public:
		_NBL_STATIC_INLINE_CONSTEXPR size_t SHADER_STAGE_COUNT = 1u;

        const SpecShaderType* getShader() const { return m_shader.get(); }
        inline const LayoutType* getLayout() const { return IPipeline<LayoutType>::m_layout.get(); }

		IComputePipeline(
			core::smart_refctd_ptr<LayoutType>&& _layout,
			core::smart_refctd_ptr<SpecShaderType>&& _cs
		) : IPipeline<LayoutType>(std::move(_layout)),
			m_shader(std::move(_cs))
		{
            assert(m_shader->getStage() == ISpecializedShader::ESS_COMPUTE);
        }

    protected:
		virtual ~IComputePipeline() = default;

		core::smart_refctd_ptr<SpecShaderType> m_shader;
};

}
}


#endif