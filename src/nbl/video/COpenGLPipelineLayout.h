// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef __NBL_VIDEO_C_OPENGL_PIPELINE_LAYOUT_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_PIPELINE_LAYOUT_H_INCLUDED__


#include "nbl/video/IGPUPipelineLayout.h"
#include "nbl/video/IOpenGL_FunctionTable.h"


namespace nbl::video
{

class COpenGLPipelineLayout : public IGPUPipelineLayout
{
public:
    struct SMultibindParams
    {
        struct SFirstCount
        {
            GLuint first = 0u;
            GLsizei count = 0u;
        };

        SFirstCount ubos;
        SFirstCount ssbos;
        SFirstCount textures;
        SFirstCount textureImages;
    };

    COpenGLPipelineLayout(
        core::smart_refctd_ptr<const ILogicalDevice>&& dev,
        const asset::SPushConstantRange* const _pcRangesBegin, const asset::SPushConstantRange* const _pcRangesEnd,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3
    ) : IGPUPipelineLayout(std::move(dev), _pcRangesBegin, _pcRangesEnd, std::move(_layout0), std::move(_layout1), std::move(_layout2), std::move(_layout3))
    {
        SMultibindParams params;

        for (size_t i = 0u; i<m_descSetLayouts.size(); ++i)
		{
            IGPUDescriptorSetLayout* descSetLayout = m_descSetLayouts[i].get();

            auto reset = [](SMultibindParams::SFirstCount& _fc) {
                _fc.first += _fc.count;
                _fc.count = 0u;
            };

            auto bindings = descSetLayout ? descSetLayout->getBindings() : core::SRange<const IGPUDescriptorSetLayout::SBinding>(nullptr, nullptr);
            for (const auto& bnd : bindings) {
                if (bnd.type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER || bnd.type == asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER_DYNAMIC)
                    params.ubos.count += bnd.count;
                else if (bnd.type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER || bnd.type == asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER_DYNAMIC)
                    params.ssbos.count += bnd.count;
                else if (bnd.type == asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER || bnd.type == asset::IDescriptor::E_TYPE::ET_UNIFORM_TEXEL_BUFFER)
                    params.textures.count += bnd.count;
                else if (bnd.type == asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE || bnd.type == asset::IDescriptor::E_TYPE::ET_STORAGE_TEXEL_BUFFER)
                    params.textureImages.count += bnd.count;
            }
            m_multibindParams[i] = params;

            reset(params.ubos);
            reset(params.ssbos);
            reset(params.textures);
            reset(params.textureImages);
        }
    }

    const SMultibindParams& getMultibindParamsForDescSet(uint32_t _setNum) const
    {
        return m_multibindParams[_setNum];
    }

private:
    SMultibindParams m_multibindParams[4];
};

}

#endif