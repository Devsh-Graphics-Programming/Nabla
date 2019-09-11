#ifndef __IRR_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/ShaderCommons.h"
#include "irr/asset/SSamplerParams.h"

namespace irr { namespace asset
{

enum E_DESCRIPTOR_TYPE
{
    EDT_SAMPLER = 0,
    EDT_COMBINED_IMAGE_SAMPLER = 1,
    EDT_SAMPLED_IMAGE = 2,
    EDT_STORAGE_IMAGE = 3,
    EDT_UNIFORM_TEXEL_BUFFER = 4,
    EDT_STORAGE_TEXEL_BUFFER = 5,
    EDT_UNIFORM_BUFFER = 6,
    EDT_STORAGE_BUFFER = 7,
    EDT_UNIFORM_BUFFER_DYNAMIC = 8,
    EDT_STORAGE_BUFFER_DYNAMIC = 9,
    EDT_INPUT_ATTACHMENT = 10
};

class IDescriptorSetLayout
{
public:
    struct SBinding
    {
        E_DESCRIPTOR_TYPE type;
        uint32_t count;
        E_SHADER_STAGE stageFlags;
        core::smart_refctd_dynamic_array<SSamplerParams> samplers;
    };

protected:
    IDescriptorSetLayout(core::smart_refctd_dynamic_array<SBinding>&& _bindings) : m_bindings(std::move(_bindings)) {}
    virtual ~IDescriptorSetLayout() = default;

    core::smart_refctd_dynamic_array<SBinding> m_bindings;
};

}}

#endif