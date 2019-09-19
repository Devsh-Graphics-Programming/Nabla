#ifndef __IRR_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/ShaderCommons.h"
#include "irr/asset/SSamplerParams.h"
#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/core/SRange.h"

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
        const SSamplerParams* samplers;
    };

protected:
    IDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : 
        m_bindings(core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SBinding>>(_end-_begin))
    {
        size_t bndCount = _end-_begin;
        size_t immSamplerCount = 0ull;
        for (size_t i = 0ull; i < bndCount; ++i) {
            const auto& bnd = _begin[i];
            if (bnd.type==EDT_COMBINED_IMAGE_SAMPLER && bnd.samplers)
                immSamplerCount += bnd.count;
        }
        m_samplers = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SSamplerParams>>(immSamplerCount);

        size_t immSamplersOffset = 0u;
        for (size_t i = 0ull; i < bndCount; ++i)
        {
            auto& bnd_out = m_bindings->operator[](i);
            const auto& bnd_in = _begin[i];

            bnd_out.type = bnd_in.type;
            bnd_out.count = bnd_in.count;
            bnd_out.stageFlags = bnd_in.stageFlags;
            bnd_out.samplers = nullptr;
            if (bnd_in.type==EDT_COMBINED_IMAGE_SAMPLER && bnd_in.samplers)
            {
                bnd_out.samplers = reinterpret_cast<const SSamplerParams*>(immSamplersOffset);
                for (uint32_t s = 0ull; s < bnd_in.count; ++s)
                    m_samplers->operator[](immSamplersOffset+s) = bnd_in.samplers[s];
                immSamplersOffset += bnd_in.count;
            }
        }

        for (size_t i = 0ull; i < m_bindings->size(); ++i)
        {
            auto& bnd = m_bindings->operator[](i);

            if (bnd.type==EDT_COMBINED_IMAGE_SAMPLER && bnd.samplers)
                bnd.samplers = m_samplers->data() + reinterpret_cast<size_t>(bnd.samplers);
        }
    }
    virtual ~IDescriptorSetLayout() = default;

    core::smart_refctd_dynamic_array<SBinding> m_bindings;
    core::smart_refctd_dynamic_array<SSamplerParams> m_samplers;

public:
    core::SRange<const SBinding> getBindings() const { return {m_bindings->data(), m_bindings->data()+m_bindings->size()}; }
};

}}

#endif