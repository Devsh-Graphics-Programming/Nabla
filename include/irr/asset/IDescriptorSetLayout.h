#ifndef __IRR_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "irr/asset/ShaderCommons.h"
#include "irr/asset/SSamplerParams.h"
#include "irr/core/memory/refctd_dynamic_array.h"

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
    template<typename SamplersType>
    struct SBinding_
    {
        E_DESCRIPTOR_TYPE type;
        uint32_t count;
        E_SHADER_STAGE stageFlags;
        SamplersType samplers;
    };
    using SBinding_internal = SBinding_<const SSamplerParams*>;
public:
    using SBinding = SBinding_<core::smart_refctd_dynamic_array<SSamplerParams>>;

protected:
    IDescriptorSetLayout(const SBinding* const _begin, const SBinding* const _end) : 
        m_bindings(core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<SBinding_internal>>(_end-_begin,nullptr))
    {
        size_t bndCount = _end-_begin;
        size_t immSamplerCount = 0ull;
        for (size_t i = 0ull; i < bndCount; ++i) {
            const auto& bnd = _begin[i];
            if (bnd.type==EDT_COMBINED_IMAGE_SAMPLER && bnd.samplers)
                immSamplerCount += bnd.samplers->size();
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
                for (size_t s = 0ull; s < bnd_in.samplers->size(); ++s)
                    m_samplers->operator[](immSamplersOffset+s) = bnd_in.samplers->operator[](s);
                immSamplersOffset += bnd_in.samplers->size();
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

    core::smart_refctd_dynamic_array<SBinding_internal> m_bindings;
    core::smart_refctd_dynamic_array<SSamplerParams> m_samplers;
};

}}

#endif