#ifndef __IRR_I_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/asset/IDescriptorSetLayout.h"//for E_DESCRIPTOR_TYPE
#include "irr/asset/format/EFormat.h"
#include "irr/asset/IDescriptor.h"
#include "irr/core/SRange.h"
#include <algorithm>

namespace irr { namespace asset
{

enum E_IMAGE_LAYOUT : uint32_t
{
    EIL_UNDEFINED = 0,
    EIL_GENERAL = 1,
    EIL_COLOR_ATTACHMENT_OPTIMAL = 2,
    EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3,
    EIL_DEPTH_STENCIL_READ_ONLY_OPTIMAL = 4,
    EIL_SHADER_READ_ONLY_OPTIMAL = 5,
    EIL_TRANSFER_SRC_OPTIMAL = 6,
    EIL_TRANSFER_DST_OPTIMAL = 7,
    EIL_PREINITIALIZED = 8,
    EIL_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL = 1000117000,
    EIL_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL = 1000117001,
    EIL_PRESENT_SRC_KHR = 1000001002,
    EIL_SHARED_PRESENT_KHR = 1000111000,
    EIL_SHADING_RATE_OPTIMAL_NV = 1000164003,
    EIL_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT = 1000218000
};

template<typename LayoutType, typename BufferType, typename TextureType, typename BufferViewType, typename SamplerType>
class IDescriptorSet
{
public:
    struct SDescriptorInfo
    {
        core::smart_refctd_ptr<IDescriptor> desc;
        union {
            struct SDescriptorBufferInfo
            {
                size_t offset;
                size_t size;//in Vulkan it's called `range` but IMO it's misleading so i changed to `size`
            } buffer;
            struct SDescriptorImageInfo
            {
                core::smart_refctd_ptr<SamplerType> sampler;
                //! Irrelevant in OpenGL backend
                E_IMAGE_LAYOUT imageLayout;
            } image;
        };
    };

    struct SWriteDescriptorSet
    {
        uint32_t binding = 0u;
        E_DESCRIPTOR_TYPE descriptorType = EDT_COMBINED_IMAGE_SAMPLER;//whatever, default value
        core::smart_refctd_dynamic_array<SDescriptorInfo> info;
    };

protected:
    /**
    @param _layout Bindings in layout must go in the same order as corresponding descriptors (SWriteDescriptorSet) in `_descriptors` parameter (this requirement should be probably dropped in the future)
    @param _descriptors Entries must be sorted by binding number
    */
    IDescriptorSet(core::smart_refctd_ptr<LayoutType>&& _layout, core::smart_refctd_dynamic_array<SWriteDescriptorSet>&& _descriptors) :
        m_layout(std::move(_layout)), m_descriptors(std::move(_descriptors)) 
    {
        auto is_not_sorted = [this] {
            for (auto it = m_descriptors->cbegin()+1; it != m_descriptors->cend(); ++it)
                if (it->binding <= (it-1)->binding)
                    return false;
            return true;
        };
        assert(!is_not_sorted);
    }
    virtual ~IDescriptorSet() = default;

    const LayoutType* getLayout() const { return m_layout.get(); }
    core::SRange<const SWriteDescriptorSet> getDescriptors() const { return {m_descriptors->begin(), m_descriptors->end()}; }

    core::smart_refctd_ptr<LayoutType> m_layout;
    core::smart_refctd_dynamic_array<SWriteDescriptorSet> m_descriptors;
};

}}

#endif