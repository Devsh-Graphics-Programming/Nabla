#ifndef __IRR_I_DESCRIPTOR_SET_H_INCLUDED__
#define __IRR_I_DESCRIPTOR_SET_H_INCLUDED__

#include "irr/core/memory/refctd_dynamic_array.h"
#include "irr/asset/SSamplerParams.h"
#include "irr/asset/IDescriptorSetLayout.h"//for E_DESCRIPTOR_TYPE
#include "irr/asset/format/EFormat.h"

namespace irr { namespace asset
{

template<typename BufferType, typename TextureType>
class IDescriptorSet
{
public:
    struct SDescriptorBufferInfo
    {
        core::smart_refctd_ptr<BufferType> buffer;
        size_t offset;
        size_t range;
    };
    struct SDescriptorImageInfo
    {
        SSamplerParams sampler;
        core::smart_refctd_ptr<TextureType> imageView;
        //VkImageLayout imageLayout;
    };
    //for texture buffers (samplerBuffer/imageBuffer in GLSL) descriptors
    struct SBufferView
    {
        core::smart_refctd_ptr<BufferType> buffer;
        size_t offset;
        size_t range;
        E_FORMAT format;
    };

    struct SWriteDescriptorSet
    {
        uint32_t binding;
        E_DESCRIPTOR_TYPE descriptorType;
        core::smart_refctd_dynamic_array<SDescriptorBufferInfo> bufferInfo;
        core::smart_refctd_dynamic_array<SDescriptorImageInfo> imageInfo;
        core::smart_refctd_dynamic_array<SBufferView> imageInfo;
    };

protected:
    IDescriptorSet(core::smart_refctd_dynamic_array<SWriteDescriptorSet>&& _descriptors) : m_descriptors(std::move(_descriptors)) {}
    virtual ~IDescriptorSet() = default;

    core::smart_refctd_dynamic_array<SWriteDescriptorSet> m_descriptors;
};

}}

#endif