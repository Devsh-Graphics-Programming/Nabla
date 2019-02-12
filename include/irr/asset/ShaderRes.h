#pragma once

#include <cstdint>
#include "irr/macros.h"
#include "irr/asset/ShaderCommons.h"

namespace irr { namespace asset
{
enum E_SHADER_RESOURCE_TYPE : uint8_t
{
    //! GLSL declaration: e.g. `sampler2D`
    ESRT_COMBINED_IMAGE_SAMPLER,
    //! GLSL declaration: e.g. `texture2D`
    ESRT_SAMPLED_IMAGE,
    //! GLSL declaration: e.g. `image2D`
    ESRT_STORAGE_IMAGE,
    //! GLSL declaration: samplerBuffer
    ESRT_UNIFORM_TEXEL_BUFFER,
    //! GLSL declaration: imageBuffer
    ESRT_STORAGE_TEXEL_BUFFER,
    //! GLSL declaration: `sampler` or `samplerShadow`
    ESRT_SAMPLER,
    //! UBO (uniform block in GLSL)
    ESRT_UNIFORM_BUFFER,
    //! GLSL declaration: `layout(push_constant) uniform name;`
    ESRT_PUSH_CONSTANT_BLOCK,
    //! GLSL declaration: subpassInput
    ESRT_INPUT_ATTACHMENT,
    //! e.g. `in vec4 Position;` in vertex shader
    ESRT_STAGE_INPUT,
    //! e.g. `out vec4 Color;` in fragment shader
    ESRT_STAGE_OUTPUT,
    //! SSBO, GLSL declaration: buffer
    ESRT_STORAGE_BUFFER
};

template<E_SHADER_RESOURCE_TYPE restype>
struct SShaderResource;

template<>
struct SShaderResource<ESRT_COMBINED_IMAGE_SAMPLER>
{

};
template<>
struct SShaderResource<ESTR_SAMPLED_IMAGE>
{

};
template<>
struct SShaderResource<ESRT_STORAGE_IMAGE>
{

};
template<>
struct SShaderResource<ESRT_UNIFORM_TEXEL_BUFFER>
{

};
template<>
struct SShaderResource<ESRT_STORAGE_TEXEL_BUFFER>
{

};
template<>
struct SShaderResource<ESRT_SAMPLER>
{

};
template<>
struct SShaderResource<ESRT_UNIFORM_BUFFER>
{

};
template<>
struct SShaderResource<ESRT_PUSH_CONSTANT_BLOCK>
{

};
template<>
struct SShaderResource<ESRT_INPUT_ATTACHMENT>
{

};
template<>
struct SShaderResource<ESRT_STAGE_INPUT>
{

};
template<>
struct SShaderResource<ESRT_STAGE_OUTPUT>
{

};
template<>
struct SShaderResource<ESRT_STORAGE_BUFFER>
{

};


struct SShaderResourceVariant
{
    //! binding or location
    uint32_t binding;
    E_SHADER_RESOURCE_TYPE type;
    //! Basically size of an array in shader (equal to 1 if individual variable)
    uint32_t descriptorCount;

    template<E_SHADER_RESOURCE_TYPE restype>
    SShaderResource<restype>& get() { return reinterpret_cast<SShaderResource<restype>&>(variant); }
    template<E_SHADER_RESOURCE_TYPE restype>
    const SShaderResource<restype>& get() const { return reinterpret_cast<const SShaderResource<restype>&>(variant); }

    union
    {
        SShaderResource<ESRT_COMBINED_IMAGE_SAMPLER> combinedImageSampler;
        SShaderResource<ESTR_SAMPLED_IMAGE> sampledImage;
        SShaderResource<ESRT_STORAGE_IMAGE> storageImage;
        SShaderResource<ESRT_UNIFORM_TEXEL_BUFFER> uniformTexelBuffer;
        SShaderResource<ESRT_STORAGE_TEXEL_BUFFER> storageTexelBuffer;
        SShaderResource<ESRT_SAMPLER> sampler;
        SShaderResource<ESRT_UNIFORM_BUFFER> uniformBuffer;
        SShaderResource<ESRT_PUSH_CONSTANT_BLOCK> pushConstantBlock;
        SShaderResource<ESRT_INPUT_ATTACHMENT> inputAttachment;
        SShaderResource<ESRT_STAGE_INPUT> stageInput;
        SShaderResource<ESRT_STAGE_OUTPUT> stageOutput;
        SShaderResource<ESRT_STORAGE_BUFFER> storageBuffer;
    } variant;
};
bool operator<(const SShaderResourceVariant& _lhs, const SShaderResourceVariant& _rhs)
{
    return _lhs.binding < _rhs.binding;
}

}}
