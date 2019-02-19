#ifndef __IRR_SHADER_RES_H_INCLUDED__
#define __IRR_SHADER_RES_H_INCLUDED__

#include <cstdint>
#include "irr/macros.h"
#include "irr/asset/ShaderCommons.h"
#include "irr/asset/EFormat.h"

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
    //! GLSL declaration: subpassInput
    ESRT_INPUT_ATTACHMENT,
    //! SSBO, GLSL declaration: buffer
    ESRT_STORAGE_BUFFER
};
enum E_SHADER_INFO_TYPE : uint8_t
{
    //! e.g. `in vec4 Position;` in vertex shader
    ESIT_STAGE_INPUT,
    //! e.g. `out vec4 Color;` in fragment shader
    ESIT_STAGE_OUTPUT
};

template<E_SHADER_RESOURCE_TYPE restype>
struct SShaderResource;

template<>
struct SShaderResource<ESRT_COMBINED_IMAGE_SAMPLER>
{
    bool arrayed;
    bool multisample;
};
template<>
struct SShaderResource<ESRT_SAMPLED_IMAGE>
{

};
template<>
struct SShaderResource<ESRT_STORAGE_IMAGE>
{
    E_FORMAT approxFormat;

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
struct SShaderResource<ESRT_INPUT_ATTACHMENT>
{
    uint32_t inputAttachmentIndex;
};

namespace impl
{
struct SShaderMemoryBlock
{
    bool restrict_;
    bool volatile_;
    bool coherent;
    bool readonly;
    bool writeonly;

    struct SMember
    {
        //! count==1 implies not array
        uint32_t count;
        uint32_t offset;
        uint32_t size;
        uint32_t arrayStride;
        //! mtxStride==0 implies not matrix
        uint32_t mtxStride;
        //! (mtxRowCnt>1 && mtxColCnt==1) implies vector
        //! (mtxRowCnt==1 && mtxColCnt==1) implies basic type (i.e. int/uint/float/...)
        uint32_t mtxRowCnt, mtxColCnt;
    };
    struct {
        SMember* array;
        size_t count;
    } members;

    //! size!=rtSizedArrayOneElementSize implies that last member is rutime-sized array (e.g. buffer SSBO { float buf[]; }).
    //! Use getRuntimeSize for size of the struct with assumption of passed number of elements.
    size_t size;
    //! If last member is runtime-sized array, rtSizedArrayOneElementSize is equal to `size+RTSZ` where RTSZ is size (bytes) of this array assuming it's of size 1.
    //! Otherwise rtSizedArrayOneElementSize==size.
    size_t rtSizedArrayOneElementSize;

    //! See docs for `size` member
    inline size_t getRuntimeSize(size_t _elmntCnt) const { return size + _elmntCnt*(rtSizedArrayOneElementSize-size); }
    inline bool isRuntimeSized() const { return size != rtSizedArrayOneElementSize; }
};
}

template<>
struct SShaderResource<ESRT_UNIFORM_BUFFER> : public impl::SShaderMemoryBlock
{

};
template<>
struct SShaderResource<ESRT_STORAGE_BUFFER> : public impl::SShaderMemoryBlock
{

};


//! push-constants are treated seprately (see SIntrospectionData in ICPUShader.h)
struct SShaderPushConstant : public impl::SShaderMemoryBlock
{
    // todo
};


struct SShaderResourceVariant
{
    //! binding
    uint32_t binding;
    E_SHADER_RESOURCE_TYPE type;
    //! Basically size of an array in shader (equal to 1 if individual variable)
    uint32_t descriptorCount;
    //! If descCountIsSpecConstant is true, than descriptorCount is ID of spec constant which is going to be size of this array
    //! Then user can look up default value of this specialization constant in SIntrospectionData::specConstants.
    bool descCountIsSpecConstant;

    template<E_SHADER_RESOURCE_TYPE restype>
    SShaderResource<restype>& get() { return reinterpret_cast<SShaderResource<restype>&>(variant); }
    template<E_SHADER_RESOURCE_TYPE restype>
    const SShaderResource<restype>& get() const { return reinterpret_cast<const SShaderResource<restype>&>(variant); }

    union Variant
    {
        SShaderResource<ESRT_COMBINED_IMAGE_SAMPLER> combinedImageSampler;
        SShaderResource<ESRT_SAMPLED_IMAGE> sampledImage;
        SShaderResource<ESRT_STORAGE_IMAGE> storageImage;
        SShaderResource<ESRT_UNIFORM_TEXEL_BUFFER> uniformTexelBuffer;
        SShaderResource<ESRT_STORAGE_TEXEL_BUFFER> storageTexelBuffer;
        SShaderResource<ESRT_SAMPLER> sampler;
        SShaderResource<ESRT_UNIFORM_BUFFER> uniformBuffer;
        SShaderResource<ESRT_INPUT_ATTACHMENT> inputAttachment;
        SShaderResource<ESRT_STORAGE_BUFFER> storageBuffer;
    } variant;
};
inline bool operator<(const SShaderResourceVariant& _lhs, const SShaderResourceVariant& _rhs)
{
    return _lhs.binding < _rhs.binding;
}

template<E_SHADER_INFO_TYPE type>
struct SShaderInfo;

template<>
struct SShaderInfo<ESIT_STAGE_INPUT>
{

};
template<>
struct SShaderInfo<ESIT_STAGE_OUTPUT>
{
    //! for dual source blending. Only relevant in Fragment Stage
    uint32_t colorIndex;
};

struct SShaderInfoVariant
{
    uint32_t location;
    E_SHADER_INFO_TYPE type;

    template<E_SHADER_INFO_TYPE type>
    SShaderInfo<type>& get() { return reinterpret_cast<SShaderInfo<type>&>(variant); }
    template<E_SHADER_INFO_TYPE type>
    const SShaderInfo<type>& get() const { return reinterpret_cast<const SShaderInfo<type>&>(variant); }

    union
    {
        SShaderInfo<ESIT_STAGE_INPUT> stageInput;
        SShaderInfo<ESIT_STAGE_OUTPUT> stageOutput;
    } variant;
};
inline bool operator<(const SShaderInfoVariant& _lhs, const SShaderInfoVariant& _rhs)
{
    return _lhs.location < _rhs.location;
}

}}

#endif//__IRR_SHADER_RES_H_INCLUDED__