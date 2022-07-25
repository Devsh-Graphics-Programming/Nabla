// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_ASSET_SHADER_RES_H_INCLUDED__
#define __NBL_ASSET_SHADER_RES_H_INCLUDED__

#include <cstdint>
#include "nbl/macros.h"
#include "nbl/asset/ICPUImageView.h"

namespace nbl
{
namespace asset
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
enum E_GLSL_VAR_TYPE
{
    EGVT_U64,
    EGVT_I64,
    EGVT_U32,
    EGVT_I32,
    EGVT_F64,
    EGVT_F32,
    EGVT_UNKNOWN_OR_STRUCT
};

template<E_SHADER_RESOURCE_TYPE restype>
struct NBL_API SShaderResource;

template<>
struct NBL_API SShaderResource<ESRT_COMBINED_IMAGE_SAMPLER>
{
    bool multisample;
    IImageView<ICPUImage>::E_TYPE viewType;
    bool shadow;
};
template<>
struct NBL_API SShaderResource<ESRT_SAMPLED_IMAGE>
{

};
template<>
struct NBL_API SShaderResource<ESRT_STORAGE_IMAGE>
{
    E_FORMAT format;
    IImageView<ICPUImage>::E_TYPE viewType;
    bool shadow;
};
template<>
struct NBL_API SShaderResource<ESRT_UNIFORM_TEXEL_BUFFER>
{

};
template<>
struct NBL_API SShaderResource<ESRT_STORAGE_TEXEL_BUFFER>
{

};
template<>
struct NBL_API SShaderResource<ESRT_SAMPLER>
{

};
template<>
struct NBL_API SShaderResource<ESRT_INPUT_ATTACHMENT>
{
    uint32_t inputAttachmentIndex;
};

namespace impl
{
struct NBL_API SShaderMemoryBlock
{
    bool restrict_;
    bool volatile_;
    bool coherent;
    bool readonly;
    bool writeonly;

    struct SMember
    {
        union {
            uint32_t count;
            uint32_t count_specID;
        };
        bool countIsSpecConstant;
        uint32_t offset;
        uint32_t size;
        //! relevant only in case of array types
        uint32_t arrayStride;
        //! mtxStride==0 implies not matrix
        uint32_t mtxStride;
        //! (mtxRowCnt>1 && mtxColCnt==1) implies vector
        //! (mtxRowCnt==1 && mtxColCnt==1) implies basic type (i.e. int/uint/float/...)
        uint32_t mtxRowCnt, mtxColCnt;
        //! rowMajor=false implies col-major
        bool rowMajor;
        E_GLSL_VAR_TYPE type;
        //TODO change to core::dynamic_array later
        struct SMembers {
            SMember* array;
            size_t count;
        } members;
        std::string name;

        bool isArray() const { return countIsSpecConstant || count > 1u; }
    };

    SMember::SMembers members;

    //! Note: for SSBOs and UBOs it's the block name, but for push_constant it's the instance name.
    std::string name;

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
struct NBL_API SShaderResource<ESRT_UNIFORM_BUFFER> : public impl::SShaderMemoryBlock
{

};
template<>
struct NBL_API SShaderResource<ESRT_STORAGE_BUFFER> : public impl::SShaderMemoryBlock
{

};


//! push-constants are treated seprately (see SIntrospectionData in ICPUShader.h)
struct NBL_API SShaderPushConstant : public impl::SShaderMemoryBlock
{
    // todo
};


struct NBL_API SShaderResourceVariant
{
    //! binding
    uint32_t binding;
    E_SHADER_RESOURCE_TYPE type;
    //! Basically size of an array in shader (equal to 1 if individual variable)
    union {
        uint32_t descriptorCount;
        uint32_t count_specID;
    };
    //! If descCountIsSpecConstant is true, than descriptorCount is ID of spec constant which is going to be size of this array
    //! Then user can look up default value of this specialization constant in SIntrospectionData::specConstants.
    bool descCountIsSpecConstant;

    template<E_SHADER_RESOURCE_TYPE restype>
    SShaderResource<restype>& get() { return reinterpret_cast<SShaderResource<restype>&>(variant); }
    template<E_SHADER_RESOURCE_TYPE restype>
    const SShaderResource<restype>& get() const { return reinterpret_cast<const SShaderResource<restype>&>(variant); }

    union Variant
    {
        Variant() {}
        Variant(const Variant& other) { memcpy(this, &other, sizeof(Variant)); }
        Variant& operator=(const Variant& other) { memcpy(this, &other, sizeof(Variant)); return *this; }
        ~Variant() {}

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
struct NBL_API SShaderInfo;

template<>
struct NBL_API SShaderInfo<ESIT_STAGE_INPUT>
{

};
template<>
struct NBL_API SShaderInfo<ESIT_STAGE_OUTPUT>
{
    //! for dual source blending. Only relevant in Fragment Stage
    uint32_t colorIndex;
};

struct NBL_API SShaderInfoVariant
{
    uint32_t location;
    struct {
        E_GLSL_VAR_TYPE basetype;
        uint32_t elements;
    } glslType;
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

#endif
