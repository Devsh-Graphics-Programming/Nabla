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
    //! Note: for SSBOs and UBOs it's the block name, but for push_constant it's the instance name.
    std::string name;
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


}}

#endif
