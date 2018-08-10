// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#define __I_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "stdint.h"
#include "IShader.h"

namespace irr
{
namespace video
{

/**
Could have been a struct, but Vulkan actually creates an object out of it.
**/
class IDescriptorSetLayout : public virtual IReferenceCounted
{
    public:
        enum E_DESCRIPTOR_TYPE
        {
            EDT_COMBINED_IMAGE_SAMPLER=0,
            EDT_STORAGE_IMAGE_AND_TEXBUF,
            EDT_STORAGE_BUFFER,
            EDT_UNIFORM_BUFFER,
            EDT_DYNAMIC_STORAGE_BUFFER,
            EDT_DYNAMIC_UNIFORM_BUFFER,
            //EDT_INPUT_ATTACHMENT = 10,
            EDT_COUNT
        };
        struct Binding
        {
            //! The binding slot, same as the binding in the `layout(.., binding=0, ...)` in your SPIR-V shader
            uint32_t binding;
            //!
            E_DESCRIPTOR_TYPE descType;
            /** Specify which shader stages use the binding resource.
            Some Vulkan implementations may see a performance benefit
            from using these conservatively.
            Still need to specify for OpenGL for (set,binding) demangling.
            **/
            IShaderStage::E_SHADER_STAGE_FLAG usedInShaderStages;
            /** Unless you declared `layout(.., binding=0, ...) glslDescType descName[N]` with N>1 in
            your SPIR-V shader, set this to 1.
            **/
            uint32_t arrayCount;
            /*
            Immutable samplers?
            */
        };
    protected:
        IDescriptorSetLayout(const Binding* bindings, const uint32_t& bindingCount) {}

        _IRR_INTERFACE_CHILD(IDescriptorSetLayout) {}
};


} // end namespace video
} // end namespace irr

#endif





