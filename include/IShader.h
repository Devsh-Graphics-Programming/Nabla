// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SHADER_H_INCLUDED__
#define __I_SHADER_H_INCLUDED__

#include "stdint.h"
#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace video
{

class IShaderStage : public virtual core::IReferenceCounted
{
    public:
        enum E_SHADER_STAGE_FLAG
        {
            ESS_INVALID_NONE=0x00u,
            ESS_VERTEX=0x01u,
            ESS_TESS_CONTROL=0x02u,
            ESS_TESS_EVAL=0x04u,
            ESS_GEOMETRY=0x08u,
            ESS_FRAGMENT=0x10u
        };

        virtual E_SHADER_STAGE_FLAG getType() const = 0;
    protected:
};

class IShader : public virtual core::IReferenceCounted
{
    public:
        enum E_SHADER_TYPE
        {
            EST_NATIVE_SPIR_V,
            EST_SEPARABLE_OPENGL,
            EST_COMBINED_OPENGL, //deprecated
            EST_COUNT
        };

        virtual E_SHADER_TYPE getType() const = 0;
    protected:
        IShader() : vertex(nullptr), control(nullptr), evaluation(nullptr), geometry(nullptr), fragment(nullptr)
        {
        }

        IShaderStage* vertex,control,evaluation,geometry,fragment;
};


} // end namespace video
} // end namespace irr

#endif




