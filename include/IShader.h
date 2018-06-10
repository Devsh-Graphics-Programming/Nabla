// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_SHADER_H_INCLUDED__
#define __I_SHADER_H_INCLUDED__

#include "stdint.h"
#include "IReferenceCounted.h"

namespace irr
{
namespace video
{

class IShaderStage : public virtual IReferenceCounted
{
    public:
        enum E_SHADER_STAGE
        {
            ESS_VERTEX,
            ESS_TESS_CONTROL,
            ESS_TESS_EVAL,
            ESS_GEOMETRY,
            ESS_FRAGMENT,
            ESS_COUNT
        };

        virtual E_SHADER_STAGE getType() const = 0;
    protected:
};

class IShader : public virtual IReferenceCounted
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




