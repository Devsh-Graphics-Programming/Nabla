// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_VULKAN_PREBAKED_H_INCLUDED__
#define __I_VULKAN_PREBAKED_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"

namespace irr
{
namespace video
{

/**
This class is the base for all Vulkan objects which are kind of immutable.
Essentially if they change state then most probably a new vulkan object needs to be
created behind the scenes, set up and validated.

A little bit like our situation with IGPUBuffer::resize() and the fact that those
buffers are actually immutable in OpenGL.
**/
class IVulkanPrebaked : public virtual core::IReferenceCounted
{
    public:
    protected:
        IVulkanPrebaked()
        {
        }

        /*
        enum E_DIRTY_FLAG
        {
            EDF_CLEAN=0,
            EDF_CANT_TELL,
            EDF_DIRTY,
            EDF_COUNT
        };
        virtual E_DIRTY_FLAG isSelfAndDepsDirty() = 0;
        */
        //virtual void computeDirtyFlag() = 0;
        virtual void rebakeSelfAndDeps() = 0;
};


} // end namespace video
} // end namespace irr

#endif






