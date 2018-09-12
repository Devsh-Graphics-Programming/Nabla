// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_VULKAN_PREBAKED_H_INCLUDED__
#define __I_VULKAN_PREBAKED_H_INCLUDED__

#include "stdint.h"
#include "IVulkanPrebaked.h"

namespace irr
{
namespace video
{

//!
class ICommandBuffer : public virtual core::IReferenceCounted
{
    public:
    protected:
        ICommandBuffer()
        {
        }

        //virtual void computeDirtyFlag() = 0;
        virtual void rebakeSelfAndDeps() = 0;
};


} // end namespace video
} // end namespace irr

#endif







