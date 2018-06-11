// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_STORAGE_IMAGE_H_INCLUDED__
#define __I_STORAGE_IMAGE_H_INCLUDED__

#include "stdint.h"
#include "IVulkanPrebaked.h"
#include "IVirtualTexture.h"

namespace irr
{
namespace video
{

/**
Class for ARB_image_load_store and Vulkan Storage Images
It's IVulkanPrebaked because texture view is IVulkanPrebaked
**/
class IStorageImage : public virtual IVulkanPrebaked
{
    public:
    protected:
        IStorageImage()
        {
        }

        IVirtualTexture* texture; //change to texture view later
};


} // end namespace video
} // end namespace irr

#endif





