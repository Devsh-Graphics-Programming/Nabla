#ifndef __NBL_I_COMMAND_BUFFER_H_INCLUDED__
#define __NBL_I_COMMAND_BUFFER_H_INCLUDED__

#include <nbl/core/IReferenceCounted.h>
#include <nbl/core/util/bitflag.h>

namespace nbl::asset
{


struct SClearAttachment
{
    asset::IImage::E_ASPECT_FLAGS aspectMask;
    uint32_t colorAttachment;
    SClearValue clearValue;
};

struct SClearRect
{
    VkRect2D rect;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
};
}

#endif