// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_FRAMEBUFFER_H_INCLUDED__
#define __NBL_I_FRAMEBUFFER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/core/parallel/IThreadBound.h"
#include "dimension2d.h"

#include "nbl/video/IGPUImageView.h"

namespace nbl
{
namespace video
{
enum E_FBO_ATTACHMENT_POINT
{
    EFAP_DEPTH_ATTACHMENT = 0,
    EFAP_STENCIL_ATTACHMENT,
    EFAP_DEPTH_STENCIL_ATTACHMENT,
    EFAP_COLOR_ATTACHMENT0,
    EFAP_COLOR_ATTACHMENT1,
    EFAP_COLOR_ATTACHMENT2,
    EFAP_COLOR_ATTACHMENT3,
    EFAP_COLOR_ATTACHMENT4,
    EFAP_COLOR_ATTACHMENT5,
    EFAP_COLOR_ATTACHMENT6,
    EFAP_COLOR_ATTACHMENT7,
    EFAP_MAX_ATTACHMENTS
};

class IFrameBuffer : public virtual core::IReferenceCounted, public core::IThreadBound
{
public:
    //! Attaches given texture to given attachment point.
    /** @param attachmentPoint Identifies attachment point.
		@param tex Texture being attached.
		@param mipMapLayer Mipmap level of the texture image to be attached. Must be 0 if `tex` has a sample count not equal to 1.
		@param layer 
		@parblock
		Layer of the framebuffer to attach to.

		values >=0 mean that a particular layer of a 2D or cubemap texture array, or 3D texture is attached to the FrameBuffer.

		value <0 means the entire 3D texture or, 2D texture or cubemap array is bound making the FrameBuffer layered, and enabling you to use gl_Layer for layered rendering.
		@endparblock
		@returns Whether attachment has been attached, can return false when you detach.
		*/
    virtual bool attach(E_FBO_ATTACHMENT_POINT attachmenPoint, core::smart_refctd_ptr<IGPUImageView>&& tex, uint32_t mipMapLayer = 0, int32_t layer = -1) = 0;

    //! Gets attachment accessible at the given index.
    /** @param ix Given index.
		@returns Attached at given index object or NULL if nothing is bound there.
		*/
    virtual const IGPUImageView* getAttachment(uint32_t ix) const = 0;

    //!
    virtual const core::dimension2du& getSize() const = 0;

protected:
    _NBL_INTERFACE_CHILD(IFrameBuffer) {}
};

}  // end namespace video
}  // end namespace nbl

#endif
