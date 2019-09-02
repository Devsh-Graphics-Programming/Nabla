// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_FRAMEBUFFER_H_INCLUDED__
#define __I_FRAMEBUFFER_H_INCLUDED__

#include "irr/core/IReferenceCounted.h"
#include "irr/core/parallel/IThreadBound.h"

#include "EDriverTypes.h"
#include "dimension2d.h"

namespace irr
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

class ITexture;
class IMultisampleTexture;
class IRenderableVirtualTexture;

class IFrameBuffer : public virtual core::IReferenceCounted, public core::IThreadBound
{
    public:
		//! Attaches given texture to given attachment point.
		/** @param attachmentPoint Identifies attachment point.
		@param tex Texture being attached.
		@param mipMapLayer Mipmap level of the texture image to be attached.
		@param layer
		@parblock
		Layer of the framebuffer to attach to.

		values >=0 mean that a particular layer of a 2D or cubemap texture array, or 3D texture is attached to the FrameBuffer.

		value <0 means the entire 3D texture or, 2D texture or cubemap array is bound making the FrameBuffer layered, and enabling you to use gl_Layer for layered rendering.
		@endparblock
		@returns Whether attachment has been attached.
			Only after rebindRevalidate() is called by the driver internally or by the user manually do the attachments drawn into by the FrameBuffer change.
		@see @ref rebindRevalidate()
		*/
        virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, ITexture* tex, const uint32_t &mipMapLayer=0, const int32_t &layer=-1) = 0;

		//! Attaches given multisample texture to given attachment point.
		/**
		@param attachmentPoint Identifies attachment point.
		@param tex Multisample texture being attached.
		@param layer Layer of the framebuffer to attach to.
		@returns Whether attachment has been attached.
			Only after rebindRevalidate() is called by the driver internally or by the user manually do the attachments drawn into by the FrameBuffer change.
		@see @ref rebindRevalidate()
		*/
        virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, IMultisampleTexture* tex, const int32_t &layer=-1) = 0;

		//! Binds possibly respecified attachments.
		/** @returns true when everything is right or when no work was necessary to do;
				false when color formats you are trying to render to are invalid or if current combination of attachments is invalid.
		*/
        virtual bool rebindRevalidate() = 0;

		//! Gets attachment accessible at the given index.
		/** @param ix Given index.
		@returns Attached at given index object or NULL if nothing is bound there.
		*/
        virtual const IRenderableVirtualTexture* getAttachment(const size_t &ix) const = 0;

    protected:
        _IRR_INTERFACE_CHILD(IFrameBuffer) {}
};


} // end namespace video
} // end namespace irr

#endif


