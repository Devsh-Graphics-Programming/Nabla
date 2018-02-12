// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __I_FRAMEBUFFER_H_INCLUDED__
#define __I_FRAMEBUFFER_H_INCLUDED__

#include "stdint.h"
#include "IReferenceCounted.h"
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

enum E_RENDERABLE_TYPE
{
    ERT_TEXTURE=0,
    ERT_RENDERBUFFER
};

//! Base class for render targets
class IRenderable : public virtual IReferenceCounted
{
    public:
		//! Returns type
        virtual E_RENDERABLE_TYPE getRenderableType() const = 0;

		//! Returns two-dimensional size
        virtual core::dimension2du getRenderableSize() const = 0;
};

class ITexture;
class IMultisampleTexture;
class IRenderBuffer;

class IFrameBuffer : public virtual IReferenceCounted
{
    public:
		//! Attaches given texture to given attachment point.
		/** @param attachmentPoint Identifies attchment point.
		@param tex Texture being attached.
		@param mipMapLayer
		@param layer
		@returns Whether attachment operation went succesfully.
		*/
        virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, ITexture* tex, const uint32_t &mipMapLayer=0, const int32_t &layer=-1) = 0;

		//! Attaches given render buffer to given attachment point.
		/** @param attachmentPoint Identifies attchment point.
		@param rbf Render buffer being attached.
		@returns Whether attachment operation went succesfully.
		*/
        virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, IMultisampleTexture* tex, const int32_t &layer=-1) = 0;

        virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, IRenderBuffer* rbf) = 0;

		//! Rebinds frame buffer object.
		/** @returns whether FBO is bound or was successfully rebound.
		*/
        virtual bool rebindRevalidate() = 0;

		//! Gets attachment accessible at the given index.
		/** @param ix Given index.
		@returns Attached at given index object or NULL if nothing is bound there.
		*/
        virtual const IRenderable* getAttachment(const size_t &ix) const = 0;

    protected:
};


} // end namespace video
} // end namespace irr

#endif


