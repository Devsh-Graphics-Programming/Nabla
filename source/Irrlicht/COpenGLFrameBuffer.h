// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_FRAMEBUFFER_H_INCLUDED__
#define __C_OPEN_GL_FRAMEBUFFER_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "IFrameBuffer.h"

#include "irr/video/COpenGLImageView.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_


namespace irr
{
namespace video
{


class COpenGLDriver;
//! OpenGL texture.
class COpenGLFrameBuffer : public IFrameBuffer
{
    protected:
        //! destructor
        virtual ~COpenGLFrameBuffer();

    public:
        //! constructor
        COpenGLFrameBuffer(COpenGLDriver* driver);

        virtual bool attach(E_FBO_ATTACHMENT_POINT attachmenPoint, core::smart_refctd_ptr<IGPUImageView>&& tex, uint32_t mipMapLayer, int32_t layer) override;

		virtual const core::dimension2du& getSize() const override { return fboSize; }

        const GLuint& getOpenGLName() const {return frameBuffer;}

        virtual const IGPUImageView* getAttachment(uint32_t ix) const override {return ix<EFAP_MAX_ATTACHMENTS ? attachments[ix].get():nullptr;}

    protected:
        COpenGLDriver*								Driver;

		core::dimension2du							fboSize;
        GLuint										frameBuffer;
        uint16_t									cachedMipLayer[EFAP_MAX_ATTACHMENTS];
        core::smart_refctd_ptr<COpenGLImageView>	attachments[EFAP_MAX_ATTACHMENTS];
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_


