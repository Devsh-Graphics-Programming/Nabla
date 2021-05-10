// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_C_OPEN_GL_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_OPEN_GL_FRAMEBUFFER_H_INCLUDED__

#include "BuildConfigOptions.h"
#include "IFrameBuffer.h"

#include "nbl/video/COpenGLImageView.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_


namespace nbl
{
namespace video
{


class COpenGLDriver;
//! OpenGL texture.
class COpenGLFrameBuffer final : public IFrameBuffer
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
} // end namespace nbl

#endif
#endif


