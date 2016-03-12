// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_FRAMEBUFFER_H_INCLUDED__
#define __C_OPEN_GL_FRAMEBUFFER_H_INCLUDED__

#include "IFrameBuffer.h"
#include "IrrCompileConfig.h"
#include "COpenGLTexture.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_


namespace irr
{
namespace video
{


class COpenGLDriver;
//! OpenGL texture.
class COpenGLFrameBuffer : public IFrameBuffer
{
public:

	//! constructor
	COpenGLFrameBuffer(COpenGLDriver* driver);

	//! destructor
	virtual ~COpenGLFrameBuffer();

    virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, ITexture* tex, const uint32_t &mipMapLayer=0, const int32_t &layer=-1);

    virtual bool attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, IRenderBuffer* rbf);

	virtual bool rebindRevalidate();

	const GLuint getOpenGLName() const {return frameBuffer;}

    virtual const IRenderable* getAttachment(const size_t &ix) const {return ix<EFAP_MAX_ATTACHMENTS ? attachments[ix]:NULL;}

protected:
	COpenGLDriver* Driver;

    GLuint frameBuffer;
    uint64_t lastValidated;
    IRenderable* attachments[EFAP_MAX_ATTACHMENTS];
    GLint cachedLevel[EFAP_MAX_ATTACHMENTS];
    GLint cachedLayer[EFAP_MAX_ATTACHMENTS];
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_


