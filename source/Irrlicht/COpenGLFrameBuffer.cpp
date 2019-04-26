// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
#include "COpenGLFrameBuffer.h"
#include "COpenGLDriver.h"
#include "COpenGLMultisampleTexture.h"

#include "os.h"



namespace irr
{
namespace video
{


bool checkFBOStatus(const GLuint &fbo, COpenGLDriver* Driver)
{
	GLenum status = Driver->extGlCheckNamedFramebufferStatus(fbo,GL_FRAMEBUFFER);

	switch (status)
	{
		//Our FBO is perfect, return true
		case GL_FRAMEBUFFER_COMPLETE:
			return true;

		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
			os::Printer::log("FBO has invalid read buffer", ELL_ERROR);
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
			os::Printer::log("FBO has invalid draw buffer", ELL_ERROR);
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
			os::Printer::log("FBO has one or several incomplete image attachments", ELL_ERROR);
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
			os::Printer::log("FBO missing an image attachment", ELL_ERROR);
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
			os::Printer::log("FBO wrong multisample setup", ELL_ERROR);
			break;

		case GL_FRAMEBUFFER_UNSUPPORTED:
			os::Printer::log("FBO format unsupported", ELL_ERROR);
			break;

		case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
			os::Printer::log("Some FBO attachment is layered, and some other is not layered, or if all populated color attachments are not from textures of the same target", ELL_ERROR);
			break;

		default:
			break;
	}
	os::Printer::log("FBO error", ELL_ERROR);
//	_IRR_DEBUG_BREAK_IF(true);
	return false;
}

//! constructor
COpenGLFrameBuffer::COpenGLFrameBuffer(COpenGLDriver* driver)
  : frameBuffer(0), Driver(driver), forceRevalidate(true), lastValidated(0)
{
#ifdef _IRR_DEBUG
	setDebugName("COpenGLFrameBuffer");
#endif
    Driver->extGlCreateFramebuffers(1,&frameBuffer);

    memset(attachments,0,sizeof(void*)*EFAP_MAX_ATTACHMENTS);
    memset(cachedLevel,-1,sizeof(GLint)*EFAP_MAX_ATTACHMENTS);
    memset(cachedLayer,-1,sizeof(GLint)*EFAP_MAX_ATTACHMENTS);
}

//! destructor
COpenGLFrameBuffer::~COpenGLFrameBuffer()
{
    if (frameBuffer)
        Driver->extGlDeleteFramebuffers(1,&frameBuffer);

    for (size_t i=0; i<EFAP_MAX_ATTACHMENTS; i++)
    {
        if (attachments[i])
            attachments[i]->drop();
    }
}

bool COpenGLFrameBuffer::attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, ITexture* tex, const uint32_t &mipMapLayer, const int32_t &layer)
{
	if (!frameBuffer||attachmenPoint>=EFAP_MAX_ATTACHMENTS)
		return false;

    COpenGLFilterableTexture* glTex = static_cast<COpenGLFilterableTexture*>(tex);
    if (tex&&COpenGLTexture::isInternalFormatCompressed(glTex->getOpenGLInternalFormat()))
        return false;

    GLenum attachment = GL_INVALID_ENUM;
    //! Need additional validation here for matching texture formats
    switch (attachmenPoint)
    {
        case EFAP_DEPTH_ATTACHMENT:
            attachment = GL_DEPTH_ATTACHMENT;
            break;
        case EFAP_STENCIL_ATTACHMENT:
            attachment = GL_STENCIL_ATTACHMENT;
            break;
        case EFAP_DEPTH_STENCIL_ATTACHMENT:
            attachment = GL_DEPTH_STENCIL_ATTACHMENT;
            break;
        default:
            attachment = GL_COLOR_ATTACHMENT0+(attachmenPoint-EFAP_COLOR_ATTACHMENT0);
            break;
    }
    /*
    If <texture> is the name of a three-dimensional texture, cube map texture,
    one- or two-dimensional array texture, cube map array texture, or two-
    dimensional multisample array texture, the texture level attached to the
    framebuffer attachment point is an array of images, and the framebuffer
    attachment is considered layered.
    */
    if (glTex)
    {
        if (layer>=0)
        {
            Driver->extGlNamedFramebufferTextureLayer(frameBuffer,attachment,glTex->getOpenGLName(),glTex->getOpenGLTextureType(),mipMapLayer,layer);
            cachedLayer[attachmenPoint] = layer;
        }
        else
        {
            Driver->extGlNamedFramebufferTexture(frameBuffer,attachment,glTex->getOpenGLName(),mipMapLayer);
            cachedLayer[attachmenPoint] = -1;
        }
        cachedLevel[attachmenPoint] = mipMapLayer;
    }
    else
    {
        Driver->extGlNamedFramebufferTexture(frameBuffer,attachment,0,0);
        cachedLevel[attachmenPoint] = -1;
        cachedLayer[attachmenPoint] = -1;
    }



    if (attachments[attachmenPoint])
        attachments[attachmenPoint]->drop();
    attachments[attachmenPoint] = tex;
	if (tex)
        tex->grab(); // grab the depth buffer, not the RTT

    forceRevalidate = true;

	return true;
}

bool COpenGLFrameBuffer::attach(const E_FBO_ATTACHMENT_POINT &attachmenPoint, IMultisampleTexture* tex, const int32_t &layer)
{
	if (!frameBuffer||attachmenPoint>=EFAP_MAX_ATTACHMENTS)
		return false;

    COpenGLMultisampleTexture* glTex = static_cast<COpenGLMultisampleTexture*>(tex);
    if (!tex)
        return false;

    GLenum attachment = GL_INVALID_ENUM;
    //! Need additional validation here for matching texture formats
    switch (attachmenPoint)
    {
        case EFAP_DEPTH_ATTACHMENT:
            attachment = GL_DEPTH_ATTACHMENT;
            break;
        case EFAP_STENCIL_ATTACHMENT:
            attachment = GL_STENCIL_ATTACHMENT;
            break;
        case EFAP_DEPTH_STENCIL_ATTACHMENT:
            attachment = GL_DEPTH_STENCIL_ATTACHMENT;
            break;
        default:
            attachment = GL_COLOR_ATTACHMENT0+(attachmenPoint-EFAP_COLOR_ATTACHMENT0);
            break;
    }
    /*
    If <texture> is the name of a three-dimensional texture, cube map texture,
    one- or two-dimensional array texture, cube map array texture, or two-
    dimensional multisample array texture, the texture level attached to the
    framebuffer attachment point is an array of images, and the framebuffer
    attachment is considered layered.
    */
    cachedLevel[attachmenPoint] = -1;
    if (glTex)
    {
        if (layer>=0)
        {
            Driver->extGlNamedFramebufferTextureLayer(frameBuffer,attachment,glTex->getOpenGLName(), glTex->getOpenGLTextureType(),0,layer);
            cachedLayer[attachmenPoint] = layer;
        }
        else
        {
            Driver->extGlNamedFramebufferTexture(frameBuffer,attachment,glTex->getOpenGLName(),0);
            cachedLayer[attachmenPoint] = -1;
        }
    }
    else
    {
        Driver->extGlNamedFramebufferTexture(frameBuffer,attachment,0,0);
        cachedLayer[attachmenPoint] = -1;
    }



    if (attachments[attachmenPoint])
        attachments[attachmenPoint]->drop();
    attachments[attachmenPoint] = tex;
	if (tex)
        tex->grab(); // grab the depth buffer, not the RTT

    forceRevalidate = true;

	return true;
}

bool COpenGLFrameBuffer::rebindRevalidate()
{
    bool noAttachments = true;
    bool revalidate = forceRevalidate;
    uint64_t highestRevalidationStamp = lastValidated;
    //
    size_t enabledBufferCnt = 0;
    GLenum drawBuffers[EFAP_MAX_ATTACHMENTS-EFAP_COLOR_ATTACHMENT0] = {0}; //GL_NONE
    for (size_t i=0; i<EFAP_MAX_ATTACHMENTS; i++)
    {
        if (!attachments[i])
            continue;
        noAttachments = false;

        if (i>=EFAP_COLOR_ATTACHMENT0)
        {
            drawBuffers[i-EFAP_COLOR_ATTACHMENT0] = GL_COLOR_ATTACHMENT0+i-EFAP_COLOR_ATTACHMENT0;
            enabledBufferCnt = i;
        }

        uint64_t revalidationStamp = 0;
        switch (attachments[i]->getVirtualTextureType())
        {
            case IVirtualTexture::EVTT_OPAQUE_FILTERABLE:
                {
                    ITexture* typeTex = static_cast<ITexture*>(attachments[i]);
                    revalidationStamp = dynamic_cast<COpenGLTexture*>(typeTex)->hasOpenGLNameChanged();
                    if (revalidationStamp>lastValidated)
                        attach((E_FBO_ATTACHMENT_POINT)i,typeTex,cachedLevel[i],cachedLayer[i]);
                }
                break;
            case IVirtualTexture::EVTT_2D_MULTISAMPLE:
                {
                    IMultisampleTexture* typeTex = static_cast<IMultisampleTexture*>(attachments[i]);
                    revalidationStamp = dynamic_cast<COpenGLTexture*>(typeTex)->hasOpenGLNameChanged();
                    if (revalidationStamp>lastValidated)
                        attach((E_FBO_ATTACHMENT_POINT)i,typeTex,cachedLayer[i]);
                }
                break;
            default:
                os::Printer::log("WTF are you trying to render into!?");
                break;
        }
        if (revalidationStamp>lastValidated)
        {
            if (revalidationStamp>highestRevalidationStamp)
                highestRevalidationStamp = revalidationStamp;
            revalidate = true;
        }
    }

    if (noAttachments)
    {
		os::Printer::log("FBO has no attachments!");
        return false;
    }

    if (revalidate)
    {
        if (!checkFBOStatus(frameBuffer,Driver))
        {
            os::Printer::log("FBO incomplete");
            return false;
        }
        forceRevalidate = false;
        lastValidated = highestRevalidationStamp;
        if (enabledBufferCnt)
            enabledBufferCnt += 1-EFAP_COLOR_ATTACHMENT0;
        COpenGLExtensionHandler::extGlNamedFramebufferDrawBuffers(frameBuffer, enabledBufferCnt, drawBuffers);
    }

	return true;
}



}
}

#endif
