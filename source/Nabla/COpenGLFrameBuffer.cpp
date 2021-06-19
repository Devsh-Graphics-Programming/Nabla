// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "BuildConfigOptions.h"

#ifdef _NBL_COMPILE_WITH_OPENGL_
#include "COpenGLFrameBuffer.h"
#include "COpenGLDriver.h"

#include "nbl_os.h"
#include "..\..\src\nbl\video\COpenGLFramebuffer.h"



namespace nbl
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
//	_NBL_DEBUG_BREAK_IF(true);
	return false;
}

//! constructor
COpenGLFrameBuffer::COpenGLFrameBuffer(COpenGLDriver* driver) : Driver(driver), fboSize(), frameBuffer(0)
{
#ifdef _NBL_DEBUG
	setDebugName("COpenGLFrameBuffer");
#endif
    Driver->extGlCreateFramebuffers(1,&frameBuffer);
	std::fill(cachedMipLayer,cachedMipLayer+EFAP_MAX_ATTACHMENTS,-1);
}

//! destructor
COpenGLFrameBuffer::~COpenGLFrameBuffer()
{
    if (frameBuffer)
        Driver->extGlDeleteFramebuffers(1,&frameBuffer);
}

// TODO : merge two below and redo with smart pointers
bool COpenGLFrameBuffer::attach(E_FBO_ATTACHMENT_POINT attachmenPoint, core::smart_refctd_ptr<IGPUImageView>&& tex, uint32_t mipMapLayer, int32_t layer)
{
	if (!frameBuffer||attachmenPoint>=EFAP_MAX_ATTACHMENTS)
		return false;

    if (tex && asset::isBlockCompressionFormat(tex->getCreationParameters().format))
        return false;

    GLenum attachment = GL_INVALID_ENUM;
    //! TODO: Need additional validation here for matching texture formats
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
	auto* glTex = static_cast<COpenGLImageView*>(tex.get());
    if (glTex)
    {
		if (glTex->getCreationParameters().image->getCreationParameters().samples!=IGPUImage::ESCF_1_BIT)
			mipMapLayer = 0;

        if (layer>=0)
            Driver->extGlNamedFramebufferTextureLayer(frameBuffer,attachment,glTex->getOpenGLName(),glTex->getOpenGLTextureType(),mipMapLayer,layer);
        else
            Driver->extGlNamedFramebufferTexture(frameBuffer,attachment,glTex->getOpenGLName(),mipMapLayer);

		cachedMipLayer[attachmenPoint] = mipMapLayer;
    }
    else
	{
        Driver->extGlNamedFramebufferTexture(frameBuffer,attachment,0,0);
		cachedMipLayer[attachmenPoint] = ~0u;
	}

    attachments[attachmenPoint] = core::smart_refctd_ptr<video::COpenGLImageView>(glTex);


    bool noAttachments = true;
    //
    size_t enabledBufferCnt = 0;
    GLenum drawBuffers[EFAP_MAX_ATTACHMENTS-EFAP_COLOR_ATTACHMENT0] = {0}; //GL_NONE
    for (size_t i=0; i<EFAP_MAX_ATTACHMENTS; i++)
    {
        if (!attachments[i])
            continue;

		auto tmp = attachments[i]->getCreationParameters().image->getMipSize(cachedMipLayer[i]);
		if (noAttachments)
		{
			fboSize.Width = tmp.x;
			fboSize.Height = tmp.y;
		}
		else
		{
			fboSize.Width = std::min(fboSize.Width, tmp.x);
			fboSize.Height = std::min(fboSize.Height, tmp.y);
		}
        noAttachments = false;

        if (i>=EFAP_COLOR_ATTACHMENT0)
        {
            drawBuffers[i-EFAP_COLOR_ATTACHMENT0] = GL_COLOR_ATTACHMENT0+i-EFAP_COLOR_ATTACHMENT0;
            enabledBufferCnt = i;
        }
    }


	// TODO: empty framebuffers
    if (noAttachments)
    {
		os::Printer::log("FBO has no attachments!");
        return false;
    }

    if (!checkFBOStatus(frameBuffer,Driver))
    {
        os::Printer::log("FBO incomplete");
        return false;
    }
    if (enabledBufferCnt)
        enabledBufferCnt += 1-EFAP_COLOR_ATTACHMENT0;
    COpenGLExtensionHandler::extGlNamedFramebufferDrawBuffers(frameBuffer, enabledBufferCnt, drawBuffers);

	return true;
}

}
}

#endif
