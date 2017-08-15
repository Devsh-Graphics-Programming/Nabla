
// Copyright (C) 2016 Mateusz "DevSH" Kielan
// This file is part of the "BAW Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_TEXTURE_BUFFER_OBJECT_H_INCLUDED__
#define __C_OPEN_GL_TEXTURE_BUFFER_OBJECT_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "COpenGLBuffer.h"
#include "COpenGLTexture.h"
#include "IImage.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{

//! OpenGL texture.
class COpenGLTextureBufferObject : public COpenGLTexture
{
public:

	//! constructor
	COpenGLTextureBufferObject(COpenGLBuffer* buffer, GLenum internalFormat, video::COpenGLDriver* driver, const size_t& offset=0, const size_t& length=0, const io::path& name="", core::LeakDebugger* dbgr=NULL)
                                : COpenGLTexture(name,driver),  lastValidated(0), currentBuffer(NULL), Offset(0), Length(0), leakTracker(dbgr)
	{
	    if (leakTracker)
            leakTracker->registerObj(this);

	    HasMipMaps = false;
	    MipLevelsStored = 1;
        COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_BUFFER,1,&TextureName);

        InternalFormat = internalFormat;

        bind(buffer,internalFormat,offset,length);
	}

	virtual ~COpenGLTextureBufferObject()
	{
	    if (leakTracker)
            leakTracker->deregisterObj(this);

	    if (currentBuffer)
            currentBuffer->drop();
	}

	//! Returns size of the texture.
	virtual const E_DIMENSION_COUNT getDimensionality() const {return EDC_ONE;}

    virtual const E_TEXTURE_TYPE getTextureType() const {return ETT_TEXTURE_BUFFER;}


    virtual core::dimension2du getRenderableSize() const {return core::dimension2du(0,0);}

	//! returns pitch of texture (in bytes)
	virtual uint32_t getPitch() const {return Length;}

	GLint getOpenGLInternalFormat() const {return InternalFormat;}

	//! returns the opengl texture type
	virtual GLenum getOpenGLTextureType() const {return GL_TEXTURE_BUFFER;}

	//! return whether this texture has mipmaps
	virtual bool hasMipMaps() const {return false;}

	//! Regenerates the mip map levels of the texture.
	/** Useful after locking and modifying the texture
	\param mipmapData Pointer to raw mipmap data, including all necessary mip levels, in the same format as the main texture image. If not set the mipmaps are derived from the main image. */
	virtual void regenerateMipMapLevels() {}


    virtual bool updateSubRegion(const ECOLOR_FORMAT &inDataColorFormat, const void* data, const uint32_t* minimum, const uint32_t* maximum, int32_t mipmap=0) {return false;}
    virtual bool resize(const uint32_t* size, const uint32_t& mipLevels=0) {return false;}

    inline bool bind(COpenGLBuffer* buffer, GLenum internalFormat, const size_t& offset=0, const size_t& length=0)
    {
        if (currentBuffer==buffer&&Offset==offset&&Length==length&&InternalFormat==internalFormat)
            return true;


        InternalFormat = internalFormat;
        if (buffer)
        {
            Offset = offset;
            if (Offset>=buffer->getSize())
                return false;

            buffer->grab();
            if (length)
                Length = length;
            else
                Length = buffer->getSize();

            if (Offset==0&&Length==buffer->getSize())
                COpenGLExtensionHandler::extGlTextureBuffer(TextureName,InternalFormat,buffer->getOpenGLName());
            else
                COpenGLExtensionHandler::extGlTextureBufferRange(TextureName,InternalFormat,buffer->getOpenGLName(),Offset,Length);
            lastValidated = buffer->getLastTimeReallocated();
            TextureSize[0] = Length*8/irr::video::getBitsPerPixelFromGLenum(InternalFormat);
        }
        else
        {
            COpenGLExtensionHandler::extGlTextureBuffer(TextureName,InternalFormat,0);
            Length = 0;
            Offset = 0;
            TextureSize[0] = 0;
        }


        if (currentBuffer)
            currentBuffer->drop();

        currentBuffer = buffer;
        return true;
    }

    inline bool rebindRevalidate()
    {
        if (!currentBuffer)
            return true;

        uint64_t revalidateStamp = currentBuffer->getLastTimeReallocated();
        if (revalidateStamp>lastValidated)
        {
            if (Offset==0&&Length==currentBuffer->getSize())
                COpenGLExtensionHandler::extGlTextureBuffer(TextureName,InternalFormat,currentBuffer->getOpenGLName());
            else
                COpenGLExtensionHandler::extGlTextureBufferRange(TextureName,InternalFormat,currentBuffer->getOpenGLName(),Offset,Length);
            lastValidated = revalidateStamp;
        }

        return true;
    }

protected:
    uint64_t lastValidated;
    COpenGLBuffer* currentBuffer;
    size_t Length,Offset;

    core::LeakDebugger* leakTracker;
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_

