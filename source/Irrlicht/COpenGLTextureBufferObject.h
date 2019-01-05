
// Copyright (C) 2016 Mateusz "DevSH" Kielan
// This file is part of the "BAW Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_TEXTURE_BUFFER_OBJECT_H_INCLUDED__
#define __C_OPEN_GL_TEXTURE_BUFFER_OBJECT_H_INCLUDED__

#include "IrrCompileConfig.h"
#include "ITextureBufferObject.h"
#include "COpenGLBuffer.h"
#include "COpenGLTexture.h"
#include "COpenGLExtensionHandler.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_



namespace irr
{
namespace video
{



//! get the amount of Bits per Pixel of the given color format
inline uint32_t getBitsPerPixelFromGLenum(const GLenum& format)
{
    switch(format)
    {
        case GL_R8:
        case GL_R8I:
        case GL_R8UI:
            return 8;
        case GL_R16:
        case GL_R16F:
        case GL_R16I:
        case GL_R16UI:
            return 16;
        case GL_R32F:
        case GL_R32I:
        case GL_R32UI:
            return 32;
        case GL_RG8:
        case GL_RG8I:
        case GL_RG8UI:
            return 16;
        case GL_RG16:
        case GL_RG16F:
        case GL_RG16I:
        case GL_RG16UI:
            return 32;
        case GL_RG32F:
        case GL_RG32I:
        case GL_RG32UI:
            return 64;
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
            return 96;
        case GL_RGBA8:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGB9_E5:
            return 32;
        case GL_RGBA16:
        case GL_RGBA16F:
        case GL_RGBA16I:
        case GL_RGBA16UI:
            return 64;
        case GL_RGBA32F:
        case GL_RGBA32I:
        case GL_RGBA32UI:
            return 128;
        default:
            return 0;
    }
}

class COpenGLTextureBufferObject : public COpenGLTexture, public ITextureBufferObject
{
    protected:
        virtual ~COpenGLTextureBufferObject()
        {
            if (leakTracker)
                leakTracker->deregisterObj(this);

            if (currentBuffer)
                currentBuffer->drop();
        }

    public:
        //! constructor
        COpenGLTextureBufferObject(COpenGLBuffer* buffer, E_TEXURE_BUFFER_OBJECT_FORMAT format, const size_t& offset=0, const size_t& length=0, core::LeakDebugger* dbgr=NULL)
                                    : COpenGLTexture(GL_TEXTURE_BUFFER), lastValidated(0), currentBuffer(NULL), Offset(0), Length(0), leakTracker(dbgr), InternalFormat(GL_INVALID_ENUM), ColorFormat(asset::EF_UNKNOWN), TextureSize(0)
        {
            if (leakTracker)
                leakTracker->registerObj(this);

            bind(buffer,format,offset,length);
        }

        //! returns driver type of texture (=the driver, that created it)
        virtual E_DRIVER_TYPE getDriverType() const {return EDT_OPENGL;}

        //!
        virtual const uint32_t* getSize() const {return &TextureSize;}

        //! Returns size of the texture.
        virtual E_DIMENSION_COUNT getDimensionality() const {return EDC_ONE;}

        //! returns color format of texture
        virtual asset::E_FORMAT getColorFormat() const {return ColorFormat;}

        //!
        virtual E_VIRTUAL_TEXTURE_TYPE getVirtualTextureType() const {return EVTT_BUFFER_OBJECT;}

        //! returns pitch of texture (in bytes)
        virtual uint64_t getByteSize() const {return video::getBitsPerPixelFromFormat(ColorFormat)*uint64_t(TextureSize)/8ull;}


        //! returns the opengl texture type
        virtual GLenum getOpenGLTextureType() const {return GL_TEXTURE_BUFFER;}


        virtual bool bind(IGPUBuffer* buffer, E_TEXURE_BUFFER_OBJECT_FORMAT format=ETBOF_COUNT, const size_t& offset=0, const size_t& length=0)
        {
            COpenGLBuffer* glbuf = static_cast<COpenGLBuffer*>(buffer);
            if (!glbuf)
                return false;

            GLenum internalFormat;
            if (format>=ETBOF_COUNT)
                internalFormat = InternalFormat;
            else
                internalFormat = getOpenGLFormatFromTBOFormat(format);

            //nothing is changed
            if (currentBuffer==glbuf&&Offset==offset&&Length==length&&InternalFormat==internalFormat)
                return true;


            InternalFormat = internalFormat;
            ColorFormat = COpenGLTexture::getColorFormatFromSizedOpenGLFormat(InternalFormat);
            if (glbuf)
            {
                Offset = offset;
                if (Offset>=glbuf->getSize())
                    return false;

                glbuf->grab();
                if (length)
                    Length = length;
                else
                    Length = glbuf->getSize();

                if (Offset==0&&Length==glbuf->getSize())
                    COpenGLExtensionHandler::extGlTextureBuffer(TextureName,InternalFormat,glbuf->getOpenGLName());
                else
                    COpenGLExtensionHandler::extGlTextureBufferRange(TextureName,InternalFormat,glbuf->getOpenGLName(),Offset,Length);
                lastValidated = glbuf->getLastTimeReallocated();
                TextureSize = Length*8/irr::video::getBitsPerPixelFromGLenum(InternalFormat);
            }
            else
            {
                COpenGLExtensionHandler::extGlTextureBuffer(TextureName,InternalFormat,0);
                Length = 0;
                Offset = 0;
                TextureSize = 0;
            }


            if (currentBuffer)
                currentBuffer->drop();

            currentBuffer = glbuf;
            return true;
        }

        virtual IGPUBuffer* getBoundBuffer() { return currentBuffer; }
        virtual const IGPUBuffer* getBoundBuffer() const { return currentBuffer; }

        virtual bool rebindRevalidate()
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


        static inline GLint getOpenGLFormatFromTBOFormat(const E_TEXURE_BUFFER_OBJECT_FORMAT &format)
        {
            switch(format)
            {
                case ETBOF_R8:
                    return GL_R8;
                    break;
                case ETBOF_R16:
                    return GL_R16;
                    break;
                case ETBOF_R16F:
                    return GL_R16F;
                    break;
                case ETBOF_R32F:
                    return GL_R32F;
                    break;
                case ETBOF_R8I:
                    return GL_R8I;
                    break;
                case ETBOF_R16I:
                    return GL_R16I;
                    break;
                case ETBOF_R32I:
                    return GL_R32I;
                    break;
                case ETBOF_R8UI:
                    return GL_R8UI;
                    break;
                case ETBOF_R16UI:
                    return GL_R16UI;
                    break;
                case ETBOF_R32UI:
                    return GL_R32UI;
                    break;
                case ETBOF_RG8:
                    return GL_RG8;
                    break;
                case ETBOF_RG16:
                    return GL_RG16;
                    break;
                case ETBOF_RG16F:
                    return GL_RG16F;
                    break;
                case ETBOF_RG32F:
                    return GL_RG32F;
                    break;
                case ETBOF_RG8I:
                    return GL_RG8I;
                    break;
                case ETBOF_RG16I:
                    return GL_RG16I;
                    break;
                case ETBOF_RG32I:
                    return GL_RG32I;
                    break;
                case ETBOF_RG8UI:
                    return GL_RG8UI;
                    break;
                case ETBOF_RG16UI:
                    return GL_RG16UI;
                    break;
                case ETBOF_RG32UI:
                    return GL_RG32UI;
                    break;
                case ETBOF_RGB32F:
                    return GL_RGB32F;
                    break;
                case ETBOF_RGB32I:
                    return GL_RGB32I;
                    break;
                case ETBOF_RGB32UI:
                    return GL_RGB32UI;
                    break;
                case ETBOF_RGBA8:
                    return GL_RGBA8;
                    break;
                case ETBOF_RGBA16:
                    return GL_RGBA16;
                    break;
                case ETBOF_RGBA16F:
                    return GL_RGBA16F;
                    break;
                case ETBOF_RGBA32F:
                    return GL_RGBA32F;
                    break;
                case ETBOF_RGBA8I:
                    return GL_RGBA8I;
                    break;
                case ETBOF_RGBA16I:
                    return GL_RGBA16I;
                    break;
                case ETBOF_RGBA32I:
                    return GL_RGBA32I;
                    break;
                case ETBOF_RGBA8UI:
                    return GL_RGBA8UI;
                    break;
                case ETBOF_RGBA16UI:
                    return GL_RGBA16UI;
                    break;
                case ETBOF_RGBA32UI:
                    return GL_RGBA32UI;
                    break;
                default:
                    os::Printer::log("Unsupported texture format", ELL_ERROR);
                    break;
            }

            return GL_INVALID_ENUM;
        }

    protected:
        uint64_t lastValidated;
        COpenGLBuffer* currentBuffer;
        size_t Length,Offset;

        core::LeakDebugger* leakTracker;

        uint32_t TextureSize;

        GLint InternalFormat;
        asset::E_FORMAT ColorFormat;
};

} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_