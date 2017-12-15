// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __C_OPEN_GL_RENDER_BUFFER_H_INCLUDED__
#define __C_OPEN_GL_RENDER_BUFFER_H_INCLUDED__

#include "IRenderBuffer.h"
#include "EDriverTypes.h"
#include "IrrCompileConfig.h"

#ifdef _IRR_COMPILE_WITH_OPENGL_
#include "COpenGLExtensionHandler.h"

namespace irr
{
namespace video
{

class COpenGLDriver;
//! OpenGL texture.
class COpenGLRenderBuffer : public IRenderBuffer
{
    public:
        //! constructor
        COpenGLRenderBuffer(GLenum internalFormat, core::dimension2du size);

        //! Returns size of the texture.
        virtual const core::dimension2d<uint32_t>& getSize() const {return RenderBufferSize;}
        virtual core::dimension2du getRenderableSize() const {return RenderBufferSize;}

        //! returns driver type of texture (=the driver, that created it)
        virtual const E_DRIVER_TYPE getDriverType() const {return EDT_OPENGL;}

        //! return open gl texture name
        const GLuint& getOpenGLName() const {return RenderBufferName;}
        GLuint* getOpenGLNamePtr() {return &RenderBufferName;}

        GLint getOpenGLInternalFormat() const {return InternalFormat;}

        virtual void resize(const core::dimension2du &newSize);

        const uint64_t& hasOpenGLNameChanged() const {return RenderBufferNameHasChanged;}


    protected:
        COpenGLRenderBuffer(GLenum internalFormat, core::dimension2du size, const float DONT_CREATE_NORMAL_RBUFFER)
            : RenderBufferSize(size), InternalFormat(internalFormat), RenderBufferName(0), RenderBufferNameHasChanged(0)
        {
        }

        //! destructor
        virtual ~COpenGLRenderBuffer();

        core::dimension2d<uint32_t> RenderBufferSize;
        GLint InternalFormat;

        GLuint RenderBufferName;
        uint64_t RenderBufferNameHasChanged;
};



class COpenGLMultisampleRenderBuffer : public COpenGLRenderBuffer
{
public:
	//! constructor
    COpenGLMultisampleRenderBuffer(GLenum internalFormat, core::dimension2du size, uint32_t sampleCount);


	virtual void resize(const core::dimension2du &newSize);

	virtual int32_t getSampleCount() const {return static_cast<uint32_t>(SampleCount);}

private:
    uint32_t SampleCount;
};


} // end namespace video
} // end namespace irr

#endif
#endif // _IRR_COMPILE_WITH_OPENGL_


