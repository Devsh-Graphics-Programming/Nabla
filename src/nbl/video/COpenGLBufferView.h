// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_OPENGL_BUFFER_VIEW_H_INCLUDED__
#define __NBL_VIDEO_C_OPENGL_BUFFER_VIEW_H_INCLUDED__

#include "nbl/video/IGPUBufferView.h"

#include "COpenGLCommon.h"
#include "COpenGLBuffer.h"


#ifdef _NBL_COMPILE_WITH_OPENGL_
namespace nbl
{
namespace video
{

class COpenGLBufferView : public IGPUBufferView
{
	public:
		COpenGLBufferView(core::smart_refctd_ptr<COpenGLBuffer>&& _buffer, asset::E_FORMAT _format, size_t _offset = 0ull, size_t _size = COpenGLBufferView::whole_buffer) :
			IGPUBufferView(std::move(_buffer), _format, _offset, _size), m_textureName(0u), m_GLformat(GL_INVALID_ENUM), m_textureSize(0u)
		{
			COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_BUFFER, 1, &m_textureName);
			m_GLformat = getSizedOpenGLFormatFromOurFormat(m_format);

			if (m_offset==0u && m_size==m_buffer->getSize())
				COpenGLExtensionHandler::extGlTextureBuffer(m_textureName, m_GLformat, static_cast<COpenGLBuffer*>(m_buffer.get())->getOpenGLName());
			else
				COpenGLExtensionHandler::extGlTextureBufferRange(m_textureName, m_GLformat, static_cast<COpenGLBuffer*>(m_buffer.get())->getOpenGLName(), m_offset, m_size);

			m_textureSize = m_size / asset::getTexelOrBlockBytesize(m_format);
		}

		GLuint getOpenGLName() const { return m_textureName; }
		GLenum getInternalFormat() const { return m_GLformat; }

	protected:
		virtual ~COpenGLBufferView()
		{
			if (m_textureName)
				glDeleteTextures(1u,&m_textureName);
		}

	private:
		GLuint m_textureName;
		GLenum m_GLformat;
		uint32_t m_textureSize;
};

}
}
#endif


#endif