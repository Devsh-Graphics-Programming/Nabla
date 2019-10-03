#ifndef __IRR_C_OPENGL_BUFFER_VIEW_H_INCLUDED__
#define __IRR_C_OPENGL_BUFFER_VIEW_H_INCLUDED__

#include "irr/video/IGPUBufferView.h"
#include "COpenGLBuffer.h"
#include "COpenGLTexture.h"
#include "COpenGLExtensionHandler.h"

namespace irr {
namespace video
{

class COpenGLBufferView : public IGPUBufferView
{
public:
    COpenGLBufferView(core::smart_refctd_ptr<COpenGLBuffer>&& _buffer, asset::E_FORMAT _format, size_t _offset = 0ull, size_t _size = COpenGLBufferView::whole_buffer) :
        IGPUBufferView(std::move(_buffer), _format, _offset, _size), m_textureName(0u), m_GLformat(GL_INVALID_ENUM), m_textureSize(0u)
    {
        COpenGLExtensionHandler::extGlCreateTextures(GL_TEXTURE_BUFFER, 1, &m_textureName);
        m_GLformat = COpenGLTexture::getOpenGLFormatAndParametersFromColorFormat(m_format);

        if (m_offset==0u && m_size==m_buffer->getSize())
            COpenGLExtensionHandler::extGlTextureBuffer(m_textureName, m_GLformat, _buffer->getOpenGLName());
        else
            COpenGLExtensionHandler::extGlTextureBufferRange(m_textureName, m_GLformat, _buffer->getOpenGLName(), m_offset, m_size);

        m_textureSize = m_size / asset::getTexelOrBlockBytesize(m_format);
    }

    GLuint getOpenGLName() const { return m_textureName; }
    GLenum getInternalFormat() const { return m_GLformat; }

private:
    GLuint m_textureName;
    GLenum m_GLformat;
    uint32_t m_textureSize;
};

}}

#endif