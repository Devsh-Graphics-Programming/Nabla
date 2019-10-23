#ifndef __IRR_C_OPENGL_TEXTURE_VIEW_H_INCLUDED__
#define __IRR_C_OPENGL_TEXTURE_VIEW_H_INCLUDED__

#include "irr/video/IGPUTextureView.h"
#include "COpenGLExtensionHandler.h"

namespace irr {
namespace video
{

class COpenGLTextureView : public IGPUTextureView
{
protected:
    virtual ~COpenGLTextureView() = default;

public:
    virtual GLuint getOpenGLName() const = 0;
    virtual GLenum getInternalFormat() const = 0;
};

}}

#endif