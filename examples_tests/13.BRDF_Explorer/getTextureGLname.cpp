#include <COpenGL2DTexture.h>

uint32_t getTextureGLname(irr::video::ITexture* _texture)
{
    if (!_texture)
        return 0u;
    return static_cast<irr::video::COpenGL2DTexture*>(_texture)->getOpenGLName();
}