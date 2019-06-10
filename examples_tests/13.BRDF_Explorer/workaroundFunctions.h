#include <cstdint>
#include <CEGUI/Size.h>
namespace CEGUI
{
class OpenGL3Renderer;
}
namespace irr
{
namespace video
{
class IVirtualTexture;
}
}

::CEGUI::Texture& irrTex2ceguiTex(uint32_t _GLname, const ::CEGUI::Sizef& _sz, const std::string& _name, CEGUI::OpenGL3Renderer& _renderer);

uint32_t getTextureGLname(irr::video::IVirtualTexture* _texture);