#include <CEGUI/RendererModules/OpenGL/Texture.h>
#include <CEGUI/RendererModules/OpenGL/GL3Renderer.h>

::CEGUI::Texture& irrTex2ceguiTex(uint32_t _GLname, const ::CEGUI::Sizef& _sz, const std::string& _name, CEGUI::OpenGL3Renderer& _renderer)
{
    ::CEGUI::Texture& ceguiTexture = _renderer.createTexture(
        _name,
        _GLname,
        _sz
    );

    return ceguiTexture;
}