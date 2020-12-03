// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <cstdint>
#include <CEGUI/Size.h>
namespace CEGUI
{
class OpenGL3Renderer;
}
namespace nbl
{
namespace video
{
class IVirtualTexture;
}
}

::CEGUI::Texture& irrTex2ceguiTex(uint32_t _GLname, const ::CEGUI::Sizef& _sz, const std::string& _name, CEGUI::OpenGL3Renderer& _renderer);

uint32_t getTextureGLname(nbl::video::IVirtualTexture* _texture);