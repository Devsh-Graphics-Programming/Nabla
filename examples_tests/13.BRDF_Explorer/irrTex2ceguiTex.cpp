// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

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